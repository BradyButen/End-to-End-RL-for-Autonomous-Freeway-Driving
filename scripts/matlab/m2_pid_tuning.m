%% M2 PID tuner on the MetaDrive kinematic bicycle
% Opens an interactive tuner: move a gain slider (or type a value) and every response redraws.
%
% Plant: kinematic bicycle [X Y psi v], inputs [delta a], fitted to MetaDrive
%        (vehicle_model static_default, physics step 0.02 s). Valid for v < 80 km/h, |a_y| <~ 3.5 m/s^2.
%        Optional first-order steering lag tau (P.tau, adjustable in the tuner) adds a 5th state
%        delta_act. Keep it on (0.12 s) when judging stability: the lag is small for trajectory
%        prediction but large at feedback crossover frequencies.
% Controller: M2's PID as implemented in src/handoff/m2.py
%   u_s  = PID_psi(psi_ref - psi) + PID_d(d_ref - d)        reference at the ego's projection on the path
%   u_tb = PID_v(v_ref(t+Ts) - v) + table^-1(a_ref(t+Ts))    reference by time (feed-forward optional)
%   Discrete, per control step: I = running sum of errors (clamped to +-integral_limit),
%   D = e_k - e_{k-1}. Gains are in command units (u_s, u_tb in [-1, 1]), i.e. the same numbers as
%   configs/m2.yaml. "Print gains" writes a YAML block you can paste back.
%
% Tabs in the tuner
%   Lateral step   linear, straight driving at 10-40 m/s (P.lin_speeds): 1 m step in d_ref, u_s, loop Bode
%   Speed step     linear, 1 m/s step in v_ref with the throttle gain and the light-brake gain
%   Circle         nonlinear closed loop, circle with sinusoidal speed
%   Spiral         nonlinear closed loop, radius shrinking with arc length
% Requires Control System Toolbox (linear tabs). Tested only by inspection; R2020b or newer.
%
% Single-loop alternative: MATLAB's pidTuner works on the speed loop with the per-step gains converted:
%   Cv = pid(Kp, Ki/Ts, Kd*Ts, 0, Ts, 'IFormula', 'BackwardEuler', 'DFormula', 'BackwardEuler');
%   pidTuner(c2d(tf(2.909, [1 0]), 0.1, 'zoh'), Cv)      % convert back: Ki_step = Ki*Ts, Kd_step = Kd/Ts

clear; close all; clc;

%% 1. Starting gains and parameters ------------------------------------------------------
% Current m2.yaml values were heading [1.7 0.01 3.5], lateral [0.3 0.002 0.05], speed [0.6 0.02 0].
gains.heading = [1.238, 0.00476, 0.0];     % [Kp Ki Kd] on psi_ref - psi (rad)      -> u_s
gains.lateral = [0.3636, 0.000824, 0.0112]; % [Kp Ki Kd] on d_ref - d (m)           -> u_s
gains.speed   = [0.36, 0.00272, 0.0];      % [Kp Ki Kd] on v_ref - v (m/s)          -> u_tb
gains.integral_limit    = 5.0;        % clamp on each integral (sum of errors)
gains.speed_feedforward = true;       % add table^-1(a_ref) to u_tb
gains.init_offset       = 0.5;        % ego lateral offset from the reference at t = 0 (m, left +)

% Vehicle (MetaDrive DefaultVehicle, static_default)
P.lf = 1.05234;  P.lr = 1.4166;  P.L = P.lf + P.lr;
P.delta_max = deg2rad(40);            % wheel angle at u_s = 1
P.v_max = 80/3.6;                     % MetaDrive cuts engine force above this
P.Ts = 0.1;                           % control period (s)
P.h  = 0.02;                          % plant integration step (= MetaDrive physics step)
% Actuation table [u_tb, accel m/s^2], configs/base.yaml (measured at 15 m/s)
P.act = [-1.0 -15.805; -0.6 -11.703; -0.3 -7.936; -0.15 -4.091; -0.05 -1.364;
          0.0  -0.364;  0.02  0.058;  1.0  2.909];
P.g_throttle = 2.909;                 % small-signal accel per unit command, throttle side
P.g_brake    = 27.27;                 % small-signal accel per unit command, light braking (|u| < 0.3)
% m/s, operating points for the linear lateral analysis. 22.2 = 80 km/h, where MetaDrive cuts the engine
% (the current v_cap); speeds above it are extrapolation of the fitted model (fit covered 8-24 m/s).
P.lin_speeds = [10 15 20 22.2 25 30 35 40];
% Steering lag of the PLANT only (s): first-order lag from the commanded to the effective wheel angle,
% measured in MetaDrive (yaw-rate time constant 0.11-0.12 s, docs/DESIGN.md D-12). 0 = pure kinematic plant.
% It matters for feedback design: at 5-12 rad/s crossover it costs 30-55 deg of phase.
P.tau = 0.12;

%% 2. Reference trajectories --------------------------------------------------------------
% Circle with sinusoidal speed
circ.T  = 60;  circ.R = 100;  circ.v0 = 15;  circ.dv = 3;  circ.Tv = 20;
refs.circle.T = circ.T;
refs.circle.name = sprintf('Circle, R = %g m, v = %g +- %g m/s', circ.R, circ.v0, circ.dv);
refs.circle.ref = make_reference(circ.T, ...
    @(t) circ.v0 + circ.dv*sin(2*pi*t/circ.Tv), ...
    @(s) ones(size(s))/circ.R);

% Spiral: radius shrinking linearly with arc length, constant speed
spir.T  = 60;  spir.R0 = 150;  spir.R1 = 45;  spir.v = 12;
S_end = spir.v*spir.T;
refs.spiral.T = spir.T;
refs.spiral.name = sprintf('Spiral, R %g -> %g m, v = %g m/s', spir.R0, spir.R1, spir.v);
refs.spiral.ref = make_reference(spir.T, ...
    @(t) spir.v*ones(size(t)), ...
    @(s) 1./(spir.R0 + (spir.R1 - spir.R0)*min(s/S_end, 1)));

%% 3. Open the tuner ----------------------------------------------------------------------
launch_tuner(gains, P, refs);


%% ===================================== tuner UI ========================================

function launch_tuner(gains0, P, refs)
fig = uifigure('Name', 'M2 PID tuner', 'Position', [40 40 1500 880]);
main = uigridlayout(fig, [1 2]);
main.ColumnWidth = {370, '1x'};

% ---------------- left: controls
left = uigridlayout(main, [18 3]);
left.ColumnWidth = {40, '1x', 85};
left.RowHeight = {22, 28, 28, 28, 22, 28, 28, 28, 22, 28, 28, 28, 28, 28, 28, 28, 32, '1x'};
left.RowSpacing = 4;

loops = {'heading', 'Heading PID:  (psi_ref - psi) -> u_s';
         'lateral', 'Lateral PID:  (d_ref - d) -> u_s';
         'speed',   'Speed PID:  (v_ref - v) -> u_tb'};
kn = {'Kp', 'Ki', 'Kd'};
H = struct();
r = 1;
for li = 1:size(loops, 1)
    name = loops{li, 1};
    lab = uilabel(left, 'Text', loops{li, 2}, 'FontWeight', 'bold');
    lab.Layout.Row = r;  lab.Layout.Column = [1 3];
    r = r + 1;
    for gi = 1:3
        v0 = gains0.(name)(gi);
        hi = 4*v0;  if hi == 0, hi = 1; end
        l = uilabel(left, 'Text', kn{gi});
        l.Layout.Row = r;  l.Layout.Column = 1;
        sl = uislider(left, 'Limits', [0 hi], 'Value', v0, 'MajorTicks', [], 'MinorTicks', []);
        sl.Layout.Row = r;  sl.Layout.Column = 2;
        ed = uieditfield(left, 'numeric', 'Value', v0, 'Limits', [0 Inf], 'ValueDisplayFormat', '%.4g');
        ed.Layout.Row = r;  ed.Layout.Column = 3;
        sl.ValueChangingFcn = @(s, e) set(ed, 'Value', e.Value);
        sl.ValueChangedFcn  = @(s, e) slider_changed(fig, s, ed);
        ed.ValueChangedFcn  = @(s, e) edit_changed(fig, s, sl);
        H.(name)(gi).sl = sl;
        H.(name)(gi).ed = ed;
        r = r + 1;
    end
end
l = uilabel(left, 'Text', 'Integral limit (sum of errors)');  l.Layout.Row = r;  l.Layout.Column = [1 2];
H.intlim = uieditfield(left, 'numeric', 'Value', gains0.integral_limit, 'Limits', [0 Inf], ...
    'ValueChangedFcn', @(s, e) update(fig));
H.intlim.Layout.Row = r;  H.intlim.Layout.Column = 3;  r = r + 1;
H.ff = uicheckbox(left, 'Text', 'Speed feed-forward  table^-1(a_ref)', 'Value', gains0.speed_feedforward, ...
    'ValueChangedFcn', @(s, e) update(fig));
H.ff.Layout.Row = r;  H.ff.Layout.Column = [1 3];  r = r + 1;
l = uilabel(left, 'Text', 'Initial lateral offset (m)');  l.Layout.Row = r;  l.Layout.Column = [1 2];
H.off = uieditfield(left, 'numeric', 'Value', gains0.init_offset, ...
    'ValueChangedFcn', @(s, e) update(fig));
H.off.Layout.Row = r;  H.off.Layout.Column = 3;  r = r + 1;
l = uilabel(left, 'Text', 'Plant steering lag tau (s, 0 = off)');  l.Layout.Row = r;  l.Layout.Column = [1 2];
H.tau = uieditfield(left, 'numeric', 'Value', P.tau, 'Limits', [0 Inf], ...
    'ValueChangedFcn', @(s, e) update(fig));
H.tau.Layout.Row = r;  H.tau.Layout.Column = 3;  r = r + 1;

btns = uigridlayout(left, [1 2], 'Padding', [0 0 0 0]);
btns.Layout.Row = r;  btns.Layout.Column = [1 3];  r = r + 1;
uibutton(btns, 'Text', 'Reset to starting gains', 'ButtonPushedFcn', @(s, e) reset_gains(fig));
uibutton(btns, 'Text', 'Print gains (YAML)', 'ButtonPushedFcn', @(s, e) print_gains(fig));

H.txt = uitextarea(left, 'Editable', 'off', 'FontName', 'Courier New', 'FontSize', 11);
H.txt.Layout.Row = r;  H.txt.Layout.Column = [1 3];

% ---------------- right: plots
tg = uitabgroup(main);
A = struct();

t1 = uitab(tg, 'Title', 'Lateral step');
g1 = uigridlayout(t1, [2 2]);
A.lat_d  = new_axes(g1, 1, 1, '1 m step in d_{ref} (\psi_{ref} = 0)', 't (s)', 'd (m)');
A.lat_u  = new_axes(g1, 2, 1, 'Steering command', 't (s)', 'u_s');
A.lat_mg = new_axes(g1, 1, 2, 'Loop gain |L| (broken at u_s)', '\omega (rad/s)', 'dB');
A.lat_ph = new_axes(g1, 2, 2, 'Loop phase', '\omega (rad/s)', 'deg');

t2 = uitab(tg, 'Title', 'Speed step');
g2 = uigridlayout(t2, [2 1]);
A.v_v = new_axes(g2, 1, 1, '1 m/s step in v_{ref} (feed-forward not included)', 't (s)', 'v - v_0 (m/s)');
A.v_u = new_axes(g2, 2, 1, 'Throttle/brake command', 't (s)', 'u_{tb}');

for nm = {'circle', 'spiral'}
    k = nm{1};
    tb = uitab(tg, 'Title', [upper(k(1)) k(2:end)]);
    gg = uigridlayout(tb, [3 2]);
    gg.ColumnWidth = {'1.2x', '1x'};
    A.(k).xy = new_axes(gg, [1 3], 1, refs.(k).name, 'X (m)', 'Y (m)');
    A.(k).err = new_axes(gg, 1, 2, 'Tracking errors', 't (s)', 'error');
    A.(k).v   = new_axes(gg, 2, 2, 'Speed', 't (s)', 'v (m/s)');
    A.(k).u   = new_axes(gg, 3, 2, 'Commands', 't (s)', 'command');
end

fig.UserData = struct('H', H, 'A', A, 'P', P, 'refs', refs, 'gains0', gains0);
update(fig);
end

function ax = new_axes(parent, row, col, ttl, xl, yl)
ax = uiaxes(parent);
ax.Layout.Row = row;  ax.Layout.Column = col;
title(ax, ttl);  xlabel(ax, xl);  ylabel(ax, yl);
grid(ax, 'on');  box(ax, 'on');
end

function slider_changed(fig, sl, ed)
ed.Value = sl.Value;
update(fig);
end

function edit_changed(fig, ed, sl)
v = ed.Value;
if v > sl.Limits(2)
    sl.Limits = [0, 2*v];
end
sl.Value = v;
update(fig);
end

function g = read_gains(D)
H = D.H;
for name = {'heading', 'lateral', 'speed'}
    n = name{1};
    g.(n) = arrayfun(@(h) h.ed.Value, H.(n));
end
g.integral_limit = H.intlim.Value;
g.speed_feedforward = H.ff.Value;
g.init_offset = H.off.Value;
end

function reset_gains(fig)
D = fig.UserData;
for name = {'heading', 'lateral', 'speed'}
    n = name{1};
    for gi = 1:3
        v0 = D.gains0.(n)(gi);
        hi = 4*v0;  if hi == 0, hi = 1; end
        D.H.(n)(gi).sl.Limits = [0 hi];
        D.H.(n)(gi).sl.Value = v0;
        D.H.(n)(gi).ed.Value = v0;
    end
end
D.H.intlim.Value = D.gains0.integral_limit;
D.H.ff.Value = D.gains0.speed_feedforward;
D.H.off.Value = D.gains0.init_offset;
D.H.tau.Value = D.P.tau;
update(fig);
end

function print_gains(fig)
g = read_gains(fig.UserData);
f = @(k) sprintf('[%.4g, %.4g, %.4g]', k(1), k(2), k(3));
tf_str = {'false', 'true'};
fprintf('\n  pid:\n');
fprintf('    heading: %s\n', f(g.heading));
fprintf('    lateral: %s\n', f(g.lateral));
fprintf('    speed: %s\n', f(g.speed));
fprintf('    integral_limit: %.4g\n', g.integral_limit);
fprintf('    speed_feedforward: %s\n', tf_str{g.speed_feedforward + 1});
end

function update(fig)
D = fig.UserData;
P = D.P;  A = D.A;
P.tau = D.H.tau.Value;                  % plant lag from the UI
fig.Pointer = 'watch';  drawnow;
g = read_gains(D);
lines_out = {sprintf('Plant steering lag tau = %.3g s%s', P.tau, repmat(' (pure kinematic)', 1, P.tau == 0)), ''};

% ---------- linear lateral
lat = lateral_linear(g, P);
cols = lines(numel(lat));
axs = [A.lat_d, A.lat_u, A.lat_mg, A.lat_ph];
for ax = axs, cla(ax); hold(ax, 'on'); end
lines_out{end + 1} = 'LATERAL STEP (linear, 1 m)';
lines_out{end + 1} = '  v   rise  settle  OS%  max|us|  GM dB  PM deg';
for i = 1:numel(lat)
    s = lat(i);
    nm = sprintf('%g m/s', s.v);
    stairs(A.lat_d, s.t, s.d, 'Color', cols(i, :), 'LineWidth', 1.3, 'DisplayName', nm);
    stairs(A.lat_u, s.t, s.u, 'Color', cols(i, :), 'LineWidth', 1.3, 'DisplayName', nm);
    plot(A.lat_mg, s.w, s.mag_db, 'Color', cols(i, :), 'LineWidth', 1.3, 'DisplayName', nm);
    plot(A.lat_ph, s.w, s.phase, 'Color', cols(i, :), 'LineWidth', 1.3, 'DisplayName', nm);
    stab = '';  if ~s.stable, stab = '  UNSTABLE'; end
    lines_out{end + 1} = sprintf('  %-3g %5.2f %6.2f %5.1f %7.2f %6.1f %7.1f%s', s.v, s.rise, s.settle, ...
        s.os, max(abs(s.u)), s.gm_db, s.pm, stab);
end
if any([lat.drift_mode])
    lines_out{end + 1} = '  note: both heading and lateral Ki > 0 -> hidden z = 1 mode';
    lines_out{end + 1} = '        (integrators can drift against each other; see circle)';
end
yline(A.lat_d, 1, 'k:', 'HandleVisibility', 'off');
yline(A.lat_u, [-1 1], 'r:', 'HandleVisibility', 'off');
yline(A.lat_mg, 0, 'k:', 'HandleVisibility', 'off');
yline(A.lat_ph, -180, 'k:', 'HandleVisibility', 'off');
A.lat_mg.XScale = 'log';  A.lat_ph.XScale = 'log';
legend(A.lat_d, 'Location', 'southeast');

% ---------- linear speed
spd = speed_linear(g, P);
cla(A.v_v); hold(A.v_v, 'on');  cla(A.v_u); hold(A.v_u, 'on');
lines_out{end + 1} = '';
lines_out{end + 1} = 'SPEED STEP (linear, 1 m/s)';
lines_out{end + 1} = '  regime    rise  settle  OS%  max|utb| GM dB  PM deg';
for i = 1:numel(spd)
    s = spd(i);
    nm = sprintf('%s (g = %.1f)', s.name, s.g);
    stairs(A.v_v, s.t, s.v, 'LineWidth', 1.3, 'DisplayName', nm);
    stairs(A.v_u, s.t, s.u, 'LineWidth', 1.3, 'DisplayName', nm);
    stab = '';  if ~s.stable, stab = '  UNSTABLE'; end
    lines_out{end + 1} = sprintf('  %-8s %5.2f %6.2f %5.1f %7.2f %6.1f %7.1f%s', s.name, s.rise, s.settle, ...
        s.os, max(abs(s.u)), s.gm_db, s.pm, stab);
end
yline(A.v_v, 1, 'k:', 'HandleVisibility', 'off');
yline(A.v_u, [-1 1], 'r:', 'HandleVisibility', 'off');
legend(A.v_v, 'Location', 'southeast');

% ---------- nonlinear trajectories
for nm = {'circle', 'spiral'}
    k = nm{1};
    R = D.refs.(k);
    res = simulate_m2(R.ref, g, P, [g.init_offset, 0], R.T);
    plot_traj(A.(k), res, R.ref);
    lines_out{end + 1} = '';
    lines_out = [lines_out, traj_metrics(res, R.ref, upper(k))]; %#ok<AGROW>
end

D.H.txt.Value = lines_out(:);
fig.Pointer = 'arrow';
drawnow;
end

function plot_traj(A, res, ref)
for ax = [A.xy, A.err, A.v, A.u], cla(ax); hold(ax, 'on'); end
plot(A.xy, ref.x, ref.y, 'k--', 'DisplayName', 'reference');
plot(A.xy, res.x(:, 1), res.x(:, 2), 'LineWidth', 1.3, 'DisplayName', 'ego');
axis(A.xy, 'equal');  legend(A.xy, 'Location', 'best');
plot(A.err, res.t, res.ed, 'DisplayName', 'cross-track (m)');
plot(A.err, res.t, rad2deg(res.epsi), 'DisplayName', 'heading (deg)');
plot(A.err, res.t, res.ev, 'DisplayName', 'speed (m/s)');
legend(A.err, 'Location', 'best');
plot(A.v, res.t, res.vref, 'k--', 'DisplayName', 'v_{ref}');
plot(A.v, res.t, res.x(:, 4), 'DisplayName', 'v');
legend(A.v, 'Location', 'best');
plot(A.u, res.t, res.us, 'DisplayName', 'u_s');
plot(A.u, res.t, res.utb, 'DisplayName', 'u_{tb}');
yline(A.u, [-1 1], 'r:', 'HandleVisibility', 'off');
legend(A.u, 'Location', 'best');
end

function out = traj_metrics(res, ref, name)
rms = @(e) sqrt(mean(e.^2));
skip = res.t >= 5;   % after the initial-offset transient
out = {sprintf('%s (t >= 5 s)', name)};
ay_ref = max(abs(ref.v.^2.*ref.kappa));
if ay_ref > 3.5
    out{end + 1} = sprintf('  WARNING ref a_y %.1f m/s^2 > 3.5 (model range)', ay_ref);
end
out{end + 1} = sprintf('  cross-track RMS %.3f  max %.3f m', rms(res.ed(skip)), max(abs(res.ed(skip))));
out{end + 1} = sprintf('  heading     RMS %.2f  max %.2f deg', rad2deg(rms(res.epsi(skip))), ...
    rad2deg(max(abs(res.epsi(skip)))));
out{end + 1} = sprintf('  speed       RMS %.3f  max %.3f m/s', rms(res.ev(skip)), max(abs(res.ev(skip))));
out{end + 1} = sprintf('  along-track RMS %.2f  max %.2f m (no fb)', rms(res.es(skip)), max(abs(res.es(skip))));
out{end + 1} = sprintf('  saturated   u_s %.1f%%  u_tb %.1f%%', 100*mean(abs(res.us) >= 1), ...
    100*mean(abs(res.utb) >= 1));
end


%% ================================== linear analysis ====================================

function out = lateral_linear(g, P)
% Small-angle kinematic bicycle along a straight path, states [d; psi]:
%   d_dot = v*psi + v*(lr/L)*delta,   psi_dot = (v/L)*delta,   delta = delta_max*u_s
% ZOH at Ts. Controller u_s = C_d*(d_ref - d) + C_psi*(psi_ref - psi). Integral clamp and command
% clipping are ignored (check max|u_s| for saturation).
Ts = P.Ts;
Cd = pid_tf(g.lateral, Ts);
Cpsi = pid_tf(g.heading, Ts);
K = ss([Cd, Cpsi]);                                 % inputs [e_d; e_psi] -> u_s
w = logspace(-2, log10(pi/Ts), 400);
out = struct([]);
for i = 1:numel(P.lin_speeds)
    v = P.lin_speeds(i);
    if P.tau > 0
        % states [d; psi; delta_act], delta_act_dot = (delta_max*u_s - delta_act)/tau
        Ac = [0 v v*P.lr/P.L; 0 0 v/P.L; 0 0 -1/P.tau];
        Bc = [0; 0; P.delta_max/P.tau];
        Cc = [1 0 0; 0 1 0];
    else
        Ac = [0 v; 0 0];
        Bc = P.delta_max*[v*P.lr/P.L; v/P.L];
        Cc = eye(2);
    end
    G = c2d(ss(Ac, Bc, Cc, 0), Ts, 'zoh');          % u_s -> [d; psi]
    Try = feedback(G*K, eye(2));                    % [d_ref; psi_ref] -> [d; psi]
    Uru = feedback(K, G);                           % [d_ref; psi_ref] -> u_s
    Lu = K*G;                                       % loop broken at u_s
    [y, t] = step(Try(1, 1), 15);
    u = step(Uru(1, 1), t);
    si = stepinfo(y, t, 1.0);
    [mag, ph] = bode(Lu, w);
    [gm, pm] = margin(Lu);
    out(i).v = v;  out(i).t = t;  out(i).d = y;  out(i).u = u;
    out(i).w = w(:);  out(i).mag_db = 20*log10(squeeze(mag));  out(i).phase = squeeze(ph);
    out(i).rise = si.RiseTime;  out(i).settle = si.SettlingTime;  out(i).os = si.Overshoot;
    out(i).gm_db = 20*log10(gm);  out(i).pm = pm;
    % Two integrators (heading and lateral) driving one input leave a closed-loop pole at z = 1 that is
    % unobservable in d, psi and u_s (the integrators can drift against each other). Judge stability on
    % the minimal realisation and report that mode separately.
    out(i).stable = isstable(minreal(Try, [], false));
    out(i).drift_mode = max(abs(pole(Try))) > 1 - 1e-6;
end
end

function out = speed_linear(g, P)
% v_dot = gain*u_tb, gain = throttle (2.909) or light brake (27.3) small-signal slope.
Ts = P.Ts;
Cv = pid_tf(g.speed, Ts);
regimes = {'throttle', P.g_throttle; 'brake', P.g_brake};
out = struct([]);
for i = 1:size(regimes, 1)
    gain = regimes{i, 2};
    Gv = c2d(tf(gain, [1 0]), Ts, 'zoh');
    Tv = feedback(Cv*Gv, 1);
    Uv = feedback(Cv, Gv);
    [y, t] = step(Tv, 30);                          % long enough for a small Ki to settle
    u = step(Uv, t);
    si = stepinfo(y, t, 1.0);
    [gm, pm] = margin(Cv*Gv);
    out(i).name = regimes{i, 1};  out(i).g = gain;
    out(i).t = t;  out(i).v = y;  out(i).u = u;
    out(i).rise = si.RiseTime;  out(i).settle = si.SettlingTime;  out(i).os = si.Overshoot;
    out(i).gm_db = 20*log10(gm);  out(i).pm = pm;  out(i).stable = isstable(Tv);
end
end

function C = pid_tf(k, Ts)
% Per-step PID of m2_pid.PID: C(z) = Kp + Ki*z/(z-1) + Kd*(z-1)/z (integral includes current error).
z = tf('z', Ts);
C = k(1) + k(2)*z/(z - 1) + k(3)*(z - 1)/z;
end


%% ================================ nonlinear simulation ==================================

function ref = make_reference(T, vfun, kappafun)
% Reference from a speed profile v(t) and a curvature profile kappa(s), sampled finely.
% Extends 3 s past T so look-ups near the end stay inside the reference.
dt = 0.01;
ref.t = (0:dt:T + 3)';
ref.v = vfun(ref.t);
ref.a = gradient(ref.v, dt);
ref.s = cumtrapz(ref.t, ref.v);
ref.kappa = kappafun(ref.s);
ref.psi = cumtrapz(ref.s, ref.kappa);             % continuous (unwrapped) heading
ref.x = cumtrapz(ref.s, cos(ref.psi));
ref.y = cumtrapz(ref.s, sin(ref.psi));
end

function res = simulate_m2(ref, gains, P, off, T)
% Closed loop: M2 PID at Ts, kinematic bicycle integrated with RK4 at h (commands held).
% off = [lateral offset m (left +), heading offset rad] of the ego at t = 0.
N = round(T/P.Ts);
nsub = round(P.Ts/P.h);
% 5th state: effective wheel angle (follows the command through the plant lag P.tau), starting at the
% reference curvature's steady-state angle.
x = [ref.x(1) - sin(ref.psi(1))*off(1);
     ref.y(1) + cos(ref.psi(1))*off(1);
     ref.psi(1) + off(2);
     ref.v(1);
     atan(P.L*ref.kappa(1))];
pid_h = pid_init();  pid_d = pid_init();  pid_v = pid_init();
lim = gains.integral_limit;
idx = 1;
L = struct('t', zeros(N, 1), 'x', zeros(N, 4), 'ed', zeros(N, 1), 'epsi', zeros(N, 1), ...
    'ev', zeros(N, 1), 'es', zeros(N, 1), 'us', zeros(N, 1), 'utb', zeros(N, 1), 'vref', zeros(N, 1));
for k = 1:N
    t = (k - 1)*P.Ts;
    % --- lateral reference: projection of the ego onto the path (spatial look-up)
    [xr, yr, psir, sproj, idx] = project(ref, x(1), x(2), idx);
    d_ego = -sin(psir)*(x(1) - xr) + cos(psir)*(x(2) - yr);   % left positive
    ed = 0 - d_ego;
    epsi = wrap(psir - x(3));
    [uh, pid_h] = pid_step(pid_h, gains.heading, epsi, lim);
    [ud, pid_d] = pid_step(pid_d, gains.lateral, ed, lim);
    us = uh + ud;
    % --- longitudinal reference: by time, one control step ahead
    vref = interp1(ref.t, ref.v, t + P.Ts);
    aref = interp1(ref.t, ref.a, t + P.Ts);
    ev = vref - x(4);
    [uv, pid_v] = pid_step(pid_v, gains.speed, ev, lim);
    utb = uv;
    if gains.speed_feedforward
        % clipped to the table range, like ActuationMap.accel_to_command (np.interp)
        aref_c = min(max(aref, P.act(1, 2)), P.act(end, 2));
        utb = utb + interp1(P.act(:, 2), P.act(:, 1), aref_c);
    end
    us = min(max(us, -1), 1);
    utb = min(max(utb, -1), 1);
    % --- log errors and state at the decision instant
    L.t(k) = t;  L.x(k, :) = x(1:4)';  L.ed(k) = ed;  L.epsi(k) = epsi;  L.ev(k) = ev;
    L.es(k) = interp1(ref.t, ref.s, t) - sproj;   % + = ego behind the reference in time
    L.us(k) = us;  L.utb(k) = utb;  L.vref(k) = interp1(ref.t, ref.v, t);
    % --- plant
    delta_cmd = P.delta_max*us;
    if P.tau == 0
        x(5) = delta_cmd;                         % no lag: wheel angle follows the command instantly
    end
    for j = 1:nsub
        a = accel(utb, x(4), P);
        x = rk4(@(xx) bicycle(xx, delta_cmd, a, P), x, P.h);
        x(4) = max(x(4), 0);
    end
end
res = L;
end

function dx = bicycle(x, delta_cmd, a, P)
% x = [X Y psi v delta_act]; the bicycle steers with delta_act.
delta = x(5);
beta = atan(P.lr/P.L*tan(delta));
if P.tau > 0
    ddelta = (delta_cmd - delta)/P.tau;
else
    ddelta = 0;
end
dx = [x(4)*cos(x(3) + beta);
      x(4)*sin(x(3) + beta);
      x(4)/P.lr*sin(beta);
      a;
      ddelta];
end

function a = accel(utb, v, P)
% MetaDrive actuation: table look-up; engine force is cut above v_max (coast instead).
if v > P.v_max && utb > 0
    utb = 0;
end
a = interp1(P.act(:, 1), P.act(:, 2), utb);
end

function x = rk4(f, x, h)
k1 = f(x);
k2 = f(x + h/2*k1);
k3 = f(x + h/2*k2);
k4 = f(x + h*k3);
x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end

function s = pid_init()
s.I = 0;
s.prev = NaN;
end

function [u, s] = pid_step(s, k, e, lim)
% Matches handoff.rl.methods.m2_pid.PID: integral += e (clamped), derivative = e - e_prev (0 first step).
s.I = min(max(s.I + e, -lim), lim);
if isnan(s.prev)
    de = 0;
else
    de = e - s.prev;
end
s.prev = e;
u = k(1)*e + k(2)*s.I + k(3)*de;
end

function [xr, yr, psir, sproj, idx] = project(ref, x, y, idx0)
% Nearest point on the sampled reference, searched forward from the previous index, then refined
% by projecting onto the neighbouring segment.
n = numel(ref.x);
w = idx0:min(idx0 + 400, n);
[~, j] = min((ref.x(w) - x).^2 + (ref.y(w) - y).^2);
i = w(j);
if i == n, i = n - 1; end
if i > 1 && seg_lambda(ref, i, x, y) <= 0
    i = i - 1;                                   % point projects onto the previous segment
end
lam = min(max(seg_lambda(ref, i, x, y), 0), 1);
xr = ref.x(i) + lam*(ref.x(i + 1) - ref.x(i));
yr = ref.y(i) + lam*(ref.y(i + 1) - ref.y(i));
psir = ref.psi(i) + lam*(ref.psi(i + 1) - ref.psi(i));
sproj = ref.s(i) + lam*(ref.s(i + 1) - ref.s(i));
idx = i;
end

function lam = seg_lambda(ref, i, x, y)
dx = ref.x(i + 1) - ref.x(i);  dy = ref.y(i + 1) - ref.y(i);
lam = ((x - ref.x(i))*dx + (y - ref.y(i))*dy)/max(dx^2 + dy^2, eps);
end

function a = wrap(a)
a = mod(a + pi, 2*pi) - pi;
end
