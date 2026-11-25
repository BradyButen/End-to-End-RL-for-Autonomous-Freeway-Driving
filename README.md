# Where Should Learning Hand Off to Control?

Self-driving stacks split the work between learned components and classical controllers, but where that line
should sit is usually decided by convention, not measured. This project measures it. On one simulated freeway
([MetaDrive](https://github.com/metadriverse/metadrive)) with slower 35–45 km/h traffic, the same reinforcement
learner (PPO, same observation, same reward) is given progressively less authority:

| method | the RL decides | the controller handles |
|---|---|---|
| **M1** end-to-end | lateral and longitudinal acceleration, every 0.1 s | nothing |
| **M2** RL + PID | lane (keep / left / right) and target speed, every 0.5 s | tracking a smooth lane-change and speed reference |
| **M3** RL + MPC | lane and maximum speed, every 1 s | trajectory, braking and collision avoidance in a model-predictive controller |
| **B0** baseline | — | everything (MetaDrive's rule-based IDM driver) |

The goal for every method is to reach the end of the road as fast as possible without crashing, which means
passing traffic rather than following it. M1 can also be warm-started from a behaviour-cloned copy of human drives.

Setup, commands, how each method works and the trained models: **[docs/GUIDE.md](docs/GUIDE.md)**.
