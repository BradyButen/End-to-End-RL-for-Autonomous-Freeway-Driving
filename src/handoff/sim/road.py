"""RoadFrame: one road-aligned Frenet frame for the whole route (spec §5.1).

Conventions (fixed here, used everywhere):

* ``s`` — arc length (m) along the reference line from the route start.
* ``d`` — signed lateral offset (m) from the reference line, **positive to the left** of the
  driving direction (counter-clockwise from the tangent, the usual right-handed convention).
* Reference line — centreline of the ego-direction lane with MetaDrive index 0 (the leftmost
  forward lane, adjacent to the median), stitched across blocks so ``s`` is continuous.
* ``lane_centers[i]`` — ``d`` of the centre of MetaDrive lane ``i``; for the 3-lane, 3.5 m road
  these are ``[0.0, -3.5, -7.0]``. ``d_max``/``d_min`` are the left (median) and right road edges.

The class is pure numpy so controllers/tests can use it without MetaDrive; ``from_metadrive``
builds it from a MetaDrive map + navigation route.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class RoadFrame:
    def __init__(self, points: np.ndarray, lane_centers: list[float], d_min: float, d_max: float,
                 lane_width: float):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("points must be (N>=2, 2)")
        # drop duplicate consecutive points
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6])
        pts = pts[keep]
        self.points = pts
        seg = np.diff(pts, axis=0)
        self._seg = seg
        self._seg_len = np.linalg.norm(seg, axis=1)
        self._seg_dir = seg / self._seg_len[:, None]
        self.s_points = np.concatenate([[0.0], np.cumsum(self._seg_len)])
        self._seg_heading = np.arctan2(seg[:, 1], seg[:, 0])
        self.length = float(self.s_points[-1])
        self.lane_centers = [float(c) for c in lane_centers]
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.lane_width = float(lane_width)

    # ------------------------------------------------------------------ construction
    @classmethod
    def straight(cls, length: float = 1000.0, lane_num: int = 3, lane_width: float = 3.5,
                 origin=(0.0, 0.0), heading: float = 0.0, spacing: float = 5.0) -> RoadFrame:
        """Synthetic straight road (tests / controller standalone runs). Lane 0 on the reference line,
        further lanes to the right (negative d)."""
        n = max(2, int(np.ceil(length / spacing)) + 1)
        s = np.linspace(0.0, length, n)
        o = np.asarray(origin, float)
        pts = o + np.stack([s * np.cos(heading), s * np.sin(heading)], axis=1)
        centers = [-i * lane_width for i in range(lane_num)]
        return cls(pts, centers, d_min=centers[-1] - lane_width / 2, d_max=lane_width / 2,
                   lane_width=lane_width)

    @classmethod
    def from_metadrive(cls, road_network, checkpoints: list[str], lane_num: int, lane_width: float,
                       spacing: float = 2.0) -> RoadFrame:
        """Stitch lane-0 centrelines of consecutive roads along ``checkpoints`` (navigation route)."""
        pts = []
        for a, b in zip(checkpoints[:-1], checkpoints[1:]):
            lane = road_network.graph[a][b][0]
            n = max(2, int(np.ceil(lane.length / spacing)) + 1)
            for lon in np.linspace(0.0, lane.length, n):
                pts.append(np.asarray(lane.position(lon, 0.0), float)[:2])
        frame = cls(np.array(pts), [0.0], 0.0, 0.0, lane_width)  # provisional, to measure lane offsets
        lane0 = road_network.graph[checkpoints[0]][checkpoints[1]]
        centers = []
        for i in range(lane_num):
            ln = lane0[i]
            p = np.asarray(ln.position(ln.length / 2, 0.0), float)[:2]
            centers.append(frame.to_frenet(p[0], p[1])[1])
        left = max(centers) + lane_width / 2
        right = min(centers) - lane_width / 2
        return cls(frame.points, centers, d_min=right, d_max=left, lane_width=lane_width)

    # ------------------------------------------------------------------ queries
    def to_frenet(self, x: float, y: float) -> tuple[float, float]:
        p = np.array([x, y], dtype=float)
        rel = p - self.points[:-1]
        t = np.einsum("ij,ij->i", rel, self._seg_dir)
        t = np.clip(t, 0.0, self._seg_len)
        proj = self.points[:-1] + self._seg_dir * t[:, None]
        dist2 = np.sum((p - proj) ** 2, axis=1)
        i = int(np.argmin(dist2))
        # extrapolate beyond the ends along the first/last segment
        if i == 0 or i == len(self._seg_len) - 1:
            t_raw = float(np.dot(rel[i], self._seg_dir[i]))
            if (i == 0 and t_raw < 0) or (i == len(self._seg_len) - 1 and t_raw > self._seg_len[i]):
                t[i] = t_raw
        s = self.s_points[i] + t[i]
        dvec = rel[i] - self._seg_dir[i] * t[i]
        cross = self._seg_dir[i, 0] * dvec[1] - self._seg_dir[i, 1] * dvec[0]
        return float(s), float(cross)

    def _locate(self, s):
        s = np.asarray(s, dtype=float)
        i = np.clip(np.searchsorted(self.s_points, s, side="right") - 1, 0, len(self._seg_len) - 1)
        return i, s - self.s_points[i]

    def to_world(self, s, d):
        """Vectorised: scalars -> (x, y) floats; arrays -> (x array, y array)."""
        i, t = self._locate(s)
        d = np.asarray(d, dtype=float)
        base = self.points[i] + self._seg_dir[i] * np.asarray(t)[..., None]
        normal = np.stack([-self._seg_dir[i, 1], self._seg_dir[i, 0]], axis=-1)
        p = base + normal * d[..., None]
        if p.ndim == 1:
            return float(p[0]), float(p[1])
        return p[..., 0], p[..., 1]

    def heading_at(self, s):
        i, _ = self._locate(s)
        h = self._seg_heading[i]
        return float(h) if np.ndim(h) == 0 else h

    def nearest_lane(self, d: float) -> int:
        return int(np.argmin([abs(d - c) for c in self.lane_centers]))

    @property
    def d_center(self) -> float:
        return 0.5 * (self.d_min + self.d_max)

    @property
    def half_width(self) -> float:
        return 0.5 * (self.d_max - self.d_min)
