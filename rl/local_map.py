"""
Egocentric occupancy map with memory (plan phase 1b).

A rolling log-odds grid in the ODOMETRY frame (what the robot actually knows), re-centred on the robot whenever it
moves more than a cell. Fused every navigation step from
  * the masked pod LiDAR: cells along each beam are observed free, the cell at a return is observed occupied
    (beams with no return clear up to MAX_R);
  * the front depth camera: pixels whose 3-D endpoint lies more than OBST_H above the local ground plane are
    obstacles, the others are free floor.
Obstacles that slide into a sensor's blind zone (the pods' forward/rear wedge, a dead LiDAR sector, the gap between
beams) stay in the map; unobserved cells relax slowly towards "unknown", so pedestrians do not leave permanent ghosts.

clearance(points) returns the distance from each point to the nearest occupied cell (Euclidean distance transform),
which the safety filter uses to test a footprint along a predicted arc.
"""
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

L_OCC, L_FREE, L_MIN, L_MAX = 0.9, -0.4, -2.0, 3.5     # log-odds increments and clamps
OCC_TH = 0.6                                            # log-odds above which a cell counts as occupied
DECAY = 0.985                                           # per update: unobserved evidence fades (~6 s half-life at 10 Hz)
OBST_H = 0.10                                           # depth endpoints this far above the ground plane are obstacles


class LocalMap:
    def __init__(self, size=96, cell=0.1, max_r=8.0):
        self.n, self.cell, self.max_r = size, cell, max_r
        self.L = np.zeros((size, size), np.float32)
        self.origin = np.zeros(2)                       # odometry-frame coordinates of cell (0, 0)'s corner
        self._dist = None
        self.reset(np.zeros(2))

    def reset(self, center):
        self.L[:] = 0
        self.origin = np.floor(np.asarray(center) / self.cell) * self.cell - self.n * self.cell / 2
        self._dist = None

    # ----------------------------------------------------------------------------------------------- geometry
    def _recenter(self, center):
        want = np.floor(np.asarray(center) / self.cell) * self.cell - self.n * self.cell / 2
        shift = np.round((want - self.origin) / self.cell).astype(int)
        if not shift.any():
            return
        L = np.zeros_like(self.L)
        sx, sy = shift
        src = self.L[max(0, sy):self.n + min(0, sy), max(0, sx):self.n + min(0, sx)]
        L[max(0, -sy):self.n + min(0, -sy), max(0, -sx):self.n + min(0, -sx)] = src
        self.L = L
        self.origin = self.origin + shift * self.cell

    def _idx(self, p):
        ij = np.floor((np.asarray(p) - self.origin) / self.cell).astype(int)
        return ij[..., 1], ij[..., 0]                   # row = y, col = x

    def _mark(self, pts, val):
        if len(pts) == 0:
            return
        r, c = self._idx(pts)
        ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
        np.add.at(self.L, (r[ok], c[ok]), val)

    # ----------------------------------------------------------------------------------------------- fusion
    def update(self, pose, beam_ang, scan, depth=None, visible=None, dead=None):
        """pose = odometry (x, y, yaw). scan: masked LiDAR ranges (MAX_R = no return).
        depth: (dirs_robot (N, 3), ranges (N,), cam_height) or None.
        visible(px, py) -> bool array: which robot-frame points the sensor can physically see (its self-occlusion);
        "no return" is only free-space evidence where the sensor could have seen. dead: bool per beam (failed sector)."""
        x, y, yaw = pose
        self._recenter((x, y))
        self.L *= DECAY
        a = beam_ang + yaw
        hit = scan < self.max_r - 1e-3
        rng = np.minimum(scan, self.max_r)
        step = self.cell * 0.7
        ks = np.arange(0.2, self.max_r, step)
        # free space along every beam up to its return (minus one cell), all beams at once
        K = ks[:, None]
        msk = K < (rng - self.cell)[None, :]
        if dead is not None:
            msk &= ~np.asarray(dead)[None, :]
        if visible is not None:
            msk &= visible(K * np.cos(beam_ang)[None, :], K * np.sin(beam_ang)[None, :])
        ca, sa = np.cos(a)[None, :], np.sin(a)[None, :]
        self._mark(np.c_[(x + K * ca)[msk], (y + K * sa)[msk]], L_FREE)
        self._mark(np.c_[x + rng[hit] * np.cos(a[hit]), y + rng[hit] * np.sin(a[hit])], L_OCC + abs(L_FREE))
        if depth is not None:
            dirs, z, h = depth
            ok = z > 0.35
            if ok.any():
                d = dirs[ok] * z[ok, None]
                above = d[:, 2] + h > OBST_H                       # endpoint height above the local ground plane
                c, s = math.cos(yaw), math.sin(yaw)
                wx = x + c * d[:, 0] - s * d[:, 1]
                wy = y + s * d[:, 0] + c * d[:, 1]
                self._mark(np.c_[wx[above], wy[above]], L_OCC)
                self._mark(np.c_[wx[~above], wy[~above]], L_FREE)
        np.clip(self.L, L_MIN, L_MAX, out=self.L)
        self._dist = None

    # ----------------------------------------------------------------------------------------------- queries
    def occupied(self):
        return self.L > OCC_TH

    def dist_field(self):
        if self._dist is None:
            occ = self.occupied()
            self._dist = distance_transform_edt(~occ) * self.cell if occ.any() else np.full(occ.shape, 99.0)
        return self._dist

    def clearance(self, pts):
        """Distance (m) from odometry-frame points to the nearest occupied cell; outside the map: 99."""
        D = self.dist_field()
        r, c = self._idx(pts)
        ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
        out = np.full(r.shape, 99.0)
        out[ok] = D[r[ok], c[ok]]
        return out

    def egocentric(self, pose, size=48, cell=0.15):
        """Robot-centred, heading-aligned crop (size x size, cell m) of occupancy probability, for a policy CNN."""
        x, y, yaw = pose
        g = (np.arange(size) - size / 2 + 0.5) * cell
        gx, gy = np.meshgrid(g, g)
        c, s = math.cos(yaw), math.sin(yaw)
        wx, wy = x + c * gx - s * gy, y + s * gx + c * gy
        r, cc = self._idx(np.stack([wx, wy], -1))
        ok = (r >= 0) & (r < self.n) & (cc >= 0) & (cc < self.n)
        out = np.zeros((size, size), np.float32)
        out[ok] = 1.0 / (1.0 + np.exp(-self.L[r[ok], cc[ok]]))
        out[~ok] = 0.5
        return out
