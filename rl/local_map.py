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
DECAY = 0.985                                           # per update: unobserved OCCUPIED evidence fades (~12 s to unknown)
OBST_H = 0.10                                           # depth endpoints this far above the ground plane are obstacles
KNOWN_FREE = -0.3                                       # log-odds below which a cell counts as observed free (safety)


class LocalMap:
    def __init__(self, size=96, cell=0.1, max_r=8.0):
        self.n, self.cell, self.max_r = size, cell, max_r
        self.L = np.zeros((size, size), np.float32)
        # sub-cell geometry: per cell, the running sum of the hit points that fell in it (and their count). Distances
        # are measured to the mean hit point, not the cell centre: a 5 cm grid plus 2 cm range noise otherwise grows
        # every surface by 2.5-5 cm, which closes a 1.0 m door for a 0.79 m robot.
        self.HX = np.zeros((size, size), np.float32)
        self.HY = np.zeros((size, size), np.float32)
        self.HN = np.zeros((size, size), np.float32)
        self.origin = np.zeros(2)                       # odometry-frame coordinates of cell (0, 0)'s corner
        self._dist = self._sdist = None
        self.reset(np.zeros(2))

    def reset(self, center):
        self.L[:] = 0
        self.HX[:] = self.HY[:] = self.HN[:] = 0
        self.origin = np.floor(np.asarray(center) / self.cell) * self.cell - self.n * self.cell / 2
        self._dist = self._sdist = None

    def mark_free_disc(self, center, radius):
        """Prior knowledge that a disc is free (e.g. the spot the robot was placed on)."""
        g = np.arange(-radius, radius + 1e-9, self.cell)
        gx, gy = np.meshgrid(g, g)
        inside = gx ** 2 + gy ** 2 <= radius ** 2
        r_, c_ = self._idx(np.c_[center[0] + gx[inside], center[1] + gy[inside]])
        ok = (r_ >= 0) & (r_ < self.n) & (c_ >= 0) & (c_ < self.n)
        self.L[r_[ok], c_[ok]] = np.minimum(self.L[r_[ok], c_[ok]], L_MIN)
        self._dist = self._sdist = None

    # ----------------------------------------------------------------------------------------------- geometry
    def _recenter(self, center):
        want = np.floor(np.asarray(center) / self.cell) * self.cell - self.n * self.cell / 2
        shift = np.round((want - self.origin) / self.cell).astype(int)
        if not shift.any():
            return
        sx, sy = shift
        for name in ("L", "HX", "HY", "HN"):
            A = getattr(self, name)
            B = np.zeros_like(A)
            B[max(0, -sy):self.n + min(0, -sy), max(0, -sx):self.n + min(0, -sx)] = \
                A[max(0, sy):self.n + min(0, sy), max(0, sx):self.n + min(0, sx)]
            setattr(self, name, B)
        d = shift * self.cell                                  # hit sums are relative to the origin, which moves
        self.HX -= d[0] * self.HN
        self.HY -= d[1] * self.HN
        self.origin = self.origin + shift * self.cell

    def _idx(self, p):
        ij = np.floor((np.asarray(p) - self.origin) / self.cell).astype(int)
        return ij[..., 1], ij[..., 0]                   # row = y, col = x

    def _mark(self, pts, val, hits=False):
        if len(pts) == 0:
            return
        r, c = self._idx(pts)
        ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
        k = r[ok] * self.n + c[ok]
        nn = self.n * self.n
        self.L += np.bincount(k, minlength=nn).reshape(self.n, self.n).astype(np.float32) * val
        if hits:
            p = np.asarray(pts)[ok]
            self.HX += np.bincount(k, p[:, 0] - self.origin[0], nn).reshape(self.n, self.n).astype(np.float32)
            self.HY += np.bincount(k, p[:, 1] - self.origin[1], nn).reshape(self.n, self.n).astype(np.float32)
            self.HN += np.bincount(k, minlength=nn).reshape(self.n, self.n).astype(np.float32)
            big = self.HN > 30                                   # keep it a recent mean
            if big.any():
                f = 30.0 / self.HN[big]
                self.HX[big] *= f
                self.HY[big] *= f
                self.HN[big] = 30.0

    # ----------------------------------------------------------------------------------------------- fusion
    def update(self, pose, beam_ang, scan, depth=None, visible=None, dead=None, free_every=1, free_max=None, robot_r=0.0,
               nearfield=None):
        """pose = odometry (x, y, yaw). scan: masked LiDAR ranges (MAX_R = no return).
        depth: (dirs_robot (N, 3), ranges (N,), cam_height, cam_forward_offset) or None.
        visible(px, py) -> bool array: which robot-frame points the sensor can physically see (its self-occlusion), or
        an array of per-beam minimum visible ranges (the same test, tabulated);
        "no return" is only free-space evidence where the sensor could have seen. dead: bool per beam (failed sector)."""
        x, y, yaw = pose
        self._recenter((x, y))
        # only occupied evidence fades (a pedestrian must not leave a permanent ghost; a faded cell becomes unknown,
        # which the safety field still treats as blocked). Free evidence persists: static free space stays free, and a
        # robot that stands still must not forget the ground around it (the pods cannot see it again: blind wedge)
        np.multiply(self.L, DECAY, out=self.L, where=self.L > 0)
        a = beam_ang + yaw
        hit = scan < self.max_r - 1e-3
        rng = np.minimum(scan, self.max_r)
        step = self.cell * 0.7
        ks = np.arange(0.2, self.max_r, step)
        # free space along every beam up to its return (minus one cell), all beams at once
        sel = slice(None, None, free_every)                              # free space from a subset of rays (cost)
        K = ks[ks < (free_max or self.max_r)][:, None]
        bs, rs = beam_ang[sel], rng[sel]
        msk = K < (rs - self.cell)[None, :]
        if dead is not None:
            msk &= ~np.asarray(dead)[sel][None, :]
        if callable(visible):
            msk &= visible(K * np.cos(bs)[None, :], K * np.sin(bs)[None, :])
        elif visible is not None:
            msk &= K >= np.asarray(visible)[sel][None, :]
        ca, sa = np.cos(a[sel])[None, :], np.sin(a[sel])[None, :]
        self._mark(np.c_[(x + K * ca)[msk], (y + K * sa)[msk]], L_FREE)
        self._mark(np.c_[x + rng[hit] * np.cos(a[hit]), y + rng[hit] * np.sin(a[hit])], L_OCC + abs(L_FREE), hits=True)
        if depth is not None:
            dirs, z, h, xo = depth
            ok = z > 0.35
            if ok.any():
                d = dirs[ok] * z[ok, None]
                d[:, 0] += xo
                above = d[:, 2] + h > OBST_H                       # endpoint height above the local ground plane
                c, s = math.cos(yaw), math.sin(yaw)
                wx = x + c * d[:, 0] - s * d[:, 1]
                wy = y + s * d[:, 0] + c * d[:, 1]
                self._mark(np.c_[wx[above], wy[above]], L_OCC, hits=True)
                self._mark(np.c_[wx[~above], wy[~above]], L_FREE)
        if nearfield is not None:                                       # ground cameras: robot-frame floor / obstacle points
            fr, oc = nearfield
            c, s = math.cos(yaw), math.sin(yaw)
            Rm = np.array([[c, s], [-s, c]])
            if len(fr):
                self._mark(np.array([x, y]) + fr @ Rm, L_FREE)
            if len(oc):
                self._mark(np.array([x, y]) + oc @ Rm, L_OCC, hits=True)
        np.clip(self.L, L_MIN, L_MAX, out=self.L)
        if robot_r > 0:                                                  # the robot's own disc is known free
            self.mark_free_disc((x, y), robot_r)
        self._dist = None
        self._sdist = None

    # ----------------------------------------------------------------------------------------------- queries
    def occupied(self):
        return self.L > OCC_TH

    def _nearest(self, blocked):
        """Per cell: the reference point (odometry frame) of the nearest blocked cell - its mean hit point when it
        holds hits, else its centre - or None if nothing is blocked."""
        if not blocked.any():
            return None
        _, (ir, ic) = distance_transform_edt(~blocked, return_indices=True)
        n = np.maximum(self.HN, 1e-6)
        has = self.HN > 0.5
        rx = np.where(has, self.HX / n, (np.arange(self.n)[None, :] + 0.5) * self.cell) + self.origin[0]
        ry = np.where(has, self.HY / n, (np.arange(self.n)[:, None] + 0.5) * self.cell) + self.origin[1]
        return rx[ir, ic], ry[ir, ic]

    def dist_field(self):
        if self._dist is None:
            self._dist = self._nearest(self.occupied())
        return self._dist

    def safety_field(self):
        """Nearest cell that is occupied OR not yet observed free (unknown counts as an obstacle)."""
        if self._sdist is None:
            self._sdist = self._nearest(self.L > KNOWN_FREE)
        return self._sdist

    def clearance(self, pts, unknown_is_obstacle=False):
        """Distance (m) from odometry-frame points to the nearest occupied cell's surface point (or, for safety, to
        anything not known to be free); outside the map: 0 when unknown counts as an obstacle, else 99."""
        F = self.safety_field() if unknown_is_obstacle else self.dist_field()
        pts = np.asarray(pts)
        r, c = self._idx(pts)
        ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
        out = np.full(r.shape, 0.0 if unknown_is_obstacle else 99.0)
        if F is None:
            out[ok] = 99.0
            return out
        rx, ry = F
        out[ok] = np.hypot(pts[ok, 0] - rx[r[ok], c[ok]], pts[ok, 1] - ry[r[ok], c[ok]])
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
