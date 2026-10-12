"""
Safety filter for navigation commands (plan phase 1c): dynamic-window admissibility on the local occupancy map.

A command (v, w) is ADMISSIBLE if, after the robot reacts to it for one control period (identified first-order lag)
and then brakes along the same arc (identified, conservative braking-distance model), the circular footprint plus a
margin never touches a cell the map holds occupied. The margin grows with the predicted horizon by the measured
path-prediction error (rl/sysid_low.py), so the guarantee covers the closed loop as it really behaves.

If the requested command is not admissible, the filter returns the admissible candidate closest to it (speed is
given up before heading). Because the footprint is a circle, turning in place never sweeps new area: if the robot is
clear now and its current motion can be braked safely, (0, w) is always admissible, so the filter never deadlocks
while the robot is still free.

With a correct map and braking model this excludes collisions with static obstacles by construction; what remains is
what the map cannot know (never-seen obstacles, moving pedestrians - see plan phase 4).
"""
import json
import math
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class SafetyFilter:
    def __init__(self, robot_r, margin=0.08, tau=0.3, period=0.1, sysid=None, dt=0.05):
        self.r, self.margin, self.tau, self.period, self.dt = robot_r, margin, tau, period, dt
        sysid = sysid or os.path.join(ROOT, "rl/runs/nav2/bench/sysid.json")
        if os.path.exists(sysid):
            S = json.load(open(sysid))
            b = S["brake_model"]
            self.brake = (b["c0"], b["c1"], b["c2"])
            e = S["arc_error"]
            self.err_rate = max(e["p95_05"] / 0.5, e["p95_10"] / 1.0)      # m of prediction error per s of horizon
        else:                                                               # conservative defaults until identified
            self.brake = (0.10, 0.25, 0.12)
            self.err_rate = 0.25
        vs = np.linspace(-1.0, 3.5, 19)
        ws = np.linspace(-2.5, 2.5, 21)
        self.cand = np.array([(v, w) for v in vs for w in ws])

    def stop_dist(self, v):
        c0, c1, c2 = self.brake
        return c0 + c1 * np.abs(v) + c2 * v * v

    def _rollout(self, pose, v0, w0, cmds):
        """Predicted paths for N commands: react for one period, then brake to rest along the arc.
        Returns points (N, K, 2) and their horizon times (N, K)."""
        x, y, th = pose
        N = len(cmds)
        v = np.full(N, float(v0))
        w = np.full(N, float(w0))
        X, Y, TH = np.full(N, x), np.full(N, y), np.full(N, th)
        a = 1 - math.exp(-self.dt / self.tau)
        pts, ts = [], []
        t = 0.0
        n_react = int(round(self.period / self.dt)) + 1
        for _ in range(n_react):                                     # reaction: lag toward the command
            v += a * (cmds[:, 0] - v)
            w += a * (cmds[:, 1] - w)
            TH = TH + w * self.dt
            X, Y = X + v * np.cos(TH) * self.dt, Y + v * np.sin(TH) * self.dt
            t += self.dt
            pts.append(np.stack([X, Y], -1))
            ts.append(t)
        # braking: travel the identified stopping distance along the current arc (curvature kept)
        d_stop = self.stop_dist(v)
        kappa = np.where(np.abs(v) > 0.05, w / np.maximum(np.abs(v), 0.05), 0.0)
        n_brake = 12
        sgn = np.sign(v)
        ds = d_stop / n_brake
        t_stop = 2.0 * d_stop / np.maximum(np.abs(v), 0.2)          # uniform deceleration: t = 2 d / v
        T = np.full(N, t)
        ts = np.tile(np.array(ts), (N, 1))
        tcols = [ts]
        for k in range(1, n_brake + 1):
            TH = TH + kappa * ds * sgn
            X, Y = X + sgn * ds * np.cos(TH), Y + sgn * ds * np.sin(TH)
            pts.append(np.stack([X, Y], -1))
            tcols.append((T + t_stop * k / n_brake)[:, None])
        return np.stack(pts, 1), np.concatenate(tcols, 1)

    def admissible(self, lmap, pose, v0, w0, cmds):
        cmds = np.atleast_2d(cmds)
        P, ts = self._rollout(pose, v0, w0, cmds)
        clear = lmap.clearance(P.reshape(-1, 2)).reshape(P.shape[:2])
        need = self.r + self.margin + self.err_rate * ts
        slack = (clear - need).min(1)
        return slack >= 0, slack

    def __call__(self, lmap, pose, v0, w0, v_cmd, w_cmd):
        """Returns (v, w, intervened, slack of the returned command)."""
        ok, slack = self.admissible(lmap, pose, v0, w0, np.array([[v_cmd, w_cmd]]))
        if ok[0]:
            return v_cmd, w_cmd, False, float(slack[0])
        okc, sl = self.admissible(lmap, pose, v0, w0, self.cand)
        if okc.any():
            c = self.cand[okc]
            # closest admissible: penalise speed loss less than heading change, never prefer going faster
            cost = 1.0 * np.abs(c[:, 0] - v_cmd) + 0.6 * np.abs(c[:, 1] - w_cmd) + 2.0 * np.maximum(c[:, 0] - v_cmd, 0)
            i = int(np.argmin(cost))
            return float(c[i, 0]), float(c[i, 1]), True, float(sl[okc][i])
        # nothing admissible (an obstacle appeared inside the braking envelope): brake as hard as possible,
        # turning in place toward the side with more room
        i = int(np.argmax(sl))
        return 0.0, float(self.cand[i, 1]), True, float(sl[i])
