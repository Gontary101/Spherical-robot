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
A_PUSH_MAX = 2.9                             # m/s^2: pendulum at 90 deg on the flat (m g L / R / M)
PED_R, PED_T, PED_GROW = 0.25, 1.5, 0.2      # person radius, prediction horizon (s), deviation growth (m per s)


class SafetyFilter:
    def __init__(self, robot_r, margin=float(os.environ.get("GYRA_MARGIN", 0.04)), period=0.1, sysid=None, dt=0.05, k_err=0.25, k_spin=0.04):
        self.r, self.margin, self.period, self.dt = robot_r, margin, period, dt
        self.k_err, self.k_spin = k_err, k_spin
        sysid = sysid or os.path.join(ROOT, "rl/runs/nav2/bench/sysid.json")
        S = json.load(open(sysid))
        self.bv, self.bd = np.array(S["brake_table"]["v"]), np.array(S["brake_table"]["d"])
        if "brake_table2" in S:                                     # stops from random dynamic states (any pendulum
            T2 = S["brake_table2"]                                  # angle): worst case over the angle bins
            self.bd = np.maximum(self.bd, np.interp(self.bv, T2["v"], np.max(np.array(T2["d"]), axis=1)))
        R = S["response"]
        self.dead, self.a_fwd, self.a_rev, self.tau_w = R["dead"], R["a_fwd"], R["a_rev"], R["tau_w"]
        e = S["arc_error"]
        self.et, self.ee = np.array([0.0, 0.5, 1.0]), np.array([0.0, e["p95_05"], e["p95_10"]])   # p95 path error vs horizon
        # measured closed-loop path-prediction error under field conditions (rough ground, wind, payload): p95 per
        # horizon and speed. Replaces the hand-set speed-proportional margin when available.
        self.err_tab = None
        if "err_table" in S:
            T = S["err_table"]
            self.tab_t = np.r_[0.0, T["t"]]
            self.tab_s = np.array(T["speed_edges"][:-1])
            self.err_tab = np.c_[np.zeros(len(T["p95"])), np.array(T["p95"])]
            self.lat_tab = self.al_tab = None
            if "p95_lat" in T:                                     # anisotropic: cross-track inflates the footprint,
                z = np.zeros(len(T["p95_lat"]))                    # along-track shifts it along the path
                self.lat_tab = np.c_[z, np.array(T["p95_lat"])]
                self.al_tab = np.c_[z, np.array(T["p95_along"])]
        vs = np.linspace(-1.0, 3.5, 19)
        ws = np.linspace(-2.5, 2.5, 21)
        self.cand = np.array([(v, w) for v in vs for w in ws])
        self.reset()

    def reset(self):
        self.prev = np.zeros(2)                                     # last output: corrections must be consistent in time
        self.scan_dir = 0.0                                         # look-around direction while active, else 0
        self.peds = []                                              # tracked people: (position, velocity), local frame
        self.a0 = 0.0                                               # current longitudinal acceleration (IMU / LIO)

    def stop_dist(self, v):
        """Worst-case stopping distance over friction / slope / payload (identified table, linear in between)."""
        av = np.abs(v)
        return np.where(av <= self.bv[-1], np.interp(av, self.bv, self.bd), self.bd[-1] * (av / self.bv[-1]) ** 2)

    def err(self, t):
        slope = (self.ee[-1] - self.ee[-2]) / (self.et[-1] - self.et[-2])
        return np.where(t <= self.et[-1], np.interp(t, self.et, self.ee), self.ee[-1] + slope * (t - self.et[-1]))

    def _rollout(self, pose, v0, w0, cmds, a0=0.0):
        """Predicted paths for N commands with the identified response (dead time, acceleration-limited speed, lagged
        yaw rate) for one control period, then a worst-case stop along the arc. Points (N, K, 2), times (N, K)."""
        x, y, th = pose
        N = len(cmds)
        v = np.full(N, float(v0))
        w = np.full(N, float(w0))
        X, Y, TH = np.full(N, x), np.full(N, y), np.full(N, th)
        aw = 1 - math.exp(-self.dt / self.tau_w)
        pts, ts, hs = [], [], []
        t = 0.0
        while t < self.period + self.dead - 1e-9:                  # react: dead time, then move toward the command
            if t < self.dead:
                # during the dead time the robot keeps its current acceleration: with the pendulum already swung deep
                # into a push it is still speeding up when a stop is commanded (measured: 2.1 -> 2.5 m/s)
                v = v + a0 * self.dt
            else:
                # worst case toward a faster command: the physical limit (pendulum at 90 deg: 133 N / 45.8 kg), not
                # the typical step response; slowing down uses the identified rate (braking is the table below)
                acc = np.where((cmds[:, 0] < v) & (v <= 0.0), self.a_rev,
                               np.where(np.abs(cmds[:, 0]) > np.abs(v), A_PUSH_MAX, self.a_fwd))
                v = v + np.clip(cmds[:, 0] - v, -acc * self.dt, acc * self.dt)
                w = w + aw * (cmds[:, 1] - w)
            TH = TH + w * self.dt
            X, Y = X + v * np.cos(TH) * self.dt, Y + v * np.sin(TH) * self.dt
            t += self.dt
            pts.append(np.stack([X, Y], -1))
            ts.append(np.full(N, t))
            hs.append(TH)
        self.t_react = t                                            # prediction error grows only until braking starts:
        d_stop = self.stop_dist(v)                                  # the braking table is itself a worst-case envelope
        kappa = np.where(np.abs(v) > 0.05, w / np.maximum(np.abs(v), 0.05), 0.0)
        n_brake = 12
        sgn = np.sign(v)
        ds = d_stop / n_brake
        t_stop = 2.0 * d_stop / np.maximum(np.abs(v), 0.2)          # uniform deceleration: t = 2 d / v
        # the braking path lies between the arc (curvature held) and a straight line (yaw rate dies out while the
        # pendulum swings through): both must be clear, or a turn that curves past an obstacle at speed is admitted
        # although stopping would run straight into it
        for kap in (kappa, 0.0):
            Xb, Yb, THb = X, Y, TH
            for k in range(1, n_brake + 1):
                THb = THb + kap * ds * sgn
                Xb, Yb = Xb + sgn * ds * np.cos(THb), Yb + sgn * ds * np.sin(THb)
                pts.append(np.stack([Xb, Yb], -1))
                ts.append(t + t_stop * k / n_brake)
                hs.append(THb)
        self.heads = np.stack(hs, 1)
        return np.stack(pts, 1), np.stack(ts, 1)

    def ped_margin(self, P, ts):
        """Clearance to the tracked people's predicted positions minus what is needed, per rollout point: a person at
        p moving at v is expected at p + v min(t, PED_T) and may deviate by PED_GROW m per second of look-ahead."""
        if not self.peds:
            return None
        out = np.full(P.shape[:2], 99.0)
        th = np.minimum(ts, PED_T)
        for p, v in self.peds:
            q = p[None, None, :] + v[None, None, :] * th[..., None]
            d = np.linalg.norm(P - q, axis=-1) - (self.r + PED_R + self.margin + PED_GROW * th)
            out = np.minimum(out, d)
        return out

    def admissible(self, lmap, pose, v0, w0, cmds, unknown_is_obstacle=True):
        cmds = np.atleast_2d(cmds)
        P, ts = self._rollout(pose, v0, w0, cmds, self.a0)
        pm = self.ped_margin(P, ts)
        speed = np.maximum(abs(v0), np.abs(cmds[:, 0]))[:, None]
        if self.err_tab is not None and self.lat_tab is not None:
            row = np.clip(np.searchsorted(self.tab_s, speed[:, 0], side="right") - 1, 0, len(self.tab_s) - 1)
            th = np.minimum(ts, self.t_react)          # error grows while reacting; braking is a worst-case envelope
            El, Ea = np.empty_like(th), np.empty_like(th)
            for j in np.unique(row):
                sel = row == j
                El[sel] = np.interp(th[sel], self.tab_t, self.lat_tab[j])
                Ea[sel] = np.interp(th[sel], self.tab_t, self.al_tab[j])
            u = np.stack([np.cos(self.heads), np.sin(self.heads)], -1) * Ea[..., None]
            Q = np.stack([P, P + u, P - u], 0)                     # nominal, ahead and behind along the path
            cl = lmap.clearance(Q.reshape(-1, 2), unknown_is_obstacle=unknown_is_obstacle).reshape(Q.shape[:3]).min(0)
            need = self.r + self.margin + El + self.k_spin * np.abs(cmds[:, 1])[:, None]
            margin = cl - need
            if pm is not None:
                margin = np.minimum(margin, pm)
            self._last = margin
            slack = margin.min(1)
            return slack >= 0, slack
        clear = lmap.clearance(P.reshape(-1, 2), unknown_is_obstacle=unknown_is_obstacle).reshape(P.shape[:2])
        # prediction-error margin: grows with the reaction horizon and the speed involved (sysid: p95 ~0.25 m per
        # s per m/s), plus the translation a fast spin causes on this pendulum robot
        if self.err_tab is not None:
            row = np.clip(np.searchsorted(self.tab_s, speed[:, 0], side="right") - 1, 0, len(self.tab_s) - 1)
            th = np.minimum(ts, self.t_react)          # error grows while reacting; braking is a worst-case envelope
            E = np.empty_like(th)
            for j in np.unique(row):                                   # few speed bins: one interpolation per bin
                sel = row == j
                E[sel] = np.interp(th[sel], self.tab_t, self.err_tab[j])
            need = self.r + self.margin + E + self.k_spin * np.abs(cmds[:, 1])[:, None]
        else:
            need = (self.r + self.margin + self.k_err * np.minimum(ts, self.t_react) * speed
                    + self.k_spin * np.abs(cmds[:, 1])[:, None])
        margin = clear - need
        self._last = margin                                         # per-point slack, for recovery
        slack = margin.min(1)
        return slack >= 0, slack

    def __call__(self, lmap, pose, v0, w0, v_cmd, w_cmd):
        """Returns (v, w, intervened, slack of the returned command)."""
        ok, slack = self.admissible(lmap, pose, v0, w0, np.array([[v_cmd, w_cmd]]))
        if ok[0]:
            self.scan_dir = 0.0
            return self._out((v_cmd, w_cmd), False, slack[0])
        okc, sl = self.admissible(lmap, pose, v0, w0, self.cand)
        if okc.any():
            c = self.cand[okc]
            # closest admissible: speed is given up before heading, aggressive turning is discouraged
            cost = (1.0 * np.abs(c[:, 0] - v_cmd) + 0.6 * np.abs(c[:, 1] - w_cmd) + 2.0 * np.maximum(c[:, 0] - v_cmd, 0)
                    + 0.2 * np.abs(c[:, 1]) + self._switch(c))
            i = int(np.argmin(cost))
            # look-around: the request is blocked only by space nobody has seen (the pods' wedge ahead of the tyre)
            # and the best alternative is to stand still. Standing still never reveals it; a yaw sweep does (the
            # wedge turns with the robot and the side sensors see what was ahead). Turn in place, keeping one
            # direction, until the request becomes admissible.
            # (forward or reverse: the wedge exists behind the tyre too, and backing out to re-approach a gap is the
            # manoeuvre a good navigator wants most often - 86 % of a privileged teacher's stalls were reverses
            # blocked by unknown space)
            if abs(v_cmd) > 0.15 and abs(c[i, 0]) < 0.15 and abs(v0) < 0.3:
                ok_known, _ = self.admissible(lmap, pose, v0, w0, np.array([[v_cmd, w_cmd]]), unknown_is_obstacle=False)
                if ok_known[0]:
                    if self.scan_dir == 0.0:
                        self.scan_dir = float(np.sign(w_cmd) or np.sign(self.prev[1]) or 1.0)
                    spin = (np.abs(c[:, 0]) < 1e-9) & (np.sign(c[:, 1]) == self.scan_dir) & (np.abs(c[:, 1]) <= 0.8)
                    if spin.any():
                        j = np.flatnonzero(spin)[int(np.argmax(np.abs(c[spin, 1])))]
                        return self._out(c[j], True, sl[okc][j])
            return self._out(c[i], True, sl[okc][i])
        # recovery (already inside the guard band, e.g. an obstacle appeared late, or a push): every path's worst point
        # is the present pose, so the minimum slack cannot rank candidates. Instead: never get closer than now (the
        # slack along the path stays above its current value), and leave toward the most room, preferring what the
        # navigator asked for. While moving, only stop (reverse while moving swings this robot's pendulum violently,
        # which is exactly what must not happen next to an obstacle).
        M = self._last
        now = M[:, 0:1]
        no_worse = (M >= np.minimum(now, 0.0) - 0.005).all(1)
        if abs(v0) > 0.3:                           # braking from speed: straight (one of the two braking paths that
            slow = (self.cand[:, 0] == 0.0) & (np.abs(self.cand[:, 1]) <= 1e-9)    # were checked); turning at a fixed
        elif abs(v0) > 0.15:                        # rate while v -> 0 tightens the curve beyond both
            slow = (self.cand[:, 0] == 0.0) & (np.abs(self.cand[:, 1]) <= 0.8)
        else:
            slow = (np.abs(self.cand[:, 0]) <= 0.4) & (np.abs(self.cand[:, 1]) <= 1.0)
        ok = slow & no_worse
        if not ok.any():
            ok = slow
        end = np.minimum(M[:, -13], M[:, -1])                        # end of the arc and of the straight braking path
        score = end - 0.05 * (np.abs(self.cand[:, 0] - v_cmd) + 0.3 * np.abs(self.cand[:, 1] - w_cmd)) - 0.2 * self._switch(self.cand)
        i = np.flatnonzero(ok)[int(np.argmax(score[ok]))]
        return self._out(self.cand[i], True, sl[i])

    def _switch(self, c):
        """Cost of changing the previous output: flipping the turn direction every step (a correction that chatters)
        swings this robot's pendulum and makes it translate while commanded to stop."""
        pv, pw = self.prev
        flip = (np.abs(pw) > 0.1) & (np.sign(c[:, 1]) == -np.sign(pw)) & (np.abs(c[:, 1]) > 0.1)
        return 0.3 * np.abs(c[:, 1] - pw) + 0.2 * np.abs(c[:, 0] - pv) + 0.8 * flip

    def _out(self, c, intervened, slack):
        self.prev = np.array([float(c[0]), float(c[1])])
        return float(c[0]), float(c[1]), intervened, float(slack)
