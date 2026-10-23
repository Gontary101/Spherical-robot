"""
Privileged navigation teacher (plan phase 2): sampling MPC on the true geodesic cost-to-go.

Candidates (v, w) are rolled out 1.5 s with the identified low-level response (dead time, acceleration-limited speed,
lagged yaw rate) from the true state. Cost = geodesic distance to the goal at the end of the rollout (the costmap-
weighted navigation function of the true static map: no local minima, doors are centred) + a clearance penalty from
the true obstacle distance field + small smoothness terms. Only commands the safety filter admits on the robot's own
map are considered, so the teacher demonstrates behaviour that is efficient *under the shield*.

    python rl/teacher.py            # benchmark it (shielded) on the four phase-1 scenarios
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import nav2_env as NE  # noqa: E402
from nav2_env import GRID, V_RANGE, W_RANGE  # noqa: E402
from sysid_low import A_FWD, A_REV, DEAD, TAU_W  # noqa: E402

# crawl speeds (GYRA_CRAWL=1): the coarsest admissible forward speed was 0.25 m/s, whose worst-case stop plus reaction
# needs ~0.3 m of guaranteed free path - in a 1.0 m door the robot was blocked by ~3 cm with nothing slower to try
CRAWL = os.environ.get("GYRA_CRAWL", "1") == "1"
VS = np.union1d(np.linspace(-0.5, 2.5, 13), [-0.1, -0.05, 0.05, 0.1]) if CRAWL else np.linspace(-0.5, 2.5, 13)
WS = np.linspace(-2.0, 2.0, 17)
CAND = np.array([(v, w) for v in VS for w in WS])
K_HEAD, K_REV = float(os.environ.get("K_HEAD", 0.4)), float(os.environ.get("K_REV", 0.2))                                # m of route per unit heading misalignment / per m/s reverse
K_PATH = float(os.environ.get("K_PATH", 1.5))            # m of route per m^2 of cross-track error (mid + end)
RECOVERY = os.environ.get("MPC_RECOVERY", "1") == "1"
# m of route per rad^2 of mean predicted pendulum swing (needs GYRA_RESP): commands that rock the pendulum make this
# non-minimum-phase robot lurch, and the filter's corrections of the lurch rock it further
K_PEND = float(os.environ.get("MPC_KPEND", 0.0))
DEBUG = os.environ.get("MPC_DEBUG") == "1"
DOOR = os.environ.get("MPC_DOOR", "1") == "1"
# people-aware rollouts (MPC_PED=1, docs/12 P4): half the simulated people never stop for the robot, and all remaining
# collisions were people walking into a robot that only the filter kept from moving toward them - plan to yield
PED_MPC = os.environ.get("MPC_PED", "0") == "1"
PED_H, PED_WANT, K_PED = 1.5, 0.4, float(os.environ.get("MPC_KPED", 3.0))          # stalled at a narrow passage: cross it on its axis (docs/12, P2)
TIGHTEN = float(os.environ.get("MPC_TIGHTEN", 0.0))     # m of spare filter slack per m/s of speed (32-episode A/B: no gain)
# Route-relative (Frenet) candidates: follow the planner's route - costmap-centred, so it runs straight through the
# middle of a door - at a speed and a lateral offset, with pure-pursuit feedback INSIDE the rollout. A body-frame arc
# held for 1.5 s cannot express "line up with the door, then go straight through"; a route-relative candidate does it
# by construction. Arcs stay in the candidate set (turning in place, backing off, leaving the route).
FRENET = os.environ.get("MPC_FRENET", "1") == "1"
FR_V = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5)
FR_Q = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)


def route_ref(env, n=60, ds=0.05):
    """The planner's route ahead of the robot in the LOCAL frame (local map, MPC rollouts): the grid staircase
    smoothed (moving average over 0.5 m) and resampled every ds. Returns points (M, 2), unit tangents (M, 2), or None."""
    path = env.plan_path(env.odo[:2], n)
    if len(path) < 4:
        return None
    P = path                                                    # from the robot's cell: its offset from the route
    k = 5                                                       # stays measurable (the pursuit blends it away)
    Ps = np.array([P[max(0, i - k // 2):i + k // 2 + 1].mean(0) for i in range(len(P))])
    seg = np.linalg.norm(np.diff(Ps, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 0.3:
        return None
    S = np.arange(0.0, s[-1], ds)
    Q = np.c_[np.interp(S, s, Ps[:, 0]), np.interp(S, s, Ps[:, 1])]
    d = env.odo[2] - env.loc[2]                                 # odometry frame -> local frame
    c_, s_ = math.cos(-d), math.sin(-d)
    r = Q - env.odo[:2]
    Q = env.loc[:2] + np.c_[c_ * r[:, 0] - s_ * r[:, 1], s_ * r[:, 0] + c_ * r[:, 1]]
    T = np.gradient(Q, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    return Q, T


def grid_bilinear(env, G, p, far=99.0):
    """Bilinear lookup of a GRID-resolution world field at points p (N, 2); unreachable (> 98) cells count as far."""
    fx = (p[:, 0] + NE.ARENA) / GRID - 0.5                 # module attribute: a showcase world may enlarge the arena
    fy = (p[:, 1] + NE.ARENA) / GRID - 0.5
    n = G.shape[0]
    j0 = np.clip(np.floor(fx).astype(int), 0, n - 2)
    i0 = np.clip(np.floor(fy).astype(int), 0, n - 2)
    tx, ty = np.clip(fx - j0, 0, 1), np.clip(fy - i0, 0, 1)
    g = np.minimum(G, far)
    return ((1 - ty) * ((1 - tx) * g[i0, j0] + tx * g[i0, j0 + 1]) + ty * ((1 - tx) * g[i0 + 1, j0] + tx * g[i0 + 1, j0 + 1]))


class GeoMPC:
    """deployable=False: privileged teacher (true pose, true geodesic, true clearance).
    deployable=True: the same MPC on what the robot knows - LiDAR-inertial odometry pose and velocity, the onboard
    planner's geodesic field on the prior map, clearance from its own local map."""
    def __init__(self, horizon=1.5, dt=0.1, deployable=False, v_max=2.5):
        self.h, self.dt, self.deployable = horizon, dt, deployable
        self.resp, self.resp_dt = None, 0.02
        from resp_model import Response, configured             # rl/resp_model.py: rollouts with the pendulum state
        if configured():
            self.resp = Response(configured())
        self.reset()
        self.cand = CAND if v_max == 2.5 else np.array([(v, w) for v in np.arange(-0.5, v_max + 1e-9, 0.25) for w in WS])

    def reset(self):
        self.prev = np.zeros(2)
        self.progress = []                                      # (time, remaining route) - stall detection
        self.mode, self.mode_t, self.boost_until, self.n_rec = "drive", 0.0, -1.0, 0

    def find_passage(self, env):
        """Narrow passage on the route ahead (local map): the route point of least clearance within 1.5 m, if that
        clearance leaves less than 0.15 m beyond the footprint. Its axis is the direction through it along which the
        clearance stays largest (through a door: perpendicular to the wall). Stores the passage point N and unit axis
        a (pointing along the route) in the local frame; returns True when found."""
        ref = route_ref(env, n=40)
        if ref is None:
            return False
        Q, T = ref
        dist = np.linalg.norm(Q - env.loc[:2], axis=1)
        sel = np.flatnonzero(dist < 1.5)
        if len(sel) == 0:
            return False
        cl = env.lmap.clearance(Q[sel])
        k = sel[int(np.argmin(cl))]
        if cl.min() > env.robot_r + 0.15:
            return False
        N = Q[k]
        th = np.linspace(0, np.pi, 36, endpoint=False)
        D = np.c_[np.cos(th), np.sin(th)]
        score = np.minimum(env.lmap.clearance(N + 0.5 * D), env.lmap.clearance(N - 0.5 * D))
        ax = D[int(np.argmax(score))]
        if ax @ T[k] < 0:
            ax = -ax
        # centre it: the route's point need not be the passage's middle (the planner's grid path ran 13 cm off-centre
        # through a 1.0 m door, which leaves this footprint 5 cm a side) - slide across the axis to the most clearance
        nrm = np.array([-ax[1], ax[0]])
        off = np.linspace(-0.3, 0.3, 61)
        N = N + nrm * off[int(np.argmax(env.lmap.clearance(N + off[:, None] * nrm)))]
        self.door_N, self.door_ax = N, ax
        return True

    def door(self, env, t):
        """Cross a narrow passage on its axis: settle the pendulum, go to the point 0.7 m before the passage on its
        axis (crawl, turning toward it first), face along the axis, then cross straight at a crawl steering onto the
        axis line, until 0.6 m past it. Any phase that takes too long hands back to normal driving."""
        N, ax = self.door_N, self.door_ax
        p, yaw = env.loc[:2], env.loc[2]
        nrm = np.array([-ax[1], ax[0]])
        if self.mode == "door_settle":
            if t - self.mode_t > 0.8:
                self.mode, self.mode_t = "door_approach", t
                d = N - 0.7 * ax - p                             # forward or reverse, decided once: flipping between
                self.door_dir = 1.0 if d @ np.array([math.cos(yaw), math.sin(yaw)]) >= 0 else -1.0   # them lurches
            return np.array([0.0, 0.0])
        if self.mode == "door_approach":
            A = N - 0.7 * ax
            d = A - p
            if np.linalg.norm(d) < 0.12 or t - self.mode_t > 12.0:
                self.mode, self.mode_t = "door_align", t
                return np.array([0.0, 0.0])
            sd = self.door_dir
            want = math.atan2(sd * d[1], sd * d[0])             # reversing: the tail points at the target
            err = (want - yaw + math.pi) % (2 * math.pi) - math.pi
            return np.array([sd * 0.1 if abs(err) < 0.4 else 0.0, float(np.clip(1.5 * err, -0.8, 0.8))])
        if self.mode == "door_align":
            err = (math.atan2(ax[1], ax[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(err) < math.radians(5) or t - self.mode_t > 5.0:
                self.mode, self.mode_t = "door_cross", t
            return np.array([0.0, float(np.clip(1.5 * err, -0.8, 0.8))])
        if self.mode == "door_cross":
            along, off = (p - N) @ ax, (p - N) @ nrm
            if along > 0.6 or t - self.mode_t > 15.0:
                self.mode, self.boost_until, self.progress = "drive", t + 3.0, []
                return None
            head = math.atan2(ax[1], ax[0]) - float(np.clip(1.5 * off, -0.5, 0.5))   # steer onto the axis line
            err = (head - yaw + math.pi) % (2 * math.pi) - math.pi
            return np.array([0.15 if abs(err) < 0.3 else 0.0, float(np.clip(2.0 * err, -0.6, 0.6))])
        return None

    def recovery(self, env):
        """Stall recovery (as production stacks do): when the remaining route has not shrunk by 0.3 m in 8 s, back off
        0.5 m the way the robot came (space it has seen free), turn in place to face the planned path 1 m ahead, then
        resume with doubled path tracking for 5 s so the gap is re-approached aligned. Returns a (v, w) command or None
        when driving normally. The safety filter still checks every command."""
        t = env.d.time
        g = float(grid_bilinear(env, env.plan_geo, env.odo[:2][None])[0])
        self.progress.append((t, g))
        self.progress = [(tt, gg) for tt, gg in self.progress if t - tt < 8.5]
        if self.mode == "drive":
            if t - self.progress[0][0] > 7.9 and self.progress[0][1] - g < 0.3 and g > 0.8:
                self.mode, self.mode_t, self.n_rec = "settle", t, self.n_rec + 1
                if DOOR and self.find_passage(env):                 # stalled at a narrow passage: cross it on its axis
                    self.mode = "door_settle"
            else:
                return None
        if self.mode.startswith("door"):
            return self.door(env, t)
        if self.mode == "settle":
            # let the pendulum come to rest first: commanding reverse while it still swings forward is non-minimum
            # phase (measured: -0.25 m/s commanded, +0.24 m/s actual for 0.8 s, into the wall)
            if t - self.mode_t > 0.8:
                self.mode, self.mode_t, self.back_from = "back", t, env.loc[:2].copy()
            else:
                return np.array([0.0, 0.0])
        if self.mode == "back":
            if np.linalg.norm(env.loc[:2] - self.back_from) > 0.5 or t - self.mode_t > 3.0:
                self.mode, self.mode_t = "align", t
            else:
                return np.array([-0.25, 0.0])
        if self.mode == "align":
            path = env.plan_path(env.odo[:2], 12)
            tgt = path[-1] if len(path) > 1 else env.goal
            want = math.atan2(tgt[1] - env.odo[1], tgt[0] - env.odo[0])
            err = (want - env.odo[2] + math.pi) % (2 * math.pi) - math.pi
            if abs(err) < math.radians(8) or t - self.mode_t > 4.0:
                self.mode, self.boost_until, self.progress = "drive", t + 5.0, []
                return None
            return np.array([0.0, float(np.clip(2.0 * err, -1.2, 1.2))])
        return None

    def plan(self, env):
        if self.deployable:
            pos, yaw = env.loc[:2], env.loc[2]                      # local frame (local map); the route field is in
            v0, w0 = env._velocity_estimate()                       # the prior map's frame: points mapped via odo
        else:
            pos, yaw, t = env._pose()
            v0, w0 = t["v"], t["w"]
        C = self.cand
        ref = route_ref(env) if (self.deployable and FRENET) else None
        nA = len(C)
        if ref is not None:                                     # route-relative candidates (speed, lateral offset)
            RQ, RT = ref
            FV, FQ = np.meshgrid(FR_V, FR_Q, indexing="ij")
            FV, FQ = FV.ravel(), FQ.ravel()
            C = np.r_[C, np.c_[FV, np.zeros_like(FV)]]          # (v, w) filled from the first pursuit step below
            e0 = float(((np.array(pos) - RQ[0]) * np.array([-RT[0, 1], RT[0, 0]])).sum())
        N = len(C)
        x, y, th = np.full(N, pos[0]), np.full(N, pos[1]), np.full(N, yaw)
        v, w = np.full(N, v0), np.full(N, w0)
        aw = 1 - math.exp(-self.dt / TAU_W)
        clr_pen = np.zeros(N)
        hit = np.zeros(N, bool)
        k = 0.0
        mid_k = int(round(self.h / self.dt / 2))
        step = 0
        mx = my = None
        peds = (env.filter.peds if (PED_MPC and self.deployable and env.shield and env.filter is not None) else [])
        ped_pen = np.zeros(N)
        RS, n_sub = None, 0
        pend_pen = np.zeros(N)
        if self.resp is not None and self.deployable:
            rs = env._resp_state()
            RS = dict(v=v.copy(), pend=np.full(N, rs["pend"]), rate=np.full(N, rs["rate"]), w=w.copy(),
                      vc=np.full(N, rs["vc"]), wc=np.full(N, rs["wc"]))
            n_sub = int(round(self.dt / self.resp_dt))
        while k < self.h - 1e-9:
            cmd = C
            if ref is not None:                                 # pure pursuit on the offset route, every step
                F = slice(nA, N)
                px, py = x[F], y[F]
                dd = (px[:, None] - RQ[None, :, 0]) ** 2 + (py[:, None] - RQ[None, :, 1]) ** 2
                i0 = dd.argmin(1)
                ld = np.clip(0.4 + 0.6 * np.maximum(v[F], FV), 0.5, 1.5)
                il = np.minimum(i0 + (ld / 0.05).astype(int), len(RQ) - 1)
                sl = il * 0.05
                q = e0 + (FQ - e0) * np.minimum(1.0, sl / np.maximum(0.8, FV))   # blend to the offset over ~1 s
                tx, ty = RQ[il, 0] - RT[il, 1] * q, RQ[il, 1] + RT[il, 0] * q
                al = np.arctan2(ty - py, tx - px) - th[F]
                al = (al + np.pi) % (2 * np.pi) - np.pi
                kap = 2.0 * np.sin(al) / np.maximum(np.hypot(tx - px, ty - py), 0.2)
                vf = np.where(np.abs(al) > 1.2, 0.0, np.minimum(FV, np.sqrt(1.5 / np.maximum(np.abs(kap), 1e-3))))
                wf = np.clip(np.where(vf > 0.0, vf * kap, 1.5 * al), -W_RANGE, W_RANGE)   # far off heading: turn first
                cmd = C.copy()
                cmd[F, 0], cmd[F, 1] = vf, wf
                if step == 0:
                    C = cmd.copy()                              # the command actually sent if this candidate wins
            if RS is not None:                                  # identified closed loop with the pendulum state
                for _ in range(n_sub):
                    self.resp.step(RS, cmd)
                    th = th + RS["w"] * self.resp_dt
                    x, y = x + RS["v"] * np.cos(th) * self.resp_dt, y + RS["v"] * np.sin(th) * self.resp_dt
                v, w = RS["v"], RS["w"]
                pend_pen += RS["pend"] ** 2
            else:
                if k >= DEAD:
                    acc = np.where((cmd[:, 0] < v) & (v <= 0.0), A_REV, A_FWD)
                    v = v + np.clip(cmd[:, 0] - v, -acc * self.dt, acc * self.dt)
                    w = w + aw * (cmd[:, 1] - w)
                th = th + w * self.dt
                x, y = x + v * np.cos(th) * self.dt, y + v * np.sin(th) * self.dt
            if peds:                                            # people-aware: keep clear of where each tracked
                tp = min(k + self.dt, PED_H)                    # person will be (constant velocity, growing doubt)
                for pp, pv in peds:
                    dd = np.hypot(x - (pp[0] + pv[0] * tp), y - (pp[1] + pv[1] * tp)) - (env.robot_r + 0.25 + 0.2 * tp)
                    ped_pen += np.maximum(0.0, PED_WANT - dd) ** 2
            if self.deployable:
                c = np.minimum(env.lmap.clearance(np.c_[x, y]), 3.0) - env.robot_r
            else:
                c = grid_bilinear(env, env.clear, np.c_[x, y]) - env.robot_r
            # wanted clearance grows with speed (slow down near obstacles); a 1.0 m door leaves 0.107 m at a crawl
            clr_pen += np.maximum(0.0, 0.15 + 0.25 * np.maximum(v, 0.0) - c) ** 2
            hit |= c < 0.0
            k += self.dt
            step += 1
            if step == mid_k:
                mx, my = x.copy(), y.copy()
        if self.deployable:
            d = env.odo[2] - env.loc[2]
            c_, s_ = math.cos(d), math.sin(d)
            rx, ry = x - env.loc[0], y - env.loc[1]
            ex, ey, eth, G = env.odo[0] + c_ * rx - s_ * ry, env.odo[1] + s_ * rx + c_ * ry, th + d, env.plan_geo
            mrx, mry = mx - env.loc[0], my - env.loc[1]
            mx, my = env.odo[0] + c_ * mrx - s_ * mry, env.odo[1] + s_ * mrx + c_ * mry
        else:
            ex, ey, eth, G = x, y, th, env.geo
        geo = grid_bilinear(env, G, np.c_[ex, ey])
        # heading: face along the route at the end of the rollout. Without it the 1.5 s horizon prefers reversing at
        # 0.5 m/s (0.75 m of progress) to turning round (none within the horizon) - and then reverses forever.
        e = 0.15
        gxd = (grid_bilinear(env, G, np.c_[ex + e, ey]) - grid_bilinear(env, G, np.c_[ex - e, ey])) / (2 * e)
        gyd = (grid_bilinear(env, G, np.c_[ex, ey + e]) - grid_bilinear(env, G, np.c_[ex, ey - e])) / (2 * e)
        gn = np.hypot(gxd, gyd)
        valid = (gn > 0.3) & (gn < 8.0) & (geo > 0.8)
        head = np.where(valid, 1.0 - (np.cos(eth) * -gxd + np.sin(eth) * -gyd) / np.maximum(gn, 1e-6), 0.0)
        # cross-track: stay on the planner's route (costmap-centred, so doors are approached aligned and centred)
        track = np.zeros(len(C))
        if self.deployable and K_PATH > 0:
            path = env.plan_path(env.odo[:2], 40)
            if len(path) >= 3:
                A, B = path[:-1], path[1:]
                AB = B - A
                for (qx, qy) in ((mx, my), (ex, ey)):
                    Q = np.c_[qx, qy][:, None, :]
                    u = np.clip(((Q - A[None]) * AB[None]).sum(-1) / np.maximum((AB ** 2).sum(-1), 1e-9)[None], 0, 1)
                    dist = np.linalg.norm(Q - (A[None] + u[..., None] * AB[None]), axis=-1).min(1)
                    track += np.minimum(dist, 1.0) ** 2
        if ref is not None:
            head[nA:], track[nA:] = 0.0, 0.0                    # on the route by construction
            track[nA:] = 0.3 * FQ ** 2                          # prefer the route itself (the costmap's centre line)
        if peds:
            # speed near people: a person seen late (from behind a corner, in the pods' blind wedge) cannot be avoided
            # at 1.5 m/s whatever the plan - limit speed by the distance to the nearest tracked person
            dmin = min(float(np.linalg.norm(pp - np.asarray(pos))) for pp, _ in peds)
            v_cap = float(np.clip(0.4 + 0.4 * (dmin - 1.0), 0.4, 3.0))
            ped_pen = ped_pen + 20.0 * (np.abs(C[:, 0]) > v_cap + 1e-6)
        cost = (geo + 2.0 * clr_pen + 50.0 * hit + K_PEND * pend_pen * self.dt / self.h + K_PED * ped_pen + 0.05 * np.abs(C[:, 1]) + K_HEAD * head + K_REV * np.maximum(0.0, -C[:, 0])
                + K_PATH * (2.0 if env.d.time < self.boost_until else 1.0) * track
                + 0.15 * np.abs(C[:, 1] - self.prev[1]) + 0.1 * np.abs(C[:, 0] - self.prev[0]))
        if env.shield and env.filter is not None:
            v0, w0 = env._velocity_estimate()
            ok, sl = env.filter.admissible(env.lmap, env.loc.copy(), v0, w0, C)
            # constraint tightening (robust MPC): a command that is only just admissible now can leave the robot, one
            # step later, faster and further turned with no admissible option at all (stop-and-go at speed). The
            # navigator picks only commands with spare slack growing with speed; the filter's own test is unchanged.
            tight = ok & (sl >= TIGHTEN * np.maximum(np.abs(C[:, 0]), np.abs(v0)))
            ok = tight if tight.any() else ok
            if ok.any():
                cost = np.where(ok, cost, cost + 1e3)
        i = int(np.argmin(cost))
        if DEBUG:                                               # cost terms per candidate, for rl/diagnose_stuck.py
            self.dbg = dict(C=C.copy(), cost=cost.copy(), geo=geo.copy(), clr=2.0 * clr_pen, hit=50.0 * hit,
                            head=K_HEAD * head, track=K_PATH * track, i=i,
                            ok=ok.copy() if (env.shield and env.filter is not None) else None)
        self.prev = C[i].copy()
        return C[i]

    def __call__(self, oa, env):
        if env is not getattr(self, "env_ref", None) or env.d.time < getattr(self, "last_t", 0.0) - 1e-6:
            self.reset()                                        # new episode (evaluators reuse one navigator)
        self.env_ref, self.last_t = env, env.d.time
        rec = self.recovery(env) if (self.deployable and RECOVERY) else None
        v, w = rec if rec is not None else self.plan(env)
        return np.array([2 * (v - V_RANGE[0]) / (V_RANGE[1] - V_RANGE[0]) - 1, w / W_RANGE])


if __name__ == "__main__":
    import json
    import multiprocessing as mp
    from collections import Counter

    from eval_nav2 import CALM, LOCO, SCEN
    from nav2_env import Nav2Env
    LOW = os.path.join(os.path.dirname(__file__), "runs/loco31/best.pt")
    scen = sys.argv[1:] or ["mixed", "rooms, 1.0 m doors", "forest (thin poles)", "depth camera dead"]

    def run(args):
        sc, ep = args
        env = Nav2Env(seed=70000 + ep, low_ckpt=LOW, level=0.8, adaptive=False, scenario={"nav": CALM | SCEN[sc], "loco": LOCO},
                      shield=True)
        oa, _ = env.reset()
        pol = GeoMPC(deployable=os.environ.get("TEACHER_DEPLOYABLE") == "1")
        while True:
            (oa, _), _, done, info = env.step(pol(oa, env))
            if done:
                e = info["episode"]
                return sc, dict(success=bool(e["success"]), collision=bool(e["collision"]), timeout=bool(e["timeout"]),
                                spl=e["spl"], time=e["time"])
    with mp.get_context("fork").Pool(4) as p:
        R = p.map(run, [(s, e) for s in scen for e in range(int(os.environ.get("EPISODES", 16)))], chunksize=1)
    for s in scen:
        r = [x for k, x in R if k == s]
        print(f"{s:28s} success {np.mean([x['success'] for x in r]):5.1%} collision {np.mean([x['collision'] for x in r]):5.1%} "
              f"timeout {np.mean([x['timeout'] for x in r]):5.1%} spl {np.mean([x['spl'] for x in r]):.2f}")
    a = [x for _, x in R]
    print(f"ALL success {np.mean([x['success'] for x in a]):.1%} collision {np.mean([x['collision'] for x in a]):.1%} "
          f"timeout {np.mean([x['timeout'] for x in a]):.1%}")
