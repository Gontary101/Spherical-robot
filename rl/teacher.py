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

VS = np.linspace(-0.5, 2.5, 13)
WS = np.linspace(-2.0, 2.0, 17)
CAND = np.array([(v, w) for v in VS for w in WS])
K_HEAD, K_REV = float(os.environ.get("K_HEAD", 0.4)), float(os.environ.get("K_REV", 0.2))                                # m of route per unit heading misalignment / per m/s reverse
K_PATH = float(os.environ.get("K_PATH", 1.5))            # m of route per m^2 of cross-track error (mid + end)
RECOVERY = os.environ.get("MPC_RECOVERY", "1") == "1"
TIGHTEN = float(os.environ.get("MPC_TIGHTEN", 0.0))     # m of spare filter slack per m/s of speed (32-episode A/B: no gain)


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
        self.reset()
        self.cand = CAND if v_max == 2.5 else np.array([(v, w) for v in np.arange(-0.5, v_max + 1e-9, 0.25) for w in WS])

    def reset(self):
        self.prev = np.zeros(2)
        self.progress = []                                      # (time, remaining route) - stall detection
        self.mode, self.mode_t, self.boost_until, self.n_rec = "drive", 0.0, -1.0, 0

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
            else:
                return None
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
        while k < self.h - 1e-9:
            if k >= DEAD:
                acc = np.where((C[:, 0] < v) & (v <= 0.0), A_REV, A_FWD)
                v = v + np.clip(C[:, 0] - v, -acc * self.dt, acc * self.dt)
                w = w + aw * (C[:, 1] - w)
            th = th + w * self.dt
            x, y = x + v * np.cos(th) * self.dt, y + v * np.sin(th) * self.dt
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
        cost = (geo + 2.0 * clr_pen + 50.0 * hit + 0.05 * np.abs(C[:, 1]) + K_HEAD * head + K_REV * np.maximum(0.0, -C[:, 0])
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
