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
from nav2_env import ARENA, GRID, V_RANGE, W_RANGE  # noqa: E402
from sysid_low import A_FWD, A_REV, DEAD, TAU_W  # noqa: E402

VS = np.linspace(-0.5, 2.5, 13)
WS = np.linspace(-2.0, 2.0, 17)
CAND = np.array([(v, w) for v in VS for w in WS])


def grid_bilinear(env, G, p, far=99.0):
    """Bilinear lookup of a GRID-resolution world field at points p (N, 2); unreachable (> 98) cells count as far."""
    fx = (p[:, 0] + ARENA) / GRID - 0.5
    fy = (p[:, 1] + ARENA) / GRID - 0.5
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
    def __init__(self, horizon=1.5, dt=0.1, deployable=False):
        self.h, self.dt, self.deployable = horizon, dt, deployable
        self.prev = np.zeros(2)

    def reset(self):
        self.prev = np.zeros(2)

    def plan(self, env):
        if self.deployable:
            pos, yaw = env.loc[:2], env.loc[2]                      # local frame (local map); the route field is in
            v0, w0 = env._velocity_estimate()                       # the prior map's frame: points mapped via odo
        else:
            pos, yaw, t = env._pose()
            v0, w0 = t["v"], t["w"]
        N = len(CAND)
        x, y, th = np.full(N, pos[0]), np.full(N, pos[1]), np.full(N, yaw)
        v, w = np.full(N, v0), np.full(N, w0)
        aw = 1 - math.exp(-self.dt / TAU_W)
        clr_pen = np.zeros(N)
        hit = np.zeros(N, bool)
        k = 0.0
        while k < self.h - 1e-9:
            if k >= DEAD:
                acc = np.where((CAND[:, 0] < v) & (v <= 0.0), A_REV, A_FWD)
                v = v + np.clip(CAND[:, 0] - v, -acc * self.dt, acc * self.dt)
                w = w + aw * (CAND[:, 1] - w)
            th = th + w * self.dt
            x, y = x + v * np.cos(th) * self.dt, y + v * np.sin(th) * self.dt
            if self.deployable:
                c = np.minimum(env.lmap.clearance(np.c_[x, y]), 3.0) - env.robot_r
            else:
                c = grid_bilinear(env, env.clear, np.c_[x, y]) - env.robot_r
            clr_pen += np.maximum(0.0, 0.15 - c) ** 2         # a 1.0 m door leaves 0.107 m: don't price doors out
            hit |= c < 0.0
            k += self.dt
        if self.deployable:
            d = env.odo[2] - env.loc[2]
            c_, s_ = math.cos(d), math.sin(d)
            rx, ry = x - env.loc[0], y - env.loc[1]
            geo = grid_bilinear(env, env.plan_geo, np.c_[env.odo[0] + c_ * rx - s_ * ry, env.odo[1] + s_ * rx + c_ * ry])
        else:
            geo = grid_bilinear(env, env.geo, np.c_[x, y])
        cost = (geo + 2.0 * clr_pen + 50.0 * hit + 0.05 * np.abs(CAND[:, 1])
                + 0.15 * np.abs(CAND[:, 1] - self.prev[1]) + 0.1 * np.abs(CAND[:, 0] - self.prev[0]))
        if env.shield and env.filter is not None:
            v0, w0 = env._velocity_estimate()
            ok, _ = env.filter.admissible(env.lmap, env.loc.copy(), v0, w0, CAND)
            if ok.any():
                cost = np.where(ok, cost, cost + 1e3)
        i = int(np.argmin(cost))
        self.prev = CAND[i].copy()
        return CAND[i]

    def __call__(self, oa, env):
        if env.steps == 0:
            self.reset()
        v, w = self.plan(env)
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
        R = p.map(run, [(s, e) for s in scen for e in range(16)], chunksize=1)
    for s in scen:
        r = [x for k, x in R if k == s]
        print(f"{s:28s} success {np.mean([x['success'] for x in r]):5.1%} collision {np.mean([x['collision'] for x in r]):5.1%} "
              f"timeout {np.mean([x['timeout'] for x in r]):5.1%} spl {np.mean([x['spl'] for x in r]):.2f}")
    a = [x for _, x in R]
    print(f"ALL success {np.mean([x['success'] for x in a]):.1%} collision {np.mean([x['collision'] for x in a]):.1%} "
          f"timeout {np.mean([x['timeout'] for x in a]):.1%}")
