"""
Evaluate autonomous navigation: learned asymmetric-PPO navigator vs a classical reactive baseline
(VFH-style gap follower using the same masked pod-LiDAR scan and the same drifting odometry).
Both drive through the same frozen learned locomotion policy, on the same 100 random maps.

    python rl/eval_nav.py --nav rl/runs/nav/best.pt --loco rl/runs/loco/best.pt [--maps 100]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from nav_env import N_BEAM, V_RANGE, W_RANGE, NavEnv, NumpyActor  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class VFHBaseline:
    """Pick the free beam direction closest to the (odometry) goal bearing; slow down near obstacles."""

    def __call__(self, env):
        scan = env.scan_prev
        g = env.goal - env.odo[:2]
        bearing = np.arctan2(g[1], g[0]) - env.odo[2]
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        ang = env.beam_ang
        # inflate obstacles: a direction is free if all beams within +-20 deg see > 1.1 m
        k = int(round(np.radians(20) / (2 * np.pi / N_BEAM)))
        free = np.array([scan[np.arange(i - k, i + k + 1) % N_BEAM].min() > 1.1 for i in range(N_BEAM)])
        if not free.any():
            return np.array([-0.6, 1.0])                         # back off and turn
        cost = np.abs(np.arctan2(np.sin(ang - bearing), np.cos(ang - bearing)))
        cost[~free] = 1e9
        target = ang[np.argmin(cost)]
        w = np.clip(2.0 * target, -W_RANGE, W_RANGE)
        front = scan[np.abs(ang) < np.radians(25)].min()
        v = np.clip(0.8 * (front - 0.8), 0.0, 2.5) * max(0.0, np.cos(target))
        v = min(v, 0.6 * np.linalg.norm(g) + 0.3)
        return np.array([2 * (v - V_RANGE[0]) / (V_RANGE[1] - V_RANGE[0]) - 1, w / W_RANGE])


def run(kind, nav, loco_ckpt, n_maps):
    env = NavEnv(seed=12345, low_level="learned", loco_ckpt=loco_ckpt, randomize=True)
    out = []
    for i in range(n_maps):
        env.rng = np.random.default_rng(9000 + i)
        env.loco.rng = np.random.default_rng(9000 + i)
        oa, oc = env.reset()
        start, goal = env.start.copy(), env.goal.copy()
        path, t = 0.0, 0
        last = env.d.qpos[0:2].copy()
        while True:
            a = nav(oa) if kind == "learned" else nav(env)
            (oa, oc), r, done, info = env.step(a)
            p = env.d.qpos[0:2].copy()
            path += np.linalg.norm(p - last)
            last = p
            t += 1
            if done:
                break
        out.append({"success": bool(info["success"]), "collision": bool(info["collision"]), "fell": bool(info["fell"]),
                    "time_s": t * 0.1, "path_m": path, "straight_m": float(np.linalg.norm(goal - start)),
                    "final_dist": float(info["dist"])})
    s = [o for o in out if o["success"]]
    return {"success_rate": float(np.mean([o["success"] for o in out])),
            "collision_rate": float(np.mean([o["collision"] for o in out])),
            "fall_rate": float(np.mean([o["fell"] for o in out])),
            "mean_time_success_s": float(np.mean([o["time_s"] for o in s])) if s else None,
            "path_efficiency": float(np.mean([o["straight_m"] / o["path_m"] for o in s])) if s else None,
            "mean_speed_success_ms": float(np.mean([o["path_m"] / o["time_s"] for o in s])) if s else None}, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", default=os.path.join(ROOT, "rl/runs/nav/best.pt"))
    ap.add_argument("--loco", default=os.path.join(ROOT, "rl/runs/loco/best.pt"))
    ap.add_argument("--maps", type=int, default=100)
    a = ap.parse_args()
    res = {}
    for kind, nav in (("vfh_baseline", VFHBaseline()), ("learned", NumpyActor(a.nav))):
        summary, _ = run("learned" if kind == "learned" else "baseline", nav, a.loco, a.maps)
        res[kind] = summary
        print(kind, json.dumps(summary, indent=1), flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(a.nav), "eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
