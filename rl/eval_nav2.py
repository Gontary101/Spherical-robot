"""
Navigation v2 benchmark: fixed-seed scenarios x controllers.

    python rl/eval_nav2.py --policies nav2 vfh oracle --episodes 20 --out docs/nav_robustness
Controllers (all drive through the same frozen locomotion v2 policy):
  nav2    learned local planner (rl/runs/nav2/actor.ts): masked LiDAR, depth, odometry, planner subgoal
  vfh     classical vector-field-histogram gap follower on the SAME masked LiDAR, odometry and planner subgoal
  oracle  privileged: true pose + true geodesic field of the full static map (an upper bound for static scenes;
          it ignores pedestrians beyond slowing down near anything)
Metrics: success, collision (static / pedestrian), fall, timeout, SPL (success weighted by geodesic / driven path).
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from nav2_env import N_BEAM, V_RANGE, W_RANGE, Nav2Env  # noqa: E402
from record_mission2 import oracle  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CALM = dict(lidar_sector=False, lidar_spurious=0.0, depth_blackout=False, depth_drop=0.0, n_dyn=0, map=True, map_stale=0.0)
SCEN = {
    "clutter": dict(family="clutter"),
    "forest (thin poles)": dict(family="forest"),
    "rooms, 1.0 m doors": dict(family="rooms", door=1.0),
    "maze": dict(family="maze"),
    "mixed": dict(family="mixed"),
    "maze, no map": dict(family="maze", map=False),
    "mixed, 40 % stale map": dict(family="mixed", map_stale=0.4),
    "6 pedestrians to 1.4 m/s": dict(family="clutter", n_dyn=6, dyn_speed=(1.0, 1.4)),
    "LiDAR 60 deg blind sector": dict(family="mixed", lidar_sector=True),
    "depth camera dead": dict(family="mixed", depth_blackout=True),
    "dust: spurious returns 3 %": dict(family="mixed", lidar_spurious=0.03),
    "everything at once": dict(family="mixed", n_dyn=4, map_stale=0.3, lidar_sector=True, lidar_spurious=0.01, depth_drop=0.2),
}
# ground / robot conditions for every scenario: rough terrain, mid friction, light disturbances
LOCO = dict(k=0.5, family="rough", terrain_k=0.5, slope=0.0, slope_dir=0.0, mu=0.7, mu2=None, mu_switch_t=8.0, tors=0.015,
            roll_fr=0.005, mscale=np.ones(4), com=np.zeros(3), payload=3.0, payload_pos=np.array([0.0, 0.05, 0.05]),
            tyre_r=0.3, strength=np.array([0.9, 0.9]), tau_noise=0.03, bob_slew=np.radians(60), delay=1, push_rate=1 / 6.0,
            push_max=150.0, kick_max=10.0, wind=10.0, wind_dir=0.5, gust=4.0, gyro_bias=np.array([0.01, -0.01, 0.01]),
            imu_tilt=np.radians([0.5, -0.5]), glitch=0.0, v_max=3.5)


class VFH:
    """Free-direction picker on the masked scan, steering to the planner subgoal (odometry frame)."""
    def __call__(self, oa, env):
        scan = env.scans[-1]
        sgp = env.subgoal(env.odo[:2])
        g = sgp - env.odo[:2]
        bearing = math.atan2(g[1], g[0]) - env.odo[2]
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        ang = env.beam_ang
        k = int(round(np.radians(20) / (2 * np.pi / N_BEAM)))
        free = np.array([scan[np.arange(i - k, i + k + 1) % N_BEAM].min() > env.robot_r + 0.6 for i in range(N_BEAM)])
        if not free.any():
            return np.array([-0.6, 1.0])
        cost = np.abs(np.arctan2(np.sin(ang - bearing), np.cos(ang - bearing)))
        cost[~free] = 1e9
        target = ang[np.argmin(cost)]
        w = np.clip(2.0 * target, -W_RANGE, W_RANGE)
        front = scan[np.abs(ang) < np.radians(25)].min()
        v = np.clip(0.8 * (front - 0.8), 0.0, 2.5) * max(0.0, math.cos(target))
        v = min(v, 0.6 * np.linalg.norm(env.goal - env.odo[:2]) + 0.3)
        return np.array([2 * (v - V_RANGE[0]) / (V_RANGE[1] - V_RANGE[0]) - 1, w / W_RANGE])


def make(name, path):
    if name == "vfh":
        return VFH()
    if name == "oracle":
        return oracle
    import torch
    torch.set_num_threads(1)
    ts = torch.jit.load(path)
    return lambda oa, env: ts(torch.as_tensor(oa[None])).detach().numpy()[0]


def job(args):
    scen, pol, path, n, low = args
    os.environ["GYRA_LOW"] = low
    nav = make(pol, path)
    out = []
    for ep in range(n):
        env = Nav2Env(seed=70000 + ep, low_ckpt=low, level=0.8, adaptive=False,
                      scenario={"nav": CALM | SCEN[scen], "loco": LOCO})
        oa, oc = env.reset()
        while True:
            (oa, oc), r, done, info = env.step(nav(oa, env))
            if done:
                out.append(info["episode"])
                break
    return scen, pol, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["nav2", "vfh", "oracle"])
    ap.add_argument("--nav", default=os.path.join(ROOT, "rl/runs/nav2/actor.ts"))
    ap.add_argument("--low", default=os.path.join(ROOT, "rl/runs/loco2/best.pt"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "nav_robustness"))
    a = ap.parse_args()
    names = a.scenarios or list(SCEN)
    jobs = [(s, p, a.nav, a.episodes, os.path.abspath(a.low)) for s in names for p in a.policies]
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(job, jobs, chunksize=1)
    table = {}
    for s, p, out in res:
        f = lambda k: float(np.mean([o[k] for o in out]))  # noqa: E731
        table.setdefault(s, {})[p] = dict(success=f("success"), collision=f("collision"), hit_dynamic=f("hit_dynamic"),
                                          fell=f("fell"), timeout=f("timeout"), spl=f("spl"))
    json.dump(table, open(a.out + ".json", "w"), indent=1)
    lines = ["| scenario | " + " | ".join(f"{p} success | {p} collision | {p} SPL" for p in a.policies) + " |",
             "|---|" + "---|---|---|" * len(a.policies)]
    for s in names:
        cells = []
        for p in a.policies:
            t = table[s][p]
            cells += [f"{100 * t['success']:.0f} %", f"{100 * t['collision']:.0f} %", f"{t['spl']:.2f}"]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    open(a.out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
