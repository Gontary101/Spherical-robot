"""
Local-map fidelity against ground truth (plan phase 1b acceptance).

Drives the privileged oracle through benchmark worlds while three maps are built from the robot's own data (odometry
pose, masked LiDAR, depth): LiDAR only, depth only, both. Every second, cells within 3 m of the robot are compared with
the true static obstacles (transformed odometry -> world with the current odometry error):
  false positive  map says occupied, true obstacle surface farther than 0.15 m
  false negative  true obstacle surface cell (seen by the LiDAR plane: top above the scan height) the map does not hold
    python rl/map_quality.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_nav2 import CALM, LOCO, SCEN  # noqa: E402
from local_map import LocalMap  # noqa: E402
from nav2_env import Nav2Env, visible_from_pods  # noqa: E402
from record_mission2 import oracle  # noqa: E402

LOW = os.path.join(os.path.dirname(__file__), "..", "rl/runs/loco31/best.pt")


def world_to_odo(env, pts, est=None):
    """Map world points into the estimate's frame using the current estimate error (rigid transform)."""
    est = env.odo if est is None else est
    pos, yaw, _ = env._pose()
    dth = est[2] - yaw
    c, s = math.cos(dth), math.sin(dth)
    rel = pts - pos
    return est[:2] + np.c_[c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]]


def evaluate(scen, n=6):
    stats = {k: dict(fp=0, occ=0, fn=0, tru=0) for k in ("lidar", "lidar+depth", "lidar@true_pose")}
    for ep in range(n):
        env = Nav2Env(seed=70000 + ep, low_ckpt=LOW, level=0.8, adaptive=False, scenario={"nav": CALM | SCEN[scen], "loco": LOCO})
        oa, _ = env.reset()
        maps = {k: LocalMap(size=192, cell=0.05) for k in stats}
        for m in maps.values():
            m.reset(env.odo[:2])
        # ground-truth obstacle surface points (world): boundary of the occupancy grid of the static obstacles
        occ = ~env.free | (env.clear < 0.06)
        surf = (env.clear < 0.06)
        tw = np.c_[env.gx[surf], env.gy[surf]]
        for step in range(400):
            dead = None
            if env.lidar_sector is not None:
                a0, w = env.lidar_sector
                dead = np.abs((env.beam_ang - a0 + np.pi) % (2 * np.pi) - np.pi) < w / 2
            dep = (env.cam_dirs, env.depth_last, 0.32, 0.25)
            maps["lidar"].update(env.odo.copy(), env.beam_ang, env.scans[-1], visible=visible_from_pods, dead=dead)
            maps["lidar+depth"].update(env.odo.copy(), env.beam_ang, env.scans[-1], depth=dep, visible=visible_from_pods, dead=dead)
            pos_t, yaw_t, _ = env._pose()
            maps["lidar@true_pose"].update(np.r_[pos_t, yaw_t], env.beam_ang, env.scans[-1], visible=visible_from_pods, dead=dead)
            (oa, _), _, done, info = env.step(oracle(oa, env))
            if step % 10 == 5:
                pos, yaw, _ = env._pose()
                near = np.linalg.norm(tw - pos, axis=1) < 3.0
                for k, m in maps.items():
                    if k == "lidar@true_pose":
                        tpts, ref = tw[near], pos
                    else:
                        tpts, ref = world_to_odo(env, tw[near]), env.odo[:2]
                    occm = m.occupied()
                    rr, cc = np.nonzero(occm)
                    if len(rr):
                        cpts = m.origin + (np.c_[cc, rr] + 0.5) * m.cell
                        close = np.linalg.norm(cpts - ref, axis=1) < 3.0
                        cpts = cpts[close]
                        if len(cpts) and len(tpts):
                            d = np.min(np.linalg.norm(cpts[:, None] - tpts[None], axis=2), axis=1)
                            stats[k]["fp"] += int((d > 0.15).sum())
                            stats[k]["occ"] += len(cpts)
                    if len(tpts):
                        cl = m.clearance(tpts)
                        stats[k]["fn"] += int((cl > 0.15).sum())
                        stats[k]["tru"] += len(tpts)
            if done:
                break
    return stats


if __name__ == "__main__":
    for scen in sys.argv[1:] or ("mixed", "rooms, 1.0 m doors", "forest (thin poles)"):
        st = evaluate(scen)
        print(scen)
        for k, s in st.items():
            print(f"   {k:6s} false-positive {s['fp'] / max(s['occ'], 1):6.1%} of occupied cells | missed true surface (> 15 cm) "
                  f"{s['fn'] / max(s['tru'], 1):6.1%}", flush=True)
