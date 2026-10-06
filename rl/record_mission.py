"""
Record an autonomous multi-waypoint mission for video rendering.

The learned navigator (10 Hz) drives the learned locomotion policy (50 Hz) through a dense
obstacle arena, visiting N waypoints in sequence. Every 20 ms the full robot state is logged
(spine pose, both tyre-half angles, pendulum and bob angles, speed, commands), plus the
masked pod-LiDAR scan at each navigation step.

Localisation note: the navigator's odometry is re-anchored to the true pose at each new
waypoint, standing in for the GNSS / LiDAR-SLAM fix a real robot would have. Between
waypoints it runs on drifting wheel/gyro odometry exactly as in training.

    python rl/record_mission.py --waypoints 5 --out media/mission.npz
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from nav_env import ARENA, NavEnv, NumpyActor  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def free_point(env, rng, near=None, min_d=4.0, max_d=9.0, clear=1.1):
    L = ARENA - 0.9
    for _ in range(2000):
        p = rng.uniform(-L, L, 2)
        if near is not None and not (min_d < np.linalg.norm(p - near) < max_d):
            continue
        if all(np.linalg.norm(p - q) > clear + 0.3 for q in env.obstacles):
            return p
    return None


def mission(seed, n_wp, nav, loco_ckpt):
    env = NavEnv(seed=seed, low_level="learned", loco_ckpt=loco_ckpt, randomize=False)
    env.rng = np.random.default_rng(seed)
    env.loco.rng = np.random.default_rng(seed)
    oa, oc = env.reset()       # new world: every obstacle/wall is a convex mesh geom (collision == rendered geometry)
    rng = np.random.default_rng(seed + 1)
    wps = [env.goal.copy()]
    for _ in range(n_wp - 1):
        p = free_point(env, rng, near=wps[-1])
        if p is None:
            return None
        wps.append(p)
    frames, scans = [], []
    lo = env.loco
    orig_step = lo.step

    def rec_step(a):
        out = orig_step(a)
        d = env.d
        t = lo._truth()
        frames.append(np.concatenate([[d.time], d.qpos[0:7], [lo._q("tyreL"), lo._q("tyreR"), lo._q("yoke"), lo._q("bob")],
                                      [t["v"], t["w"], t["roll"]], lo.cmd]))
        return out
    lo.step = rec_step
    wp_i, reached = 0, []
    while True:
        (oa, oc), r, done, info = env.step(nav(oa))
        scans.append(np.concatenate([[env.d.time], env.scan_prev]))
        if info["success"]:
            reached.append(float(env.d.time))
            wp_i += 1
            if wp_i == len(wps):
                return dict(env=env, frames=np.array(frames), scans=np.array(scans), wps=np.array(wps),
                            reached=reached, ok=True)
            env.goal = wps[wp_i].copy()
            pos, yaw, _ = env._pose()
            env.odo = np.array([pos[0], pos[1], yaw])          # localisation fix at the waypoint
            env.prev_dist = np.linalg.norm(env.goal - pos)
            env.steps = 0
            continue
        if done:
            return dict(env=env, frames=np.array(frames), scans=np.array(scans), wps=np.array(wps),
                        reached=reached, ok=False, info=info)


def obstacles(env):
    """Compiled MuJoCo collision geometry (world vertices + faces) and a check against the spec."""
    geo = env.mesh_world_geometry()
    worst = 0.0
    for g, spec in zip(geo, env.mesh_specs):
        assert g["geom_type"] == int(mujoco.mjtGeom.mjGEOM_MESH), g["name"]
        assert len(g["verts"]) == g["n_input_verts"], f"{g['name']}: hull dropped vertices"
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(spec["quat"], float))
        want = np.asarray(spec["verts"]) @ R.reshape(3, 3).T + spec["pos"]
        got = np.asarray(g["verts"])
        d = max(np.min(np.linalg.norm(want[:, None] - got[None], axis=2), axis=1).max(),
                np.min(np.linalg.norm(got[:, None] - want[None], axis=2), axis=1).max())
        worst = max(worst, float(d))
    return geo, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waypoints", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "media", "mission.npz"))
    ap.add_argument("--seeds", type=int, default=40)
    a = ap.parse_args()
    nav = NumpyActor(os.path.join(ROOT, "rl/runs/nav/best.pt"))
    loco = os.path.join(ROOT, "rl/runs/loco/best.pt")
    for seed in range(100, 100 + a.seeds):
        res = mission(seed, a.waypoints, nav, loco)
        if res is None:
            continue
        n_obs = len(res["env"].obs_gid)
        print(f"seed {seed}: ok={res['ok']} reached {len(res['reached'])}/{a.waypoints} obstacles={n_obs} "
              f"t={res['frames'][-1, 0]:.1f}s", flush=True)
        if res["ok"] and n_obs >= 14:
            geo, worst = obstacles(res["env"])
            print(f"collision meshes: {len(geo)} convex mesh geoms, max |spec - compiled| vertex error = {worst:.2e} m")
            np.savez_compressed(a.out, frames=res["frames"], scans=res["scans"], wps=res["wps"],
                                reached=np.array(res["reached"]), beam_ang=res["env"].beam_ang,
                                obstacles=json.dumps(geo), seed=seed)
            print("saved", a.out)
            return
    print("no fully successful mission found")


if __name__ == "__main__":
    main()
