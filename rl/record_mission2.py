"""
Record an autonomous multi-waypoint mission in a navigation-v2 world, for rendering.

    python rl/record_mission2.py --family mixed --waypoints 5 --out media/mission2.npz

Logged at 50 Hz: spine pose, tyre-half / pendulum / bob angles, speed, yaw rate, roll, commands, and every pedestrian's
position. Logged at 10 Hz: the masked pod-LiDAR scan. Geometry stored for the renderer, all read back from the COMPILED
MuJoCo model (what the collision detector uses):
  static   world-space vertices + hull faces of every obstacle and wall mesh geom
  dyn      the pedestrian mesh (local vertices + faces)
  terrain  the heightfield (N x N heights) - rendered with MuJoCo's own triangulation
  probes   ray-cast samples (origin, direction, MuJoCo hit distance) against terrain + static obstacles; the renderer
           re-casts them against the Blender scene and reports the largest disagreement
Localisation: odometry is re-anchored to the true pose at each waypoint (GNSS / SLAM fix); between waypoints the
navigator runs on drifting encoder + gyro odometry, exactly as in training.
"""
import argparse
import json
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import terrain as TR  # noqa: E402
from nav2_env import ARENA, Nav2Env  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def compiled_geometry(env):
    m, d = env.m, env.d
    mujoco.mj_forward(m, d)
    out, worst = [], 0.0
    for o in env.static_specs:
        gid = m.geom(o["name"]).id
        assert m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH, o["name"]
        mid = m.geom_dataid[gid]
        va, vn, fa, fn = m.mesh_vertadr[mid], m.mesh_vertnum[mid], m.mesh_faceadr[mid], m.mesh_facenum[mid]
        assert vn == len(o["verts"]), f"{o['name']}: convex hull dropped vertices"
        Rm = d.geom_xmat[gid].reshape(3, 3)
        world = m.mesh_vert[va:va + vn].astype(float) @ Rm.T + d.geom_xpos[gid]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(o["quat"], float))
        want = np.asarray(o["verts"]) @ R.reshape(3, 3).T + o["pos"]
        worst = max(worst, float(np.abs(np.sort(want, 0) - np.sort(world, 0)).max()))
        out.append(dict(name=o["name"], kind=o["kind"], wall=o["wall"], verts=world.tolist(), faces=m.mesh_face[fa:fa + fn].tolist()))
    dyn = None
    if env.dyn:
        gid = m.geom("dyn0").id
        mid = m.geom_dataid[gid]
        va, vn, fa, fn = m.mesh_vertadr[mid], m.mesh_vertnum[mid], m.mesh_faceadr[mid], m.mesh_facenum[mid]
        dyn = dict(verts=m.mesh_vert[va:va + vn].astype(float).tolist(), faces=m.mesh_face[fa:fa + fn].tolist(),
                   offset=(d.geom_xpos[gid] - d.mocap_pos[0]).tolist())
    return out, dyn, worst


def probes(env, n=4000, seed=0):
    """Ray samples against terrain + static obstacles (robot and pedestrians excluded)."""
    m, d = env.m, env.d
    r = np.random.default_rng(seed)
    hide = [m.geom(f"dyn{i}").id for i in range(len(env.dyn))] + list(env.loco.tyres)
    saved = m.geom_group[hide].copy()
    m.geom_group[hide] = 5
    P = []
    for i in range(n):
        if i % 2 == 0:                                     # vertical rays: terrain and obstacle tops
            o = np.r_[r.uniform(-ARENA - 1, ARENA + 1, 2), 4.0]
            v = np.array([0, 0, -1.0])
        else:                                              # oblique rays: obstacle sides
            o = np.r_[r.uniform(-ARENA, ARENA, 2), r.uniform(0.15, 1.0)]
            a, el = r.uniform(-np.pi, np.pi), r.uniform(-0.4, 0.1)
            v = np.array([math.cos(a) * math.cos(el), math.sin(a) * math.cos(el), math.sin(el)])
        g = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(m, d, o, v, np.array([1, 1, 0, 0, 0, 0], np.uint8), 1, -1, g)
        if dist >= 0 and dist < 12:
            P.append(np.r_[o, v, dist])
    m.geom_group[hide] = saved
    return np.array(P)


def new_goal(env, rng, lo=4.0, hi=10.0):
    from scipy.ndimage import distance_transform_edt  # noqa: F401
    pos = env.d.qpos[0:2]
    geo_from = env._geodesic(env.free, env._cell(pos))
    ok = np.argwhere(env.free & np.isfinite(geo_from) & (geo_from > lo) & (geo_from < hi))
    clear_ok = [c for c in ok if env.free[max(c[0] - 3, 0):c[0] + 4, max(c[1] - 3, 0):c[1] + 4].all()]
    if not clear_ok:
        return None
    c = clear_ok[int(rng.integers(len(clear_ok)))]
    return np.array([env.gx[tuple(c)], env.gy[tuple(c)]]), tuple(c)


def oracle(oa, env):
    """Privileged baseline: follow the geodesic gradient of the static map (true pose), slow down near anything."""
    pos, yaw, t = env._pose()
    _, g = env._geo_at(pos)
    if np.linalg.norm(g) < 1e-6:
        g = env.goal - pos
    err = (math.atan2(g[1], g[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
    clear = float(env._true_scan(pos, yaw).min()) - 0.36
    v = float(np.clip(min(2.0, 0.4 + 1.5 * clear), 0.2, 2.0)) * max(0.0, math.cos(err)) ** 2
    w = float(np.clip(2.0 * err, -2.5, 2.5))
    return np.array([(v - (-1.0)) / 4.5 * 2 - 1, w / 2.5])


def mission(seed, family, n_wp, k, n_dyn, nav, low):
    env = Nav2Env(seed=seed, low_ckpt=low, level=k, adaptive=False,
                  scenario={"nav": dict(family=family, n_dyn=n_dyn, lidar_sector=False, depth_blackout=False),
                            "loco": None})
    oa, oc = env.reset()
    rng = np.random.default_rng(seed + 5)
    lo = env.loco
    frames, scans = [], []
    orig = lo.step

    def rec(a):
        out = orig(a)
        t = lo._truth()
        dyn = np.concatenate([o["p"] for o in env.dyn]) if env.dyn else np.zeros(0)
        frames.append(np.concatenate([[env.d.time], env.d.qpos[0:7], [lo.q("tyreL"), lo.q("tyreR"), lo.q("yoke"), lo.q("bob")],
                                      [t["v"], t["w"], t["roll"]], lo.cmd, dyn]))
        return out
    lo.step = rec
    static, dyn, worst = compiled_geometry(env)
    wps, reached = [env.goal.copy()], []
    while True:
        (oa, oc), r, done, info = env.step(nav(oa, env))
        scans.append(np.concatenate([[env.d.time], env.scans[-1]]))
        if info["success"]:
            reached.append(float(env.d.time))
            if len(reached) == n_wp:
                break
            g = new_goal(env, rng)
            if g is None:
                break
            env.goal = g[0]
            env.geo = env._geodesic(env.free, g[1])
            env.geo[~np.isfinite(env.geo)] = 99.0
            env.geo_prev, _ = env._geo_at(env.d.qpos[0:2])
            wps.append(env.goal.copy())
            pos, yaw, _ = env._pose()
            env.odo = np.array([pos[0], pos[1], yaw])
            env.steps = 0
            continue
        if done:
            break
    return dict(env=env, frames=np.array(frames), scans=np.array(scans), wps=np.array(wps), reached=reached,
                ok=len(reached) == n_wp, info=info, static=static, dyn=dyn, worst=worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="mixed")
    ap.add_argument("--waypoints", type=int, default=5)
    ap.add_argument("--level", type=float, default=0.8)
    ap.add_argument("--n-dyn", type=int, default=4)
    ap.add_argument("--nav", default=os.path.join(ROOT, "rl/runs/nav2/actor.ts"))
    ap.add_argument("--low", default=os.path.join(ROOT, "rl/runs/loco2/best.pt"))
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--min-time", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "media", "mission2.npz"))
    a = ap.parse_args()
    if a.nav == "oracle":
        nav = oracle
    else:
        import torch
        torch.set_num_threads(1)
        ts = torch.jit.load(a.nav)
        nav = lambda o, env: ts(torch.as_tensor(o[None])).detach().numpy()[0]  # noqa: E731
    for seed in range(a.seed0, a.seed0 + a.seeds):
        res = mission(seed, a.family, a.waypoints, a.level, a.n_dyn, nav, a.low)
        env = res["env"]
        print(f"seed {seed}: reached {len(res['reached'])}/{a.waypoints} t={res['frames'][-1, 0]:.1f}s terrain={env.loco.S['family']} "
              f"static={len(res['static'])} dyn={len(env.dyn)} end={ {k: v for k, v in res['info'].items() if v is True} }", flush=True)
        if res["ok"] and res["frames"][-1, 0] >= a.min_time:
            pr = probes(env)
            h = env.loco.hmap
            print(f"collision meshes: {len(res['static'])} convex mesh geoms, max |spec - compiled| vertex error {res['worst']:.2e} m; "
                  f"{len(pr)} ray probes", flush=True)
            np.savez_compressed(a.out, frames=res["frames"], scans=res["scans"], wps=res["wps"], reached=np.array(res["reached"]),
                                beam_ang=env.beam_ang, static=json.dumps(res["static"]), dyn=json.dumps(res["dyn"]),
                                n_dyn=len(env.dyn), dyn_z=np.array([o["z"] for o in env.dyn]), terrain=(h.astype(np.float32) if h is not None else np.zeros((0, 0), np.float32)),
                                terrain_half=TR.HALF, probes=pr, seed=seed, family=a.family, terrain_family=env.loco.S["family"],
                                mu=env.loco.S["mu"])
            print("saved", a.out)
            return
    print("no fully successful mission found")


if __name__ == "__main__":
    main()
