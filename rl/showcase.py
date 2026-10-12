"""
Showcase mission for the video: one continuous 50 x 50 m course, recorded with the deployed stack
(deployable MPC navigator -> safety filter -> locomotion policy v3.1 on GYRA Mk2.1).

Zones, in route order:
  A  rolling hills (max slope ~8 deg) with boulders to weave around        south band
  B  a forest of thin poles (r 6-12 cm)                                     east
  C  a building: three rooms in a row joined by 1.1 m doors                north-east
  D  open field with long straights (speed up to 3.5 m/s)                   north-west / west
  E  boulder garden, finish                                                 centre
Wind is on (5 N + gusts); no pedestrians and no shoves, so the footage shows the stack as deployed.

    python rl/showcase.py --out media/showcase.npz
The .npz follows rl/record_mission2.py (read by render/animate_mission.py) plus the HUD channels.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import nav2_env as NE  # noqa: E402

NE.ARENA = 25.0                                   # before any env is built: grids are sized from it
import terrain as TR  # noqa: E402
from eval_nav2 import LOCO  # noqa: E402
from local_map import LocalMap  # noqa: E402
from nav2_env import GRID, PLAN_MARGIN, Nav2Env, _box, _cyl, _footprint  # noqa: E402
from gyra2_model import obstacle_vertices  # noqa: E402
from teacher import GeoMPC  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
START = np.array([-21.0, -19.5])
# (waypoint, zone label shown on screen)
ROUTE = [((-8.0, -19.0), "Rolling hills"), ((6.0, -18.0), "Rolling hills"), ((18.0, -17.0), "Rolling hills"),
         ((17.5, -6.0), "Pole forest"), ((17.0, 6.5), "Pole forest"),
         ((16.0, 16.0), "Doorways"), ((8.5, 16.0), "Doorways"), ((1.0, 16.0), "Doorways"), ((-5.0, 16.0), "Doorways"),
         ((-21.0, 17.0), "Open field"), ((-21.0, -5.0), "Open field"),
         ((-10.0, -2.0), "Boulder garden"), ((2.0, 2.0), "Boulder garden"),
         ((-9.0, -14.5), "Back over the hills"), ((-21.0, -10.0), "Back over the hills"),
         ((-14.0, 12.0), "Final sprint"), ((-7.0, 20.0), "Final sprint")]


def showcase_terrain():
    """Rolling hills in the south band (two crossing swells, max slope ~8 deg), a gentle swell in the open field,
    fine gravel texture everywhere, flat under the building."""
    X, Y = TR._grid()
    rng = np.random.default_rng(7)
    h = np.zeros_like(X)
    band = np.clip((-11.0 - Y) / 3.0, 0, 1) * np.clip((Y + 24.0) / 2.0, 0, 1)          # zone A, -24 < y < -11
    h += band * (0.22 * np.sin(2 * np.pi * X / 14.0 + 0.6) + 0.12 * np.sin(2 * np.pi * (X * 0.6 + Y) / 11.0))
    field = np.clip((-3.0 - X) / 3.0, 0, 1) * np.clip((Y + 8.0) / 3.0, 0, 1) * np.clip((24.0 - Y) / 2.0, 0, 1)
    h += field * 0.10 * np.sin(2 * np.pi * Y / 16.0)
    building = (X > -3.5) & (X < 21.5) & (Y > 8.5) & (Y < 23.5)
    h += 0.008 * TR._band_noise(rng, 0.3, 0.6) * ~building
    edge = np.clip((TR.HALF - 1.0 - np.maximum(np.abs(X), np.abs(Y))) / 2.0, 0, 1)
    h = h * edge
    i, j = np.argmin(np.abs(np.linspace(-TR.HALF, TR.HALF, TR.N) - START[1])), np.argmin(np.abs(np.linspace(-TR.HALF, TR.HALF, TR.N) - START[0]))
    return h - h[i, j]


def showcase_layout():
    r = np.random.default_rng(11)
    L = NE.ARENA
    obs = [_box(f"wall{i}", x, y, sx, sy, 0.8, wall=True)
           for i, (x, y, sx, sy) in enumerate([(L, 0, .1, L + .1), (-L, 0, .1, L + .1), (0, L, L + .1, .1), (0, -L, L + .1, .1)])]
    # A: boulders on the hills, staggered across the route
    for k, (x, y) in enumerate([(-15, -18.2), (-11.5, -20.8), (-4, -17.6), (0.5, -20.5), (4, -16.6), (10, -19.8), (13.5, -16.8)]):
        obs.append(_cyl(f"rock{k}", x + r.uniform(-0.3, 0.3), y, r.uniform(0.45, 0.75), r.uniform(0.5, 0.9)))
    # B: pole forest, jittered grid
    for ix, x in enumerate(np.arange(13.0, 23.5, 2.2)):        # gaps >= ~1.1 m: a slalom for a 0.79 m robot
        for iy, y in enumerate(np.arange(-10.0, 5.5, 2.2)):
            px, py = x + 1.1 * (iy % 2) + r.uniform(-0.3, 0.3), y + r.uniform(-0.3, 0.3)
            obs.append(_cyl(f"pole{ix}_{iy}", px, py, r.uniform(0.06, 0.12), r.uniform(1.3, 2.0)))
    # C: building x in [-2, 20], y in [11, 21], three rooms, wall 0.12 thick, 0.9 m high, doors 1.1 m
    t, hgt, dw = 0.06, 0.9, 1.1

    def wall_x(y, x0, x1, doors=()):                     # wall along x at y, with door centres
        segs, a = [], x0
        for dc in sorted(doors):
            segs.append((a, dc - dw / 2))
            a = dc + dw / 2
        segs.append((a, x1))
        for s0, s1 in segs:
            if s1 - s0 > 0.05:
                obs.append(_box(f"bw{len(obs)}", (s0 + s1) / 2, y, (s1 - s0) / 2, t, hgt, wall=True))

    def wall_y(x, y0, y1, doors=()):
        segs, a = [], y0
        for dc in sorted(doors):
            segs.append((a, dc - dw / 2))
            a = dc + dw / 2
        segs.append((a, y1))
        for s0, s1 in segs:
            if s1 - s0 > 0.05:
                obs.append(_box(f"bw{len(obs)}", x, (s0 + s1) / 2, t, (s1 - s0) / 2, hgt, wall=True))
    wall_x(11.0, -2.0, 20.0, doors=(17.0,))              # south facade, entrance
    wall_x(21.0, -2.0, 20.0)                             # north facade
    wall_y(20.0, 11.0, 21.0)                             # east facade
    wall_y(-2.0, 11.0, 21.0, doors=(16.0,))              # west facade, exit
    wall_y(12.5, 11.0, 21.0, doors=(16.0,))              # interior walls
    wall_y(5.0, 11.0, 21.0, doors=(15.0,))
    for k, (x, y, sx, sy) in enumerate([(18.0, 19.5, 0.6, 0.4), (14.0, 12.2, 0.5, 0.35), (9.0, 19.6, 0.8, 0.35),
                                        (3.0, 12.3, 0.6, 0.3)]):
        obs.append(_box(f"crate{k}", x, y, sx, sy, 0.7))  # furniture
    # D: open field, a few slalom markers along the west straight
    for k, y in enumerate((10.0, 4.0, -1.0)):
        obs.append(_cyl(f"cone{k}", -21.0 + (1.4 if k % 2 else -1.4), y, 0.2, 0.7))
    # E: boulder garden
    for k in range(16):
        while True:
            x, y = r.uniform(-17, 6), r.uniform(-8, 6)
            if np.hypot(x - 2.0, y - 2.0) > 2.2 and np.hypot(x + 21, y + 5) > 3:
                break
        if r.random() < 0.5:
            obs.append(_cyl(f"bould{k}", x, y, r.uniform(0.3, 0.7), r.uniform(0.4, 1.0)))
        else:
            obs.append(_box(f"block{k}", x, y, r.uniform(0.25, 0.6), r.uniform(0.25, 0.6), r.uniform(0.4, 1.1), yaw=r.uniform(0, np.pi)))
    return obs


class ShowcaseEnv(Nav2Env):
    def __init__(self, low_ckpt):
        loco = dict(LOCO, push_rate=0.0, wind=5.0, gust=2.0, payload=2.0)
        super().__init__(seed=0, low_ckpt=low_ckpt, level=0.6, adaptive=False, scenario={"nav": {}, "loco": loco}, shield=True)
        H = showcase_terrain()
        self.loco._terrain = lambda family, k: H
        self.lmap = LocalMap(size=320, cell=0.05)              # 16 m: braking from 3.5 m/s needs ~6.3 m verified free

    def _sample_params(self):
        P = super()._sample_params()
        P.update(family="showcase", n_dyn=0, lidar_sector=False, lidar_spurious=0.0, depth_blackout=False, depth_drop=0.0,
                 map=True, map_stale=0.0, odom="lio", goal_max=1e9)
        return P

    def snap(self, p):
        """Nearest planner-free cell centre to p (waypoints must be reachable)."""
        i, j = self._cell(p)
        if self.plan_free[i, j]:
            return np.asarray(p, float)
        free = np.argwhere(self.plan_free)
        k = int(np.argmin(np.hypot(free[:, 0] - i, free[:, 1] - j)))
        return np.array([self.gx[tuple(free[k])], self.gy[tuple(free[k])]])

    def _world(self, S, hmap):
        from scipy.ndimage import distance_transform_edt as edt
        obs = showcase_layout()
        occ = self._occupancy(obs)
        clear = edt(~occ) * GRID
        self.free = clear > self.robot_r + PLAN_MARGIN
        self.clear = clear
        self.plan_clear, self.plan_free = clear, self.free
        self.start = START.copy()
        self.set_goal(self.snap(ROUTE[0][0]))
        self.static_specs = []
        for o in obs:
            fp = _footprint(o)
            zs = TR.height_at(hmap, fp[:, 0], fp[:, 1]) if hmap is not None else np.zeros(len(fp))
            zb, zt = float(zs.min()) - 0.05, float(zs.max()) + o["h"]
            hh = (zt - zb) / 2
            size = (o["size"][0], hh) if o["kind"] == "cylinder" else (o["size"][0], o["size"][1], hh)
            self.static_specs.append(dict(name=o["name"], kind=o["kind"], wall=o["wall"], verts=obstacle_vertices(o["kind"], size),
                                          pos=(o["xy"][0], o["xy"][1], zb + hh), quat=(np.cos(o["yaw"] / 2), 0, 0, np.sin(o["yaw"] / 2))))
        self.dyn, self.dyn_free = [], clear > 0.5
        return self.static_specs, []


def run(low, out, max_time=420.0):
    from record_mission2 import compiled_geometry, probes
    env = ShowcaseEnv(low)
    oa, _ = env.reset()
    # face the first waypoint at the start
    nav = GeoMPC(deployable=True, v_max=3.5)
    lo = env.loco
    frames, scans, hud = [], [], []
    orig = lo.step

    def rec(a):
        res = orig(a)
        t = lo._truth()
        frames.append(np.concatenate([[env.d.time], env.d.qpos[0:7], [lo.q("tyreL"), lo.q("tyreR"), lo.q("yoke"), lo.q("bob")],
                                      [t["v"], t["w"], t["roll"]], lo.cmd]))
        return res
    lo.step = rec
    static, dyn, worst = compiled_geometry(env)
    wp, reached = 0, []
    t0 = env.d.time
    while env.d.time - t0 < max_time:
        a = nav(oa, env)
        (oa, _), _, done, info = env.step(a)
        t = lo._truth()
        ego = env.lmap.egocentric2(env.loc, 64, 0.2)          # 12.8 m HUD map: occupied / known-free
        hud.append(dict(t=float(env.d.time), wp=wp, zone=ROUTE[wp][1], v=float(t["v"]), w=float(t["w"]), pitch=float(t["pitch"]),
                        roll=float(t["roll"]), pend=float(t["pend"]), interv=bool(info.get("intervened", False)),
                        env=oa[-NE.N_ENV - NE.MAP_OBS:-NE.MAP_OBS].tolist()))
        scans.append(np.concatenate([[env.d.time], env.scans[-1]]))
        if not hasattr(run, "maps"):
            run.maps = []
        run.maps.append((np.clip(ego[0] * 255, 0, 255).astype(np.uint8), (ego[1] * 255).astype(np.uint8)))
        if info.get("collision") or info.get("fell"):
            print("FAILED", {k: v for k, v in info.items() if v is True}, "at t", env.d.time, flush=True)
            break
        if info["success"]:
            reached.append(float(env.d.time))
            print(f"waypoint {wp + 1}/{len(ROUTE)} ({ROUTE[wp][1]}) at t={env.d.time - t0:.1f}s", flush=True)
            wp += 1
            if wp == len(ROUTE):
                break
            env.set_goal(env.snap(ROUTE[wp][0]))
            env.geo_prev, _ = env._geo_at(env.d.qpos[0:2])
            env.steps = 0
            continue
        if done:
            print("ended", {k: v for k, v in info.items() if v is True}, flush=True)
            break
    ok = wp == len(ROUTE)
    print(f"{'COMPLETE' if ok else 'INCOMPLETE'}: {len(reached)}/{len(ROUTE)} waypoints, {env.d.time - t0:.1f} s, "
          f"path {env.path_len:.0f} m, interventions {np.mean([h['interv'] for h in hud]):.0%}", flush=True)
    pr = probes(env)
    M = run.maps
    np.savez_compressed(out, frames=np.array(frames), scans=np.array(scans), wps=np.array([w for w, _ in ROUTE]),
                        zones=json.dumps([z for _, z in ROUTE]), reached=np.array(reached), beam_ang=env.beam_ang,
                        static=json.dumps(static), dyn=json.dumps(dyn), n_dyn=0, dyn_z=np.zeros(0),
                        terrain=lo.hmap.astype(np.float32), terrain_half=TR.HALF, probes=pr, seed=0, family="showcase",
                        terrain_family="showcase", mu=lo.S["mu"], hud=json.dumps(hud),
                        map_occ=np.array([m[0] for m in M]), map_free=np.array([m[1] for m in M]), complete=ok)
    print("saved", out)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", default=os.path.join(ROOT, "rl/runs/loco31/best.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "media/showcase.npz"))
    ap.add_argument("--max-time", type=float, default=420.0)
    a = ap.parse_args()
    run(a.low, a.out, a.max_time)
