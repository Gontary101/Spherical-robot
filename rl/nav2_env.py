"""
GYRA Mk2 navigation v2: "works everywhere" autonomous driving environment.

Hierarchy: navigation policy (10 Hz, this env) -> (v_cmd, w_cmd) -> locomotion v2 policy (50 Hz, frozen, run inside
the env) -> motor torques. Every episode compiles a fresh MuJoCo world: terrain heightfield (from the locomotion v2
generator), every static obstacle and wall as a convex mesh geom, every moving obstacle as a mocap body with a mesh
geom. Collision, LiDAR, depth and rendering all use those same vertices.

World families (curriculum level k in [0, 1] scales density, door width, dynamics, disturbances):
  clutter   boxes and cylinders of all sizes
  forest    many thin poles (5-15 cm): hard for a 72-beam LiDAR
  rooms     2x2 / 3x3 rooms joined by doors down to 1.0 m (robot is 0.70 m wide)
  maze      corridors of a random maze, 2.4-3 m wide, with clutter
  mixed     rooms + clutter + poles
Moving obstacles: up to 6 pedestrians (r 0.25 m, 1.6 m tall) at 0.4-1.4 m/s, half of them attentive (stop when the
robot is in front of them), half not.
Ground and robot: terrain, friction, slopes, payload, pushes, wind, weak motors, IMU bias (locomotion v2 sampler).
Sensor faults: blind LiDAR sector (dirty window), spurious returns (dust/rain), depth-camera blackout and frame drops,
encoder-scale and gyro-bias odometry drift.

Global planner: in 70 % of episodes a prior map exists (walls + a random 60-100 % of the free-standing obstacles:
the map is stale, and pedestrians are never in it); Dijkstra on it gives a subgoal 2.5 m ahead along the path from the
odometry pose. Without a map the subgoal is the goal itself. The learned policy is the local planner / controller.

Actor observation (deployable):
  LiDAR 72 beams x 3 frames (0.2 s), front stereo depth 24 x 12, odometry goal (dist, bearing), planner subgoal
  (dist, bearing) + map flag, encoder speed,
  gyro yaw rate, previous action, episode-time fraction, the locomotion policy's own estimate of body velocity / slope /
  friction / payload / wind / motor strength (10)
Critic observation (privileged): true scan, true goal, geodesic distance + direction to goal, true v/w, friction,
odometry error, nearest 3 pedestrians (relative position and velocity), level, clearance.
Reward: geodesic progress, success +10, collision/fall -10, clearance shaping, action rate, reverse driving, time.
"""
import math
import os
import sys

import mujoco
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import terrain as TR  # noqa: E402
from gyra2_model import obstacle_vertices  # noqa: E402
from loco2_env import FRAME, HIST, N_ACT, N_CMD, N_EST, Loco2Env  # noqa: E402
from nav_env import visible_from_pods  # noqa: E402
from ppo2 import Actor2  # noqa: E402

N_BEAM = 72
LIDAR_FRAMES = 3
MAX_R = 8.0
DEPTH_W, DEPTH_H = 24, 12
LOW_STEPS = 5
EP_STEPS = 600
ARENA = 8.0                      # half-size of the walled arena (16 x 16 m)
V_RANGE = (-1.0, 3.5)
W_RANGE = 2.5
ROBOT_R = 0.36                   # half-width incl. pods (footprint radius used for planning / clearance)
GRID = 0.1
N_DYN = 6
LOOKAHEAD = 2.5                  # planner subgoal distance along the path (m)
ACT_OBS = N_BEAM * LIDAR_FRAMES + DEPTH_W * DEPTH_H + 3 + 4 + 2 + 2 + 1 + N_EST
FAMILIES = ("clutter", "forest", "rooms", "maze", "mixed")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class LowLevel:
    """Frozen locomotion v2 policy (python module so its estimator head is available to navigation)."""
    def __init__(self, ckpt):
        torch.set_num_threads(1)               # one env per core: intra-op threads only oversubscribe
        ck = torch.load(ckpt, weights_only=False)
        self.actor = Actor2(HIST, FRAME, N_CMD, N_EST, N_ACT)
        self.actor.load_state_dict(ck["actor"])
        self.actor.eval()
        self.m = np.array(ck["na"]["mean"], np.float32)
        self.s = np.sqrt(np.array(ck["na"]["var"], np.float32) + 1e-8)

    @torch.no_grad()
    def __call__(self, oa):
        x = torch.as_tensor(np.clip((oa - self.m) / self.s, -8, 8)[None].astype(np.float32))
        mu, e = self.actor.mean(x)
        return np.clip(mu.numpy()[0], -1, 1), e.numpy()[0]


def _box(name, cx, cy, sx, sy, h, yaw=0.0, wall=False):
    return dict(name=name, kind="box", size=(sx, sy, h / 2), xy=(cx, cy), yaw=yaw, h=h, wall=wall)


def _cyl(name, cx, cy, r, h):
    return dict(name=name, kind="cylinder", size=(r, h / 2), xy=(cx, cy), yaw=0.0, h=h, wall=False)


def _footprint(o):
    """2-D convex footprint polygon (CCW) of an obstacle spec."""
    cx, cy = o["xy"]
    if o["kind"] == "cylinder":
        a = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        return np.c_[cx + o["size"][0] * np.cos(a), cy + o["size"][0] * np.sin(a)]
    sx, sy = o["size"][:2]
    c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
    pts = np.array([[sx, sy], [-sx, sy], [-sx, -sy], [sx, -sy]])
    return pts @ np.array([[c, s], [-s, c]]) + [cx, cy]


class Nav2Env:
    def __init__(self, seed=0, low_ckpt=None, level=0.0, adaptive=True, scenario=None):
        self.rng = np.random.default_rng(seed + 991)
        self.level, self.adaptive, self.scenario = level, adaptive and scenario is None, scenario
        self.low = LowLevel(low_ckpt or os.environ.get("GYRA_LOW", os.path.join(ROOT, "rl/runs/loco2/best_train.pt")))
        self.loco = Loco2Env(seed=seed, adaptive=False, world_fn=self._world, ep_len=10 ** 9, external_cmd=True)
        self.beam_ang = np.linspace(-np.pi, np.pi, N_BEAM, endpoint=False)
        hf, vf = np.radians(45), np.radians(22.5)
        u, v = np.meshgrid(np.linspace(-hf, hf, DEPTH_W), np.linspace(vf, -vf, DEPTH_H) - np.radians(8))
        self.cam_dirs = np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)], -1).reshape(-1, 3)
        self.g_obs = np.array([0, 1, 0, 0, 0, 0], np.uint8)
        self.g_all = np.array([1, 1, 0, 0, 0, 0], np.uint8)
        n = int(round(2 * ARENA / GRID))
        self.n_grid = n
        xs = (np.arange(n) + 0.5) * GRID - ARENA
        self.gx, self.gy = np.meshgrid(xs, xs)
        self._graph_cache = None

    # ================================================================== world generation
    def _sample_params(self):
        r = self.rng
        k = self.level
        if self.adaptive and r.random() < 0.15:
            k = r.uniform(0, self.level)
        P = dict(k=k)
        fams = ["clutter"] + (["forest"] if k > 0.15 else []) + (["rooms", "maze"] if k > 0.3 else []) + (["mixed"] if k > 0.5 else [])
        P["family"] = str(r.choice(fams))
        P["n_dyn"] = int(r.integers(0, 1 + round(N_DYN * k))) if k > 0.2 else 0
        P["dyn_speed"] = (0.4, 0.4 + 1.0 * k)
        P["door"] = 1.6 - 0.6 * k * r.uniform(0.5, 1.0)
        P["density"] = 0.4 + 0.6 * k
        P["goal_max"] = 4 + 10 * k
        P["loco_level"] = 0.6 * k
        P["lidar_sector"] = r.random() < 0.3 * k
        P["lidar_spurious"] = 0.01 * k
        P["depth_blackout"] = r.random() < 0.1 * k
        P["depth_drop"] = 0.1 * k
        P["map"] = r.random() < 0.7
        P["map_stale"] = r.uniform(0, 0.4)
        if self.scenario:
            P.update(self.scenario.get("nav", {}))
        return P

    def _static_layout(self, P):
        r = self.rng
        L = ARENA
        obs = [_box(f"wall{i}", x, y, sx, sy, 0.8, wall=True)
               for i, (x, y, sx, sy) in enumerate([(L, 0, .1, L + .1), (-L, 0, .1, L + .1), (0, L, L + .1, .1), (0, -L, L + .1, .1)])]
        fam = P["family"]
        if fam in ("rooms", "mixed"):
            nr = int(r.integers(2, 4))
            edges = np.linspace(-L, L, nr + 1)
            for ax in (0, 1):
                for e in edges[1:-1]:
                    segs = []
                    for j in range(nr):                      # one door per room boundary segment
                        a0, a1 = edges[j], edges[j + 1]
                        dc = r.uniform(a0 + 0.9, a1 - 0.9)
                        dw = P["door"]
                        segs += [(a0, dc - dw / 2), (dc + dw / 2, a1)]
                    for s0, s1 in segs:
                        if s1 - s0 < 0.05:
                            continue
                        c, hl = (s0 + s1) / 2, (s1 - s0) / 2
                        nm = f"iw{len(obs)}"
                        obs.append(_box(nm, e, c, 0.06, hl, 0.8, wall=True) if ax == 0 else _box(nm, c, e, hl, 0.06, 0.8, wall=True))
        if fam == "maze":
            cs = r.uniform(2.4, 3.0)
            nc = int(2 * L // cs)
            off = -nc * cs / 2
            seen = np.zeros((nc, nc), bool)
            open_e = set()
            stack = [(0, 0)]
            seen[0, 0] = True
            while stack:                                     # randomized DFS maze
                cx, cy = stack[-1]
                nb = [(cx + dx, cy + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                      if 0 <= cx + dx < nc and 0 <= cy + dy < nc and not seen[cx + dx, cy + dy]]
                if not nb:
                    stack.pop()
                    continue
                nx, ny = nb[int(r.integers(len(nb)))]
                open_e.add(frozenset(((cx, cy), (nx, ny))))
                seen[nx, ny] = True
                stack.append((nx, ny))
            for i in range(nc):                              # extra openings: loops make it less of a pure tree
                for j in range(nc):
                    for dx, dy in ((1, 0), (0, 1)):
                        if i + dx < nc and j + dy < nc and r.random() < 0.15:
                            open_e.add(frozenset(((i, j), (i + dx, j + dy))))
            for i in range(nc):
                for j in range(nc):
                    if i + 1 < nc and frozenset(((i, j), (i + 1, j))) not in open_e:
                        obs.append(_box(f"mw{len(obs)}", off + (i + 1) * cs, off + (j + 0.5) * cs, 0.06, cs / 2 + 0.06, 0.8, wall=True))
                    if j + 1 < nc and frozenset(((i, j), (i, j + 1))) not in open_e:
                        obs.append(_box(f"mw{len(obs)}", off + (i + 0.5) * cs, off + (j + 1) * cs, cs / 2 + 0.06, 0.06, 0.8, wall=True))
        # free-standing obstacles
        n_cl = {"clutter": int(10 + 22 * P["density"]), "forest": int(6 * P["density"]), "rooms": int(6 + 8 * P["density"]),
                "maze": int(3 + 5 * P["density"]), "mixed": int(6 + 8 * P["density"])}[fam]
        n_pole = {"forest": int(25 + 35 * P["density"]), "mixed": int(8 + 10 * P["density"])}.get(fam, 0)
        for i in range(n_cl):
            x, y = r.uniform(-L + 0.5, L - 0.5, 2)
            h = r.uniform(0.25, 1.2)
            if r.random() < 0.5:
                obs.append(_cyl(f"c{len(obs)}", x, y, r.uniform(0.12, 0.45), h))
            else:
                obs.append(_box(f"b{len(obs)}", x, y, r.uniform(0.15, 0.6), r.uniform(0.15, 0.6), h, yaw=r.uniform(0, np.pi)))
        for i in range(n_pole):
            x, y = r.uniform(-L + 0.4, L - 0.4, 2)
            obs.append(_cyl(f"p{len(obs)}", x, y, r.uniform(0.05, 0.15), r.uniform(0.8, 2.0)))
        return obs

    def _occupancy(self, obs):
        occ = np.zeros((self.n_grid, self.n_grid), bool)
        X, Y = self.gx, self.gy
        for o in obs:
            poly = _footprint(o)
            lo, hi = poly.min(0) - GRID, poly.max(0) + GRID
            sel = (X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1])
            if not sel.any():
                continue
            px, py = X[sel], Y[sel]
            inside = np.ones(px.shape, bool)
            for j in range(len(poly)):
                a, b = poly[j], poly[(j + 1) % len(poly)]
                inside &= (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]) >= -1e-9
            occ[sel] |= inside
        return occ

    def _graph(self):
        if self._graph_cache is None:
            n = self.n_grid
            idx = np.arange(n * n).reshape(n, n)
            rows, cols, w = [], [], []
            for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                a = idx[0:n - dy, max(0, -dx):n - max(0, dx)]
                b = idx[dy:n, max(0, dx):n + min(0, dx)]
                rows.append(a.ravel())
                cols.append(b.ravel())
                w.append(np.full(a.size, GRID * math.hypot(dx, dy)))
            self._graph_cache = (np.concatenate(rows), np.concatenate(cols), np.concatenate(w))
        return self._graph_cache

    def _geodesic(self, free, goal_cell):
        n = self.n_grid
        r_, c_, w = self._graph()
        fl = free.ravel()
        keep = fl[r_] & fl[c_]
        G = csr_matrix((w[keep], (r_[keep], c_[keep])), shape=(n * n, n * n))
        dist = dijkstra(G, directed=False, indices=goal_cell[0] * n + goal_cell[1])
        return dist.reshape(n, n)

    def _cell(self, p):
        j = int(np.clip((p[0] + ARENA) / GRID, 0, self.n_grid - 1))
        i = int(np.clip((p[1] + ARENA) / GRID, 0, self.n_grid - 1))
        return i, j

    def _world(self, S, hmap):
        """Called by Loco2Env._build after the terrain is generated: returns the mesh + mocap obstacle specs."""
        P = self.P
        r = self.rng
        for attempt in range(20):
            obs = self._static_layout(P)
            occ = self._occupancy(obs)
            from scipy.ndimage import distance_transform_edt
            clear = distance_transform_edt(~occ) * GRID
            free = clear > ROBOT_R + 0.05
            cand = np.argwhere(clear > 0.75)
            if len(cand) < 10:
                continue
            si = cand[int(r.integers(len(cand)))]
            geo = self._geodesic(free, tuple(si))            # distance from the start to every cell
            ok = np.argwhere((clear > 0.6) & np.isfinite(geo) & (geo > 3.0) & (geo < P["goal_max"]))
            if len(ok) == 0:
                continue
            gi = ok[int(r.integers(len(ok)))]
            break
        else:
            raise RuntimeError("no feasible layout")
        self.start = np.array([self.gx[tuple(si)], self.gy[tuple(si)]])
        self.free = free
        # prior map for the global planner: walls + a random subset of the free-standing obstacles
        from scipy.ndimage import distance_transform_edt as edt
        kept = [o for o in obs if o["wall"] or r.random() >= P["map_stale"]]
        self.plan_free = edt(~self._occupancy(kept)) * GRID > ROBOT_R + 0.05
        self.set_goal(np.array([self.gx[tuple(gi)], self.gy[tuple(gi)]]))
        self.static_specs = []
        for o in obs:
            fp = _footprint(o)
            zs = TR.height_at(hmap, fp[:, 0], fp[:, 1]) if hmap is not None else np.zeros(len(fp))
            zb, zt = float(zs.min()) - 0.05, float(zs.max()) + o["h"]
            hh = (zt - zb) / 2
            size = (o["size"][0], hh) if o["kind"] == "cylinder" else (o["size"][0], o["size"][1], hh)
            self.static_specs.append(dict(name=o["name"], kind=o["kind"], wall=o["wall"], verts=obstacle_vertices(o["kind"], size),
                                          pos=(o["xy"][0], o["xy"][1], zb + hh), quat=(np.cos(o["yaw"] / 2), 0, 0, np.sin(o["yaw"] / 2))))
        # pedestrians: start somewhere free, away from the robot
        self.dyn = []
        dyn_specs = []
        cand = np.argwhere(clear > 0.6)
        for i in range(P["n_dyn"]):
            for _ in range(50):
                c = cand[int(r.integers(len(cand)))]
                p = np.array([self.gx[tuple(c)], self.gy[tuple(c)]])
                if np.linalg.norm(p - self.start) > 2.5:
                    break
            z0 = float(TR.height_at(hmap, np.array([p[0]]), np.array([p[1]]))[0]) if hmap is not None else 0.0
            self.dyn.append(dict(p=p.copy(), z=z0 + 0.8, speed=r.uniform(*P["dyn_speed"]), attentive=r.random() < 0.5,
                                 target=p.copy(), v=np.zeros(2), wait=0.0))
            dyn_specs.append(dict(name=f"dyn{i}", verts=obstacle_vertices("cylinder", (0.25, 0.8), n_seg=32), pos=(p[0], p[1], z0 + 0.8)))
        self.dyn_free = clear > 0.5
        return self.static_specs, dyn_specs

    def set_goal(self, goal):
        """New goal: true geodesic field (reward / critic) + planner field and descent pointers (actor subgoal)."""
        self.goal = np.asarray(goal, float)
        gc = self._cell(self.goal)
        self.geo = self._geodesic(self.free, gc)
        self.geo[~np.isfinite(self.geo)] = 99.0
        pg = self._geodesic(self.plan_free, gc)
        pg[~np.isfinite(pg)] = 1e6
        n = self.n_grid
        best = pg.copy()
        nxt = np.arange(n * n).reshape(n, n)
        idx = nxt.copy()
        P = np.pad(pg, 1, constant_values=1e7)
        I = np.pad(idx, 1, constant_values=-1)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                cand = P[1 + dy:1 + dy + n, 1 + dx:1 + dx + n]
                ci = I[1 + dy:1 + dy + n, 1 + dx:1 + dx + n]
                better = cand < best
                best = np.where(better, cand, best)
                nxt = np.where(better, ci, nxt)
        self.plan_geo, self.plan_next = pg, nxt.ravel()

    def subgoal(self, pos):
        """Point LOOKAHEAD metres down the planner's shortest path from pos (goal itself without a map)."""
        if not self.P["map"]:
            return self.goal
        i, j = self._cell(pos)
        c = i * self.n_grid + j                    # blocked cells point to their best neighbour, so drift is tolerated
        for _ in range(int(LOOKAHEAD / GRID)):
            c2 = self.plan_next[c]
            if c2 == c:
                break
            c = c2
        if self.plan_geo.flat[c] < GRID * 1.5 or self.plan_geo.flat[c] >= 1e6:
            return self.goal
        i, j = divmod(int(c), self.n_grid)
        return np.array([self.gx[i, j], self.gy[i, j]])

    # ================================================================== pedestrians
    def _move_dyn(self, dt):
        r = self.rng
        rp = self.d.qpos[0:2]
        for i, o in enumerate(self.dyn):
            if o["wait"] > 0:
                o["wait"] -= dt
                o["v"] = np.zeros(2)
            else:
                to = o["target"] - o["p"]
                dist = np.linalg.norm(to)
                if dist < 0.2:
                    cand = np.argwhere(self.dyn_free)
                    c = cand[int(r.integers(len(cand)))]
                    o["target"] = np.array([self.gx[tuple(c)], self.gy[tuple(c)]])
                    o["wait"] = r.uniform(0, 2.0)
                    continue
                v = to / dist * o["speed"]
                if o["attentive"]:                          # stop if the robot is close in front
                    rel = rp - o["p"]
                    if np.linalg.norm(rel) < 1.3 and rel @ v > 0:
                        v = np.zeros(2)
                nxt = o["p"] + v * dt
                ci = self._cell(nxt)
                if not self.dyn_free[ci]:                   # blocked by a static obstacle: new target
                    o["target"] = o["p"].copy()
                    v = np.zeros(2)
                    nxt = o["p"]
                o["p"], o["v"] = nxt, v
            self.d.mocap_pos[i] = [o["p"][0], o["p"][1], o["z"]]

    # ================================================================== sensing
    def _pose(self):
        t = self.loco._truth()
        return self.d.qpos[0:2].copy(), t["yaw"], t

    def _rays(self, origin, dirs, group):
        n = len(dirs)
        dist = np.zeros(n)
        gid = np.zeros(n, np.int32)
        mujoco.mj_multiRay(self.m, self.d, origin, dirs.reshape(-1), group, 1, self.loco.bid["spine"], gid, dist, None, n, MAX_R)
        dist[(gid < 0) | (dist < 0)] = MAX_R
        return np.minimum(dist, MAX_R)

    def _true_scan(self, pos, yaw):
        a = self.beam_ang + yaw
        z = self.d.qpos[2]
        return self._rays(np.array([pos[0], pos[1], z]), np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1), self.g_obs)

    def _lidar(self, true):
        r, P = self.rng, self.P
        px, py = true * np.cos(self.beam_ang), true * np.sin(self.beam_ang)
        vis = visible_from_pods(px, py) | (true >= MAX_R - 1e-6)
        meas = np.where(vis, true, MAX_R) + r.normal(0, 0.02, N_BEAM)
        meas[r.random(N_BEAM) < 0.02] = MAX_R
        spur = r.random(N_BEAM) < P["lidar_spurious"]
        meas[spur] = r.uniform(0.3, 3.0, spur.sum())
        if self.lidar_sector is not None:
            a0, w = self.lidar_sector
            dd = np.abs((self.beam_ang - a0 + np.pi) % (2 * np.pi) - np.pi)
            meas[dd < w / 2] = MAX_R
        return np.clip(meas, 0, MAX_R)

    def _depth(self, pos, yaw):
        r, P = self.rng, self.P
        if P["depth_blackout"] or r.random() < P["depth_drop"]:
            return np.zeros(DEPTH_W * DEPTH_H)
        c, s = np.cos(yaw), np.sin(yaw)
        dirs = self.cam_dirs @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
        z = self._rays(np.array([pos[0] + 0.25 * c, pos[1] + 0.25 * s, self.d.qpos[2] + 0.02]), dirs, self.g_all)
        z = z + r.normal(0, 1, z.shape) * 0.01 * z ** 2
        z[z < 0.35] = 0.0
        return np.clip(z, 0, MAX_R)

    def _geo_at(self, p):
        """Bilinear geodesic distance to goal (static map) at a world point, plus its descent direction."""
        fx = (p[0] + ARENA) / GRID - 0.5
        fy = (p[1] + ARENA) / GRID - 0.5
        j0 = int(np.clip(np.floor(fx), 0, self.n_grid - 2))
        i0 = int(np.clip(np.floor(fy), 0, self.n_grid - 2))
        tx, ty = np.clip(fx - j0, 0, 1), np.clip(fy - i0, 0, 1)
        g = self.geo[i0:i0 + 2, j0:j0 + 2]
        g = np.where(g > 98, np.nan, g)
        if np.isnan(g).all():
            return 99.0, np.zeros(2)
        g = np.where(np.isnan(g), np.nanmax(g) + GRID, g)
        val = (1 - ty) * ((1 - tx) * g[0, 0] + tx * g[0, 1]) + ty * ((1 - tx) * g[1, 0] + tx * g[1, 1])
        gxd = (1 - ty) * (g[0, 1] - g[0, 0]) + ty * (g[1, 1] - g[1, 0])
        gyd = (1 - tx) * (g[1, 0] - g[0, 0]) + tx * (g[1, 1] - g[0, 1])
        d = -np.array([gxd, gyd])
        n = np.linalg.norm(d)
        return float(val), (d / n if n > 1e-9 else d)

    # ================================================================== API
    def reset(self):
        self.P = P = self._sample_params()
        lo = self.loco
        lo.level = P["loco_level"]
        sc = self.scenario.get("loco") if self.scenario else None
        lo.scenario = sc
        lo.spawn = None
        # Loco2Env.reset -> _sample_world -> _build -> self._world (places start/goal) -> compile
        # the spawn is applied after compile, so build first, then place the robot
        oa_lo, _ = lo.reset()
        self.m, self.d = lo.m, lo.d
        yaw = self.rng.uniform(-np.pi, np.pi)
        z0 = float(TR.height_at(lo.hmap, np.array([self.start[0]]), np.array([self.start[1]]))[0]) if lo.hmap is not None else 0.0
        self.d.qpos[0:3] = [self.start[0], self.start[1], z0 + lo.S["tyre_r"] + 0.002]
        self.d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        lo.hist[:] = lo._frame(lo._truth())
        self.oa_lo = np.concatenate([lo.hist.reshape(-1), lo.cmd * np.array([0.25, 0.5])]).astype(np.float32)
        self.obs_gid = set(self.m.geom(o["name"]).id for o in self.static_specs)
        self.dyn_gid = set(self.m.geom(f"dyn{i}").id for i in range(len(self.dyn)))
        self.tyre_gid = set(lo.tyres)
        self.lidar_sector = (self.rng.uniform(-np.pi, np.pi), self.rng.uniform(np.radians(30), np.radians(70))) if P["lidar_sector"] else None
        pos, yaw, t = self._pose()
        self.odo = np.array([pos[0], pos[1], yaw])
        self.odo_bias = self.rng.normal(0, 0.01) + lo.S["gyro_bias"][2]
        self.odo_scale = 1 + self.rng.normal(0, 0.02)
        self.est = np.zeros(N_EST)
        self.last_a = np.zeros(2)
        self.steps = 0
        self.geo_prev, _ = self._geo_at(pos)
        self.geo0 = self.geo_prev
        self.path_len = 0.0
        true = self._true_scan(pos, yaw)
        scan = self._lidar(true)
        self.scans = [scan] * LIDAR_FRAMES
        return self._obs(pos, yaw, t, true)

    def _goal_feat(self, pos, yaw, goal=None):
        dv = (self.goal if goal is None else goal) - pos
        b = math.atan2(dv[1], dv[0]) - yaw
        return np.array([min(np.linalg.norm(dv), 10) / 10, math.sin(b), math.cos(b)])

    def _obs(self, pos, yaw, t, true):
        lo = self.loco
        v_est = (lo.qd("tyreL") + lo.qd("tyreR")) * 0.5 * 0.3 * self.odo_scale
        depth = self._depth(pos, yaw)
        actor = np.concatenate([1.0 - np.concatenate(self.scans[::-1]) / MAX_R, 1.0 - depth / MAX_R,
                                self._goal_feat(self.odo[:2], self.odo[2]),
                                self._goal_feat(self.odo[:2], self.odo[2], self.subgoal(self.odo[:2])), [float(self.P["map"])],
                                [v_est * 0.3, (t["w"] + self.rng.normal(0, 0.02)) * 0.4], self.last_a,
                                [self.steps / EP_STEPS], self.est]).astype(np.float32)
        geo, gdir = self._geo_at(pos)
        c, s = math.cos(yaw), math.sin(yaw)
        gd = np.array([c * gdir[0] + s * gdir[1], -s * gdir[0] + c * gdir[1]])
        ped = np.zeros((3, 4))
        if self.dyn:
            rel = [(np.linalg.norm(o["p"] - pos), o) for o in self.dyn]
            rel.sort(key=lambda x: x[0])
            for j, (dd, o) in enumerate(rel[:3]):
                dp, dv = o["p"] - pos, o["v"]
                ped[j] = [(c * dp[0] + s * dp[1]) / 5, (-s * dp[0] + c * dp[1]) / 5, c * dv[0] + s * dv[1], -s * dv[0] + c * dv[1]]
        critic = np.concatenate([1.0 - true / MAX_R, self._goal_feat(pos, yaw), [min(geo, 20) / 10], gd,
                                 [t["v"] * 0.3, t["w"] * 0.4, lo.mu, (true.min() - ROBOT_R) / 2, self.P["k"]],
                                 [np.linalg.norm(self.odo[:2] - pos), math.sin(self.odo[2] - yaw)], ped.ravel(), self.last_a]).astype(np.float32)
        return actor, critic

    def _collided(self):
        n = self.d.ncon
        if n == 0:
            return False, False
        hit_s = hit_d = False
        for c in self.d.contact[:n].geom:
            a, b = int(c[0]), int(c[1])
            if a in self.tyre_gid or b in self.tyre_gid:
                o = b if a in self.tyre_gid else a
                hit_s |= o in self.obs_gid
                hit_d |= o in self.dyn_gid
        return hit_s, hit_d

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        lo = self.loco
        v_cmd = V_RANGE[0] + (a[0] + 1) / 2 * (V_RANGE[1] - V_RANGE[0])
        lo.cmd = np.array([v_cmd, a[1] * W_RANGE])
        hit_s = hit_d = fell = False
        p0 = self.d.qpos[0:2].copy()
        for _ in range(LOW_STEPS):
            la, self.est = self.low(self.oa_lo)
            (self.oa_lo, _), _, _, info = lo.step(la)
            self._move_dyn(0.02)
            v_enc = (lo.qd("tyreL") + lo.qd("tyreR")) * 0.5 * 0.3 * self.odo_scale
            t = lo._truth()
            self.odo[2] += (t["w"] + self.odo_bias) * 0.02
            self.odo[:2] += v_enc * 0.02 * np.array([math.cos(self.odo[2]), math.sin(self.odo[2])])
            hs, hd = self._collided()
            hit_s |= hs
            hit_d |= hd
            fell = info["fell"]
            if fell or hs or hd:
                break
        self.steps += 1
        pos, yaw, t = self._pose()
        self.path_len += float(np.linalg.norm(pos - p0))
        true = self._true_scan(pos, yaw)
        self.scans = self.scans[1:] + [self._lidar(true)]
        geo, _ = self._geo_at(pos)
        progress = float(np.clip(self.geo_prev - geo, -0.5, 0.5))
        self.geo_prev = geo
        clearance = float(true.min()) - ROBOT_R
        success = np.linalg.norm(self.goal - pos) < 0.5
        collided = hit_s or hit_d
        rew = (2.0 * progress - 0.01 - 0.5 * max(0.0, 0.5 - clearance) ** 2 - 0.05 * float(np.sum((a - self.last_a) ** 2))
               - 0.05 * max(0.0, -t["v"]))
        if success:
            rew += 10.0
        if collided or fell:
            rew -= 10.0
        self.last_a = a
        timeout = self.steps >= EP_STEPS and not (success or collided or fell)
        done = success or collided or fell or timeout
        info = dict(success=success, collision=collided, hit_static=hit_s, hit_dynamic=hit_d, fell=fell, timeout=timeout,
                    level=self.level, family=self.P["family"])
        if done:
            info["episode"] = dict(success=success, collision=collided, hit_static=hit_s, hit_dynamic=hit_d, fell=fell,
                                   timeout=timeout, k=self.P["k"], family=self.P["family"], n_dyn=self.P["n_dyn"],
                                   spl=float(success) * self.geo0 / max(self.path_len, self.geo0), time=self.steps * 0.1)
            if self.adaptive:
                if success and self.P["k"] >= self.level - 1e-9:
                    self.level = min(1.0, self.level + 0.05)
                elif collided or fell:
                    self.level = max(0.0, self.level - 0.05)
        return self._obs(pos, yaw, t, true), float(rew), done, info

    @staticmethod
    def mirror_spec():
        """Left/right mirror of the actor observation (index permutation, signs) and of the action."""
        perm, sg = [], []
        beams = [(-k) % N_BEAM for k in range(N_BEAM)]          # angle a -> -a
        for f in range(LIDAR_FRAMES):
            perm += [f * N_BEAM + b for b in beams]
            sg += [1] * N_BEAM
        base = N_BEAM * LIDAR_FRAMES
        for i in range(DEPTH_H):
            perm += [base + i * DEPTH_W + (DEPTH_W - 1 - j) for j in range(DEPTH_W)]
            sg += [1] * DEPTH_W
        base += DEPTH_W * DEPTH_H
        perm += [base + i for i in range(7)]
        sg += [1, -1, 1, 1, -1, 1, 1]                             # goal / subgoal (dist, sin, cos), map flag
        base += 7
        perm += [base, base + 1, base + 2, base + 3, base + 4]
        sg += [1, -1, 1, -1, 1]                                   # v, w, last a (v, w), time
        base += 5
        # estimator: vb(3) slope(2) mu payload wind(2) strength
        perm += [base + i for i in range(N_EST)]
        sg += [1, -1, 1, 1, -1, 1, 1, 1, -1, 1]
        return np.array(perm), np.array(sg, np.float32), np.array([1, -1], np.float32)


if __name__ == "__main__":
    import time
    for fam in FAMILIES:
        e = Nav2Env(0, level=1.0, adaptive=False, scenario={"nav": {"family": fam}})
        t0 = time.time()
        oa, oc = e.reset()
        tr = time.time() - t0
        t0 = time.time()
        n = 0
        for i in range(60):
            (oa, oc), r, d, inf = e.step(np.array([0.2, 0.0]))
            n += 1
            if d:
                break
        print(f"{fam:8s} reset {tr*1000:5.0f} ms, {n / (time.time() - t0):5.1f} nav steps/s, actor {oa.shape} critic {oc.shape}, "
              f"static {len(e.static_specs)} dyn {len(e.dyn)} geo0 {e.geo0:.1f} m, terrain {e.loco.S['family']}, done {d} {inf.get('episode', '')}")
