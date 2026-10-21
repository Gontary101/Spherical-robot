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
from local_map import OBST_H, LocalMap  # noqa: E402
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
COST_ALPHA, COST_RANGE = 4.0, 0.6   # costmap: up to 5x traversal cost within 0.6 m of the footprint touching anything
PLAN_MARGIN = 0.08               # planner keeps this much air between the robot's footprint and any mapped obstacle
# Live replanning (as production planners replan on their costmap): every REPLAN_EVERY navigation steps the route is
# recomputed with, within REPLAN_R of the robot, the clearance from the robot's own local map (5 cm cells, measured
# surface points) and the safety filter's own passability requirement (PASS_NEED of air beside the footprint at a
# crawl: its margin + the crawl prediction error); beyond it, the prior map. A route fixed once from the prior map's
# 0.1 m grid (+-7 cm quantisation, as large as the margin) can lead through a gap the filter will never accept, or into
# an obstacle the prior map lacks, and nothing revised it.
REPLAN = os.environ.get("GYRA_REPLAN", "1") == "1"
REPLAN_EVERY, REPLAN_R, PASS_NEED = 10, 4.0, 0.06
MAP_RAYS = 1152                  # the pods' 3-D LiDARs return dense clouds: 0.31 deg rays for the local map
HOLD_K = 2.0                     # position-hold gain (1/s) at rest; 0.5 ... 4 tested, 2 best under field disturbances
SHIELD_COST = 0.3                # reward per unit of safety-filter correction (action units, 0..2 per step); 1.0
                                 # with no envelope in the observation taught the navigator to stand still
SCAN_MATCH = True                # without LIO, register each dense scan against the local map
NEARFIELD = True                 # pod ground cameras observe the floor around the robot (except the wedge ahead)
MAP_3D = True                    # map scan cut from 3-D returns (low obstacles below the hub plane are seen)
# Sensor model. "ideal": the design-study model (2-D tyre-box occlusion of the pod LiDARs, a forward stereo depth
# camera 0.25 m ahead of the axle, one ground camera per pod). "asbuilt": the build CAD's sensors (cad/build): each
# Mid-360 at its pod's pole (|y| 0.395 m), axis outboard, field -7..+52 deg about the axle (it cannot look down and
# inward past the sphere); the six pod cameras ray-cast against the CAD (render/camera_views.py ->
# rl/sensor_tables.py); no forward depth camera. "asbuilt_lane": as built, with the front / rear cameras moved to the
# pod rims (lane cameras, docs/11_build.md).
SENSORS = os.environ.get("GYRA_SENSORS", "ideal")
LIDAR_POD_Y = 0.395
LIDAR_EL = (math.sin(math.radians(-7.0)), math.sin(math.radians(52.0)))
LOW_H = -0.22                    # the low-ring returns lie ~0.22 m below the LiDARs (obstacles 0.1 m tall)


def MAP_ELEV(z0, d_near=0.4, d_far=6.0, band=0.10):
    """Downward elevations (rad) whose rays cross the height band [0, band] at geometrically spaced distances that
    tile [d_near, d_far] without gaps: ray k covers d in [(z0 - band) / tan e_k, z0 / tan e_k]."""
    ratio = z0 / max(z0 - band, 1e-3)
    n = int(np.ceil(np.log(d_far / d_near) / np.log(ratio))) + 1
    d_hi = d_far * ratio ** -np.arange(n)                       # far end of each ray's band
    return -np.arctan(z0 / d_hi)


LIO_SCALE = float(os.environ.get("LIO_SCALE", 1.0))
SPAWN_FREE = 0.6                 # radius known clear at the start (the start keeps > 0.75 m from every obstacle)
LIDAR_OVERSAMPLE = 4             # rays per beam bin; a beam reports the minimum (how a 2-D scan is cut from a 3-D cloud)


def pod_visibility(hw):
    """Pod-LiDAR self-occlusion for a hardware variant: is robot-frame point (px, py) visible from either pod?
    Mk2 geometry (pods at |y| = 0.335 m, tyre pair |x| < 0.30, |y| < 0.285); the crown offset widens the tyre pair and
    moves both pods out by the same amount."""
    dc = hw["crown_d"] - 0.050
    pod_y, tyre_x, tyre_y = 0.335 + dc, 0.30, 0.285 + dc

    def visible(px, py):
        vis = np.zeros_like(px, dtype=bool)
        for sy in (pod_y, -pod_y):
            dy = py - sy
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (tyre_y * np.sign(sy) - sy) / dy
            enters = (np.abs(py) < tyre_y) | (np.sign(py) != np.sign(sy))
            blocked = enters & (t1 > 0) & (t1 < 1) & (np.abs(t1 * px) < tyre_x)
            blocked |= enters & (np.abs(px) < tyre_x) & (np.abs(py) < tyre_y)
            vis |= ~blocked
        return vis
    return visible


def lidar_visible_asbuilt(h=0.0):
    """As-built Mid-360s: is robot-frame point (px, py), h below/above the LiDARs, inside either sensor's field?
    The elevation about the outboard axle axis must lie in -7..+52 deg (below that the pod shell and the sphere hide
    it; above it the sensor has no beams: straight sideways close in is blind)."""
    def visible(px, py):
        vis = np.zeros(np.broadcast(px, py).shape, bool)
        for sg in (1.0, -1.0):
            dy = sg * py - LIDAR_POD_Y
            se = dy / np.maximum(np.sqrt(px ** 2 + dy ** 2 + h ** 2), 1e-6)
            vis |= (se >= LIDAR_EL[0]) & (se <= LIDAR_EL[1])
        return vis
    return visible


def min_visible_range(visible, ang, r_max=MAX_R, dr=0.005):
    """Per ray from the robot centre: the range beyond which every point is visible (the blind part of a ray is one
    interval starting at the robot). Turns the per-point occlusion test into one comparison per ray."""
    rs = np.arange(0.0, r_max + dr, dr)
    V = visible(rs[:, None] * np.cos(ang)[None], rs[:, None] * np.sin(ang)[None])
    blind = ~V
    last = np.where(blind.any(0), len(rs) - 1 - np.argmax(blind[::-1], 0), -1)
    rmin = np.where(last >= 0, rs[np.minimum(last + 1, len(rs) - 1)], 0.0)
    return rmin


def nearfield_table(hw, r_min=0.35, r_max=1.5, cell=0.1, cam_z=0.33):
    """Ground/near-field cameras (one per pod, Mk1 design): the floor points (robot frame, grid `cell`) one of the
    pods can see past the tyre, and which pod. The tyre is modelled as the two crown spheres (R 0.30, offset
    +-crown_d) clipped at the tyre half-width; on Mk2.1 this sees 91 % of the floor between 0.3 and 1.5 m and the
    centreline from 1.14 m ahead (design doc: 1.1-2.3 m); the wedge just ahead of the tyre stays blind."""
    dc = hw["crown_d"] - 0.050
    pod_y, tyre_y, cd = 0.335 + dc, 0.285 + dc, hw["crown_d"]
    g = np.arange(-r_max, r_max + 1e-9, cell)
    X, Y = np.meshgrid(g, g)
    X, Y = X.ravel(), Y.ravel()
    sel = (np.hypot(X, Y) > r_min) & (np.hypot(X, Y) < r_max)
    X, Y = X[sel], Y[sel]
    ts = np.linspace(0, 1, 40)[:, None]
    pod = np.full(len(X), -1)
    for k, sy in enumerate((pod_y, -pod_y)):
        px, py, pz = ts * X, sy + ts * (Y - sy), cam_z * (1 - ts)
        cy = np.sign(py) * cd
        inside = (np.abs(py) < tyre_y) & (px ** 2 + (py - cy) ** 2 + (pz - 0.30) ** 2 < 0.30 ** 2)
        vis = ~inside.any(0)
        pod = np.where((pod < 0) & vis, k, pod)
    ok = pod >= 0
    return np.c_[X[ok], Y[ok]], pod[ok], pod_y


def footprint_radius(hw):
    """Circumscribed footprint radius from the hardware (CAD: sensor pods reach |y| = 0.353 m on Mk2 and move out with
    the crown offset; the two crown spheres stay inside that circle)."""
    return 0.353 + (hw["crown_d"] - 0.050)
GRID = 0.1
N_DYN = 6
LOOKAHEAD = 2.5                  # planner subgoal distance along the path (m)
N_ENV = 7                        # safe-speed envelope: max admissible forward speed for 7 yaw rates (shielded)
ENV_W = np.linspace(-W_RANGE, W_RANGE, N_ENV)
ENV_V = np.linspace(0.0, V_RANGE[1], 15)
MAP_N, MAP_C, MAP_CELL = 48, 2, 0.15   # egocentric local map for the navigator (7.2 m square), at the end of the obs
MAP_OBS = MAP_C * MAP_N * MAP_N
ACT_OBS = N_BEAM * LIDAR_FRAMES + DEPTH_W * DEPTH_H + 3 + 4 + 2 + 2 + 1 + N_EST + N_ENV + MAP_OBS
FAMILIES = ("clutter", "forest", "rooms", "maze", "mixed")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class LowLevel:
    """Frozen locomotion policy (incl. its estimator head) evaluated in NumPy: identical maths to the PyTorch Actor2
    (checked in __main__), ~5x faster than a batch-1 PyTorch call inside the 50 Hz loop."""
    def __init__(self, ckpt):
        ck = torch.load(ckpt, weights_only=False)
        W = {k: v.numpy().astype(np.float64) for k, v in ck["actor"].items()}
        self.W = W
        self.m = np.array(ck["na"]["mean"], np.float64)
        self.s = np.sqrt(np.array(ck["na"]["var"], np.float64) + 1e-8)
        # conv weights as (C_in*k, C_out) matrices for an im2col + matmul formulation
        self.convs = [(W[f"enc.conv.{i}.weight"].shape[2], W[f"enc.conv.{i}.weight"].transpose(1, 2, 0).reshape(-1, W[f"enc.conv.{i}.weight"].shape[0]),
                       W[f"enc.conv.{i}.bias"]) for i in (0, 2, 4)]
        self.pi = [(W[f"pi.{i}.weight"], W[f"pi.{i}.bias"]) for i in (0, 2, 4, 6)]

    @staticmethod
    def _elu(x):
        return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))

    @staticmethod
    def _conv(x, k, wm, b):                   # x (L, C_in) time-major, stride 2, no padding -> (L_out, C_out)
        n = (x.shape[0] - k) // 2 + 1
        cols = np.lib.stride_tricks.as_strided(x, (n, x.shape[1], k), (x.strides[0] * 2, x.strides[1], x.strides[0]))
        return cols.reshape(n, -1) @ wm + b

    def __call__(self, oa):
        x = np.clip((oa - self.m) / self.s, -8, 8)
        h = x[:HIST * FRAME].reshape(HIST, FRAME)
        z = np.ascontiguousarray(h)
        for k, wm, b in self.convs:
            z = np.ascontiguousarray(self._elu(self._conv(z, k, wm, b)))
        z = self._elu(self.W["enc.fc.0.weight"] @ z.T.reshape(-1) + self.W["enc.fc.0.bias"])     # torch flattens (C, L)
        e = self.W["est.weight"] @ z + self.W["est.bias"]
        y = np.concatenate([h[-1], x[HIST * FRAME:], z, e])
        for i, (w, b) in enumerate(self.pi):
            y = w @ y + b
            if i < 3:
                y = self._elu(y)
        return np.clip(y, -1, 1), e


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
    def __init__(self, seed=0, low_ckpt=None, level=0.0, adaptive=True, scenario=None, shield=False):
        self.rng = np.random.default_rng(seed + 991)
        self.level, self.adaptive, self.scenario = level, adaptive and scenario is None, scenario
        self.low = LowLevel(low_ckpt or os.environ.get("GYRA_LOW", os.path.join(ROOT, "rl/runs/loco2/best_train.pt")))
        self.loco = Loco2Env(seed=seed, adaptive=False, world_fn=self._world, ep_len=10 ** 9, external_cmd=True)
        # navigation resets are frequent (short early episodes): reuse a per-worker terrain bank (flips/transposes add variety)
        self.loco.terrain_tol, self.loco.terrain_reuse, self.loco.terrain_cache = 0.3, 0.95, 24
        self.loco.lite = True
        self.shield = shield                        # plan phase 1: safety filter on a local occupancy map
        self.lmap = LocalMap(size=192, cell=0.05) if shield else None
        self.filter = None
        self.robot_r = footprint_radius(self.loco.hw)
        self.beam_ang = np.linspace(-np.pi, np.pi, N_BEAM, endpoint=False)
        self.map_ang = np.linspace(-np.pi, np.pi, MAP_RAYS, endpoint=False)    # dense rays for the local map
        self.visible = pod_visibility(self.loco.hw)
        self.map_rmin = min_visible_range(self.visible, self.map_ang)
        if SENSORS != "ideal":
            self.visible = lidar_visible_asbuilt(0.0)
            self.visible_low = lidar_visible_asbuilt(LOW_H)
            T = np.load(os.path.join(ROOT, "rl", "sensors", f"{SENSORS}.npz"))
            self.cam_pts, self.cam_idx, self.cam_pos = T["pts"], T["cam"], T["cam_pos"]
        self.nf_pts, self.nf_pod, self.pod_y = nearfield_table(self.loco.hw)
        k = np.arange(LIDAR_OVERSAMPLE) - (LIDAR_OVERSAMPLE - 1) / 2
        self.fine_ang = (self.beam_ang[:, None] + k[None, :] * (2 * np.pi / N_BEAM / LIDAR_OVERSAMPLE)).ravel()
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
        # localisation: GYRA's pods carry two 3-D LiDARs + IMU, so the production stack runs LiDAR-inertial odometry
        # (drift ~0.1-1 % of distance); "wheel" = LiDAR odometry lost, encoder + gyro dead reckoning only
        P["odom"] = "wheel" if r.random() < 0.1 * k else "lio"
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

    def _geodesic(self, free, goal_cell, clear=None):
        """Dijkstra distance-to-goal over free cells. With a clearance map the edge cost grows near obstacles
        (costmap inflation, as in production planners): paths run down the middle of corridors and doorways instead
        of grazing corners at exactly the inflation radius, where any tracking lag in a turn clips the jamb."""
        n = self.n_grid
        r_, c_, w = self._graph()
        fl = free.ravel()
        keep = fl[r_] & fl[c_]
        w = w[keep]
        if clear is not None:
            cl = clear.ravel()
            c_mid = 0.5 * (cl[r_[keep]] + cl[c_[keep]]) - self.robot_r
            w = w * (1.0 + COST_ALPHA * np.clip((COST_RANGE - c_mid) / COST_RANGE, 0, 1) ** 2)
        G = csr_matrix((w, (r_[keep], c_[keep])), shape=(n * n, n * n))
        dist = dijkstra(G, directed=False, indices=goal_cell[0] * n + goal_cell[1])
        return dist.reshape(n, n)

    def _replan(self):
        """Route on the live map (see REPLAN): local-map clearance and the filter's requirement near the robot, the
        prior map beyond. Kept as is when the live map leaves no route (the prior one is the better guess then)."""
        near = np.hypot(self.gx - self.odo[0], self.gy - self.odo[1]) < REPLAN_R
        pts = np.c_[self.gx[near], self.gy[near]]
        d = self.odo[2] - self.loc[2]
        c, s_ = math.cos(-d), math.sin(-d)
        r = pts - self.odo[:2]
        q = self.loc[:2] + np.c_[c * r[:, 0] - s_ * r[:, 1], s_ * r[:, 0] + c * r[:, 1]]
        ri, ci = self.lmap._idx(q)
        inside = (ri >= 0) & (ri < self.lmap.n) & (ci >= 0) & (ci < self.lmap.n)
        seen = np.zeros(len(q), bool)
        seen[inside] = np.abs(self.lmap.L[ri[inside], ci[inside]]) > 0.05
        clear = self.plan_clear0.copy()
        need = np.full(clear.shape, self.robot_r + PLAN_MARGIN)
        cl = clear[near]
        # a planner cell is passable if the robot fits ANYWHERE in it: the best local-map clearance over the cell (3 x 3
        # samples, the local map's own resolution), not the value at its centre - a 1.0 m door leaves a +-8.7 cm band
        # of feasible centre positions, narrower than the 0.1 m grid, so centre samples flickered across the threshold
        o = (np.arange(3) - 1) * GRID / 3
        ox, oy = np.meshgrid(o, o)
        sub = q[seen][:, None, :] + np.c_[ox.ravel(), oy.ravel()][None]
        cl[seen] = self.lmap.clearance(sub.reshape(-1, 2)).reshape(-1, 9).max(1)
        clear[near] = cl
        nd = need[near]
        nd[seen] = self.robot_r + PASS_NEED
        need[near] = nd
        free = clear > need
        pg = self._geodesic(free, self._cell(self.goal), clear)
        if np.isfinite(pg[self._cell(self.odo[:2])]):
            if os.environ.get("GYRA_REPLAN_LOG"):
                old = self.plan_geo[self._cell(self.odo[:2])]
                path = self.plan_path(self.odo[:2], 30)
                ij = [self._cell(p) for p in path]
                m = min(float(clear[i, j] - need[i, j]) for i, j in ij) if ij else 0.0
                self.replan_log = getattr(self, "replan_log", []) + [(float(self.d.time), float(old),
                                                                       float(pg[self._cell(self.odo[:2])]), m)]
            self.plan_clear, self.plan_free = clear, free
            self._set_plan(pg)

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
            free = clear > self.robot_r + PLAN_MARGIN
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
        self.plan_clear = edt(~self._occupancy(kept)) * GRID
        self.plan_free = self.plan_clear > self.robot_r + PLAN_MARGIN
        self.plan_clear0, self.plan_free0 = self.plan_clear, self.plan_free
        self.clear = clear
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
        self.geo = self._geodesic(self.free, gc, self.clear)            # potential for the progress reward / critic
        self.geo[~np.isfinite(self.geo)] = 99.0
        self.geo_len = self._geodesic(self.free, gc)                     # plain shortest length (SPL metric)
        self._set_plan(self._geodesic(self.plan_free, gc, self.plan_clear))

    def _set_plan(self, pg):
        """Planner field and its descent pointers (blocked cells point to their best neighbour)."""
        pg = pg.copy()
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

    def plan_path(self, pos, n=40):
        """The planner's route from pos (global frame), following its descent pointers: up to n cells (GRID each)."""
        i, j = self._cell(pos)
        c = i * self.n_grid + j
        pts = []
        for _ in range(n):
            ii, jj = divmod(int(c), self.n_grid)
            pts.append((self.gx[ii, jj], self.gy[ii, jj]))
            c2 = self.plan_next[c]
            if c2 == c:
                break
            c = c2
        return np.array(pts)

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
        a = self.fine_ang + yaw
        z = self.d.qpos[2]
        r = self._rays(np.array([pos[0], pos[1], z]), np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1), self.g_obs)
        return r.reshape(N_BEAM, LIDAR_OVERSAMPLE).min(1)

    def _lidar(self, true):
        r, P = self.rng, self.P
        px, py = true * np.cos(self.beam_ang), true * np.sin(self.beam_ang)
        vis = self.visible(px, py) | (true >= MAX_R - 1e-6)
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
        if P["depth_blackout"] or r.random() < P["depth_drop"] or SENSORS != "ideal":
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
        self.is_obst = np.zeros(self.m.ngeom + 1, bool)               # geom id -> obstacle (static or pedestrian)
        self.is_obst[list(self.obs_gid | self.dyn_gid)] = True
        self.tyre_gid = set(lo.tyres)
        self.lidar_sector = (self.rng.uniform(-np.pi, np.pi), self.rng.uniform(np.radians(30), np.radians(70))) if P["lidar_sector"] else None
        pos, yaw, t = self._pose()
        self.odo = np.array([pos[0], pos[1], yaw])
        self.odo_bias = self.rng.normal(0, 0.01) + lo.S["gyro_bias"][2]
        self.odo_scale = 1 + self.rng.normal(0, 0.02)
        # LIO error model: per-episode drift direction (0.5 % of distance per axis) + noise, heading 0.1 deg per metre
        self.lio_drift = self.rng.normal(0, 0.005, 2)
        self.lio_hdrift = self.rng.normal(0, np.radians(0.1))
        self.lio_err = np.zeros(3)
        self.loc_err = np.zeros(3)                     # prior-map localisation error (OU process), used when map=True
        self.tracks = {}                               # people tracker: id -> dict(p, v, seen) in the local frame
        self.loc = self.odo.copy()                     # local pose: what the local map / filter / MPC are expressed in
        self.wheel = self.odo.copy()                   # wheel + gyro dead reckoning (used when LIO is lost)
        self.est = np.zeros(N_EST)
        self.last_a = np.zeros(2)
        self.applied_a = np.zeros(2)
        self.hold_anchor = None
        self.v_prev = 0.0                # the command actually executed (after the safety filter), action units
        self.steps = 0
        self.geo_prev, _ = self._geo_at(pos)
        gl = self.geo_len[self._cell(pos)]
        self.geo0 = float(gl) if np.isfinite(gl) else self.geo_prev
        self.path_len = 0.0
        true = self._true_scan(pos, yaw)
        scan = self._lidar(true)
        self.scans = [scan] * LIDAR_FRAMES
        obs = self._obs(pos, yaw, t, true)
        self.n_interv, self.interv_mag = 0, 0.0
        if self.shield:
            if self.filter is None:
                from safety_filter import SafetyFilter
                self.filter = SafetyFilter(self.robot_r)
            self.filter.reset()
            self.lmap.reset(self.loc[:2])
            # the spot the robot was put down on is known clear (start clearance > 0.75 m by construction): without
            # this prior, the area inside the pods' forward/rear blind wedge would stay unknown and block every motion
            self.lmap.mark_free_disc(self.loc[:2], SPAWN_FREE)
            self._map_update()
            self._put_envelope(obs)
        return obs

    def _velocity_estimate(self):
        """(v, w) as the safety filter sees them: LiDAR-inertial odometry, or encoders + gyro when LIO is lost."""
        tr = self.loco._truth()
        if self.P["odom"] == "lio":
            return tr["v"] + self.rng.normal(0, 0.03), tr["w"] + self.rng.normal(0, 0.02)
        lo = self.loco
        return (lo.qd("tyreL") + lo.qd("tyreR")) * 0.5 * 0.3 * self.odo_scale, tr["w"] + self.odo_bias

    def _put_envelope(self, obs):
        self.filter.pend = float(self.loco._truth()["pend"])
        """What the filter knows, in a form the navigator can act on: for each of N_ENV yaw rates, the fastest
        admissible forward speed (fraction of v_max). Without it, the navigator is penalised for corrections it cannot
        predict (map memory, blind wedge, braking envelope) and learns to stand still."""
        v0, w0 = self._velocity_estimate()
        C = np.array([(v, w) for w in ENV_W for v in ENV_V])
        ok, _ = self.filter.admissible(self.lmap, self.loc.copy(), v0, w0, C)
        ok = ok.reshape(N_ENV, len(ENV_V))
        # fastest speed with every slower speed also admissible (the envelope a navigator can safely stay inside)
        first_bad = np.where(ok.all(1), len(ENV_V), np.argmin(ok, 1))
        env = np.where(first_bad > 0, ENV_V[np.maximum(first_bad - 1, 0)], -0.1) / V_RANGE[1]
        obs[0][-N_ENV - MAP_OBS:-MAP_OBS] = env
        obs[1][-N_ENV:] = env
        # the map the filter uses, robot-centred: memory of the blind wedge, of openings passed, of where it has been
        obs[0][-MAP_OBS:] = self.lmap.egocentric2(self.loc, MAP_N, MAP_CELL).ravel()

    def _scan3d_low(self, pos, yaw):
        """The pods' Mid-360s sweep densely in elevation: per azimuth (1.25 deg), the nearest obstacle return over
        downward rings whose ground footprints are spaced so that every obstacle taller than OBST_H (0.10 m) between
        0.4 and 6 m is hit (ray height z0 - d tan(e) passes through [0, 0.10 m] at every distance). Returns the
        horizontal range of the nearest obstacle hit per azimuth; ground hits (terrain geom: ground segmentation)
        are not obstacles."""
        z0 = self.d.qpos[2]
        # every map ray (0.31 deg): casting at 1.25 deg and copying each result to its 4 neighbours widened every
        # obstacle edge sideways by up to a bin (2-6 cm at 1-3 m) - doors narrowed by the sensor model, not the world
        a = self.map_ang + yaw
        el = MAP_ELEV(z0)
        ce, se = np.cos(el)[:, None], np.sin(el)[:, None]
        dirs = np.stack([ce * np.cos(a)[None], ce * np.sin(a)[None], np.broadcast_to(se, (len(el), len(a)))], -1).reshape(-1, 3)
        n = len(dirs)
        dist, gid = np.zeros(n), np.zeros(n, np.int32)
        # obstacles only (ray-heightfield tests are 5x the cost); a return below the terrain surface at its xy would
        # have hit the ground first: rejected
        mujoco.mj_multiRay(self.m, self.d, np.array([pos[0], pos[1], z0]), dirs.reshape(-1), self.g_obs, 1,
                           self.loco.bid["spine"], gid, dist, None, n, MAX_R)
        hit = (gid >= 0) & (dist >= 0)
        if hit.any() and self.loco.hmap is not None:
            p = np.array([pos[0], pos[1], z0]) + dirs[hit] * dist[hit, None]
            hit[np.flatnonzero(hit)] = p[:, 2] > TR.height_at(self.loco.hmap, p[:, 0], p[:, 1]) + 0.01
        horiz = np.where(hit, dist * np.cos(np.repeat(el, len(a))), MAX_R).reshape(len(el), len(a))
        return horiz.min(0)

    def _map_scan(self):
        """Dense masked LiDAR for the map: same sensor model as the policy scan (pod self-occlusion, noise, dropouts,
        dust, dead sector) at 0.31 deg resolution, cut from the 3-D returns (horizontal ring + downward rings)."""
        pos, yaw, _ = self._pose()
        a = self.map_ang + yaw
        n = len(a)
        true, gid = np.zeros(n), np.zeros(n, np.int32)
        mujoco.mj_multiRay(self.m, self.d, np.array([pos[0], pos[1], self.d.qpos[2]]),
                           np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1).reshape(-1), self.g_obs, 1,
                           self.loco.bid["spine"], gid, true, None, n, MAX_R)
        true[(gid < 0) | (true < 0)] = MAX_R
        true = np.minimum(true, MAX_R)
        self.map_gid = gid
        if SENSORS != "ideal":                      # each return must lie inside a Mid-360's field (horizontal ring and
            ca, sa = np.cos(self.map_ang), np.sin(self.map_ang)        # low rings tested at their own heights)
            true = np.where(self.visible(true * ca, true * sa) | (true >= MAX_R - 1e-6), true, MAX_R)
            if MAP_3D:
                low = self._scan3d_low(pos, yaw)
                true = np.minimum(true, np.where(self.visible_low(low * ca, low * sa), low, MAX_R))
            vis = np.ones(n, bool)
        elif MAP_3D:
            true = np.minimum(true, self._scan3d_low(pos, yaw))
        r, P = self.rng, self.P
        if SENSORS == "ideal":
            vis = (true >= self.map_rmin) | (true >= MAX_R - 1e-6)
        meas = np.where(vis, true, MAX_R) + r.normal(0, 0.02, len(true))
        meas[r.random(len(true)) < 0.02] = MAX_R
        spur = r.random(len(true)) < P["lidar_spurious"]
        meas[spur] = r.uniform(0.3, 3.0, spur.sum())
        dead = None
        if self.lidar_sector is not None:
            a0, w = self.lidar_sector
            dead = np.abs((self.map_ang - a0 + np.pi) % (2 * np.pi) - np.pi) < w / 2
            meas[dead] = MAX_R
        meas = np.clip(meas, 0, MAX_R)
        # isolated-return filter (what LiDAR drivers / outlier removal do): every real obstacle is >= 0.10 m wide, i.e.
        # >= 2 adjacent 0.31 deg rays out to 8 m; a return with no neighbour within 0.15 m in range is dust/noise
        hit = meas < MAX_R - 1e-3
        nb = (np.abs(meas - np.roll(meas, 1)) < 0.15) | (np.abs(meas - np.roll(meas, -1)) < 0.15)
        meas = np.where(hit & ~nb, MAX_R, meas)
        return meas, dead

    def _nearfield(self):
        """The pods' ground cameras: for every visible floor point, a sight line from its pod (true pose, obstacles
        only, returns under the terrain rejected). Robot-frame (free floor points, obstacle points)."""
        if SENSORS != "ideal":
            return self._cameras_asbuilt()
        pos, yaw, _ = self._pose()
        c, s_ = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s_], [s_, c]])
        z0 = self.d.qpos[2] + 0.03
        free, occ = [], []
        for k, sy in enumerate((self.pod_y, -self.pod_y)):
            pts = self.nf_pts[self.nf_pod == k]
            pts = pts + self.rng.uniform(-0.05, 0.05, 2)              # jitter: the 0.1 m sample grid covers every 5 cm cell
            o = np.array([pos[0], pos[1], z0]) + np.r_[R @ np.array([0.0, sy]), 0.0]
            tw = pos + pts @ R.T
            zg = TR.height_at(self.loco.hmap, tw[:, 0], tw[:, 1]) if self.loco.hmap is not None else np.zeros(len(tw))
            d = np.c_[tw - o[:2], zg - o[2]]
            L = np.linalg.norm(d, axis=1)
            d = d / L[:, None]
            n = len(d)
            dist, gid = np.zeros(n), np.zeros(n, np.int32)
            mujoco.mj_multiRay(self.m, self.d, o, d.reshape(-1), self.g_obs, 1, self.loco.bid["spine"], gid, dist, None, n, MAX_R)
            hit = (gid >= 0) & (dist >= 0) & (dist < L - 0.03)
            free.append(pts[~hit])
            if hit.any():
                hw_ = o[:2] + d[hit, :2] * dist[hit, None]
                occ.append((hw_ - pos) @ R)                      # world -> robot frame
        return np.concatenate(free), (np.concatenate(occ) if occ else np.zeros((0, 2)))

    def _cameras_asbuilt(self):
        """The as-built pod cameras: every floor point a camera sees past the tyre (CAD ray-cast table), cast from that
        camera's real position into the world (true pose, obstacles only, returns under the terrain rejected): free
        floor, or the obstacle point the sight line meets. Robot-frame (free points, obstacle points)."""
        pos, yaw, _ = self._pose()
        c, s_ = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s_], [s_, c]])
        free, occ = [], []
        for k in range(len(self.cam_pos)):
            pts = self.cam_pts[self.cam_idx == k]
            if not len(pts):
                continue
            pts = pts + self.rng.uniform(-0.06, 0.06, 2)
            cp = self.cam_pos[k]
            o = np.r_[pos + R @ cp[:2], self.d.qpos[2] + cp[2]]
            tw = pos + pts @ R.T
            zg = TR.height_at(self.loco.hmap, tw[:, 0], tw[:, 1]) if self.loco.hmap is not None else np.zeros(len(tw))
            d = np.c_[tw - o[:2], zg - o[2]]
            L = np.linalg.norm(d, axis=1)
            d = d / L[:, None]
            n = len(d)
            dist, gid = np.zeros(n), np.zeros(n, np.int32)
            mujoco.mj_multiRay(self.m, self.d, o, d.reshape(-1), self.g_obs, 1, self.loco.bid["spine"], gid, dist, None, n, MAX_R)
            hit = (gid >= 0) & (dist >= 0) & (dist < L - 0.03)
            free.append(pts[~hit])
            if hit.any():
                occ.append((o[:2] + d[hit, :2] * dist[hit, None] - pos) @ R)
        return (np.concatenate(free) if free else np.zeros((0, 2))), (np.concatenate(occ) if occ else np.zeros((0, 2)))

    def _scan_match(self, scan):
        """Scan-to-map registration (the fallback when LiDAR-inertial odometry is lost): around the dead-reckoned
        local pose, a grid search (+-6 cm, +-1.5 deg) for the pose whose dense-scan returns lie closest to the mapped
        surfaces (mean distance to the nearest occupied cell's mean hit point, clipped at 0.2 m). Needs enough
        structure in view; otherwise the dead-reckoned pose is kept."""
        hit = scan < MAX_R - 1e-3
        if hit.sum() < 60 or not self.lmap.occupied().any():
            return self.loc
        idx = np.flatnonzero(hit)[::max(1, hit.sum() // 300)]
        pr = np.c_[scan[idx] * np.cos(self.map_ang[idx]), scan[idx] * np.sin(self.map_ang[idx])]
        best, best_s = self.loc, None
        for dth in np.radians([-1.5, -0.75, 0.0, 0.75, 1.5]):
            th = self.loc[2] + dth
            c, s_ = math.cos(th), math.sin(th)
            pw = pr @ np.array([[c, s_], [-s_, c]])
            for dx in (-0.06, -0.03, 0.0, 0.03, 0.06):
                for dy in (-0.06, -0.03, 0.0, 0.03, 0.06):
                    q = pw + self.loc[:2] + np.array([dx, dy])
                    sc = float(np.minimum(self.lmap.clearance(q), 0.2).mean())
                    if best_s is None or sc < best_s - 1e-4:
                        best, best_s = np.array([self.loc[0] + dx, self.loc[1] + dy, th]), sc
        return best

    def _track_people(self, scan):
        """People tracker: a person is detected when one of the dense map rays that survived the sensor model
        (self-occlusion, dead sector, dropouts) returns from it; measured centroid noise 5 cm. Velocity by smoothed
        finite differences; a track coasts for 1 s without detection. Local frame."""
        if not self.dyn:
            return
        pos, yaw, _ = self._pose()
        R = lambda th: np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        to_loc = lambda pw: self.loc[:2] + R(self.loc[2] - yaw) @ (pw - pos)
        seen = set()
        valid = scan < MAX_R - 1e-3
        for i, o in enumerate(self.dyn):
            gid = self.m.geom(f"dyn{i}").id
            if not np.any(valid & (self.map_gid == gid)):
                continue
            seen.add(i)
            z = to_loc(o["p"]) + self.rng.normal(0, 0.05, 2)
            tr = self.tracks.get(i)
            if tr is None:
                self.tracks[i] = dict(p=z, v=np.zeros(2), seen=0.0)
            else:
                v_new = (z - tr["p"]) / 0.1
                tr["v"] = 0.7 * tr["v"] + 0.3 * v_new
                tr["p"], tr["seen"] = z, 0.0
        for i in list(self.tracks):
            if i not in seen:
                tr = self.tracks[i]
                tr["p"] = tr["p"] + tr["v"] * 0.1
                tr["seen"] += 0.1
                if tr["seen"] > 1.0:
                    del self.tracks[i]

    def _near_people(self, pts_robot):
        """Robot-frame points within 0.15 m of a tracked person's surface (dynamic-object removal for the static map)."""
        if not self.tracks or len(pts_robot) == 0:
            return np.zeros(len(pts_robot), bool)
        c, s_ = math.cos(self.loc[2]), math.sin(self.loc[2])
        w = self.loc[:2] + pts_robot @ np.array([[c, s_], [-s_, c]])
        near = np.zeros(len(w), bool)
        for tr in self.tracks.values():
            near |= np.linalg.norm(w - tr["p"], axis=1) < 0.25 + 0.15
        return near

    def _map_update(self):
        """Fuse the dense masked LiDAR, the depth image and the near-field ground cameras into the local map at the
        odometry pose (robot knowledge only)."""
        scan, dead = self._map_scan()
        if self.P["odom"] != "lio" and SCAN_MATCH:
            self.loc = self._scan_match(scan)
            self.odo = self.loc.copy()
        self._track_people(scan)
        fr, oc = self._nearfield() if NEARFIELD else (np.zeros((0, 2)), np.zeros((0, 2)))
        dfr, doc = self._depth_points()
        oc = np.concatenate([oc, doc])
        oc = oc[~self._near_people(oc)]
        hp = np.c_[scan * np.cos(self.map_ang), scan * np.sin(self.map_ang)]
        self.lmap.update(self.loc.copy(), self.map_ang, scan, depth=None,
                         visible=self.map_rmin if SENSORS == "ideal" else self.visible, dead=dead, free_every=4, free_max=6.0, robot_r=self.robot_r,
                         nearfield=(np.concatenate([fr, dfr]), oc), hit_mask=~self._near_people(hp))
        if self.filter is not None:
            self.filter.peds = [(tr["p"].copy(), tr["v"].copy()) for tr in self.tracks.values()]

    def _depth_points(self):
        """Depth image endpoints classified against the ground surface (ground segmentation, as for the LiDAR rings
        and ground cameras): obstacle if more than OBST_H above the terrain at its xy. Classifying against a flat plane
        at the robot's own level marked bumps of rough ground as obstacles; in the blind wedge such a cell decays to
        unknown and can never be re-observed, blocking every forward command (the cause of most stalls of a
        privileged teacher). Robot-frame (free points, obstacle points)."""
        z = self.depth_last
        ok = z > 0.35
        if not ok.any():
            return np.zeros((0, 2)), np.zeros((0, 2))
        pos, yaw, _ = self._pose()
        c, s_ = math.cos(yaw), math.sin(yaw)
        d = self.cam_dirs[ok] * z[ok, None]                      # camera frame = robot axes, origin 0.25 m ahead
        rx, ry = 0.25 + d[:, 0], d[:, 1]
        wx, wy = pos[0] + c * rx - s_ * ry, pos[1] + s_ * rx + c * ry
        wz = self.d.qpos[2] + 0.02 + d[:, 2]
        zg = TR.height_at(self.loco.hmap, wx, wy) if self.loco.hmap is not None else np.zeros(len(wx))
        obst = wz - zg > OBST_H
        pts = np.c_[rx, ry]
        return pts[~obst], pts[obst]

    def _goal_feat(self, pos, yaw, goal=None):
        dv = (self.goal if goal is None else goal) - pos
        b = math.atan2(dv[1], dv[0]) - yaw
        return np.array([min(np.linalg.norm(dv), 10) / 10, math.sin(b), math.cos(b)])

    def _obs(self, pos, yaw, t, true):
        lo = self.loco
        v_est = (lo.qd("tyreL") + lo.qd("tyreR")) * 0.5 * 0.3 * self.odo_scale
        depth = self._depth(pos, yaw)
        self.depth_last = depth
        actor = np.concatenate([1.0 - np.concatenate(self.scans[::-1]) / MAX_R, 1.0 - depth / MAX_R,
                                self._goal_feat(self.odo[:2], self.odo[2]),
                                self._goal_feat(self.odo[:2], self.odo[2], self.subgoal(self.odo[:2])), [float(self.P["map"])],
                                [v_est * 0.3, (t["w"] + self.rng.normal(0, 0.02)) * 0.4], self.applied_a,
                                [self.steps / EP_STEPS], self.est, np.ones(N_ENV), np.zeros(MAP_OBS)]).astype(np.float32)
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
                                 [t["v"] * 0.3, t["w"] * 0.4, lo.mu, (true.min() - self.robot_r) / 2, self.P["k"]],
                                 [np.linalg.norm(self.odo[:2] - pos), math.sin(self.odo[2] - yaw)], ped.ravel(), self.applied_a,
                                 np.ones(N_ENV)]).astype(np.float32)
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
        w_cmd = a[1] * W_RANGE
        intervened, corr = False, 0.0
        if self.shield:
            # velocity from the LiDAR-inertial odometry (encoders mislead while the pendulum swings the spine);
            # with LIO lost, fall back to encoders + gyro
            v_now, w_now = self._velocity_estimate()
            # longitudinal acceleration (IMU), smoothed: the filter extrapolates it through the dead time
            self.filter.a0 = 0.6 * self.filter.a0 + 0.4 * float(np.clip((v_now - getattr(self, "v_prev", v_now)) / 0.1, -4, 4))
            self.v_prev = v_now
            self.filter.pend = float(self.loco._truth()["pend"])         # yoke encoder + IMU on the robot
            v_f, w_f, intervened, _ = self.filter(self.lmap, self.loc.copy(), v_now, w_now, v_cmd, w_cmd)
            corr = abs(v_f - v_cmd) / (V_RANGE[1] - V_RANGE[0]) + abs(w_f - w_cmd) / (2 * W_RANGE)
            self.interv_mag += corr
            self.n_interv += intervened
            v_cmd, w_cmd = v_f, w_f
        # position hold: "stop" means "stay here", not "zero velocity". Under rough ground and wind a velocity-zero
        # command lets disturbances integrate into drift (up to 0.4 m in 5 s, measured); once the robot is at rest, a
        # slow outer loop on the along-track displacement from that rest point bounds it (0.13-0.17 m worst case).
        # The anchor is taken only once the robot has slowed, so hold never pulls back through the braking distance.
        if abs(v_cmd) < 0.05:
            if self.hold_anchor is None and abs(self.loco._truth()["v"]) < 0.1:
                self.hold_anchor = self.loc[:2].copy()
            if self.hold_anchor is not None:
                hd = np.array([math.cos(self.loc[2]), math.sin(self.loc[2])])
                v_cmd = float(np.clip(-HOLD_K * (self.loc[:2] - self.hold_anchor) @ hd, -0.3, 0.3))
        else:
            self.hold_anchor = None
        lo.cmd = np.array([v_cmd, w_cmd])
        hit_s = hit_d = fell = False
        p0 = self.d.qpos[0:2].copy()
        for _ in range(LOW_STEPS):
            la, self.est = self.low(self.oa_lo)
            (self.oa_lo, _), _, _, info = lo.step(la)
            self._move_dyn(0.02)
            v_enc = (lo.qd("tyreL") + lo.qd("tyreR")) * 0.5 * 0.3 * self.odo_scale
            t = lo._truth()
            if self.P["odom"] == "lio":
                ds = float(np.linalg.norm(self.d.qvel[0:2])) * 0.02
                self.lio_err[:2] += LIO_SCALE * ds * (self.lio_drift + self.rng.normal(0, 0.005, 2))
                self.lio_err[2] += LIO_SCALE * ds * (self.lio_hdrift + self.rng.normal(0, np.radians(0.05)))
                pos_t, yaw_t = self.d.qpos[0:2], t["yaw"]
                # two estimates, as a LiDAR-inertial stack keeps them:
                #  local: each scan is registered against the local map, so pose and map stay mutually consistent
                #         (registration noise ~1 cm); building the map with a drifting pose smeared every wall by the
                #         drift accumulated over its memory (a 1.0 m door lost ~25 % success to that alone)
                #  global: relative to the prior map - localised against it when there is one (bounded error),
                #         otherwise the drifting LIO estimate
                self.loc = np.array([pos_t[0], pos_t[1], yaw_t]) + self.rng.normal(0, [0.01, 0.01, np.radians(0.2)]) * 0.2
                if self.P["map"]:
                    a = 0.02 / 5.0                                   # OU, tau 5 s, std 3 cm / 0.3 deg
                    self.loc_err += -a * self.loc_err + np.sqrt(2 * a) * self.rng.normal(0, [0.03, 0.03, np.radians(0.3)])
                    err = self.loc_err
                else:
                    err = self.lio_err
                self.odo = np.array([pos_t[0] + err[0], pos_t[1] + err[1], yaw_t + err[2]])
            else:
                d_odo = self.wheel.copy()
                self.wheel[2] += (t["w"] + self.odo_bias) * 0.02
                self.wheel[:2] += v_enc * 0.02 * np.array([math.cos(self.wheel[2]), math.sin(self.wheel[2])])
                # no LIO: dead-reckon the local pose with the wheel+gyro increment; it is re-registered against the
                # local map once per navigation step (scan matching, in _map_update). The registered pose is a LiDAR
                # odometry in its own right, so it is also the global estimate (no prior-map localisation assumed).
                c, s_ = math.cos(self.loc[2] - d_odo[2]), math.sin(self.loc[2] - d_odo[2])
                dx, dy = self.wheel[0] - d_odo[0], self.wheel[1] - d_odo[1]
                self.loc = self.loc + np.array([c * dx - s_ * dy, s_ * dx + c * dy, self.wheel[2] - d_odo[2]])
                self.odo = self.loc.copy() if SCAN_MATCH else self.wheel.copy()
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
        clearance = float(true.min()) - self.robot_r
        success = np.linalg.norm(self.goal - pos) < 0.5
        collided = hit_s or hit_d
        rew = (2.0 * progress - 0.01 - 0.5 * max(0.0, 0.5 - clearance) ** 2 - 0.05 * float(np.sum((a - self.last_a) ** 2))
               - 0.05 * max(0.0, -t["v"]))
        # shielded: corrections cost (the navigator should propose what the filter would accept, not lean on it)
        rew -= SHIELD_COST * corr
        if success:
            rew += 10.0
        if collided or fell:
            rew -= 10.0
        self.last_a = a
        self.applied_a = np.array([2 * (v_cmd - V_RANGE[0]) / (V_RANGE[1] - V_RANGE[0]) - 1, w_cmd / W_RANGE])
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
        obs = self._obs(pos, yaw, t, true)
        if self.shield:
            self._map_update()
            if REPLAN and self.steps % REPLAN_EVERY == 0:
                self._replan()
            self._put_envelope(obs)
            info["intervened"] = intervened
            if done:
                info["episode"]["interventions"] = self.n_interv / max(self.steps, 1)
        return obs, float(rew), done, info

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
        base += N_EST
        perm += [base + N_ENV - 1 - i for i in range(N_ENV)]                 # envelope: yaw rate w -> -w
        sg += [1] * N_ENV
        base += N_ENV
        for c in range(MAP_C):                                               # map: rows are robot-frame y -> -y
            for i in range(MAP_N):
                perm += [base + c * MAP_N * MAP_N + (MAP_N - 1 - i) * MAP_N + j for j in range(MAP_N)]
        sg += [1] * MAP_OBS
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
