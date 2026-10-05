"""
GYRA Mk2 autonomous navigation environment (hierarchical, asymmetric actor-critic).

The navigation policy runs at 10 Hz and outputs (v_cmd, w_cmd). A frozen low-level controller
(the learned locomotion policy, or the classical controller) turns them into motor torques at 50 Hz.

Actor observation (deployable sensors only):
  * 2-D LiDAR scan, 72 beams x 2 frames: synthesised from the two pod Mid-360s. Points the tyre
    hides from both pods (the forward/rear blind wedge) return nothing; 2 cm noise, 2 % dropouts
  * front stereo depth image 16 x 8 (90 x 45 deg FOV): noise grows with range^2, < 0.35 m invalid
  * goal (distance, bearing) from encoder+gyro dead-reckoning odometry, which drifts
  * speed estimate from encoders, gyro yaw rate, previous nav action
Critic observation (privileged): true goal vector, true v/w, unmasked noiseless 72-beam scan,
nearest-obstacle distance, odometry error.
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from loco_env import LocoEnv  # noqa: E402

N_BEAM = 72
MAX_R = 8.0
DEPTH_W, DEPTH_H = 16, 8
NAV_DT = 0.1
LOW_STEPS = 5
N_OBS = 22
ARENA = 7.0
ACT_OBS = N_BEAM * 2 + DEPTH_W * DEPTH_H + 8
CRIT_OBS = N_BEAM + 10
EP_STEPS = 600
V_RANGE = (-1.0, 3.5)
W_RANGE = 2.5
POD_Y = 0.335
TYRE_X, TYRE_Y = 0.30, 0.285


class NumpyActor:
    """Frozen locomotion actor (TorchScript weights -> numpy MLP, ELU)."""

    def __init__(self, ckpt_path):
        import torch
        ck = torch.load(ckpt_path, weights_only=False)
        sd = ck["actor"]
        self.W = [sd[k].numpy().T for k in sd if k.startswith("mu.") and k.endswith("weight")]
        self.b = [sd[k].numpy() for k in sd if k.startswith("mu.") and k.endswith("bias")]
        self.mean = np.array(ck["na"]["mean"])
        self.std = np.sqrt(np.array(ck["na"]["var"]) + 1e-8)

    def __call__(self, obs):
        x = np.clip((obs - self.mean) / self.std, -8, 8)
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = x @ w + b
            if i < len(self.W) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        return np.clip(x, -1, 1)


def visible_from_pods(px, py):
    """Is robot-frame point (px, py) visible from either pod, given the tyre footprint rectangle?"""
    vis = np.zeros_like(px, dtype=bool)
    for sy in (POD_Y, -POD_Y):
        # segment from (0, sy) to (px, py); blocked if it enters |x|<TYRE_X, |y|<TYRE_Y
        # parametrise y(t) = sy + t (py - sy); the part with |y| < TYRE_Y
        dy = py - sy
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (TYRE_Y * np.sign(sy) - sy) / dy           # where the ray crosses the near tyre edge
        enters = (np.abs(py) < TYRE_Y) | (np.sign(py) != np.sign(sy))
        xt = t1 * px
        blocked = enters & (t1 > 0) & (t1 < 1) & (np.abs(xt) < TYRE_X)
        blocked |= enters & (np.abs(px) < TYRE_X) & (np.abs(py) < TYRE_Y)
        vis |= ~blocked
    return vis


class NavEnv:
    def __init__(self, seed=0, low_level="classical", loco_ckpt=None, randomize=True):
        self.rng = np.random.default_rng(seed + 777)
        self.loco = LocoEnv(seed=seed, v_max=3.5, randomize=randomize, push=False, n_obstacles=N_OBS, arena=ARENA)
        self.loco._sample_cmd = self._hold_cmd
        m = self.loco.m
        self.m, self.d = m, self.loco.d
        self.obs_gid = np.array([m.geom(f"obs{i}").id for i in range(N_OBS)])
        self.wall_gid = np.array([m.geom(f"wall{i}").id for i in range(4)])
        self.tyre_gid = np.array([m.geom("tyreL").id, m.geom("tyreR").id])
        self.low_level = low_level
        if low_level == "learned":
            self.policy = NumpyActor(loco_ckpt)
        else:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
            from gyra2_sim import ClassicalController  # noqa: E402
            self._Classical = ClassicalController
        ang = np.linspace(-np.pi, np.pi, N_BEAM, endpoint=False)
        self.beam_ang = ang
        hf, vf = np.radians(45), np.radians(22.5)
        u = np.linspace(-hf, hf, DEPTH_W)
        v = np.linspace(vf, -vf, DEPTH_H) - np.radians(8)       # pitched 8 deg down
        uu, vv = np.meshgrid(u, v)
        self.cam_dirs = np.stack([np.cos(vv) * np.cos(uu), np.cos(vv) * np.sin(uu), np.sin(vv)], -1).reshape(-1, 3)
        self.geomgroup = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)   # floor (0) + obstacles/walls (1)
        self.geomgroup_obs = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)

    def _hold_cmd(self):
        if not hasattr(self.loco, "cmd"):
            self.loco.cmd = np.zeros(2)
        self.loco.next_cmd_t = 1e9

    # -------------------------------------------------------------------- world generation
    def _layout(self):
        r, m = self.rng, self.m
        L = ARENA - 0.8
        self.start = np.array([r.uniform(-L, L), r.uniform(-L, L)])
        while True:
            self.goal = np.array([r.uniform(-L, L), r.uniform(-L, L)])
            if 4.0 < np.linalg.norm(self.goal - self.start) < 10.0:
                break
        n_active = int(r.integers(8, N_OBS + 1))
        placed = []
        for i, gid in enumerate(self.obs_gid):
            if i >= n_active:
                m.geom_pos[gid] = [200 + 3 * i, 200, 0.4]
                continue
            for _ in range(50):
                p = np.array([r.uniform(-L, L), r.uniform(-L, L)])
                if np.linalg.norm(p - self.start) > 1.3 and np.linalg.norm(p - self.goal) > 1.2 and \
                        all(np.linalg.norm(p - q) > 1.1 for q in placed):
                    break
            placed.append(p)
            h = r.uniform(0.25, 0.9)
            if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                rad = r.uniform(0.12, 0.45)
                m.geom_size[gid] = [rad, h / 2, 0]
                m.geom_rbound[gid] = np.hypot(rad, h / 2)
            else:
                sx, sy = r.uniform(0.15, 0.6), r.uniform(0.15, 0.6)
                m.geom_size[gid] = [sx, sy, h / 2]
                m.geom_rbound[gid] = np.linalg.norm([sx, sy, h / 2])
                yaw = r.uniform(0, np.pi)
                m.geom_quat[gid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            m.geom_pos[gid] = [p[0], p[1], h / 2]
        self.obstacles = placed

    # -------------------------------------------------------------------- sensing
    def _pose(self):
        t = self.loco._truth()
        return self.d.qpos[0:2].copy(), float(np.arctan2(t["fwd"][1], t["fwd"][0])), t

    def _rays(self, origin, dirs, group):
        n = len(dirs)
        dist = np.zeros(n)
        gid = np.zeros(n, dtype=np.int32)
        mujoco.mj_multiRay(self.m, self.d, origin, dirs.reshape(-1), group, 1, -1, gid, dist, None, n, MAX_R)
        dist[(gid < 0) | (dist < 0)] = MAX_R
        return np.minimum(dist, MAX_R)

    def _true_scan(self, pos, yaw):
        a = self.beam_ang + yaw
        dirs = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1)
        return self._rays(np.array([pos[0], pos[1], 0.30]), dirs, self.geomgroup_obs)

    def _lidar(self, pos, yaw):
        true = self._true_scan(pos, yaw)
        px, py = true * np.cos(self.beam_ang), true * np.sin(self.beam_ang)
        vis = visible_from_pods(px, py) | (true >= MAX_R - 1e-6)
        meas = np.where(vis, true, MAX_R) + self.rng.normal(0, 0.02, N_BEAM)
        meas[self.rng.random(N_BEAM) < 0.02] = MAX_R
        return np.clip(meas, 0, MAX_R), true

    def _depth(self, pos, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        dirs = self.cam_dirs @ Rz.T
        origin = np.array([pos[0], pos[1], 0.32])
        z = self._rays(origin, dirs, self.geomgroup)
        z = z + self.rng.normal(0, 1, z.shape) * 0.01 * z ** 2
        z[z < 0.35] = 0.0
        return np.clip(z, 0, MAX_R)

    # -------------------------------------------------------------------- API
    def reset(self):
        self._layout()
        lo = self.loco
        oa, _ = lo.reset()
        self.d.qpos[0:2] = self.start
        yaw = self.rng.uniform(-np.pi, np.pi)
        self.d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.m, self.d)
        lo.cmd = np.zeros(2)
        lo.hist[:] = lo._frame(lo._truth())
        self.loco_obs = np.concatenate([lo.hist.reshape(-1), lo.cmd * np.array([0.25, 0.5])]).astype(np.float32)
        if self.low_level == "classical":
            self.ctrl = self._Classical(0.02)
        pos, yaw, t = self._pose()
        self.odo = np.array([pos[0], pos[1], yaw])
        self.odo_bias = self.rng.normal(0, 0.01)                   # gyro bias (rad/s)
        self.odo_scale = 1 + self.rng.normal(0, 0.02)              # wheel radius error
        self.prev_dist = np.linalg.norm(self.goal - pos)
        self.last_a = np.zeros(2)
        self.steps = 0
        scan, _ = self._lidar(pos, yaw)
        self.scan_prev = scan
        return self._obs(pos, yaw, t, scan)

    def _goal_feat(self, pos, yaw):
        dv = self.goal - pos
        dist = np.linalg.norm(dv)
        b = np.arctan2(dv[1], dv[0]) - yaw
        return np.array([min(dist, 10) / 10, np.sin(b), np.cos(b)])

    def _obs(self, pos, yaw, t, scan, true=None):
        depth = self._depth(pos, yaw)
        v_est = float(self.loco._qd("tyreL") + self.loco._qd("tyreR")) * 0.5 * 0.3 * self.odo_scale
        actor = np.concatenate([
            1.0 - scan / MAX_R, 1.0 - self.scan_prev / MAX_R, 1.0 - depth / MAX_R,
            self._goal_feat(self.odo[:2], self.odo[2]),
            [v_est * 0.3, (t["w"] + self.rng.normal(0, 0.02)) * 0.4], self.last_a, [self.steps / EP_STEPS],
        ]).astype(np.float32)
        if true is None:
            true = self._true_scan(pos, yaw)
        critic = np.concatenate([
            1.0 - true / MAX_R, self._goal_feat(pos, yaw), [t["v"] * 0.3, t["w"] * 0.4, true.min() / MAX_R],
            [np.linalg.norm(self.odo[:2] - pos), np.sin(self.odo[2] - yaw)], self.last_a,
        ]).astype(np.float32)
        return actor, critic

    def _collided(self):
        n = self.d.ncon
        if n == 0:
            return False
        gg = self.d.contact.geom[:n]
        hit_t = np.isin(gg, self.tyre_gid)
        hit_o = np.isin(gg, self.obs_gid) | np.isin(gg, self.wall_gid)
        return bool((hit_t.any(1) & hit_o.any(1)).any())

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        v_cmd = V_RANGE[0] + (a[0] + 1) / 2 * (V_RANGE[1] - V_RANGE[0])
        w_cmd = a[1] * W_RANGE
        lo = self.loco
        lo.cmd = np.array([v_cmd, w_cmd])
        collided = False
        for _ in range(LOW_STEPS):
            if self.low_level == "learned":
                la = self.policy(self.loco_obs)
            else:
                la = self._classical_action()
            (self.loco_obs, _), _, done_lo, info = lo.step(la)
            # odometry: encoder speed + biased gyro
            v_enc = float(lo._qd("tyreL") + lo._qd("tyreR")) * 0.5 * 0.3 * self.odo_scale
            t = lo._truth()
            self.odo[2] += (t["w"] + self.odo_bias) * 0.02
            self.odo[:2] += v_enc * 0.02 * np.array([np.cos(self.odo[2]), np.sin(self.odo[2])])
            collided |= self._collided()
            if info["fell"]:
                break
        self.steps += 1
        pos, yaw, t = self._pose()
        dist = np.linalg.norm(self.goal - pos)
        scan, true = self._lidar(pos, yaw)
        progress = self.prev_dist - dist
        self.prev_dist = dist
        near = max(0.0, 0.9 - true.min())
        rew = 3.0 * progress - 0.01 - 0.3 * near ** 2 - 0.02 * float(np.sum((a - self.last_a) ** 2))
        success = dist < 0.6
        fell = info["fell"]
        if success:
            rew += 10.0
        if collided or fell:
            rew -= 10.0
        self.last_a = a
        done = success or collided or fell or self.steps >= EP_STEPS
        obs = self._obs(pos, yaw, t, scan, true)
        self.scan_prev = scan
        info = {"success": success, "collision": collided, "fell": fell, "timeout": self.steps >= EP_STEPS and not (success or collided or fell),
                "dist": dist}
        return obs, float(rew), done, info

    def _classical_action(self):
        """Classical low-level expressed in the policy's action space (for smoke tests / baselines)."""
        lo = self.loco
        g = _Adapter(lo)
        s = g.state()
        tl, tr, _, bob = self.ctrl(g, s, lo.cmd[0], lo.cmd[1])
        tau_sum, tau_diff = tl + tr, tr - tl
        return np.array([tau_sum / 70, tau_diff / 50, bob / 0.698])


class _Adapter:
    """Lets ClassicalController read state from a LocoEnv."""

    def __init__(self, lo):
        self.lo = lo

    def q(self, n):
        return float(self.lo._q(n))

    def qd(self, n):
        return float(self.lo._qd(n))

    def state(self):
        t = self.lo._truth()
        return dict(v=t["v"], yaw_rate=t["w"], roll=t["roll"], roll_rate=t["roll_rate"], pitch=t["pitch"],
                    pitch_rate=t["pitch_rate"])


if __name__ == "__main__":
    import time
    e = NavEnv(0, low_level="classical")
    oa, oc = e.reset()
    print("actor", oa.shape, ACT_OBS, "critic", oc.shape, CRIT_OBS)
    t0 = time.time()
    n, res = 300, []
    for i in range(n):
        (oa, oc), r, d, inf = e.step(np.array([0.3, 0.0]))
        if d:
            res.append(inf)
            e.reset()
    print(f"{n / (time.time() - t0):.0f} nav steps/s", res[:3])
