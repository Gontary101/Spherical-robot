"""
GYRA Mk2 locomotion v2: "works everywhere" teleoperation environment.

Every episode compiles a fresh MuJoCo world (~5 ms), so every randomised quantity (terrain, tyre radius,
payload, masses, gravity/slope) is a real model parameter rather than a run-time patch.

Task: track (v_cmd, w_cmd) from a virtual operator at 50 Hz on whatever ground, whatever load,
whatever is pushing the robot.

World / robot randomisation (scaled by the per-env curriculum level k in [0, 1]):
  terrain    flat | hills | rough (hills + gravel) | curbs/steps/ditches | ramps | mixed   (sim/terrain.py)
  slope      uniform incline via gravity tilt, up to 14 deg
  ground     Coulomb friction 0.25-1.2 (ice/wet tile ... rubber on asphalt), torsional and rolling friction,
             mid-episode surface change (driving from asphalt onto ice)
  robot      +-10 % link masses, CoM shift, 0-10 kg payload anywhere on the spine, tyre radius -1.5/+1 %
  actuators  per-motor strength 0.75-1.1 (battery sag, heating), torque noise, bob worm-drive slew 45-75 deg/s,
             0-40 ms action latency
Disturbances:
  pushes     impulses up to 350 N for 60-300 ms (kicks, collisions), yaw torque kicks up to 40 N m
  wind       steady 0-40 N plus Ornstein-Uhlenbeck gusts
Sensors (actor):
  IMU        gyro noise + per-episode bias up to 0.03 rad/s, accelerometer-gravity noise, mounting misalignment up
             to 1.5 deg, rare glitch frames
  encoders   rate noise on pendulum and both drive motors, bob angle noise

Action (3, [-1, 1]): residual on top of the classical controller with FULL authority:
  tau_sum += 60 a0, tau_diff += 60 a1, bob target += 0.5 a2   (then motor torque-speed limits apply)

Actor observation: HIST = 50 frames (1 s) of FRAME = 18 proprioceptive values + command (2).
Critic observation: estimator targets (N_EST) + current frame + command + privileged state + 7 x 7 terrain
height scan around the robot (sampled from the heightfield the tyres collide with).
"""
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import terrain as TR  # noqa: E402
from gyra2_model import R as R_NOM  # noqa: E402
from gyra2_model import MotorModel, build_xml2  # noqa: E402
from gyra2_sim import ClassicalController  # noqa: E402

DT = 0.02
N_SUB = 10                       # 2 ms physics
HIST = 50
FRAME = 18
N_CMD = 2
ACT_OBS = HIST * FRAME + N_CMD
N_EST = 10
SCAN = 7
SCAN_D = 0.25
N_ACT = 3
EP_LEN = 1000                    # 20 s
RES_SUM, RES_DIFF, RES_BOB = 60.0, 60.0, 0.5
GEOFENCE = TR.HALF - 4.0


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def _rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


class Loco2Env:
    def __init__(self, seed=0, level=0.0, adaptive=True, scenario=None, record=False, world_fn=None, ep_len=EP_LEN,
                 external_cmd=False):
        """scenario: fixed dict of overrides for evaluation (see eval_robust.py); disables the curriculum.
        world_fn(S, hmap) -> (mesh_obstacles, mocap_obstacles): extra geometry compiled into the world (navigation).
        external_cmd: commands are written to self.cmd by a higher level (no virtual operator)."""
        self.world_fn, self.ep_len, self.external_cmd = world_fn, ep_len, external_cmd
        self.spawn = None
        self.rng = np.random.default_rng(seed)
        self.level = level
        self.adaptive = adaptive and scenario is None
        self.scenario = scenario
        self.record = record
        self.cache = []
        self.hist = np.zeros((HIST, FRAME))
        self.m = None
        sx = np.arange(SCAN) - SCAN // 2
        gx, gy = np.meshgrid(sx * SCAN_D, sx * SCAN_D)
        self.scan_pts = np.stack([gx.ravel(), gy.ravel()], 1)
        self.ep_stats = None

    # ------------------------------------------------------------------ world
    def _terrain(self, family, k):
        r = self.rng
        for i, (fam, kk, h) in enumerate(self.cache):
            if fam == family and abs(kk - k) < 0.12 and r.random() < 0.8:
                h = h[::-1] if r.random() < 0.5 else h
                h = h[:, ::-1] if r.random() < 0.5 else h
                return h.T if r.random() < 0.5 else h
        h = TR.make(r, family, k)
        self.cache.append((family, k, h))
        self.cache = self.cache[-10:]
        return h

    def _sample_world(self):
        r, k = self.rng, self.level
        if self.adaptive and r.random() < 0.15:
            k = r.uniform(0, self.level)                       # replay easier levels: no forgetting
        S = dict(k=k)
        S["family"] = "flat" if r.random() < 0.2 else str(r.choice(TR.FAMILIES[1:]))
        S["terrain_k"] = k * r.uniform(0.5, 1.0)
        S["slope"] = np.radians(r.uniform(0, 14 * k)) if r.random() < 0.4 else 0.0
        S["slope_dir"] = r.uniform(-np.pi, np.pi)
        S["mu"] = r.uniform(max(0.25, 0.8 - 0.55 * k), 1.2)
        S["mu2"] = r.uniform(max(0.25, 0.8 - 0.55 * k), 1.2) if r.random() < 0.3 else None
        S["mu_switch_t"] = r.uniform(4, 16)
        S["tors"], S["roll_fr"] = r.uniform(0.008, 0.025), r.uniform(0.002, 0.008)
        S["mscale"] = r.uniform(0.9, 1.1, 4)
        S["com"] = r.uniform(-6, 6, 3) * 1e-3
        S["payload"] = r.uniform(0, 10 * k) if r.random() < 0.5 else 0.0
        S["payload_pos"] = np.array([r.uniform(-0.08, 0.08), r.uniform(-0.15, 0.15), r.uniform(-0.05, 0.10)])
        S["tyre_r"] = R_NOM * r.uniform(0.985, 1.01)
        lo = 1 - 0.25 * k
        S["strength"] = r.uniform(lo, 1.1, 2)
        S["tau_noise"] = 0.05 * k
        S["bob_slew"] = np.radians(r.uniform(45, 75))
        S["delay"] = int(r.integers(0, 1 + int(round(2 * k))))
        S["push_rate"] = 1 / 3.0 if r.random() < 0.8 else 0.0
        S["push_max"] = 50 + 300 * k
        S["kick_max"] = 40 * k
        S["wind"] = r.uniform(0, 40 * k) if r.random() < 0.5 else 0.0
        S["wind_dir"] = r.uniform(-np.pi, np.pi)
        S["gust"] = 10 * k
        S["gyro_bias"] = r.uniform(-0.03, 0.03, 3) * k
        S["imu_tilt"] = np.radians(r.uniform(-1.5, 1.5, 2)) * k
        S["glitch"] = 0.002 * k
        S["v_max"] = 2.0 + 6.5 * k
        if self.scenario:
            S.update(self.scenario)
        return S

    def _build(self, S):
        h = self._terrain(S["family"], S["terrain_k"]) if S["family"] != "flat" else None
        terrain = None
        if h is not None:
            asset, lo, data = TR.hfield_xml(h)
            terrain = (asset, lo)
        sl, sd = S["slope"], S["slope_dir"]
        g = 9.81 * np.array([-np.sin(sl) * np.cos(sd), -np.sin(sl) * np.sin(sd), -np.cos(sl)])
        extra = self.world_fn(S, h) if self.world_fn else ((), ())
        ms = dict(spine=S["mscale"][0], yoke=S["mscale"][1], bob=S["mscale"][2], tyre=S["mscale"][3])
        xml = build_xml2(friction=S["mu"], torsional=S["tors"], rolling=S["roll_fr"], mass_scale=ms, com_shift=S["com"],
                         timestep=DT / N_SUB, terrain=terrain, tyre_r=S["tyre_r"], gravity=g,
                         payload=(S["payload"], S["payload_pos"]) if S["payload"] > 0.05 else None,
                         mesh_obstacles=extra[0], mocap_obstacles=extra[1])
        self.m = mujoco.MjModel.from_xml_string(xml)
        if h is not None:
            self.m.hfield_data[:] = data.ravel()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.jid = {n: m.joint(n) for n in ("tyreL", "tyreR", "yoke", "bob")}
        self.qa = {n: j.qposadr[0] for n, j in self.jid.items()}
        self.va = {n: j.dofadr[0] for n, j in self.jid.items()}
        self.aid = {n: m.actuator(n).id for n in ("driveL", "driveR", "level", "lean")}
        self.bid = {n: m.body(n).id for n in ("spine", "tyreL", "tyreR", "yoke", "bob")}
        self.ground = [m.geom("floor").id] + ([m.geom("terrain").id] if h is not None else [])
        self.tyres = [m.geom("tyreL").id, m.geom("tyreR").id]
        self.gtilt = np.array([np.sin(sl) * np.cos(sd), np.sin(sl) * np.sin(sd)])
        self.hmap = h

    def _set_mu(self, mu):
        for g in self.ground + self.tyres:
            self.m.geom_friction[g, 0] = mu
        self.mu = mu

    # ------------------------------------------------------------------ state
    def q(self, n):
        return float(self.d.qpos[self.qa[n]])

    def qd(self, n):
        return float(self.d.qvel[self.va[n]])

    def _truth(self):
        d = self.d
        Rm = d.xmat[self.bid["spine"]].reshape(3, 3)
        fx, fy = Rm[0, 0], Rm[1, 0]
        nf = max(math.hypot(fx, fy), 1e-9)
        fwd = np.array([fx / nf, fy / nf, 0.0])
        w_world = Rm @ d.qvel[3:6]
        qv = d.qvel
        t = dict(Rm=Rm, fwd=fwd, v=float(qv[0] * fwd[0] + qv[1] * fwd[1]), w=float(w_world[2]),
                 roll=math.asin(_clamp(Rm[2, 1], -1.0, 1.0)), pitch=-math.asin(_clamp(Rm[2, 0], -1.0, 1.0)),
                 pitch_rate=float(w_world @ Rm[:, 1]), roll_rate=float(w_world[0] * fwd[0] + w_world[1] * fwd[1]))
        t["yaw"] = math.atan2(fwd[1], fwd[0])
        t["pend"] = t["pitch"] + self.q("yoke")
        t["pend_rate"] = t["pitch_rate"] + self.qd("yoke")
        return t

    def _frame(self, t):
        r, S = self.rng, self.S
        gyro = self.d.qvel[3:6] + S["gyro_bias"] + r.normal(0, 0.02, 3)
        if r.random() < S["glitch"]:
            gyro = gyro + r.normal(0, 1.0, 3)
        Rm = t["Rm"]
        ax, ay = S["imu_tilt"]
        Rimu = np.array([[1, 0, ay], [0, 1, -ax], [-ay, ax, 1]])              # small mounting misalignment
        grav = Rimu @ (Rm.T @ np.array([0, 0, -1.0])) + r.normal(0, 0.015, 3)
        yk = self.q("yoke") + r.normal(0, 0.002)
        return np.concatenate([
            gyro * 0.25, grav,
            [np.sin(yk), np.cos(yk), (self.qd("yoke") + r.normal(0, 0.05)) * 0.1,
             (self.qd("tyreL") - self.qd("yoke") + r.normal(0, 0.05)) * 0.05,
             (self.qd("tyreR") - self.qd("yoke") + r.normal(0, 0.05)) * 0.05,
             self.q("bob") + r.normal(0, 0.003)],
            self.last_a, self.base_a])

    def _height_scan(self, t):
        """Terrain height on a 7 x 7 grid (0.25 m) around the robot, yaw-aligned, relative to the tyre bottom."""
        p = self.d.qpos[0:3]
        if self.hmap is None:
            z = np.zeros(len(self.scan_pts))
        else:
            pts = self.scan_pts @ _rotz(t["yaw"]).T + p[:2]
            z = TR.height_at(self.hmap, pts[:, 0], pts[:, 1])
        return np.clip(z - (p[2] - self.S["tyre_r"]), -0.5, 0.5)

    def _est_targets(self, t):
        Rm = t["Rm"]
        vb = Rm.T @ self.d.qvel[0:3]
        c, s = np.cos(t["yaw"]), np.sin(t["yaw"])
        slope_h = np.array([c * self.gtilt[0] + s * self.gtilt[1], -s * self.gtilt[0] + c * self.gtilt[1]])
        wind_h = np.array([c * self.wind_f[0] + s * self.wind_f[1], -s * self.wind_f[0] + c * self.wind_f[1]])
        return np.concatenate([vb * 0.25, slope_h * 5, [self.mu, self.S["payload"] / 10], wind_h / 40,
                               [np.mean(self.S["strength"]) - 1]])

    def _obs(self, t):
        cmd = self.cmd * np.array([0.25, 0.5])
        actor = np.concatenate([self.hist.reshape(-1), cmd]).astype(np.float32)
        c, s = np.cos(t["yaw"]), np.sin(t["yaw"])
        pf = self.push_f if self.push_left > 0 else np.zeros(3)
        n = self.d.ncon
        gg = self.d.contact.geom[:n] if n else np.zeros((0, 2), int)
        cont = [float((gg == g).any()) for g in self.tyres]
        S = self.S
        priv = np.concatenate([
            [t["w"] * 0.5, t["roll"], t["pitch"], np.sin(t["pend"]), np.cos(t["pend"]), t["pend_rate"] * 0.1,
             self.q("bob"), t["roll_rate"], t["pitch_rate"]],
            cont, [S["tors"] * 50, S["roll_fr"] * 100], S["mscale"] - 1, S["com"] * 100, S["payload_pos"] * 5,
            S["strength"] - 1, [(S["tyre_r"] - R_NOM) * 50, S["delay"] * 0.5, self.level],
            [(c * pf[0] + s * pf[1]) / 200, (-s * pf[0] + c * pf[1]) / 200, self.kick / 40, float(self.push_left > 0)],
            S["gyro_bias"] * 20, [(self.cmd[0] - t["v"]) * 0.25, (self.cmd[1] - t["w"]) * 0.5],
            self._height_scan(t) * 4,
        ])
        critic = np.concatenate([self._est_targets(t), self.hist[-1], cmd, priv]).astype(np.float32)
        return actor, critic

    # ------------------------------------------------------------------ operator
    def _sample_cmd(self):
        r, S = self.rng, self.S
        vm = S["v_max"]
        a_lat = min(4.0, 0.6 * 9.81 * self.mu)
        u = r.random()
        w_max = min(3.0, 1.5 + vm / 4)
        if u < 0.10:
            v, w = 0.0, 0.0
        elif u < 0.25:
            v, w = 0.0, r.uniform(-w_max, w_max)
        elif u < 0.45:
            v, w = r.uniform(-min(vm, 2.5), vm), 0.0
        else:
            v = r.uniform(-min(vm, 2.5), vm)
            wl = w_max if r.random() < 0.2 else min(w_max, a_lat / max(abs(v), 0.5))
            w = r.uniform(-wl, wl)
        # geofence: the operator steers back towards the middle of the terrain patch
        p = self.d.qpos[0:2]
        if np.linalg.norm(p) > GEOFENCE - 8:
            t = self._truth()
            err = np.arctan2(-p[1], -p[0]) - t["yaw"]
            err = (err + np.pi) % (2 * np.pi) - np.pi
            v = abs(v) if abs(err) < 1.5 else 0.5 * abs(v)
            w = float(np.clip(1.5 * err, -w_max, w_max))
            w = float(np.clip(w, -a_lat / max(abs(v), 0.5), a_lat / max(abs(v), 0.5)))
        self.cmd_target = np.array([v, w])
        self.cmd_ramp = 0.0 if r.random() < 0.5 else r.uniform(0.5, 1.5)
        self.cmd_from = self.cmd.copy()
        self.cmd_t0 = self.d.time
        self.next_cmd_t = self.d.time + r.uniform(1.5, 5.0)

    def _update_cmd(self):
        if self.external_cmd:
            return
        p = self.d.qpos[0:2]
        if self.d.time >= self.next_cmd_t:
            self._sample_cmd()
        elif math.hypot(p[0], p[1]) > GEOFENCE - 8 and self.d.time - self.cmd_t0 > 1.0 and p @ self.d.qvel[0:2] > 0:
            self._sample_cmd()                                  # heading out: operator turns back now
        if self.cmd_ramp <= 0:
            self.cmd = self.cmd_target.copy()
        else:
            a = min(1.0, (self.d.time - self.cmd_t0) / self.cmd_ramp)
            self.cmd = self.cmd_from + a * (self.cmd_target - self.cmd_from)

    # ------------------------------------------------------------------ API
    def reset(self):
        self.S = S = self._sample_world()
        self._build(S)
        self.mu = S["mu"]
        d = self.d
        yaw = self.rng.uniform(-np.pi, np.pi)
        if self.spawn is not None:
            x, y, yaw = self.spawn
            z0 = float(TR.height_at(self.hmap, np.array([x]), np.array([y]))[0]) if self.hmap is not None else 0.0
            d.qpos[0:3] = [x, y, z0 + S["tyre_r"] + 0.002]
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.m, d)
        self.last_a = np.zeros(N_ACT)
        self.prev_a = np.zeros(N_ACT)
        self.base_a = np.zeros(N_ACT)
        self.ctrl = ClassicalController(DT)
        self.a_queue = [np.zeros(N_ACT)] * 3
        self.bob_cmd = 0.0
        self.steps = 0
        self.push_left, self.push_f, self.kick = 0, np.zeros(3), 0.0
        self.gust = np.zeros(2)
        self.wind_f = np.zeros(2)
        self.cmd = np.zeros(2)
        if not self.external_cmd:
            self._sample_cmd()
            self._update_cmd()
        self.ep_stats = dict(track=0.0, n=0, ev=0.0, ew=0.0, tilt=0.0, energy=0.0)
        t = self._truth()
        self.hist[:] = self._frame(t)
        return self._obs(t)

    def _disturb(self):
        r, S, d = self.rng, self.S, self.d
        if self.push_left <= 0 and r.random() < DT * S["push_rate"]:
            ang = r.uniform(-np.pi, np.pi)
            mag = r.uniform(50, S["push_max"])
            self.push_f = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0])
            self.push_left = int(r.integers(3, 16))
            self.kick = r.uniform(-S["kick_max"], S["kick_max"]) if r.random() < 0.4 else 0.0
        # wind: steady + OU gusts (tau = 1 s)
        self.gust += -self.gust * DT + S["gust"] * np.sqrt(2 * DT) * r.normal(size=2)
        self.wind_f = S["wind"] * np.array([np.cos(S["wind_dir"]), np.sin(S["wind_dir"])]) + self.gust
        f = np.array([self.wind_f[0], self.wind_f[1], 0.0])
        tq = np.zeros(3)
        if self.push_left > 0:
            f = f + self.push_f
            tq[2] = self.kick
        d.xfrc_applied[self.bid["spine"], :3] = f
        d.xfrc_applied[self.bid["spine"], 3:] = tq
        self.push_left -= 1
        if S["mu2"] is not None and d.time >= S["mu_switch_t"] and self.mu != S["mu2"]:
            self._set_mu(S["mu2"])

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        S, d, r = self.S, self.d, self.rng
        self.a_queue.append(a)
        a_eff = self.a_queue[-1 - S["delay"]]
        self.a_queue = self.a_queue[-4:]
        self._disturb()
        t = self._truth()
        est = dict(v=t["v"] + r.normal(0, 0.05), yaw_rate=t["w"] + S["gyro_bias"][2] + r.normal(0, 0.02),
                   roll=t["roll"] + r.normal(0, 0.01), roll_rate=t["roll_rate"] + r.normal(0, 0.02),
                   pitch=t["pitch"] + r.normal(0, 0.005), pitch_rate=t["pitch_rate"] + r.normal(0, 0.02))
        tl, tr, _, bob_c = self.ctrl(self, est, self.cmd[0], self.cmd[1])
        self.base_a = np.array([(tl + tr) / 100.0, (tr - tl) / 100.0, bob_c / 0.698])
        tau_sum = (tl + tr) + RES_SUM * a_eff[0]
        tau_diff = (tr - tl) + RES_DIFF * a_eff[1]
        bob_target = _clamp(bob_c + RES_BOB * a_eff[2], -0.698, 0.698)
        step = S["bob_slew"] * DT
        self.bob_cmd += _clamp(bob_target - self.bob_cmd, -step, step)
        wL, wR = self.qd("tyreL") - self.qd("yoke"), self.qd("tyreR") - self.qd("yoke")
        nz = 1 + S["tau_noise"] * r.normal(size=2)
        cmdL, cmdR = 0.5 * tau_sum - 0.5 * tau_diff, 0.5 * tau_sum + 0.5 * tau_diff
        tauL = MotorModel.limit_s(cmdL, wL, S["strength"][0]) * nz[0]
        tauR = MotorModel.limit_s(cmdR, wR, S["strength"][1]) * nz[1]
        d.ctrl[self.aid["driveL"]] = tauL
        d.ctrl[self.aid["driveR"]] = tauR
        d.ctrl[self.aid["level"]] = _clamp(90 * t["pitch"] + 9 * t["pitch_rate"], -15, 15)
        d.ctrl[self.aid["lean"]] = self.bob_cmd
        mujoco.mj_step(self.m, d, nstep=N_SUB)
        self.steps += 1
        self._update_cmd()
        t = self._truth()
        # ---------------- reward (each term logged)
        ev, ew = self.cmd[0] - t["v"], self.cmd[1] - t["w"]
        sat = (max(abs(cmdL) - abs(tauL), 0) + max(abs(cmdR) - abs(tauR), 0)) / MotorModel.TAU_STALL
        power = max(tauL * wL, 0) + max(tauR * wR, 0)
        terms = dict(
            track_v=1.0 * math.exp(-ev ** 2 / 0.25),
            track_w=0.75 * math.exp(-ew ** 2 / 0.25),
            steady=0.25 * math.exp(-(t["pitch_rate"] ** 2 + t["roll_rate"] ** 2) / 0.25),
            tilt=-1.0 * t["pitch"] ** 2 - 0.2 * max(abs(t["roll"]) - 0.15, 0) ** 2,
            act_rate=-0.02 * float(np.sum((a - self.last_a) ** 2)),
            act_smooth=-0.01 * float(np.sum((a - 2 * self.last_a + self.prev_a) ** 2)),
            power=-4e-4 * power,
            saturation=-0.3 * sat,
            pendulum=-0.01 * t["pend_rate"] ** 2,
            bob_limit=-0.5 * max(abs(self.q("bob")) - 0.6, 0),
        )
        rew = sum(terms.values())
        fell = abs(t["pend"]) > 2.4 or abs(t["roll"]) > 1.1 or abs(t["pitch"]) > 0.6 or not np.isfinite(d.qpos[2])
        out = math.hypot(d.qpos[0], d.qpos[1]) > GEOFENCE
        if fell:
            rew -= 20.0
        self.prev_a, self.last_a = self.last_a, a
        self.hist[:-1] = self.hist[1:]
        self.hist[-1] = self._frame(t)
        es = self.ep_stats
        es["n"] += 1
        es["track"] += (terms["track_v"] + terms["track_w"] / 0.75) / 2
        es["ev"] += ev ** 2
        es["ew"] += ew ** 2
        es["tilt"] += t["pitch"] ** 2
        es["energy"] += (abs(tauL * wL) + abs(tauR * wR)) * DT
        timeout = (self.steps >= self.ep_len or out) and not fell
        done = fell or timeout
        info = {"ev": abs(ev), "ew": abs(ew), "fell": fell, "timeout": timeout, "terms": terms, "level": self.level}
        if done:
            n = es["n"]
            info["episode"] = dict(track=es["track"] / n, ev_rms=np.sqrt(es["ev"] / n), ew_rms=np.sqrt(es["ew"] / n),
                                   tilt_rms=np.degrees(np.sqrt(es["tilt"] / n)), energy=es["energy"], fell=fell,
                                   level=self.S["k"], family=self.S["family"], length=n)
            if self.adaptive:
                if fell:
                    self.level = max(0.0, self.level - 0.05)
                elif es["track"] / n > 0.6 and self.S["k"] >= self.level - 1e-9:
                    self.level = min(1.0, self.level + 0.05)
        obs = self._obs(t)
        return obs, float(rew) * DT * 5, done, info

    # mirror maps (robot is left/right symmetric) for the symmetry loss ------------------------------------
    @staticmethod
    def mirror_frame_signs():
        # gyro x,y,z | grav x,y,z | sin cos rate(yoke) | wL wR (swapped) | bob | last a (sum,diff,bob) | base a
        return np.array([-1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1], np.float32)


CRIT_OBS = None


def _crit_dim():
    e = Loco2Env(0)
    return len(e.reset()[1])


if __name__ == "__main__":
    import time
    e = Loco2Env(0, level=1.0)
    oa, oc = e.reset()
    print("actor", oa.shape, ACT_OBS, "critic", oc.shape)
    t0 = time.time()
    n, falls, eps = 3000, 0, 0
    for i in range(n):
        o, r, dn, inf = e.step(np.zeros(3))
        if dn:
            falls += inf["fell"]
            eps += 1
            print(inf["episode"])
            e.reset()
    print(f"{n / (time.time() - t0):.0f} env steps/s (incl. resets), falls {falls}/{eps}")
