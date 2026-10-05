"""
GYRA Mk2 locomotion / teleoperation environment (MuJoCo), built for asymmetric actor-critic PPO.

Task: track joystick commands (v_cmd forward speed, w_cmd yaw rate) at 50 Hz.

Action (3, in [-1, 1]):  tau_sum = 50*a0 (pendulum drive),  tau_diff = 50*a1 (right - left),
                         bob target = 0.698*a2 (worm drive, rate-limited to 60 deg/s)
The spine levelling motor stays on its own hardware PD loop (it is not part of the policy).

Actor observation = what the real robot measures, with noise, 4-frame history:
    spine IMU gyro (3) + gravity direction (3), pendulum encoder (sin, cos, rate),
    both drive-motor encoder rates (2), bob encoder (1), previous action (3)   -> 15 per frame
    + command (2)                                                               -> 62 total
Critic observation = actor's current frame + privileged simulator state (true body velocity,
attitude, pendulum absolute angle, contacts, friction, mass scales, CoM shift, motor strength,
gravity tilt, push force) -> 45.
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from gyra2_model import MotorModel, build_xml2  # noqa: E402
from gyra2_sim import ClassicalController  # noqa: E402

DT = 0.02
N_SUB = 20
HIST = 4
FRAME = 18
ACT_OBS = FRAME * HIST + 2
CRIT_OBS = FRAME + 2 + 28
RES_SUM, RES_DIFF, RES_BOB = 25.0, 25.0, 0.35   # residual authority on top of the classical loop
N_ACT = 3
EP_LEN = 1000                      # 20 s
TAU_SUM, TAU_DIFF = 50.0, 50.0     # action scales (N m at the tyre)


class LocoEnv:
    def __init__(self, seed=0, v_max=3.0, randomize=True, push=True, n_obstacles=0, arena=None,
                 residual=True, difficulty=1.0):
        self.rng = np.random.default_rng(seed)
        self.residual = residual
        self.difficulty = difficulty
        # obstacle slots are pre-allocated far away and moved/resized per episode (no recompiles)
        obst = [("cylinder" if i % 2 == 0 else "box", (200.0 + 3 * i, 200.0, 0.4), (0.2, 0.4) if i % 2 == 0 else (0.2, 0.2, 0.4))
                for i in range(n_obstacles)]
        self.m = mujoco.MjModel.from_xml_string(build_xml2(timestep=DT / N_SUB, obstacles=obst, arena=arena))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.jid = {n: m.joint(n) for n in ("tyreL", "tyreR", "yoke", "bob")}
        self.qa = {n: j.qposadr[0] for n, j in self.jid.items()}
        self.va = {n: j.dofadr[0] for n, j in self.jid.items()}
        self.aid = {n: m.actuator(n).id for n in ("driveL", "driveR", "level", "lean")}
        self.bid = {n: m.body(n).id for n in ("spine", "tyreL", "tyreR", "yoke", "bob")}
        self.gid = {n: m.geom(n).id for n in ("floor", "tyreL", "tyreR")}
        self.base_mass = m.body_mass.copy()
        self.base_inertia = m.body_inertia.copy()
        self.base_ipos = m.body_ipos.copy()
        self.v_max = v_max
        self.randomize = randomize
        self.push = push
        self.hist = np.zeros((HIST, FRAME))

    # ------------------------------------------------------------------ randomisation
    def _randomize(self):
        m, r = self.m, self.rng
        if self.randomize:
            self.fric = r.uniform(0.5, 1.2)
            self.tors = r.uniform(0.008, 0.025)
            self.roll_fr = r.uniform(0.002, 0.008)
            self.mscale = r.uniform(0.9, 1.1, size=4)
            self.com = np.array([r.uniform(-5, 5), r.uniform(-5, 5), r.uniform(-5, 5)]) * 1e-3
            self.strength = r.uniform(0.8, 1.1)
            slope = np.radians(r.uniform(0, 12 * self.difficulty)) if r.random() < 0.5 else 0.0
            sdir = r.uniform(-np.pi, np.pi)
            self.delay = int(r.integers(0, 2))
        else:
            self.fric, self.tors, self.roll_fr = 1.0, 0.015, 0.005
            self.mscale, self.com, self.strength = np.ones(4), np.zeros(3), 1.0
            slope, sdir, self.delay = 0.0, 0.0, 0
        for g in ("floor", "tyreL", "tyreR"):
            m.geom_friction[self.gid[g]] = [self.fric, self.tors, self.roll_fr]
        for k, b in enumerate(("spine", "yoke", "bob")):
            i = self.bid[b]
            m.body_mass[i] = self.base_mass[i] * self.mscale[k]
            m.body_inertia[i] = self.base_inertia[i] * self.mscale[k]
        for b in ("tyreL", "tyreR"):
            i = self.bid[b]
            m.body_mass[i] = self.base_mass[i] * self.mscale[3]
            m.body_inertia[i] = self.base_inertia[i] * self.mscale[3]
        m.body_ipos[self.bid["spine"]] = self.base_ipos[self.bid["spine"]] + self.com
        self.gtilt = np.array([np.sin(slope) * np.cos(sdir), np.sin(slope) * np.sin(sdir)])
        m.opt.gravity[:] = [-9.81 * self.gtilt[0], -9.81 * self.gtilt[1], -9.81 * np.cos(slope)]

    # ------------------------------------------------------------------ commands
    def _sample_cmd(self):
        r = self.rng
        u = r.random()
        vm = self.v_max
        w_max = min(4.0, 1.5 + vm / 3)                     # yaw-rate range grows with the curriculum
        if u < 0.15:
            v, w = 0.0, r.uniform(-w_max, w_max)              # turn in place
        elif u < 0.35:
            v, w = r.uniform(-min(vm, 3), vm), 0.0            # straight
        elif u < 0.45:
            v, w = 0.0, 0.0                                  # stand still
        else:
            v = r.uniform(-min(vm, 3), vm)
            wl = min(w_max, 4.0 / max(abs(v), 0.5))          # tip-over bound |v w| <= 4 m/s^2
            w = r.uniform(-wl, wl)
        self.cmd = np.array([v, w])
        self.next_cmd_t = self.d.time + r.uniform(2.0, 5.0)

    # ------------------------------------------------------------------ state helpers
    def q(self, n):                 # adapter API used by ClassicalController
        return float(self.d.qpos[self.qa[n]])

    def qd(self, n):
        return float(self.d.qvel[self.va[n]])

    def _q(self, n):
        return self.d.qpos[self.qa[n]]

    def _qd(self, n):
        return self.d.qvel[self.va[n]]

    def _truth(self):
        d = self.d
        Rm = d.xmat[self.bid["spine"]].reshape(3, 3)
        fwd = Rm[:, 0].copy()
        fwd[2] = 0
        fwd /= max(np.linalg.norm(fwd), 1e-9)
        w_world = Rm @ d.qvel[3:6]
        t = dict(Rm=Rm, fwd=fwd, v=float(d.qvel[0:3] @ fwd), w=float(w_world[2]),
                 roll=float(np.arcsin(np.clip(Rm[2, 1], -1, 1))),
                 pitch=float(-np.arcsin(np.clip(Rm[2, 0], -1, 1))),
                 pitch_rate=float(w_world @ Rm[:, 1]), roll_rate=float(w_world @ fwd))
        t["pend"] = t["pitch"] + self._q("yoke")
        t["pend_rate"] = t["pitch_rate"] + self._qd("yoke")
        return t

    def _frame(self, t):
        n = self.rng.normal
        gyro = self.d.qvel[3:6] + n(0, 0.02, 3)
        grav = t["Rm"].T @ np.array([0, 0, -1.0]) + n(0, 0.015, 3)
        yk = self._q("yoke") + n(0, 0.002)
        f = np.concatenate([
            gyro * 0.25, grav,
            [np.sin(yk), np.cos(yk), (self._qd("yoke") + n(0, 0.05)) * 0.1,
             (self._qd("tyreL") - self._qd("yoke") + n(0, 0.05)) * 0.05,
             (self._qd("tyreR") - self._qd("yoke") + n(0, 0.05)) * 0.05,
             self._q("bob") + n(0, 0.003)],
            self.last_a, self.base_a,
        ])
        return f

    def _contacts(self):
        n = self.d.ncon
        if n == 0:
            return np.zeros(2)
        gg = self.d.contact.geom[:n]
        return np.array([float((gg == self.gid["tyreL"]).any()), float((gg == self.gid["tyreR"]).any())])

    def _obs(self, t):
        cmd = self.cmd * np.array([0.25, 0.5])
        actor = np.concatenate([self.hist.reshape(-1), cmd]).astype(np.float32)
        Rm = t["Rm"]
        vb = Rm.T @ self.d.qvel[0:3]
        priv = np.concatenate([
            vb * 0.25, [t["w"] * 0.5, t["roll"], t["pitch"], np.sin(t["pend"]), np.cos(t["pend"]),
                        t["pend_rate"] * 0.1, self._q("bob")],
            self._contacts(), [self.fric, self.tors * 50, self.roll_fr * 100],
            self.mscale - 1, self.com * 100, [self.strength - 1], self.gtilt * 5,
            [float(self.push_left > 0)], [(self.cmd[0] - t["v"]) * 0.25, (self.cmd[1] - t["w"]) * 0.5],
        ])
        critic = np.concatenate([self.hist[-1], cmd, priv]).astype(np.float32)
        return actor, critic

    # ------------------------------------------------------------------ API
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self._randomize()
        yaw = self.rng.uniform(-np.pi, np.pi)
        self.d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.m, self.d)
        self.last_a = np.zeros(N_ACT)
        self.base_a = np.zeros(N_ACT)
        self.ctrl = ClassicalController(DT)
        self.a_queue = [np.zeros(N_ACT)] * 2
        self.bob_cmd = 0.0
        self.steps = 0
        self.push_left = 0
        self.push_f = np.zeros(3)
        self._sample_cmd()
        t = self._truth()
        f = self._frame(t)
        self.hist[:] = f
        return self._obs(t)

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        self.a_queue.append(a)
        a_eff = self.a_queue[-1 - self.delay]
        self.a_queue = self.a_queue[-3:]
        d = self.d
        tau_sum, tau_diff = TAU_SUM * a_eff[0], TAU_DIFF * a_eff[1]
        bob_target = 0.698 * a_eff[2]
        # pushes
        if self.push and self.push_left <= 0 and self.rng.random() < DT / 4.0 * self.difficulty:
            ang = self.rng.uniform(-np.pi, np.pi)
            mag = self.rng.uniform(50, 50 + 200 * self.difficulty)
            self.push_f = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0])
            self.push_left = int(self.rng.integers(3, 8))
        d.xfrc_applied[self.bid["spine"], :3] = self.push_f if self.push_left > 0 else 0.0
        self.push_left -= 1
        t = self._truth()
        if self.residual:
            # classical loop on *estimated* state (IMU + encoders, with noise), policy adds a residual
            n = self.rng.normal
            est = dict(v=t["v"] + n(0, 0.05), yaw_rate=t["w"] + n(0, 0.02), roll=t["roll"] + n(0, 0.01),
                       roll_rate=t["roll_rate"] + n(0, 0.02), pitch=t["pitch"] + n(0, 0.005),
                       pitch_rate=t["pitch_rate"] + n(0, 0.02))
            tl, tr, _, bob_c = self.ctrl(self, est, self.cmd[0], self.cmd[1])
            self.base_a = np.array([(tl + tr) / TAU_SUM, (tr - tl) / TAU_DIFF, bob_c / 0.698])
            tau_sum = (tl + tr) + RES_SUM * a_eff[0]
            tau_diff = (tr - tl) + RES_DIFF * a_eff[1]
            bob_target = float(np.clip(bob_c + RES_BOB * a_eff[2], -0.698, 0.698))
        step = np.radians(60) * DT                       # worm drive slew limit
        self.bob_cmd += float(np.clip(bob_target - self.bob_cmd, -step, step))
        wL = self._qd("tyreL") - self._qd("yoke")
        wR = self._qd("tyreR") - self._qd("yoke")
        d.ctrl[self.aid["driveL"]] = MotorModel.limit(0.5 * tau_sum - 0.5 * tau_diff, wL, self.strength)
        d.ctrl[self.aid["driveR"]] = MotorModel.limit(0.5 * tau_sum + 0.5 * tau_diff, wR, self.strength)
        d.ctrl[self.aid["level"]] = np.clip(90 * t["pitch"] + 9 * t["pitch_rate"], -15, 15)
        d.ctrl[self.aid["lean"]] = self.bob_cmd
        mujoco.mj_step(self.m, d, nstep=N_SUB)
        self.steps += 1
        if d.time >= self.next_cmd_t:
            self._sample_cmd()
        t = self._truth()
        # reward
        ev, ew = self.cmd[0] - t["v"], self.cmd[1] - t["w"]
        r_track = np.exp(-ev ** 2 / 0.5) + np.exp(-ew ** 2 / 0.5) - 0.15 * min(abs(ev), 3) - 0.15 * min(abs(ew), 3)
        p_roll = 0.05 * t["roll_rate"] ** 2 + 0.5 * max(abs(t["roll"]) - 0.35, 0) ** 2
        p_spine = 2.0 * t["pitch"] ** 2
        p_act = 0.05 * float(np.sum((a - self.last_a) ** 2)) + (0.02 * float(np.sum(a ** 2)) if self.residual else 0.0)
        p_energy = 2e-4 * abs(d.ctrl[self.aid["driveL"]] * wL) + 2e-4 * abs(d.ctrl[self.aid["driveR"]] * wR)
        p_pend = 0.02 * t["pend_rate"] ** 2 * 0.1
        rew = r_track - p_roll - p_spine - p_act - p_energy - p_pend
        fell = abs(t["pend"]) > 2.4 or abs(t["roll"]) > 1.1 or not np.isfinite(d.qpos).all()
        if fell:
            rew -= 10.0
        self.last_a = a
        f = self._frame(t)
        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = f
        done = fell or self.steps >= EP_LEN
        info = {"ev": abs(ev), "ew": abs(ew), "fell": fell, "timeout": self.steps >= EP_LEN and not fell}
        obs = self._obs(t)
        return obs, float(rew) * DT * 5, done, info


if __name__ == "__main__":
    import time
    e = LocoEnv(0)
    o = e.reset()
    print("actor", o[0].shape, "critic", o[1].shape)
    assert o[0].shape[0] == ACT_OBS and o[1].shape[0] == CRIT_OBS
    t0 = time.time()
    n = 2000
    for i in range(n):
        o, r, dn, inf = e.step(np.random.uniform(-1, 1, 3) * 0.3)
        if dn:
            e.reset()
    print(f"{n / (time.time() - t0):.0f} env steps/s")
