"""
GYRA Mk1 - closed-loop MuJoCo experiments.

    python sim/scenarios.py            # runs all scenarios, writes sim/out/*.npz + media/sim_*.png

Scenarios
    drive   accelerate to 3 m/s, cruise, brake: tyre speed, pendulum pitch vs levelled-spine pitch
    turn    2 m/s, lean-steer step (bob 25 deg); CMG roll damping OFF vs ON
    spin    zero-speed turn-in-place using the CMG pair in "yaw mode"
    slope   climbing 8..16 deg grades at 1 m/s
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gyra_model import build_xml  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "sim", "out")
os.makedirs(OUT, exist_ok=True)
W_FLY = 10000 * 2 * np.pi / 60


class Gyra:
    def __init__(self, **kw):
        self.m = mujoco.MjModel.from_xml_string(build_xml(**kw))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.j = {n: m.joint(n) for n in ("tyre", "yoke", "bob", "gimL", "gimR", "flyL", "flyR")}
        self.a = {n: m.actuator(n).id for n in ("drive", "level", "lean", "gimL", "gimR", "spinL", "spinR")}
        # flywheels already at speed (spin-up takes ~60 s with the 40 W spin motors)
        self.d.qvel[self.j["flyL"].dofadr[0]] = W_FLY
        self.d.qvel[self.j["flyR"].dofadr[0]] = -W_FLY
        mujoco.mj_forward(m, self.d)
        self.I_fly = 0.0
        b = m.body("flyL")
        self.h = float(b.inertia[2]) * W_FLY        # principal inertia about the spin axis
        self.int_v = 0.0
        self.KV, self.KI, self.KP_TH, self.KD_TH = 0.9, 0.45, 70.0, 9.0
        self.bob_cmd = 0.0
        self.log = []

    # ---- state ----------------------------------------------------------
    def q(self, n):
        return float(self.d.qpos[self.j[n].qposadr[0]])

    def qd(self, n):
        return float(self.d.qvel[self.j[n].dofadr[0]])

    def state(self):
        d = self.d
        Rm = d.xmat[self.m.body("spine").id].reshape(3, 3)
        fwd = Rm[:, 0].copy()
        fwd[2] = 0
        fwd /= np.linalg.norm(fwd)
        yaw = np.arctan2(fwd[1], fwd[0])
        roll = np.arcsin(np.clip(Rm[2, 1], -1, 1))            # axle tilt (lean)
        pitch = -np.arcsin(np.clip(Rm[2, 0], -1, 1))          # spine (sensor) pitch
        w = d.qvel[3:6]                                       # free joint ang. vel (body frame)
        w_world = Rm @ w
        v_world = d.qvel[0:3]
        return dict(
            x=float(d.qpos[0]), y=float(d.qpos[1]), yaw=yaw, roll=roll, pitch=pitch,
            v=float(v_world @ fwd), roll_rate=float(w_world @ fwd), yaw_rate=float(w_world[2]),
            pitch_rate=float(w_world @ Rm[:, 1]),
        )

    # ---- one control step -------------------------------------------------
    def step(self, v_des, bob_des=0.0, cmg_mode="roll", cmg_gain=0.0, yaw_delta_rate=0.0,
             level=True, dt_ctrl=0.002, gim_rates=(0.0, 0.0), yaw_hold=None):
        s = self.state()
        d = self.d
        if yaw_hold is not None and abs(s["v"]) > 0.3:
            # heading hold through lean-steer (precession): +bob -> CCW yaw
            err = np.arctan2(np.sin(yaw_hold - s["yaw"]), np.cos(yaw_hold - s["yaw"]))
            bob_des = bob_des + float(np.clip(1.6 * err - 0.6 * s["yaw_rate"], -0.35, 0.35)) * np.sign(s["v"])
        # 1) speed: cascade. outer PI -> pendulum swing setpoint (clamped to +-65 deg so the
        #    pendulum can never be driven over the top), inner PD -> drive torque.
        e = v_des - s["v"]
        pend = s["pitch"] + self.qd("yoke") * 0 + self.q("yoke")
        pend_rate = s["pitch_rate"] + self.qd("yoke")
        lim = np.radians(65)
        u = self.KV * e + self.KI * self.int_v
        if abs(u) < lim or np.sign(e) != np.sign(u):          # anti-windup
            self.int_v += e * dt_ctrl
        th_ref = -float(np.clip(u, -lim, lim))                # forward swing is negative
        tau = self.KP_TH * (pend - th_ref) + self.KD_TH * pend_rate
        d.ctrl[self.a["drive"]] = np.clip(tau, -32, 32)
        # 2) spine levelling (motor between spine and yoke)
        d.ctrl[self.a["level"]] = (90 * s["pitch"] + 9 * s["pitch_rate"]) if level else 0.0
        # 3) lean actuator: self-locking worm, rate-limited to 45 deg/s
        step_max = np.radians(45) * dt_ctrl
        self.bob_cmd += np.clip(bob_des - self.bob_cmd, -step_max, step_max)
        d.ctrl[self.a["lean"]] = self.bob_cmd
        # 4) CMG pair
        gL, gR = self.q("gimL"), self.q("gimR")
        if cmg_mode == "roll":
            # scissored: body roll torque = -2 h cos(g) g_dot  -> damp roll rate
            g = 0.5 * (gL - gR)
            gdot = cmg_gain * s["roll_rate"] / (2 * self.h * max(np.cos(g), 0.3)) - 1.2 * g
            gdot = float(np.clip(gdot, -4, 4))
            if abs(g) > np.radians(70) and np.sign(gdot) == np.sign(g):
                gdot = 0.0
            tL, tR = gdot, -gdot
        elif cmg_mode == "yaw":   # gimbals parked near +-90 deg, swept together (pure yaw torque)
            tL = tR = yaw_delta_rate
        else:                      # "manual"
            tL, tR = gim_rates
        d.ctrl[self.a["gimL"]] = tL
        d.ctrl[self.a["gimR"]] = tR
        d.ctrl[self.a["spinL"]] = W_FLY
        d.ctrl[self.a["spinR"]] = -W_FLY
        n = int(round(dt_ctrl / self.m.opt.timestep))
        mujoco.mj_step(self.m, self.d, nstep=n)
        s.update(t=float(d.time), tau=float(d.ctrl[self.a["drive"]]), yoke=self.q("yoke"),
                 pend=s["pitch"] + self.q("yoke"), bob=self.q("bob"), gimL=gL, gimR=gR,
                 level_tau=float(d.ctrl[self.a["level"]]), tyre_w=self.qd("tyre"))
        self.log.append(s)
        return s

    def arrays(self):
        return {k: np.array([r[k] for r in self.log]) for k in self.log[0]}


# ---------------------------------------------------------------------------
def ramp(t, pts):
    ts, vs = zip(*pts)
    return float(np.interp(t, ts, vs))


def run_drive():
    g = Gyra()
    prof = [(0, 0), (1, 0), (3, 3.0), (8, 3.0), (10.5, 0), (14, 0)]
    while g.d.time < 14:
        g.step(ramp(g.d.time, prof), yaw_hold=0.0, cmg_gain=10.0)
    return g.arrays()


def run_turn(cmg_on, v=2.0, bob=25.0, T=16.0):
    g = Gyra()
    while g.d.time < T:
        t = g.d.time
        b = np.radians(bob) if 5 <= t < 13 else 0.0
        g.step(ramp(t, [(0, 0), (2, v), (T, v)]), bob_des=b, cmg_gain=10.0 if cmg_on else 0.0)
    return g.arrays()


def run_spin(n_sweeps=4, noslip=8, ret_rate=0.5):
    """Park the gimbals at +-90 deg (slowly), then fast sweep / slow return cycles.
    noslip: MuJoCo's no-slip friction pass, models rubber stiction (no creep under sub-friction torque)."""
    g = Gyra(noslip=noslip)
    # park: L -> +90, R -> -90 at 0.35 rad/s (scissor) - slow so the roll impulse is small
    while g.d.time < 0.5:
        g.step(0.0)
    while g.q("gimL") < np.radians(90) - 0.01:
        g.step(0.0, cmg_mode="manual", gim_rates=(0.35, -0.35))
    # settle, then go to delta = -60 deg slowly (torque stays below pivot friction)
    def hold(T):
        t0 = g.d.time
        while g.d.time < t0 + T:
            g.step(0.0, cmg_mode="yaw", yaw_delta_rate=0.0)

    def move_delta(target, rate):
        while True:
            dlt = g.q("gimL") - np.pi / 2
            if abs(dlt - target) < 0.02:
                break
            g.step(0.0, cmg_mode="yaw", yaw_delta_rate=float(np.sign(target - dlt) * rate))

    hold(1.0)
    move_delta(np.radians(-60), 0.25)
    hold(1.0)
    for _ in range(n_sweeps):
        move_delta(np.radians(60), 4.0)        # fast: yaw torque > pivot friction
        hold(1.2)
        move_delta(np.radians(-60), ret_rate)  # slow return: below pivot friction
        hold(0.4)
    return g.arrays()


def run_slope(deg, T=10.0):
    g = Gyra(slope_deg=deg)
    while g.d.time < T:
        g.step(ramp(g.d.time, [(0, 0), (2, 1.0), (T, 1.0)]), yaw_hold=0.0, cmg_gain=10.0)
    return g.arrays()


# ---------------------------------------------------------------------------
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8.5)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    deg = np.degrees

    # --- drive ---
    a = res["drive"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax[0].plot(a["t"], a["v"], color=C1, lw=2)
    style(ax[0], "Speed tracking (0 -> 3 m/s -> 0)", "time (s)", "forward speed (m/s)")
    ax[1].plot(a["t"], deg(a["pend"]), color=C2, lw=2, label="pendulum / RT-G-style rigid pods")
    ax[1].plot(a["t"], deg(a["pitch"]), color=C1, lw=2, label="GYRA levelled sensor spine")
    style(ax[1], "Sensor pitch while accelerating & braking", "time (s)", "pitch (deg)")
    ax[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "sim_drive.png"), facecolor=SURF)

    # --- turn ---
    off, on = res["turn_off"], res["turn_on"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax[0].plot(off["t"], deg(off["roll"]), color=C2, lw=2, label="CMG off (pendulum only)")
    ax[0].plot(on["t"], deg(on["roll"]), color=C1, lw=2, label="CMG roll damping on")
    ax[0].plot(on["t"], deg(on["bob"]), color=C3, lw=1.5, ls="--", label="lean actuator (bob) angle")
    style(ax[0], "Lean-steer step at 2 m/s: axle roll (wobble)", "time (s)", "angle (deg)")
    ax[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    ax[1].plot(off["x"], off["y"], color=C2, lw=2, label="CMG off")
    ax[1].plot(on["x"], on["y"], color=C1, lw=2, label="CMG on")
    ax[1].set_aspect("equal", adjustable="datalim")
    style(ax[1], "Ground track", "x (m)", "y (m)")
    ax[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "sim_turn.png"), facecolor=SURF)

    # --- spin ---
    a = res["spin"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax[0].plot(a["t"], deg(np.unwrap(a["yaw"])), color=C1, lw=2)
    style(ax[0], "Turn-in-place with the CMG pair (yaw mode)", "time (s)", "heading (deg)")
    ax[1].plot(a["t"], deg(a["gimL"]), color=C1, lw=2, label="gimbal L")
    ax[1].plot(a["t"], deg(a["gimR"]), color=C2, lw=2, label="gimbal R")
    style(ax[1], "Gimbal angles: park, fast sweep, slow return", "time (s)", "angle (deg)")
    ax[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "sim_spin.png"), facecolor=SURF)

    # --- slope ---
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=150)
    fig.patch.set_facecolor(SURF)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
    for (k, a), c in zip(sorted(((k, v) for k, v in res.items() if k.startswith("slope")),
                                key=lambda kv: int(kv[0].split("_")[1])), cols):
        ax.plot(a["t"], a["x"], color=c, lw=2, label=k.split("_")[1] + " deg")
    style(ax, "Hill climb at 1 m/s command", "time (s)", "distance up-slope (m)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "sim_slope.png"), facecolor=SURF)


def summary(res):
    out = {}
    a = res["drive"]
    cr = (a["t"] > 4) & (a["t"] < 8)
    out["drive_cruise_speed_ms"] = float(a["v"][cr].mean())
    out["drive_peak_pendulum_pitch_deg"] = float(np.degrees(np.abs(a["pend"]).max()))
    out["drive_peak_spine_pitch_deg"] = float(np.degrees(np.abs(a["pitch"]).max()))
    for k in ("turn_off", "turn_on"):
        a = res[k]
        w = (a["t"] > 8) & (a["t"] < 13)
        out[f"{k}_roll_std_deg"] = float(np.degrees(a["roll"][w].std()))
        out[f"{k}_roll_p2p_deg"] = float(np.degrees(np.ptp(a["roll"][w])))
        out[f"{k}_mean_yaw_rate_dps"] = float(np.degrees(a["yaw_rate"][w].mean()))
        out[f"{k}_turn_radius_m"] = float(abs(a["v"][w].mean() / a["yaw_rate"][w].mean()))
    a = res["spin"]
    yaw = np.unwrap(a["yaw"])
    out["spin_total_yaw_deg"] = float(np.degrees(yaw[-1] - yaw[0]))
    out["spin_drift_m"] = float(np.hypot(a["x"][-1] - a["x"][0], a["y"][-1] - a["y"][0]))
    for k, a in res.items():
        if k.startswith("slope"):
            out[f"{k}_climbed_m_in_10s"] = float(a["x"][-1])
    return out


if __name__ == "__main__":
    import json
    import time
    t0 = time.time()
    res = {"drive": run_drive(), "turn_off": run_turn(False), "turn_on": run_turn(True), "spin": run_spin()}
    for dgr in (6, 10, 12, 14, 16):
        res[f"slope_{dgr}"] = run_slope(dgr)
    np.savez_compressed(os.path.join(OUT, "runs.npz"), **{f"{k}__{kk}": vv for k, v in res.items() for kk, vv in v.items()})
    S = summary(res)
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for k, v in S.items():
        print(f"{k:36s} {v:9.3f}")
    plots(res)
    print(f"done in {time.time() - t0:.0f}s")
