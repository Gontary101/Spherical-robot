"""
GYRA Mk2 - simulation wrapper, classical controller, and Mk1-vs-Mk2 benchmark.

    python sim/gyra2_sim.py          # runs the benchmark, writes sim/out/mk2_summary.json + media/mk2_*.png
"""
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gyra2_model import MotorModel, build_xml2  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "sim", "out")
os.makedirs(OUT, exist_ok=True)


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


class Gyra2:
    """Thin stateful wrapper: physics + sensor-level state extraction."""

    def __init__(self, **kw):
        self.m = mujoco.MjModel.from_xml_string(build_xml2(**kw))
        self.d = mujoco.MjData(self.m)
        self.j = {n: self.m.joint(n) for n in ("tyreL", "tyreR", "yoke", "bob")}
        self.qa = {n: j.qposadr[0] for n, j in self.j.items()}
        self.va = {n: j.dofadr[0] for n, j in self.j.items()}
        self.a = {n: self.m.actuator(n).id for n in ("driveL", "driveR", "level", "lean")}
        self.spine = self.m.body("spine").id
        self.motor_strength = 1.0
        mujoco.mj_forward(self.m, self.d)

    def q(self, n):
        return float(self.d.qpos[self.qa[n]])

    def qd(self, n):
        return float(self.d.qvel[self.va[n]])

    def state(self):
        d = self.d
        Rm = d.xmat[self.spine].reshape(3, 3)
        fwd = Rm[:, 0].copy()
        fwd[2] = 0
        fwd /= max(np.linalg.norm(fwd), 1e-9)
        w_world = Rm @ d.qvel[3:6]
        return dict(
            x=float(d.qpos[0]), y=float(d.qpos[1]), z=float(d.qpos[2]),
            yaw=float(np.arctan2(fwd[1], fwd[0])),
            roll=float(np.arcsin(np.clip(Rm[2, 1], -1, 1))),
            pitch=float(-np.arcsin(np.clip(Rm[2, 0], -1, 1))),
            v=float(d.qvel[0:3] @ fwd), v_lat=float(d.qvel[0:3] @ np.array([-fwd[1], fwd[0], 0])),
            roll_rate=float(w_world @ fwd), yaw_rate=float(w_world[2]), pitch_rate=float(w_world @ Rm[:, 1]),
            gyro=d.qvel[3:6].copy(), grav_body=Rm.T @ np.array([0, 0, -1.0]),
        )

    def apply(self, tauL, tauR, level_tau, bob_target, n_sub):
        d = self.d
        wL = self.qd("tyreL") - self.qd("yoke")
        wR = self.qd("tyreR") - self.qd("yoke")
        d.ctrl[self.a["driveL"]] = float(MotorModel.limit(tauL, wL, self.motor_strength))
        d.ctrl[self.a["driveR"]] = float(MotorModel.limit(tauR, wR, self.motor_strength))
        d.ctrl[self.a["level"]] = float(np.clip(level_tau, -15, 15))
        d.ctrl[self.a["lean"]] = float(np.clip(bob_target, -0.698, 0.698))
        mujoco.mj_step(self.m, self.d, nstep=n_sub)


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class ClassicalController:
    """Cascade speed loop + differential yaw-rate loop + counter-lean + spine levelling."""

    def __init__(self, dt, lean_rate_deg=60.0):
        self.dt = dt
        self.lean_rate = np.radians(lean_rate_deg)       # bob actuator slew limit (worm drive: 60 deg/s)
        self.iv = 0.0
        self.iw = 0.0
        self.bob = 0.0

    def __call__(self, g: Gyra2, s, v_cmd, w_cmd):
        # command shaping: limit yaw-rate command slew and lateral acceleration (tip-over bound)
        a_lat_max = 3.8
        w_lim = a_lat_max / max(abs(s["v"]), 0.5)
        w_cmd = _clamp(w_cmd, -w_lim, w_lim)
        self.wc = getattr(self, "wc", 0.0)
        self.wc += _clamp(w_cmd - self.wc, -1.5 * self.dt, 1.5 * self.dt)
        self.vc = getattr(self, "vc", 0.0)
        self.vc += _clamp(v_cmd - self.vc, -2.5 * self.dt, 2.5 * self.dt)
        e = self.vc - s["v"]
        lim = np.radians(80)
        u = 0.9 * e + 0.45 * self.iv
        if abs(u) < lim or np.sign(e) != np.sign(u):
            self.iv += e * self.dt
        pend = s["pitch"] + g.q("yoke")
        pend_rate = s["pitch_rate"] + g.qd("yoke")
        th_ref = -_clamp(u, -lim, lim)
        tau_sum = 80 * (pend - th_ref) + 10 * pend_rate
        ew = self.wc - s["yaw_rate"]
        sched = 1.0 / (1.0 + abs(s["v"]) / 1.5)    # yaw authority grows with speed -> lower gains
        self.iw = _clamp(self.iw + ew * self.dt, -2, 2)
        tau_diff = sched * (30 * ew + 25 * self.iw) - 4.0 * s["roll_rate"] * 0
        # counter-lean: feed-forward from the commanded lateral acceleration + roll feedback
        a_ff = s["v"] * self.wc
        bob_des = _clamp(1.1 * a_ff / 9.81 + 1.2 * s["roll"] + 0.25 * s["roll_rate"], -0.698, 0.698)
        step = self.lean_rate * self.dt
        self.bob += _clamp(bob_des - self.bob, -step, step)
        level = 90 * s["pitch"] + 9 * s["pitch_rate"]
        return 0.5 * tau_sum - 0.5 * tau_diff, 0.5 * tau_sum + 0.5 * tau_diff, level, self.bob


def run(cmd_fn, T, dt=0.02, **kw):
    g = Gyra2(**kw)
    c = ClassicalController(dt)
    n_sub = int(round(dt / g.m.opt.timestep))
    log = []
    while g.d.time < T:
        s = g.state()
        v_cmd, w_cmd = cmd_fn(g.d.time, s)
        tl, tr, lv, bb = c(g, s, v_cmd, w_cmd)
        g.apply(tl, tr, lv, bb, n_sub)
        s.update(t=float(g.d.time), v_cmd=v_cmd, w_cmd=w_cmd, bob=g.q("bob"), pend=s["pitch"] + g.q("yoke"),
                 tauL=float(g.d.ctrl[g.a["driveL"]]), tauR=float(g.d.ctrl[g.a["driveR"]]))
        s.pop("gyro"), s.pop("grav_body")
        log.append(s)
    return {k: np.array([r[k] for r in log]) for k in log[0]}


def ramp(t, pts):
    ts, vs = zip(*pts)
    return float(np.interp(t, ts, vs))


def benchmark():
    R = {}
    # 1. straight line 0 -> 3 -> 0 (sensor pitch)
    R["drive"] = run(lambda t, s: (ramp(t, [(0, 0), (1, 0), (3, 3), (8, 3), (10.5, 0), (14, 0)]), 0.0), 14)
    # 2. turn in place at 3 rad/s (and 6 rad/s)
    R["spin3"] = run(lambda t, s: (0.0, 3.0 if t > 1 else 0.0), 6)
    R["spin6"] = run(lambda t, s: (0.0, 6.0 if t > 1 else 0.0), 6)
    # 3. turns at speed: 6 m/s with increasing yaw-rate commands
    for w in (0.4, 0.6, 0.8, 1.0):
        R[f"turn6_{w}"] = run(lambda t, s, w=w: (ramp(t, [(0, 0), (4, 6), (20, 6)]), w if t > 6 else 0.0), 14)
    # 4. lateral push at 3 m/s (wobble / disturbance rejection)
    def push(t, s):
        return (ramp(t, [(0, 0), (2, 3), (20, 3)]), 0.0)
    R["push"] = run_push(push, 10)
    # 5. grades at 1 m/s
    for sl in (10, 14, 16, 18, 20):
        R[f"slope_{sl}"] = run(lambda t, s: (ramp(t, [(0, 0), (3, 1.0), (20, 1.0)]), 0.0), 12, slope_deg=sl)
    # 6. top speed
    R["vmax"] = run(lambda t, s: (8.5, 0.0), 14)
    return R


def run_push(cmd_fn, T, dt=0.02):
    g = Gyra2()
    c = ClassicalController(dt)
    n_sub = int(round(dt / g.m.opt.timestep))
    log = []
    while g.d.time < T:
        s = g.state()
        g.d.xfrc_applied[g.spine, :] = 0
        if 5.0 <= g.d.time < 5.1:                         # 300 N lateral shove for 0.1 s at the axle
            g.d.xfrc_applied[g.spine, 1] = 300.0
        tl, tr, lv, bb = c(g, s, *cmd_fn(g.d.time, s))
        g.apply(tl, tr, lv, bb, n_sub)
        s.update(t=float(g.d.time), bob=g.q("bob"))
        s.pop("gyro"), s.pop("grav_body")
        log.append(s)
    return {k: np.array([r[k] for r in log]) for k in log[0]}


def summarize(R):
    S = {}
    a = R["drive"]
    S["drive_peak_pendulum_deg"] = float(np.degrees(np.abs(a["pend"]).max()))
    S["drive_peak_spine_pitch_deg"] = float(np.degrees(np.abs(a["pitch"]).max()))
    for k in ("spin3", "spin6"):
        a = R[k]
        w = a["t"] > 2.5
        S[f"{k}_yaw_rate_dps"] = float(np.degrees(a["yaw_rate"][w].mean()))
        S[f"{k}_drift_m"] = float(np.hypot(a["x"][-1], a["y"][-1]))
    best = None
    for w in (0.4, 0.6, 0.8, 1.0):
        a = R[f"turn6_{w}"]
        win = (a["t"] > 9) & (a["t"] < 14)
        ok = np.abs(a["roll"]).max() < np.radians(35) and abs(a["v"][win].mean()) > 5.0
        rad = float(abs(a["v"][win].mean() / max(abs(a["yaw_rate"][win].mean()), 1e-6)))
        S[f"turn6_{w}_radius_m"] = rad
        S[f"turn6_{w}_ok"] = bool(ok)
        if ok:
            best = rad if best is None else min(best, rad)
    S["min_turn_radius_at_6ms_m"] = best
    a = R["push"]
    w = a["t"] > 5
    S["push_peak_roll_deg"] = float(np.degrees(np.abs(a["roll"][w]).max()))
    S["push_roll_settle_s"] = float(a["t"][w][np.where(np.abs(a["roll"][w]) > np.radians(2))[0][-1]] - 5.0) \
        if (np.abs(a["roll"][w]) > np.radians(2)).any() else 0.0
    for sl in (10, 14, 16, 18, 20):
        S[f"slope_{sl}_climbed_m"] = float(R[f"slope_{sl}"]["x"][-1])
    S["v_max_ms"] = float(R["vmax"]["v"][R["vmax"]["t"] > 9].mean())
    return S


def plots(R, mk1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"

    def style(ax, t, xl, yl):
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=.8)
        [s_.set_visible(False) for s_ in ax.spines.values()]
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.set_title(t, color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel(xl, color=INK2, fontsize=9)
        ax.set_ylabel(yl, color=INK2, fontsize=9)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    fig.patch.set_facecolor(SURF)
    a = R["spin3"]
    ax[0].plot(a["t"], np.degrees(np.unwrap(a["yaw"])), color=C1, lw=2, label="Mk2 differential, 3 rad/s cmd")
    a = R["spin6"]
    ax[0].plot(a["t"], np.degrees(np.unwrap(a["yaw"])), color=C3, lw=2, label="Mk2 differential, 6 rad/s cmd")
    if mk1 is not None:
        ax[0].plot(mk1["spin__t"], np.degrees(np.unwrap(mk1["spin__yaw"])), color=C2, lw=2, label="Mk1 CMG yaw mode")
    ax[0].set_xlim(0, 12)
    style(ax[0], "Turn in place", "time (s)", "heading (deg)")
    ax[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    for w, c in zip((0.4, 0.6, 0.8, 1.0), ("#9ec5f4", "#5b9be5", "#2a78d6", "#1c4f90")):
        a = R[f"turn6_{w}"]
        ax[1].plot(a["x"], a["y"], color=c, lw=1.8, label=f"{w} rad/s @ 6 m/s")
    ax[1].set_aspect("equal", adjustable="datalim")
    style(ax[1], "Mk2 turns at 6 m/s (21.6 km/h)", "x (m)", "y (m)")
    ax[1].legend(frameon=False, fontsize=8, labelcolor=INK2)
    for sl, c in zip((10, 14, 16, 18, 20), ("#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#e34948")):
        a = R[f"slope_{sl}"]
        ax[2].plot(a["t"], a["x"], color=c, lw=2, label=f"{sl} deg")
    style(ax[2], "Mk2 hill climb at 1 m/s", "time (s)", "distance up-slope (m)")
    ax[2].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "mk2_bench.png"), facecolor=SURF)


if __name__ == "__main__":
    R = benchmark()
    S = summarize(R)
    json.dump(S, open(os.path.join(OUT, "mk2_summary.json"), "w"), indent=1)
    for k, v in S.items():
        print(f"{k:34s} {v}")
    mk1_path = os.path.join(OUT, "runs.npz")
    plots(R, np.load(mk1_path) if os.path.exists(mk1_path) else None)
