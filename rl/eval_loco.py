"""
Evaluate the learned locomotion policy against the classical controller on identical tests.

    python rl/eval_loco.py --ckpt rl/runs/loco/best.pt
Writes rl/runs/loco/eval.json and media/rl_loco_eval.png
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from loco_env import LocoEnv  # noqa: E402
from nav_env import NumpyActor  # noqa: E402


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def make_env(seed, randomize, push=False):
    e = LocoEnv(seed=seed, randomize=randomize, push=push)
    e._sample_cmd = lambda: setattr(e, "next_cmd_t", 1e9) or (None if hasattr(e, "cmd") else setattr(e, "cmd", np.zeros(2)))
    return e


def episode(kind, policy, sched, T, seed=0, randomize=False, push=False, push_at=None):
    env = make_env(seed, randomize, push)
    oa, _ = env.reset()
    env.d.qpos[3:7] = [1, 0, 0, 0]
    log = {k: [] for k in ("t", "v", "w", "roll", "x", "y", "pitch", "pend", "v_cmd", "w_cmd")}
    fell = False
    for k in range(int(T / 0.02)):
        t = k * 0.02
        env.cmd = np.array(sched(t), dtype=float)
        if push_at is not None and push_at <= t < push_at + 0.1:
            env.push_f, env.push_left = np.array([0.0, 300.0, 0.0]), 1
        if kind == "classical":
            a = np.zeros(3)          # residual env: zero residual == the classical controller alone
        else:
            oa = np.concatenate([env.hist.reshape(-1), env.cmd * np.array([0.25, 0.5])]).astype(np.float32)
            a = policy(oa)
        _, _, done, info = env.step(a)
        tr_ = env._truth()
        for key, val in (("t", t), ("v", tr_["v"]), ("w", tr_["w"]), ("roll", tr_["roll"]), ("x", env.d.qpos[0]),
                         ("y", env.d.qpos[1]), ("pitch", tr_["pitch"]), ("pend", tr_["pend"]),
                         ("v_cmd", env.cmd[0]), ("w_cmd", env.cmd[1])):
            log[key].append(float(val))
        if info["fell"]:
            fell = True
            break
    out = {k: np.array(v) for k, v in log.items()}
    out["fell"] = fell
    return out


def ramp(t, pts):
    ts, vs = zip(*pts)
    return float(np.interp(t, ts, vs))


TESTS = {
    "speed_steps": (lambda t: (ramp(t, [(0, 0), (1, 0), (1.01, 2), (5, 2), (5.01, 4), (9, 4), (9.01, 6), (13, 6), (13.01, 8), (18, 8)]), 0.0), 18),
    "spin_3": (lambda t: (0.0, 3.0 if t > 1 else 0.0), 6),
    "turn_6ms": (lambda t: (ramp(t, [(0, 0), (4, 6), (30, 6)]), 0.6 if t > 7 else 0.0), 16),
    "slalom_4ms": (lambda t: (ramp(t, [(0, 0), (3, 4), (30, 4)]), 0.8 * np.sign(np.sin(2 * np.pi * (t - 4) / 4)) if t > 4 else 0.0), 18),
}


def metrics(r, warm=1.0):
    m = r["t"] > warm
    return {"rms_v_err": float(np.sqrt(np.mean((r["v"][m] - r["v_cmd"][m]) ** 2))),
            "rms_w_err": float(np.sqrt(np.mean((r["w"][m] - r["w_cmd"][m]) ** 2))),
            "max_roll_deg": float(np.degrees(np.abs(r["roll"]).max())),
            "max_spine_pitch_deg": float(np.degrees(np.abs(r["pitch"]).max())), "fell": bool(r["fell"])}


def random_cmd_suite(kind, policy, n=12):
    errs_v, errs_w, falls = [], [], 0
    for s in range(n):
        rng = np.random.default_rng(100 + s)
        segs = []
        for _ in range(6):
            v = rng.uniform(-2, 7)
            wl = min(2.5, 3.5 / max(abs(v), 0.5))
            segs.append((v, rng.uniform(-wl, wl)))
        r = episode(kind, policy, lambda t, segs=segs: segs[min(int(t / 3.5), 5)], 21, seed=500 + s, randomize=True, push=True)
        m = r["t"] > 1
        errs_v.append(np.mean(np.abs(r["v"][m] - r["v_cmd"][m])))
        errs_w.append(np.mean(np.abs(r["w"][m] - r["w_cmd"][m])))
        falls += int(r["fell"])
    return {"mean_abs_v_err": float(np.mean(errs_v)), "mean_abs_w_err": float(np.mean(errs_w)), "fall_rate": falls / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "rl/runs/loco/best.pt"))
    a = ap.parse_args()
    pol = NumpyActor(a.ckpt)
    res, runs = {}, {}
    for kind in ("classical", "learned"):
        res[kind] = {}
        for name, (sched, T) in TESTS.items():
            r = episode(kind, pol, sched, T)
            runs[(kind, name)] = r
            res[kind][name] = metrics(r)
        pr = episode(kind, pol, lambda t: (3.0, 0.0), 10, push_at=5.0)
        res[kind]["push_300N_peak_roll_deg"] = float(np.degrees(np.abs(pr["roll"][pr["t"] > 5]).max()))
        res[kind]["random_commands_randomised_dynamics"] = random_cmd_suite(kind, pol)
        print(kind, json.dumps(res[kind], indent=1))
    out_dir = os.path.dirname(a.ckpt)
    json.dump(res, open(os.path.join(out_dir, "eval.json"), "w"), indent=1)
    plot(runs)


def plot(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    COL = {"classical": "#eb6834", "learned": "#2a78d6"}

    def style(ax, t, xl, yl):
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=.8)
        [s_.set_visible(False) for s_ in ax.spines.values()]
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.set_title(t, color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel(xl, color=INK2, fontsize=9)
        ax.set_ylabel(yl, color=INK2, fontsize=9)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8), dpi=140)
    fig.patch.set_facecolor(SURF)
    r = runs[("learned", "speed_steps")]
    ax[0, 0].plot(r["t"], r["v_cmd"], color=INK2, lw=1, ls="--", label="command")
    for k in ("classical", "learned"):
        r = runs[(k, "speed_steps")]
        ax[0, 0].plot(r["t"], r["v"], color=COL[k], lw=2, label=k)
    style(ax[0, 0], "Speed steps to 8 m/s", "time (s)", "speed (m/s)")
    for k in ("classical", "learned"):
        r = runs[(k, "turn_6ms")]
        ax[0, 1].plot(r["x"], r["y"], color=COL[k], lw=2, label=k + (" (fell)" if r["fell"] else ""))
    ax[0, 1].set_aspect("equal", adjustable="datalim")
    style(ax[0, 1], "Turn at 6 m/s, 0.6 rad/s command", "x (m)", "y (m)")
    r = runs[("learned", "slalom_4ms")]
    ax[1, 0].plot(r["t"], r["w_cmd"], color=INK2, lw=1, ls="--", label="command")
    for k in ("classical", "learned"):
        r = runs[(k, "slalom_4ms")]
        ax[1, 0].plot(r["t"], r["w"], color=COL[k], lw=1.6, label=k)
    style(ax[1, 0], "Slalom at 4 m/s: yaw-rate tracking", "time (s)", "yaw rate (rad/s)")
    for k in ("classical", "learned"):
        r = runs[(k, "slalom_4ms")]
        ax[1, 1].plot(r["t"], np.degrees(r["roll"]), color=COL[k], lw=1.6, label=k)
    style(ax[1, 1], "Slalom at 4 m/s: axle roll", "time (s)", "roll (deg)")
    for a_ in ax.flat:
        a_.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "rl_loco_eval.png"), facecolor=SURF)


if __name__ == "__main__":
    main()
