"""
Identify the navigation layer's models from a log recorded on the robot (docs/12, P7).

The navigation layer needs, per low level: the closed-loop response model (rl/resp_model.py), the braking table and
the error envelope of the model's predictions (rl/safety_filter.py). In simulation they come from privileged runs;
on the build they must come from what the robot measures. This tool takes a log with only those channels and
produces the same files.

Identification run (about 10 minutes, open floor >= 8 x 8 m, ideally repeated on each surface of interest): the
low level is driven by `nav_commands` sequences (the navigator's candidate speeds and turn rates switched every
0.1-0.5 s, a third of them stops, plus longer holds at any speed up to 2.5 m/s), with 21 deliberate stops from
0.5 ... 2.5 m/s. Logged at 50 Hz:

    t          s       time
    cmd_v      m/s     command sent to the low level        cmd_w   rad/s
    v          m/s     LiDAR-inertial odometry speed        w       rad/s   gyro yaw rate
    pend       rad     pendulum angle (yoke encoder + IMU pitch), signed with the motion (< 0 = accelerating)
    pend_rate  rad/s   its rate (encoder + gyro)
    x, y, yaw  m, rad  LiDAR-inertial pose
    vc         m/s     the base controller's speed reference (optional: reconstructed from cmd_v if absent)
    seg        -       run index (a new value wherever the log was interrupted)

    python rl/hw_ident.py --simulate-log rl/runs/nav2/bench/hwlog_sim.npz      # a simulated log, sensor noise included
    python rl/hw_ident.py --log rl/runs/nav2/bench/hwlog_sim.npz --name sim    # -> resp_sim.json, sysid_sim_hw.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import resp_model as RM  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
B = os.path.join(ROOT, "rl/runs/nav2/bench")
DT = RM.DT
STOPS = [(v, k) for v in (0.5, 1.0, 1.5, 2.0, 2.5) for k in range(4)][:21]


# ---------------------------------------------------------------------------------------------- simulated log
def sim_run(args):
    """One run of the identification protocol in simulation, logging only what the robot can measure (with the
    estimator's noise levels: LIO speed 0.03 m/s, gyro 0.02 rad/s, encoder + IMU pendulum 0.003 rad / 0.03 rad/s)."""
    seed, n_periods, cond = args
    import sysid_low as SI
    from nav2_env import LowLevel
    low = LowLevel(SI.LOW)
    rng = np.random.default_rng(seed)
    e = SI.env_for(cond, seed)
    oa, _ = e.reset()
    rows, seg = [], seed * 1000

    def log(cv, cw):
        t = e._truth()
        rows.append([seg, e.d.time, cv, cw, t["v"] + rng.normal(0, 0.03), t["w"] + rng.normal(0, 0.02),
                     t["pend"] + rng.normal(0, 0.003), t["pend_rate"] + rng.normal(0, 0.03),
                     e.d.qpos[0], e.d.qpos[1], t["yaw"], getattr(e.ctrl, "vc", t["v"])])
    cmds = RM.nav_commands(rng, n_periods)
    stops = list(STOPS)
    rng.shuffle(stops)
    i = 0
    while i < len(cmds):
        if stops and rng.random() < 0.02:                 # a deliberate stop: hold a speed, then command zero
            v0, _ = stops.pop()
            seq = [(v0, 0.0)] * 25 + [(0.0, 0.0)] * 30
        else:
            seq, i = [cmds[i]], i + 1
        for cv, cw in seq:
            for _ in range(5):
                e.cmd = np.array([cv, cw])
                (oa, _), *_ = e.step(low(oa)[0])
                log(cv, cw)
        if e.d.qpos[0] ** 2 + e.d.qpos[1] ** 2 > 400 or abs(e._truth()["pend"]) > 2.0:
            oa, _ = e.reset()
            seg += 1
    return np.array(rows)


def simulate_log(path, seeds=range(4), cond=RM.FIELD):
    with mp.get_context("fork").Pool(4) as pool:
        R = pool.map(sim_run, [(500 + s, 1500, cond) for s in seeds])
    L = np.concatenate(R)
    keys = ["seg", "t", "cmd_v", "cmd_w", "v", "w", "pend", "pend_rate", "x", "y", "yaw", "vc"]
    np.savez_compressed(path, **{k: L[:, i] for i, k in enumerate(keys)})
    print(f"simulated log: {len(L)} samples ({len(L) * DT / 60:.1f} min), {len(np.unique(L[:, 0]))} runs -> {path}")


# ---------------------------------------------------------------------------------------------- identification
def to_rows(Z, has_vc=True):
    """Log -> the transition rows rl/resp_model.py fits: [seg, v, pend, rate, w, vc, ac, wc, v', pend', rate', w',
    cmd_v, cmd_w, x', y', yaw']. Without a logged reference, vc is rebuilt from the commands with the base controller's
    slew limit (it is deterministic)."""
    n = len(Z["t"])
    vc = np.array(Z["vc"], float) if has_vc and "vc" in Z else np.zeros(n)
    if not (has_vc and "vc" in Z):
        for i in range(n):
            prev = vc[i - 1] if i and Z["seg"][i] == Z["seg"][i - 1] else Z["v"][i]
            vc[i] = prev + np.clip(Z["cmd_v"][i] - prev, -RM.V_SLEW * DT, RM.V_SLEW * DT)
    rows = []
    for i in range(n - 1):
        if Z["seg"][i + 1] != Z["seg"][i]:
            continue
        ac = (vc[i + 1] - vc[i]) / DT
        rows.append([Z["seg"][i], Z["v"][i], Z["pend"][i], Z["pend_rate"][i], Z["w"][i], vc[i + 1], ac, Z["cmd_w"][i + 1],
                     Z["v"][i + 1], Z["pend"][i + 1], Z["pend_rate"][i + 1], Z["w"][i + 1], Z["cmd_v"][i + 1],
                     Z["cmd_w"][i + 1], Z["x"][i + 1], Z["y"][i + 1], Z["yaw"][i + 1]])
    return np.array(rows)


def stop_events(Z):
    """Commanded stops from >= 0.4 m/s: distance travelled until |v| < 0.05 (LIO pose), per starting speed."""
    out = []
    cv, seg = Z["cmd_v"], Z["seg"]
    v = np.convolve(Z["v"], np.ones(5) / 5, mode="same")         # 0.1 s average: LIO speed noise is ~0.03 m/s
    for i in range(1, len(cv)):
        if abs(cv[i]) < 1e-9 and abs(cv[i - 1]) >= 0.4 and abs(v[i]) >= 0.3 and seg[i] == seg[i - 1]:
            j = i
            while j < len(cv) - 1 and abs(cv[j]) < 1e-9 and seg[j] == seg[i] and (abs(v[j]) >= 0.05 or j - i < 5):
                j += 1
            if abs(cv[j]) < 1e-9 and abs(v[j]) < 0.05:
                out.append(dict(v0=float(abs(v[i])), dist=float(math.hypot(Z["x"][j] - Z["x"][i], Z["y"][j] - Z["y"][i])),
                                time=float((j - i) * DT)))
    return out


def envelope(D, R, K=50, n=1500):
    """The filter's error envelope for the model's predictions, from the log: p95 along-track, lateral and total
    error per horizon (0.1 ... 1.0 s) and speed bin."""
    rng = np.random.default_rng(1)
    st = RM.windows(D, K, n)
    th = (np.arange(10) + 1) * 0.1
    sb = [0.0, 0.3, 0.7, 1.5, 2.5, 4.0]
    E, Ea, El, spd = [], [], [], []
    for i0 in st:
        S = dict(v=D[i0, 1:2].copy(), pend=D[i0, 2:3].copy(), rate=D[i0, 3:4].copy(), w=D[i0, 4:5].copy(),
                 vc=np.array([D[i0 - 1, 5]]), wc=np.array([D[i0, 4]]))
        x0, y0, th0 = D[i0 - 1, 14:17]
        X, Y, TH, P = 0.0, 0.0, th0, []
        for k in range(K):
            R.step(S, D[i0 + k, 12:14][None])
            TH += S["w"][0] * DT
            X += S["v"][0] * math.cos(TH) * DT
            Y += S["v"][0] * math.sin(TH) * DT
            P.append((X, Y, TH))
        P = np.array(P)
        d = D[i0:i0 + K, 14:16] - np.array([x0, y0]) - P[:, :2]
        al = np.abs(d[:, 0] * np.cos(P[:, 2]) + d[:, 1] * np.sin(P[:, 2]))
        la = np.abs(-d[:, 0] * np.sin(P[:, 2]) + d[:, 1] * np.cos(P[:, 2]))
        idx = (th / DT).astype(int) - 1
        E.append(np.hypot(al, la)[idx]), Ea.append(al[idx]), El.append(la[idx])
        spd.append(max(abs(D[i0, 1]), np.abs(D[i0:i0 + K, 12]).max()))
    spd = np.array(spd)
    out = dict(t=th.tolist(), speed_edges=sb, cond="identification log")
    for name, A in (("p95", np.array(E)), ("p95_along", np.array(Ea)), ("p95_lat", np.array(El))):
        tab = []
        for lo, hi in zip(sb[:-1], sb[1:]):
            sel = (spd >= lo) & (spd < hi)
            tab.append(np.percentile(A[sel], 95, axis=0).tolist() if sel.sum() >= 10 else None)
        for i in range(len(tab)):
            if tab[i] is None:
                tab[i] = next(tab[j] for j in sorted(range(len(tab)), key=lambda j: abs(j - i)) if tab[j] is not None)
        out[name] = np.maximum.accumulate(np.maximum.accumulate(np.array(tab), axis=1), axis=0).tolist()
    return out


def identify(log, name, template=os.path.join(B, "sysid_loco31_pipe.json")):
    Z = dict(np.load(log))
    D = to_rows(Z)
    M = RM.fit(D)
    M["source"] = os.path.relpath(log, ROOT)
    resp = os.path.join(B, f"resp_{name}.json")
    json.dump(M, open(resp, "w"), indent=1)
    R = RM.Response(resp)
    V = RM.validate(D, R)
    S = json.load(open(template))                              # fields the log cannot give keep the template's values
    ev = stop_events(Z)
    v0s = [0.5, 1.0, 1.5, 2.0, 2.5]
    d = []
    for v in v0s:
        x = [e["dist"] for e in ev if abs(e["v0"] - v) < 0.25]
        d.append(max(x) + 0.05 if x else None)
    S["brake_table_hw"] = dict(v=[0.0] + v0s, d=[0.05] + d, stops=len(ev))
    S["err_table"] = envelope(D, R)
    out = os.path.join(B, f"sysid_{name}_hw.json")
    json.dump(S, open(out, "w"))
    print(f"{len(D)} transitions, {len(ev)} stops; response model -> {resp}, tables -> {out}")
    print("open-loop prediction (along p95 at 0.5 / 1.0 / 1.5 s):",
          " / ".join(f"{V['model'][h]['along_p95']:.2f}" for h in ('0.5s', '1.0s', '1.5s')), "m (model)",
          " / ".join(f"{V['deadtime'][h]['along_p95']:.2f}" for h in ('0.5s', '1.0s', '1.5s')), "m (dead-time model)")
    print("stopping distance from the log (worst + 5 cm), m:", {v: (round(x, 2) if x else None) for v, x in zip(v0s, d)})
    return resp, out


def compare(resp_hw, resp_ref, cond=RM.FIELD):
    """How close is the log-identified model to the privileged one? Both predict fresh simulated data."""
    D = RM.gather(range(300, 302))
    for name, p in (("from the log", resp_hw), ("privileged", resp_ref)):
        V = RM.validate(D, RM.Response(p))["model"]
        print(f"{name:13s} along p95 0.5 / 1.0 / 1.5 s: " + " / ".join(f"{V[h]['along_p95']:.2f}" for h in ("0.5s", "1.0s", "1.5s"))
              + "   lateral p95: " + " / ".join(f"{V[h]['lat_p95']:.2f}" for h in ("0.5s", "1.0s", "1.5s")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate-log")
    ap.add_argument("--log")
    ap.add_argument("--name", default="hw")
    ap.add_argument("--compare", help="privileged response model to compare the log-identified one with")
    a = ap.parse_args()
    if a.simulate_log:
        simulate_log(a.simulate_log)
    if a.log:
        resp, _ = identify(a.log, a.name)
        if a.compare:
            compare(resp, a.compare)
