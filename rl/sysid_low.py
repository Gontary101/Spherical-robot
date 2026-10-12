"""
System identification of the locomotion policy as seen from navigation (plan phase 1a).

The safety filter needs to know, for the real closed loop (locomotion v3.1 on Mk2.1), how far the robot travels
after a stop command and how far its path departs from the unicycle prediction of a (v, w) command.

  braking   accelerate to v0, command 0: stopping distance / time, across friction, slope (along the path), payload
  arcs      random (v, w) commands held 1 s from random initial motion: deviation of the real path from the unicycle
            arc (lag-compensated), as a function of horizon

    python rl/sysid_low.py --out rl/runs/nav2/bench/sysid.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import loco2_env as L  # noqa: E402
from eval_robust import NOMINAL  # noqa: E402
from nav2_env import LowLevel  # noqa: E402

L.GEOFENCE = 1e9
DEAD, A_FWD, A_REV, TAU_W = 0.22, 1.7, 1.1, 0.15      # identified from step responses (see docs/09)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOW = os.path.join(ROOT, "rl/runs/loco31/best.pt")
from eval_nav2 import LOCO  # noqa: E402
CONDS = {"nominal": {}, "field (rough, wind, payload; no pushes)": dict(LOCO, push_rate=0.0), "ice mu 0.3": dict(mu=0.3), "wet mu 0.6": dict(mu=0.6), "downhill 8 deg": dict(slope=np.radians(8)),
         "uphill 8 deg": dict(slope=np.radians(8)), "payload 10 kg": dict(payload=10.0, payload_pos=np.array([0.0, 0.0, 0.05]))}


def env_for(cond, seed):
    sc = (dict(NOMINAL) | CONDS[cond]) if not cond.startswith("field") else dict(CONDS[cond])
    e = L.Loco2Env(seed, scenario=sc, external_cmd=True)
    # slope: gravity tilt direction chosen so that driving along +x is downhill (or uphill)
    if "slope" in CONDS[cond]:
        sc["slope_dir"] = 0.0 if "uphill" in cond else math.pi
        e.scenario = sc
    e.spawn = (0.0, 0.0, 0.0)
    return e


def braking(args):
    cond, v0, seed = args
    low = LowLevel(LOW)
    e = env_for(cond, seed)
    oa, _ = e.reset()
    for k in range(int(4.0 / 0.02)):
        e.cmd = np.array([v0, 0.0])
        (oa, _), *_ = e.step(low(oa)[0])
    p0, vstart = e.d.qpos[0:2].copy(), e._truth()["v"]
    for k in range(int(6.0 / 0.02)):
        e.cmd = np.zeros(2)
        (oa, _), *_ = e.step(low(oa)[0])
        if abs(e._truth()["v"]) < 0.05 and k > 10:
            break
    # along-path distance actually travelled (sign: forward positive)
    d = float(np.linalg.norm(e.d.qpos[0:2] - p0))
    return dict(cond=cond, v0=v0, v_start=float(vstart), dist=d, time=(k + 1) * 0.02)


def dyn_braking(args):
    """Stops triggered at random moments of random speed profiles: stopping distance vs (speed, pendulum angle)."""
    seed, n = args[:2]
    cond = args[2] if len(args) > 2 else "nominal"
    low = LowLevel(LOW)
    r = np.random.default_rng(seed)
    out = []
    e = env_for(cond, seed)
    oa, _ = e.reset()
    for i in range(n):
        v_a = r.uniform(-0.5, 3.5)
        for k in range(int(r.uniform(0.3, 2.5) / 0.02)):       # random phase of a speed change
            e.cmd = np.array([v_a, r.uniform(-0.3, 0.3)])
            (oa, _), *_ = e.step(low(oa)[0])
        t = e._truth()
        v0, th0 = t["v"], t["pend"]
        p0 = e.d.qpos[0:2].copy()
        for k in range(int(6.0 / 0.02)):
            e.cmd = np.zeros(2)
            (oa, _), *_ = e.step(low(oa)[0])
            if abs(e._truth()["v"]) < 0.05 and k > 10:
                break
        out.append(dict(v=float(v0), pend=float(th0), dist=float(np.linalg.norm(e.d.qpos[0:2] - p0))))
        if e.d.qpos[0] ** 2 + e.d.qpos[1] ** 2 > 400:
            oa, _ = e.reset()
    return out


def arcs(args):
    seed, n, cond = args
    low = LowLevel(LOW)
    r = np.random.default_rng(seed)
    out = []
    e = env_for(cond, seed)
    oa, _ = e.reset()
    for i in range(n):
        v1, w1 = r.uniform(-0.5, 3.0), r.uniform(-1.5, 1.5)
        for k in range(int(r.uniform(1, 2) / 0.02)):        # random initial motion
            e.cmd = np.array([v1, w1])
            (oa, _), *_ = e.step(low(oa)[0])
        v, w = r.uniform(-0.8, 3.0), r.uniform(-2.0, 2.0)
        t = e._truth()
        x, y, th = e.d.qpos[0], e.d.qpos[1], t["yaw"]
        vc, wc = t["v"], t["w"]
        P, U, TH = [], [], []
        if r.random() < 0.4:                               # low-speed manoeuvring near obstacles matters most
            v, w = r.uniform(-0.5, 0.8), r.uniform(-1.0, 1.0)
        for k in range(int(1.0 / 0.02)):
            e.cmd = np.array([v, w])
            (oa, _), *_ = e.step(low(oa)[0])
            # identified response (the filter's model): dead time, then acceleration-limited speed, lagged yaw rate
            if k * 0.02 >= DEAD:
                acc = A_REV if (v < vc and vc <= 0.0) else A_FWD          # speeding up in reverse is slower
                vc += float(np.clip(v - vc, -acc * 0.02, acc * 0.02))
                wc += (1 - math.exp(-0.02 / TAU_W)) * (w - wc)
            th += wc * 0.02
            x += vc * math.cos(th) * 0.02
            y += vc * math.sin(th) * 0.02
            P.append(e.d.qpos[0:2].copy())
            U.append((x, y))
            TH.append(th)
        D = np.array(P) - np.array(U)
        err = np.linalg.norm(D, axis=1)
        # split into the predicted path's along-track and cross-track components (heading of the prediction)
        ths = np.array(TH)
        e_al = np.abs(D[:, 0] * np.cos(ths) + D[:, 1] * np.sin(ths))
        e_lat = np.abs(-D[:, 0] * np.sin(ths) + D[:, 1] * np.cos(ths))
        out.append(dict(v=v, w=w, speed=float(max(abs(t["v"]), abs(v))), err=err[4::5].tolist(),
                        err_al=e_al[4::5].tolist(), err_lat=e_lat[4::5].tolist(),
                        err_05=float(err[24]), err_10=float(err[-1])))
        if e.d.qpos[0] ** 2 + e.d.qpos[1] ** 2 > 400:
            oa, _ = e.reset()
    return out


def error_tables(res, AF):
    """Path-prediction error envelope under field conditions: p95 per horizon (0.1 ... 1.0 s) and speed bin, for the
    total, along-track and cross-track error (along-track error is harmless in a doorway; cross-track is not)."""
    th = (np.arange(10) + 1) * 0.1
    sb = [0.0, 0.3, 0.7, 1.5, 2.5, 4.0]
    spd = np.array([x["speed"] for x in AF])
    out = dict(t=th.tolist(), speed_edges=sb, cond="field (rough, wind, payload; no pushes)")
    for key, name in (("err", "p95"), ("err_al", "p95_along"), ("err_lat", "p95_lat")):
        if key not in AF[0]:
            continue
        E = np.array([x[key] for x in AF])
        tab = []
        for lo, hi in zip(sb[:-1], sb[1:]):
            sel = (spd >= lo) & (spd < hi)
            tab.append(np.percentile(E[sel], 95, axis=0).tolist() if sel.sum() >= 10 else None)
        for i in range(len(tab)):                                 # empty bins: nearest filled one; monotone in speed
            if tab[i] is None:
                tab[i] = next(tab[j] for j in sorted(range(len(tab)), key=lambda j: abs(j - i)) if tab[j] is not None)
        tab = np.maximum.accumulate(np.maximum.accumulate(np.array(tab), axis=1), axis=0)
        out[name] = tab.tolist()
        print(f"field path-prediction error {name} (m), rows speed bins {sb}, columns horizon 0.1 ... 1.0 s:")
        for lo, row in zip(sb, tab):
            print(f"   >= {lo:3.1f} m/s " + " ".join(f"{x:5.2f}" for x in row))
    res["err_table"] = out
    res["arcs_field"] = AF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "rl/runs/nav2/bench/sysid.json"))
    ap.add_argument("--arcs-only", action="store_true", help="re-measure only the field error tables")
    ap.add_argument("--braking-only", action="store_true", help="re-measure the dynamic braking table under field conditions")
    a = ap.parse_args()
    if a.braking_only:                               # re-identify the dynamic braking table under field conditions
        res = json.load(open(a.out))
        with mp.get_context("fork").Pool(4) as pool:
            DB = sum(pool.map(dyn_braking, [(300 + s, 150, "field (rough, wind, payload; no pushes)") for s in range(8)]), [])
        vb = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        old_T = np.array(res["brake_table2"]["d"])
        T = np.zeros_like(old_T)
        pb = np.array(res["brake_table2"]["pend_edges"])
        for x in DB:
            sv = np.sign(x["v"]) if abs(x["v"]) > 0.05 else 1.0
            i = int(np.clip(np.searchsorted(vb, abs(x["v"])), 0, len(vb) - 1))
            j = int(np.searchsorted(pb, sv * x["pend"]))
            T[i, j] = max(T[i, j], x["dist"] + 0.05)
        T = np.maximum.accumulate(T, axis=0)
        res["brake_table2"]["d"] = np.maximum(old_T, T).tolist()
        res["brake_table2"]["cond"] = "max(nominal, field)"
        res["dyn_braking_field"] = DB
        json.dump(res, open(a.out, "w"), indent=1)
        print("field dynamic braking, worst per speed (m):", dict(zip(vb.tolist(), np.round(T.max(1), 2).tolist())))
        print("table used by the filter now:", dict(zip(vb.tolist(), np.round(np.maximum(old_T, T).max(1), 2).tolist())))
        return
    if a.arcs_only:
        res = json.load(open(a.out))
        with mp.get_context("fork").Pool(2) as pool:
            AF = sum(pool.map(arcs, [(50 + s, 120, "field (rough, wind, payload; no pushes)") for s in range(8)]), [])
        error_tables(res, AF)
        json.dump(res, open(a.out, "w"), indent=1)
        return
    jobs = [(c, v0, s) for c in CONDS for v0 in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5) for s in range(3)]
    with mp.get_context("fork").Pool(4) as pool:
        B = pool.map(braking, jobs)
        A = sum(pool.map(arcs, [(s, 60, "nominal") for s in range(8)]), [])
        AF = sum(pool.map(arcs, [(50 + s, 120, "field (rough, wind, payload; no pushes)") for s in range(8)]), [])
        DB = sum(pool.map(dyn_braking, [(100 + s, 120) for s in range(8)]), [])
    res = dict(braking=B, arcs=A, dyn_braking=DB)
    # state-dependent worst case: the pendulum angle "against" the motion (accelerating) delays braking
    vb = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    pb = np.radians([-60, -30, -10, 10, 30, 60])                  # signed with the direction of motion: <0 = accelerating
    T = np.zeros((len(vb), len(pb) + 1))
    for x in DB:
        sv = np.sign(x["v"]) if abs(x["v"]) > 0.05 else 1.0
        i = int(np.clip(np.searchsorted(vb, abs(x["v"])), 0, len(vb) - 1))
        j = int(np.searchsorted(pb, sv * x["pend"]))
        T[i, j] = max(T[i, j], x["dist"])
    T = np.maximum.accumulate(T, axis=0)                          # monotone in speed
    flat = np.interp(vb, [0.0] + sorted({b["v0"] for b in B}), [0.05] + [max(b["dist"] for b in B if b["v0"] == v) for v in sorted({b["v0"] for b in B})])
    T = np.maximum(T, flat[:, None]) + 0.05                        # never below the steady-state worst case (slopes etc.)
    res["brake_table2"] = dict(v=vb.tolist(), pend_edges=pb.tolist(), d=T.tolist())
    # conservative braking model: per commanded speed, the WORST stopping distance over all conditions (+5 cm)
    v0s = sorted({b["v0"] for b in B})
    res["brake_table"] = dict(v=[0.0] + v0s, d=[0.05] + [max(b["dist"] for b in B if b["v0"] == v) + 0.05 for v in v0s])
    res["response"] = dict(dead=DEAD, a_fwd=A_FWD, a_rev=A_REV, tau_w=TAU_W)
    e05 = np.array([x["err_05"] for x in A])
    e10 = np.array([x["err_10"] for x in A])
    res["arc_error"] = dict(p50_05=float(np.median(e05)), p95_05=float(np.percentile(e05, 95)),
                            p50_10=float(np.median(e10)), p95_10=float(np.percentile(e10, 95)))
    error_tables(res, AF)
    json.dump(res, open(a.out, "w"), indent=1)
    print("braking distance (m) by condition, v0 = 0.5 ... 3.5 m/s:")
    for c in CONDS:
        row = [np.mean([b["dist"] for b in B if b["cond"] == c and b["v0"] == v0]) for v0 in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)]
        print(f"  {c:16s} " + " ".join(f"{x:5.2f}" for x in row))
    print("state-dependent braking table (rows |v|, columns pendulum angle with the motion, deg edges", np.degrees(pb).round().tolist(), "):")
    for i, v in enumerate(vb):
        print(f"   {v:3.1f} m/s " + " ".join(f"{x:5.2f}" for x in T[i]))
    print("worst-case braking table:", dict(zip(res["brake_table"]["v"], [round(x, 2) for x in res["brake_table"]["d"]])))
    print("identified-model prediction error: 0.5 s p50 %.2f / p95 %.2f m, 1.0 s p50 %.2f / p95 %.2f m" % tuple(res["arc_error"].values()))


if __name__ == "__main__":
    main()
