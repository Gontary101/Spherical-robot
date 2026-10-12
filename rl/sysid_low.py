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
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOW = os.path.join(ROOT, "rl/runs/loco31/best.pt")
CONDS = {"nominal": {}, "ice mu 0.3": dict(mu=0.3), "wet mu 0.6": dict(mu=0.6), "downhill 8 deg": dict(slope=np.radians(8)),
         "uphill 8 deg": dict(slope=np.radians(8)), "payload 10 kg": dict(payload=10.0, payload_pos=np.array([0.0, 0.0, 0.05]))}


def env_for(cond, seed):
    sc = dict(NOMINAL) | CONDS[cond]
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


def arcs(args):
    seed, n = args
    low = LowLevel(LOW)
    r = np.random.default_rng(seed)
    out = []
    e = env_for("nominal", seed)
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
        P, U = [], []
        for k in range(int(1.0 / 0.02)):
            e.cmd = np.array([v, w])
            (oa, _), *_ = e.step(low(oa)[0])
            # first-order-lag unicycle prediction (tau 0.3 s), the model the filter will use
            a = 1 - math.exp(-0.02 / 0.3)
            vc += a * (v - vc)
            wc += a * (w - wc)
            th += wc * 0.02
            x += vc * math.cos(th) * 0.02
            y += vc * math.sin(th) * 0.02
            P.append(e.d.qpos[0:2].copy())
            U.append((x, y))
        err = np.linalg.norm(np.array(P) - np.array(U), axis=1)
        out.append(dict(v=v, w=w, err_05=float(err[24]), err_10=float(err[-1])))
        if e.d.qpos[0] ** 2 + e.d.qpos[1] ** 2 > 400:
            oa, _ = e.reset()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "rl/runs/nav2/bench/sysid.json"))
    a = ap.parse_args()
    jobs = [(c, v0, s) for c in CONDS for v0 in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5) for s in range(3)]
    with mp.get_context("fork").Pool(4) as pool:
        B = pool.map(braking, jobs)
        A = sum(pool.map(arcs, [(s, 60) for s in range(8)]), [])
    res = dict(braking=B, arcs=A)
    # conservative braking model d(v) = c0 + c1 v + c2 v^2 fitted to the per-speed maximum over all conditions
    v = np.array([b["v_start"] for b in B])
    d = np.array([b["dist"] for b in B])
    X = np.c_[np.ones_like(v), np.abs(v), v ** 2]
    coef = np.linalg.lstsq(X, d, rcond=None)[0]
    resid = d - X @ coef
    coef[0] += float(resid.max())                  # shift up: the model upper-bounds every measured stop
    res["brake_model"] = dict(c0=float(coef[0]), c1=float(coef[1]), c2=float(coef[2]))
    e05 = np.array([x["err_05"] for x in A])
    e10 = np.array([x["err_10"] for x in A])
    res["arc_error"] = dict(p50_05=float(np.median(e05)), p95_05=float(np.percentile(e05, 95)),
                            p50_10=float(np.median(e10)), p95_10=float(np.percentile(e10, 95)))
    json.dump(res, open(a.out, "w"), indent=1)
    print("braking distance (m) by condition, v0 = 0.5 ... 3.5 m/s:")
    for c in CONDS:
        row = [np.mean([b["dist"] for b in B if b["cond"] == c and b["v0"] == v0]) for v0 in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)]
        print(f"  {c:16s} " + " ".join(f"{x:5.2f}" for x in row))
    print("conservative model d(v) = %.3f + %.3f |v| + %.3f v^2" % tuple(res["brake_model"].values()))
    print("unicycle (tau 0.3 s) prediction error: 0.5 s p50 %.2f / p95 %.2f m, 1.0 s p50 %.2f / p95 %.2f m" % tuple(res["arc_error"].values()))


if __name__ == "__main__":
    main()
