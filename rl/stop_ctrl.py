"""
Stop-and-hold controller for the pendulum-driven sphere (low level, model-based).

The learned locomotion policy tracks speed commands well, but "track zero smoothly" is not "stop": from field states
its stops took 0.79 m per m/s (median), 1.45 (p95), up to 3.4, and it crept 0.3 m afterwards (rl/stop_ctrl.py
--bench). While a stop is commanded this controller owns the common drive torque (steering and lean stay with the
policy): an LQR on a longitudinal model identified from the simulator (rolling speed, pendulum angle and rate,
travelled distance and its integral), so the robot stops short, then holds its spot on slopes, against wind and with
an off-centre payload (integral action) instead of creeping.

    python rl/stop_ctrl.py --identify     # longitudinal model from excitation data -> rl/runs/nav2/bench/stop_model.json
    python rl/stop_ctrl.py --bench        # stopping distance / time / creep: policy vs stop controller, field conditions
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = os.path.join(ROOT, "rl/runs/nav2/bench/stop_model.json")
DT = 0.02
FIELD = "field (rough, wind, payload; no pushes)"


# ---------------------------------------------------------------------------------------------- identification
def excite(args):
    """Drive with the policy to a random state, then hand the common torque to the classical cascade plus a random
    binary excitation for 1.5 s; record (v, pend, pend_rate) and the torque actually applied."""
    seed, n, cond = args
    import sysid_low as SI
    from nav2_env import LowLevel
    low = LowLevel(SI.LOW)
    r = np.random.default_rng(seed)
    e = SI.env_for(cond, seed)
    oa, _ = e.reset()
    rows = []
    for i in range(n):
        v_a = r.uniform(-0.5, 2.0)
        for k in range(int(r.uniform(0.5, 2.0) / DT)):
            e.long_ctrl = None
            e.cmd = np.array([v_a, r.uniform(-0.3, 0.3)])
            (oa, _), *_ = e.step(low(oa)[0])
        amp = r.uniform(10.0, 40.0)
        prbs = {"u": 0.0, "k": 0}

        def ctrl(env, est, t):
            if prbs["k"] % 5 == 0:
                prbs["u"] = amp * r.choice([-1.0, 1.0])
            prbs["k"] += 1
            tl, tr, _, _ = env.ctrl(env, est, env.cmd[0], env.cmd[1])
            return (tl + tr) + prbs["u"]
        e.long_ctrl = ctrl
        e.cmd = np.array([r.uniform(-0.3, 0.8), 0.0])
        for k in range(int(1.5 / DT)):
            t0 = e._truth()
            (oa, _), *_ = e.step(low(oa)[0])
            t1 = e._truth()
            rows.append([t0["v"], t0["pend"], t0["pend_rate"], e.diag["tauL"] + e.diag["tauR"], t1["v"], t1["pend"], t1["pend_rate"]])
            if abs(t1["pend"]) > 1.4:
                break
        e.long_ctrl = None
        if e.d.qpos[0] ** 2 + e.d.qpos[1] ** 2 > 400 or abs(e._truth()["pend"]) > 1.4:
            oa, _ = e.reset()
    return rows


def identify():
    with mp.get_context("fork").Pool(4) as pool:
        R = sum(pool.map(excite, [(900 + s, 40, c) for s in range(4) for c in ("nominal", FIELD)]), [])
    D = np.array(R)
    D = D[np.abs(D[:, 1]) < 0.6]                     # linear region of the pendulum (|angle| < 34 deg)
    X = np.c_[D[:, 0], D[:, 1], D[:, 2], D[:, 3], np.ones(len(D))]
    Y = D[:, 4:7]
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A, B, c = W[:3].T, W[3:4].T, W[4]
    res = Y - X @ W
    out = dict(A=A.tolist(), B=B.ravel().tolist(), c=c.tolist(), dt=DT, samples=len(D),
               resid_std=res.std(0).tolist(), r2=(1 - res.var(0) / Y.var(0)).tolist())
    json.dump(out, open(MODEL, "w"), indent=1)
    print(f"{len(D)} samples; x = [v, pend, pend_rate], u = common drive torque (N m), dt {DT}")
    print("A =", np.round(A, 4).tolist())
    print("B =", np.round(B.ravel(), 5).tolist(), " c =", np.round(c, 4).tolist())
    print("one-step residual std", np.round(res.std(0), 4).tolist(), " R^2", np.round(out["r2"], 4).tolist())


# ---------------------------------------------------------------------------------------------- controller
TAU_MAX = 2 * 82.1 * 0.95                            # both drive motors at their 60 A limit (N m at the tyres)
ENABLED = os.environ.get("GYRA_STOPCTRL", "0") == "1"


def dlqr(A, B, Q, R):
    from scipy.linalg import solve_discrete_are
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


class StopHold:
    """While the commanded speed is ~0: brake on [v, pendulum, pendulum rate] until the robot is (nearly) still, then
    hold that spot: [distance from it, v, pendulum, rate, integral of the distance]. One LQR on the identified model
    augmented with distance and its integral; during braking the distance terms are zero (stop where the brakes
    stop you, do not drive back)."""

    def __init__(self, q=(40.0, 30.0, 2.0, 0.05, 60.0), r=1e-4, path=MODEL):
        M = json.load(open(path))
        A3, B3 = np.array(M["A"]), np.array(M["B"])
        A = np.zeros((5, 5))
        A[0, 0], A[0, 1] = 1.0, DT                     # distance
        A[1:4, 1:4] = A3
        A[4, 4], A[4, 0] = 1.0, DT                     # integral of the distance
        B = np.r_[0.0, B3, 0.0][:, None]
        self.K = dlqr(A, B, np.diag(q), np.array([[r]]))[0]
        self.reset()

    def reset(self):
        self.active, self.hold, self.anchor, self.integ, self.fwd = False, False, None, 0.0, None

    def __call__(self, env, est, t):
        if abs(env.cmd[0]) >= 0.05:
            self.reset()
            return None
        if not self.active:
            self.active, self.fwd = True, t["fwd"][:2].copy()
        v = est["v"]
        pos = float(env.d.qpos[0:2] @ self.fwd)        # wheel odometry along the heading at the stop command
        if not self.hold and abs(t["v"]) < 0.05:
            self.hold, self.anchor, self.integ = True, pos, 0.0
        if self.hold:
            s = pos - self.anchor
            self.integ = float(np.clip(self.integ + s * DT, -0.5, 0.5))
            z = np.array([s, v, t["pend"], t["pend_rate"], self.integ])
        else:
            z = np.array([0.0, v, t["pend"], t["pend_rate"], 0.0])
        return float(np.clip(-self.K @ z, -TAU_MAX, TAU_MAX))


class StopCascade:
    """Stop and hold as a cascade that respects the pendulum by construction. Outer: the pendulum angle wanted -
    braking in proportion to the speed (no command ramp), then, once still, holding the spot with PI action on the
    distance - capped at TH_MAX (beyond ~60 deg gravity no longer restores the pendulum; a linear regulator that
    ignored this looped it over the top). Inner: the classical pendulum-angle loop (sim/gyra2_sim.py)."""

    def __init__(self, k_v=2.0, th_max=math.radians(60), k_s=2.0, k_i=1.0, kp=130.0, kd=13.0, kp_h=80.0, kd_h=10.0):
        self.k_v, self.th_max, self.k_s, self.k_i, self.kp, self.kd = k_v, th_max, k_s, k_i, kp, kd
        self.kp_h, self.kd_h = kp_h, kd_h
        self.reset()

    def reset(self):
        self.active, self.hold, self.anchor, self.integ, self.fwd = False, False, None, 0.0, None

    def __call__(self, env, est, t):
        if abs(env.cmd[0]) >= 0.05:
            self.reset()
            return None
        if not self.active:
            self.active, self.fwd = True, t["fwd"][:2].copy()
        v = est["v"]
        pos = float(env.d.qpos[0:2] @ self.fwd)        # wheel odometry along the heading at the stop command
        if not self.hold and abs(t["v"]) < 0.05:
            self.hold, self.anchor, self.integ = True, pos, 0.0
        th_ref = self.k_v * v
        if self.hold:
            s = pos - self.anchor
            self.integ = float(np.clip(self.integ + s * DT, -0.3, 0.3))
            th_ref += self.k_s * s + self.k_i * self.integ
        th_ref = float(np.clip(th_ref, -self.th_max, self.th_max))
        kp, kd = (self.kp_h, self.kd_h) if self.hold else (self.kp, self.kd)
        return float(np.clip(kp * (t["pend"] - th_ref) + kd * t["pend_rate"], -TAU_MAX, TAU_MAX))


# ---------------------------------------------------------------------------------------------- benchmark
def bench_run(args):
    mode, seed, n, q, r_ = args
    import sysid_low as SI
    from nav2_env import LowLevel
    low = LowLevel(SI.LOW)
    rng = np.random.default_rng(seed)
    e = SI.env_for(FIELD, seed)
    oa, _ = e.reset()
    ctrl = StopHold(q=q, r=r_) if mode == "stop" else (StopCascade(k_v=q[0], th_max=math.radians(q[1]), kp=q[2], kd=q[3], kp_h=q[4], kd_h=q[5],
                                                                     k_s=q[6], k_i=q[7]) if mode == "cascade" else None)
    out = []
    for i in range(n):
        e.long_ctrl = None
        v_a = rng.uniform(-0.5, 3.0)
        for k in range(int(rng.uniform(0.3, 2.5) / DT)):
            e.cmd = np.array([v_a, rng.uniform(-0.3, 0.3)])
            (oa, _), *_ = e.step(low(oa)[0])
        t = e._truth()
        v0, th0 = t["v"], t["pend"]
        p0 = e.d.qpos[0:2].copy()
        if ctrl is not None:
            ctrl.reset()
            e.long_ctrl = ctrl
        k_stop, d_stop, pk = None, None, 0.0
        for k in range(int(4.0 / DT)):
            e.cmd = np.zeros(2)
            (oa, _), *_ = e.step(low(oa)[0])
            tt = e._truth()
            pk = max(pk, abs(tt["pend"]))
            if k_stop is None and abs(tt["v"]) < 0.05 and k > 5:
                k_stop, d_stop = k, float(np.linalg.norm(e.d.qpos[0:2] - p0))
            if abs(tt["pend"]) > 2.4 or abs(tt["roll"]) > 1.4:
                break
        d_end = float(np.linalg.norm(e.d.qpos[0:2] - p0))
        out.append(dict(v=v0, pend=th0, dist=d_stop if d_stop is not None else d_end, t=(k_stop or 200) * DT,
                        creep=d_end - (d_stop if d_stop is not None else d_end), peak=pk, fell=abs(e._truth()["pend"]) > 2.4))
        if e.d.qpos[0] ** 2 + e.d.qpos[1] ** 2 > 400 or out[-1]["fell"]:
            oa, _ = e.reset()
    return mode, q, r_, out


def summarise(o):
    v = np.array([x["v"] for x in o]); d = np.array([x["dist"] for x in o])
    sel = np.abs(v) > 0.15
    q = d[sel] / np.abs(v[sel])
    return (f"dist per m/s p50 {np.median(q):.2f} p95 {np.percentile(q, 95):.2f} max {q.max():.2f}; "
            f"time to stop p95 {np.percentile([x['t'] for x in o], 95):.2f} s; creep p95 {np.percentile([x['creep'] for x in o], 95):.3f} m; "
            f"peak pendulum p95 {np.degrees(np.percentile([x['peak'] for x in o], 95)):.0f} deg; falls {sum(x['fell'] for x in o)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--identify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="a few weightings of the regulator")
    ap.add_argument("--sweep-cascade", action="store_true")
    a = ap.parse_args()
    if a.identify:
        identify()
    if a.sweep_cascade:
        # braking gains x hold gains: (k_v, th_max, brake kp/kd, hold kp/kd, hold k_s/k_i)
        cfgs = [("cascade", (kv, 60.0, 130.0, 13.0, kph, kdh, ks, ki), 0) for kv in (2.0, 3.0)
                for kph, kdh in ((80.0, 10.0), (60.0, 12.0)) for ks, ki in ((2.0, 1.0), (1.0, 0.3))]
        with mp.get_context("fork").Pool(4) as pool:
            R = pool.map(bench_run, [(m, s_, 60, q, r_) for m, q, r_ in cfgs for s_ in (31, 32)])
        for m, q, r_ in cfgs:
            o = sum([x[3] for x in R if x[0] == m and x[1] == q], [])
            print(f"k_v {q[0]:3.1f} brake {q[2]:.0f}/{q[3]:.0f} hold {q[4]:.0f}/{q[5]:.0f} k_s {q[6]:.1f} k_i {q[7]:.1f}: " + summarise(o))
    if a.bench or a.sweep:
        base = (40.0, 30.0, 2.0, 0.05, 60.0)
        cfgs = [("policy", base, 1e-4)] + ([("stop", base, 1e-4)] if a.bench else
                [("stop", (40.0, w_v, 2.0, 0.05, 60.0), r_) for w_v in (10.0, 30.0, 100.0) for r_ in (1e-4, 1e-3)])
        with mp.get_context("fork").Pool(4) as pool:
            R = pool.map(bench_run, [(m, s_, 60, q, r_) for m, q, r_ in cfgs for s_ in (31, 32)])
        for m, q, r_ in cfgs:
            o = sum([x[3] for x in R if x[0] == m and x[1] == q and x[2] == r_], [])
            print(f"{m:6s} q_v {q[1]:6.1f} r {r_:.0e}: " + summarise(o))
