"""
Stop-and-hold controller for the pendulum-driven sphere (low level, model-based).

The learned locomotion policy tracks speed commands well, but "track zero smoothly" is not "stop": from field states
its stops took 0.75 m per m/s (median), 1.51 (p95), up to 2.8, and it crept 0.26 m afterwards. While a stop is
commanded StopCascade owns the common drive torque (steering and lean stay with the policy): a pendulum-capped braking
cascade with a disturbance observer, both built on a longitudinal model identified from the simulator (rolling speed,
pendulum angle and rate). Field stops: 0.45 / 0.77 / 0.91 m per m/s, creep 0.04 m; 8 deg downhill 0.58 / 0.91 / 1.26,
creep 0.03 m (policy 0.68 / 1.32 / 2.03, 0.17 m); no falls. Enabled in the navigation stack with GYRA_STOPCTRL=1.
(StopHold, one LQR on the same model, looped the pendulum over the top from fast stops; kept for reference.)

    python rl/stop_ctrl.py --identify     # longitudinal model from excitation data -> rl/runs/nav2/bench/stop_model.json
    python rl/stop_ctrl.py --bench        # stopping distance / time / creep: policy vs LQR, field conditions
    python rl/stop_ctrl.py --sweep-cascade   # StopCascade variants on field ground and an 8 deg downhill
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
    act = lambda o: low(o)[0]
    r = np.random.default_rng(seed)
    e = SI.env_for(cond, seed)
    oa, _ = e.reset()
    rows = []
    for i in range(n):
        v_a = r.uniform(-0.5, 2.0)
        for k in range(int(r.uniform(0.5, 2.0) / DT)):
            e.long_ctrl = None
            e.cmd = np.array([v_a, r.uniform(-0.3, 0.3)])
            (oa, _), *_ = e.step(act(oa))
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
            (oa, _), *_ = e.step(act(oa))
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
    braking in proportion to the speed (no command ramp), then, once still, holding the spot (PD on the distance) -
    capped at TH_MAX (beyond ~60 deg gravity no longer restores the pendulum; a linear regulator that ignored this
    looped it over the top). Inner: the classical pendulum-angle loop (sim/gyra2_sim.py), stiffer while braking,
    softer and better damped while holding, plus the torque that holds the pendulum at the wanted angle, so the angle
    asked for is the angle reached.

    A slope, wind or an off-centre payload needs a standing pendulum angle to stand still, which a speed-proportional
    brake never asks for: downhill it rolled on at the speed where k_v * v balanced the slope. A disturbance observer
    on the identified longitudinal model estimates the acceleration the model does not explain and adds the standing
    pendulum angle that cancels it (static gain of the model: the torque that holds the pendulum also drives the
    shell), so the brake and the hold act on a level robot whatever the ground does. It is the hold's integral action
    too (a separate integrator on the distance only fought it)."""

    def __init__(self, k_v=1.5, th_max=math.radians(60), k_s=2.0, kp=130.0, kd=13.0, kp_h=60.0, kd_h=12.0,
                 l_obs=0.03, path=MODEL):
        self.k_v, self.th_max, self.k_s, self.kp, self.kd = k_v, th_max, k_s, kp, kd
        self.kp_h, self.kd_h, self.l_obs = kp_h, kd_h, l_obs
        M = json.load(open(path))
        A, B = np.array(M["A"]), np.array(M["B"])
        self.a_v, self.b_v, self.c_v = A[0], B[0], M["c"][0]   # rolling-speed row of the model
        self.k_g = -A[2, 1] / B[2]                     # drive torque that holds the pendulum still at an angle (per rad)
        self.g_s = A[0, 1] + B[0] * self.k_g           # speed change per step of a pendulum held at an angle (per rad)
        self.reset()

    def reset(self):
        self.active, self.hold, self.anchor, self.fwd = False, False, None, None
        self.x_prev, self.dist = None, 0.0

    def __call__(self, env, est, t):
        if abs(env.cmd[0]) >= 0.05:
            self.reset()
            return None
        if not self.active:
            self.active, self.fwd = True, t["fwd"][:2].copy()
        v = est["v"]
        x = np.array([v, t["pend"], t["pend_rate"]])
        if self.x_prev is not None and self.l_obs > 0:   # speed change the model does not explain (per step)
            u = env.diag["tauL"] + env.diag["tauR"]        # torque actually applied over the last step
            self.dist += self.l_obs * (v - (self.a_v @ self.x_prev + self.b_v * u + self.c_v) - self.dist)
        self.x_prev = x
        pos = float(env.d.qpos[0:2] @ self.fwd)        # wheel odometry along the heading at the stop command
        if not self.hold and abs(t["v"]) < 0.05:
            self.hold, self.anchor = True, pos
        th_ref = self.k_v * v - self.dist / self.g_s     # plus the standing angle that cancels the disturbance
        if self.hold:
            th_ref += self.k_s * (pos - self.anchor)
        th_ref = float(np.clip(th_ref, -self.th_max, self.th_max))
        kp, kd = (self.kp_h, self.kd_h) if self.hold else (self.kp, self.kd)
        u_eq = self.k_g * math.sin(th_ref)             # the pendulum's own weight at the wanted angle: no P-loop offset
        # steering keeps priority: each motor carries half the common torque plus or minus half the steering torque,
        # so the common torque gets only what the steering leaves (else the motors saturate and turns stall)
        head = max(TAU_MAX - abs(getattr(env, "tau_diff", 0.0)), 0.0)
        return float(np.clip(u_eq + kp * (t["pend"] - th_ref) + kd * t["pend_rate"], -head, head))


# ---------------------------------------------------------------------------------------------- benchmark
def bench_run(args):
    mode, seed, n, q, r_ = args[:5]
    cond = args[5] if len(args) > 5 else FIELD
    import sysid_low as SI
    from nav2_env import LowLevel
    low = LowLevel(SI.LOW)
    act = (lambda o: np.zeros(3)) if os.environ.get("GYRA_ZERO_RES") == "1" else (lambda o: low(o)[0])
    rng = np.random.default_rng(seed)
    e = SI.env_for(cond, seed)
    oa, _ = e.reset()
    ctrl = StopHold(q=q, r=r_) if mode == "stop" else (StopCascade(k_v=q[0], th_max=math.radians(q[1]), kp=q[2], kd=q[3], kp_h=q[4], kd_h=q[5],
                                                                     k_s=q[6], l_obs=q[8]) if mode == "cascade" else None)
    out = []
    for i in range(n):
        e.long_ctrl = None
        v_a = rng.uniform(-0.5, 3.0)
        for k in range(int(rng.uniform(0.3, 2.5) / DT)):
            e.cmd = np.array([v_a, rng.uniform(-0.3, 0.3)])
            (oa, _), *_ = e.step(act(oa))
        t = e._truth()
        v0, th0 = t["v"], t["pend"]
        p0 = e.d.qpos[0:2].copy()
        if ctrl is not None:
            ctrl.reset()
            e.long_ctrl = ctrl
        k_stop, d_stop, pk = None, None, 0.0
        for k in range(int(4.0 / DT)):
            e.cmd = np.zeros(2)
            (oa, _), *_ = e.step(act(oa))
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


def nav_run(args):
    """Command sequences like the navigator's: speeds from its candidate set held 0.1-0.5 s, a third of them zero
    (brief stops, stops while turning, stops cut short). For every zero segment: how far the robot goes on in the
    direction it was moving, and how much it speeds up after the stop command (pendulum reaction surges)."""
    mode, seed, n, q = args[:4]
    cond = args[4] if len(args) > 4 else FIELD
    import sysid_low as SI
    from nav2_env import LowLevel
    low = LowLevel(SI.LOW)
    act = (lambda o: np.zeros(3)) if os.environ.get("GYRA_ZERO_RES") == "1" else (lambda o: low(o)[0])
    rng = np.random.default_rng(seed)
    e = SI.env_for(cond, seed)
    e.long_ctrl = (StopCascade(**q) if mode == "cascade" else None)
    oa, _ = e.reset()
    out, absv = [], []
    for i in range(n):
        v_c = 0.0 if rng.random() < 0.35 else float(rng.choice([-0.25, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]))
        w_c = float(rng.choice([-0.75, -0.25, 0.0, 0.0, 0.25, 0.75]))
        t0 = e._truth()
        sv, p0 = (np.sign(t0["v"]) or 1.0), e.d.qpos[0:2].copy()
        fwd0, vmax = t0["fwd"][:2].copy(), 0.0
        for k in range(int(rng.integers(1, 6)) * 5):
            e.cmd = np.array([v_c, w_c])
            (oa, _), *_ = e.step(act(oa))
            t = e._truth()
            if v_c == 0.0:
                vmax = max(vmax, sv * t["v"])
                absv.append(abs(t["v"]))
        if v_c == 0.0 and abs(t0["v"]) > 0.1:
            dp = e.d.qpos[0:2] - p0
            out.append(dict(v=t0["v"], go_on=float(sv * dp @ fwd0), side=float(abs(dp[0] * fwd0[1] - dp[1] * fwd0[0])),
                            surge=vmax - abs(t0["v"]), w_err=float(abs(e._truth()["w"] - w_c)), path=float(np.linalg.norm(dp))))
        if abs(e._truth()["pend"]) > 2.4 or e.d.qpos[0] ** 2 + e.d.qpos[1] ** 2 > 400:
            out.append(dict(v=0.0, go_on=0.0, surge=0.0, fell=abs(e._truth()["pend"]) > 2.4))
            oa, _ = e.reset()
    return mode, q, out, absv


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
    ap.add_argument("--policy-only", action="store_true", help="--sweep-cascade: only the low-level policy (GYRA_LOW)")
    ap.add_argument("--nav-bench", default=None, help="JSON list of StopCascade kwargs to compare with the policy on navigator-like commands")
    a = ap.parse_args()
    if a.identify:
        identify()
    if a.sweep_cascade:
        # (k_v, th_max, brake kp/kd, hold kp/kd, hold k_s, unused, observer gain) on flat field ground and an 8 deg slope
        cfgs = [("policy", (0,) * 9, 0)] + ([] if a.policy_only else [("cascade", (kv, 60.0, 130.0, 13.0, 60.0, 12.0, 2.0, 0.0, lo), 0)
                                                                for kv in (1.5, 2.0) for lo in (0.0, 0.03, 0.1)])
        conds = (FIELD, "downhill 8 deg")
        with mp.get_context("fork").Pool(4) as pool:
            R = pool.map(bench_run, [(m, s_, 40, q, r_, c) for m, q, r_ in cfgs for s_ in (31, 32, 33) for c in conds])
        for m, q, r_ in cfgs:
            for c in conds:
                o = sum([x[3] for x, (mm, qq, _, cc) in zip(R, [(m_, q_, r__, c_) for m_, q_, r__ in cfgs for _ in (31, 32, 33) for c_ in conds])
                         if mm == m and qq == q and cc == c], [])
                print(f"{m:7s} k_v {q[0]:.1f} obs {q[8]:.2f} {c[:8]:8s}: " + summarise(o))
    if a.nav_bench:
        cfgs = [("policy", {})] + [("cascade", q) for q in json.loads(a.nav_bench)]
        with mp.get_context("fork").Pool(4) as pool:
            R = pool.map(nav_run, [(m, s_, 150, q) for m, q in cfgs for s_ in (41, 42, 43, 44)])
        for m, q in cfgs:
            o = sum([x[2] for x in R if x[0] == m and x[1] == q], [])
            av = np.array(sum([x[3] for x in R if x[0] == m and x[1] == q], []))
            o = [x for x in o if "side" in x]
            g, su = np.array([x["go_on"] for x in o]), np.array([x["surge"] for x in o])
            sd, we = np.array([x["side"] for x in o]), np.array([x["w_err"] for x in o])
            print(f"{m:7s} {json.dumps(q):60s} stops {len(o)}: go-on p50 {np.median(g):.2f} p95 {np.percentile(g, 95):.2f} m; "
                  f"side p95 {np.percentile(sd, 95):.2f} m; yaw-rate error p50 {np.median(we):.2f}; "
                  f"surge p95 {np.percentile(su, 95):.2f} max {su.max():.2f} m/s; |v| while stopped mean {av.mean():.2f}; "
                  f"falls {sum(x.get('fell', False) for x in o)}")
    if a.bench or a.sweep:
        base = (40.0, 30.0, 2.0, 0.05, 60.0)
        cfgs = [("policy", base, 1e-4)] + ([("stop", base, 1e-4)] if a.bench else
                [("stop", (40.0, w_v, 2.0, 0.05, 60.0), r_) for w_v in (10.0, 30.0, 100.0) for r_ in (1e-4, 1e-3)])
        with mp.get_context("fork").Pool(4) as pool:
            R = pool.map(bench_run, [(m, s_, 60, q, r_) for m, q, r_ in cfgs for s_ in (31, 32)])
        for m, q, r_ in cfgs:
            o = sum([x[3] for x in R if x[0] == m and x[1] == q and x[2] == r_], [])
            print(f"{m:6s} q_v {q[1]:6.1f} r {r_:.0e}: " + summarise(o))
