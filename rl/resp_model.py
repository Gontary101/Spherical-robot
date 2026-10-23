"""
Closed-loop response model of the low level, for the navigation layer (safety filter and MPC).

The navigation layer used to predict the robot's speed as a dead time followed by an acceleration-limited approach to
the command. That cannot represent what a pendulum-driven sphere does: it is non-minimum phase. To brake, the drive
swings the pendulum forward, and the reaction first drives the shell forward; to speed up, the shell first rolls
back. The harder the low level brakes, the larger that lurch, so a faster stop made navigation worse (docs/09, low-
level stopping study).

This model keeps the pendulum in the state. Per low level (policy checkpoint + base controller) it identifies, from
navigator-like command sequences under nominal and field conditions,

    [v, pend, pend_rate]_{k+1} = A [v, pend, pend_rate]_k + B [vc, ac]_k + c          (dt 0.02 s)
    w                          = first-order lag (w_tau) of wc delayed by w_dead steps

where vc / wc are the base controller's speed and yaw-rate references (slew-limited commands, reproduced exactly from
the commands the navigation layer sent) and ac the reference's acceleration. The pendulum is signed with the motion
(< 0 = accelerating), as everywhere in this code.

    python rl/resp_model.py --identify [--out rl/runs/nav2/bench/resp_<name>.json]   # GYRA_LOW / GYRA_BASE select
    python rl/resp_model.py --validate                                                  # vs the dead-time model
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
DT = 0.02
V_SLEW, W_SLEW = 2.5, 1.5                       # the base controllers' reference slew limits (sim/gyra2_sim.py)
FIELD = "field (rough, wind, payload; no pushes)"
# the deployed low level: loco31 with the base's slope feedforward (docs/12, P5); its tables are the worst case over two
# identifications (rl/lowlevel_pipeline.py --merge)
DEFAULT = os.path.join(ROOT, "rl/runs/nav2/bench/resp_loco31m.json")
DEFAULT_SYSID = os.path.join(ROOT, "rl/runs/nav2/bench/sysid_loco31m_pipe.json")


def configured():
    """The response model the navigation layer uses: GYRA_RESP (a path, or "off" for the dead-time model), else the
    deployed low level's. Held-out benchmark, 416 episodes over both seed sets: at-fault collisions 4 -> 0 at the
    same success (93.8 / 92.3 % vs 94.2 / 93.3 %)."""
    p = os.environ.get("GYRA_RESP", DEFAULT)
    return None if p.lower() in ("off", "0", "none", "") else p


def nav_commands(rng, n):
    """Command sequence (one per 0.1 s navigation period): mostly the navigator's pattern (candidate speeds and turn
    rates switched every 0.1-0.5 s, a third of them stops), plus longer holds at any speed."""
    out = []
    while len(out) < n:
        if rng.random() < 0.7:
            v = 0.0 if rng.random() < 0.35 else float(rng.choice([-0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]))
            w = float(rng.choice(np.arange(-1.5, 1.51, 0.25)))
            hold = int(rng.integers(1, 6))
        else:
            v, w = float(rng.uniform(-0.5, 3.0)), float(rng.uniform(-1.5, 1.5))
            hold = int(rng.integers(5, 25))
        if abs(v) > 1.0:
            w = float(np.clip(w, -3.0 / abs(v), 3.0 / abs(v)))
        out += [(v, w)] * hold
    return out[:n]


def collect(args):
    seed, n_periods, cond = args
    import sysid_low as SI
    from nav2_env import LowLevel
    low = LowLevel(SI.LOW)
    rng = np.random.default_rng(seed)
    e = SI.env_for(cond, seed)
    e.long_ctrl = None
    oa, _ = e.reset()
    rows, seg = [], 0
    for v, w in nav_commands(rng, n_periods):
        for _ in range(5):                                     # 5 low-level steps per navigation period
            t0 = e._truth()
            e.cmd = np.array([v, w])
            vc0 = getattr(e.ctrl, "vc", 0.0)
            (oa, _), *_ = e.step(low(oa)[0])
            t1 = e._truth()
            vc, wc = e.ctrl.vc, e.ctrl.wc
            rows.append([seg, t0["v"], t0["pend"], t0["pend_rate"], t0["w"], vc, (vc - vc0) / DT, wc,
                         t1["v"], t1["pend"], t1["pend_rate"], t1["w"], v, w,
                         e.d.qpos[0], e.d.qpos[1], t1["yaw"]])
        if e.d.qpos[0] ** 2 + e.d.qpos[1] ** 2 > 400 or abs(e._truth()["pend"]) > 2.0:
            oa, _ = e.reset()
            seg += 1
    return np.array(rows), seed


def windows(D, K, n, seed=0):
    """n random windows of K steps that do not cross a reset: start rows."""
    rng = np.random.default_rng(seed)
    st = [i for i in rng.integers(1, len(D) - K, 4 * n) if (D[i - 1:i + K, 0] == D[i, 0]).all()]
    return np.array(st[:n])


def simulate(P, D, st, K):
    """Longitudinal rollout of parameters P = (A, B, c) from rows st, driven by the logged references: (n, K, 3)."""
    A, B, c = P
    x = D[st, 1:4].T
    out = []
    for k in range(K):
        x = A @ x + B @ D[st + k, 5:7].T + c[:, None]
        out.append(x.T)
    return np.stack(out, 1)


def fit(D, K=75):
    """One-step least squares, refined on the multi-step (output) error over K steps: a model fitted one step at a
    time is accurate locally but drifts over the MPC's 1.5 s horizon. Yaw rate: the base's yaw reference through a
    dead time and a first-order lag, chosen on the simulated yaw rate (the measured one is too noisy for one-step
    regression)."""
    from scipy.optimize import least_squares
    X = np.c_[D[:, 1:4], D[:, 5:7], np.ones(len(D))]
    Y = D[:, 8:11]
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    p0 = np.r_[W[:3].T.ravel(), W[3:5].T.ravel(), W[5]]
    st = windows(D, K, 400)
    T = np.stack([D[st + k, 8:11] for k in range(K)], 1)
    wgt = np.array([1.0, 0.3, 0.03])                     # speed matters most; pendulum rate is noisy

    def unpack(p):
        return p[:9].reshape(3, 3), p[9:15].reshape(3, 2), p[15:18]

    def resid(p):
        return ((simulate(unpack(p), D, st, K) - T) * wgt).ravel()
    r = least_squares(resid, p0, max_nfev=60)
    A, B, c = unpack(r.x)
    # yaw: w = lag(dead(reference)), the reference being the command slew-limited at w_slew (the base controller
    # slews at 1.5 rad/s^2, the policy was trained to track one slewed at 4; which one the closed loop follows is
    # identified, not assumed)
    stw = windows(D, 50, 600, seed=1)
    Wt = np.stack([D[stw + k, 11] for k in range(50)], 1)
    best = None
    for slew in (1.5, 4.0, 1e9):
        ref_all = np.empty(len(D))
        r_ = 0.0
        for i in range(len(D)):
            if i == 0 or D[i, 0] != D[i - 1, 0]:
                r_ = D[i, 4]
            r_ += float(np.clip(D[i, 13] - r_, -slew * DT, slew * DT))
            ref_all[i] = r_
        for dead in range(0, 12):
            for tau in (0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3):
                aw = 1 - math.exp(-DT / tau)
                w = D[stw, 4].copy()
                E = 0.0
                for k in range(50):
                    ref = ref_all[stw + k - dead] if k >= dead else w
                    w = w + aw * (ref - w)
                    E += float(((w - Wt[:, k]) ** 2).mean())
                if best is None or E < best[0]:
                    best = (E, dead, tau, slew)
    return dict(A=A.tolist(), B=B.tolist(), c=c.tolist(), w_dead=best[1], w_tau=best[2], dt=DT,
                v_slew=V_SLEW, w_slew=best[3], samples=len(D), output_error_fit=dict(cost0=float((resid(p0) ** 2).sum()),
                                                                                 cost=float(r.cost * 2)))


class Response:
    """Vectorised rollout of the identified model for N candidates. State per candidate: v, pend, pend_rate, w and the
    base controller's references vc, wc (the navigation layer knows them: they follow its own commands)."""

    def __init__(self, path=None):
        path = path or configured() or DEFAULT
        M = json.load(open(path))
        self.A, self.B, self.c = np.array(M["A"]), np.array(M["B"]), np.array(M["c"])
        self.w_dead, self.aw = M["w_dead"], 1 - math.exp(-DT / M["w_tau"])
        self.v_slew, self.w_slew = M["v_slew"], M["w_slew"]
        self.adapter = None

    def step(self, S, cmd):
        """S: dict of arrays v, pend, rate, w, vc, wc (modified in place); cmd: (N, 2). One 0.02 s step."""
        vc0 = S["vc"]
        S["vc"] = vc0 + np.clip(cmd[:, 0] - vc0, -self.v_slew * DT, self.v_slew * DT)
        S["wc"] = S["wc"] + np.clip(cmd[:, 1] - S["wc"], -self.w_slew * DT, self.w_slew * DT)
        ac = (S["vc"] - vc0) / DT
        x = np.stack([S["v"], S["pend"], S["rate"]], 0)
        u = np.stack([S["vc"], ac], 0)
        xn = self.A @ x + self.B @ u + self.c[:, None]
        if self.adapter is not None:                     # online correction (slope, wind, payload, friction, wear)
            xn[:2] += self.adapter.theta @ np.stack([np.ones_like(S["vc"]), S["vc"], ac], 0)
        S["v"], S["pend"], S["rate"] = xn[0], xn[1], xn[2]
        # yaw: the reference w_dead steps ago (a short queue of past references per candidate), through the lag
        q = S.setdefault("wq", [])
        q.append(S["wc"])
        ref = q.pop(0) if len(q) > self.w_dead else S.get("w0", S["w"])
        S["w"] = S["w"] + self.aw * (ref - S["w"])
        return S


class Adapter:
    """Online correction of the identified model (docs/12, P6) on the error that matters: every step, re-run the model
    from the state H steps ago through the references since then and compare its speed with the measured one; the
    difference nudges a constant acceleration offset of the speed row (slope, wind, an off-centre payload). A one-step
    (RLS) correction made 1 s predictions worse under every condition: the model is fitted on multi-step error, so it is
    deliberately not the best one-step predictor, and one-step adaptation undoes that. The identified model stays the
    prior: the offset starts at zero and is clipped."""

    def __init__(self, model, H=25, gain=0.02, clip=0.012, dead=float(os.environ.get("GYRA_ADAPT_DEAD", 0.004))):
        self.m, self.H, self.gain, self.clip, self.dead = model, H, gain, clip, dead
        self.est = 0.0                                         # estimated speed change per step (m/s)
        self.theta = np.zeros((2, 3))                          # applied: the estimate beyond a dead band, so noise
        self.hist = []                                         # around zero (flat ground) changes nothing
        self.n = 0

    def update(self, x, vc, ac, x_next):
        m = self.m
        self.hist.append((np.asarray(x, float), vc, ac))
        if len(self.hist) > self.H:
            self.hist.pop(0)
        self.n += 1
        if len(self.hist) < self.H:
            return
        z = self.hist[0][0].copy()
        for _, vc_k, ac_k in self.hist:
            z = m.A @ z + m.B @ np.array([vc_k, ac_k]) + m.c
            z[0] += self.est
        e = x_next[0] - z[0]                                   # speed error after H steps
        self.est = float(np.clip(self.est + self.gain * e / self.H, -self.clip, self.clip))
        self.theta[0, 0] = math.copysign(max(abs(self.est) - self.dead, 0.0), self.est)


def validate_adaptive(D, R, K=50, every=25):
    """Causal check under a condition: walk through the log updating the adapter; every `every` steps predict K steps
    ahead with the fixed and with the adapted model (correction frozen at that moment). Along-track error p50 / p95."""
    ad = Adapter(R)
    errs = {"fixed": [], "adapted": []}
    for i in range(1, len(D) - K):
        if D[i, 0] != D[i - 1, 0]:
            ad = Adapter(R)                                   # new run: start from the prior again
            continue
        ad.update(D[i, 1:4], D[i, 5], D[i, 6], D[i, 8:11])
        if i % every or ad.n < 100 or not (D[i:i + K + 1, 0] == D[i, 0]).all():
            continue
        x0 = D[i + 1, 1:4]
        true = np.sum(D[i + 1:i + 1 + K, 8] * DT)
        for name, use in (("fixed", None), ("adapted", ad)):
            R.adapter = use
            S = dict(v=x0[0:1].copy(), pend=x0[1:2].copy(), rate=x0[2:3].copy(), w=np.zeros(1),
                     vc=np.array([D[i, 5]]), wc=np.zeros(1))
            dist = 0.0
            for k in range(K):
                R.step(S, D[i + 1 + k, 12:14][None])
                dist += S["v"][0] * DT
            errs[name].append(abs(dist - true))
    R.adapter = None
    return {k: (float(np.median(v)), float(np.percentile(v, 95)), len(v)) for k, v in errs.items() if v}


def deadtime_paths(D, i0, K):
    """The previous model (rl/sysid_low.py constants) from row i0 over K steps: speed and yaw rate."""
    from sysid_low import A_FWD, A_REV, DEAD, TAU_W
    v, w = D[i0, 1], D[i0, 4]
    aw = 1 - math.exp(-DT / TAU_W)
    V, Wr = [], []
    for k in range(K):
        cv, cw = D[i0 + k, 12], D[i0 + k, 13]
        if k * DT >= DEAD:
            acc = A_REV if (cv < v and v <= 0.0) else A_FWD
            v += float(np.clip(cv - v, -acc * DT, acc * DT))
            w += aw * (cw - w)
        V.append(v)
        Wr.append(w)
    return np.array(V), np.array(Wr)


def validate(D, R, K=75, n=1500, seed=0):
    """Open-loop prediction from random starts over K steps: along-track and lateral position error of the path (the
    quantities the filter's margins cover), the new model against the dead-time model."""
    rng = np.random.default_rng(seed)
    starts = [i for i in rng.integers(0, len(D) - K, 4 * n) if (D[i:i + K, 0] == D[i, 0]).all()][:n]
    errs = {"model": [], "deadtime": []}
    for i0 in starts:
        S = dict(v=np.array([D[i0, 1]]), pend=np.array([D[i0, 2]]), rate=np.array([D[i0, 3]]), w=np.array([D[i0, 4]]),
                 vc=np.array([D[i0 - 1, 5] if i0 > 0 and D[i0 - 1, 0] == D[i0, 0] else D[i0, 1]]),
                 wc=np.array([D[i0, 4]]))                 # yaw reference: starts at the current yaw rate
        Vm, Wm = [], []
        for k in range(K):
            R.step(S, D[i0 + k, 12:14][None])
            Vm.append(S["v"][0])
            Wm.append(S["w"][0])
        Vd, Wd = deadtime_paths(D, i0, K)
        x0, y0, th0 = D[i0 - 1, 14:17] if i0 > 0 else D[i0, 14:17]
        true = D[i0:i0 + K, 14:16] - np.array([x0, y0])
        for name, V, Wr in (("model", np.array(Vm), np.array(Wm)), ("deadtime", Vd, Wd)):
            th = th0 + np.cumsum(Wr) * DT
            P = np.cumsum(np.c_[V * np.cos(th), V * np.sin(th)] * DT, 0)
            d = true - P
            al = np.abs(d[:, 0] * np.cos(th) + d[:, 1] * np.sin(th))
            lat = np.abs(-d[:, 0] * np.sin(th) + d[:, 1] * np.cos(th))
            errs[name].append(np.c_[al, lat])
    out = {}
    for name, E in errs.items():
        E = np.array(E)                                         # (n, K, 2)
        out[name] = {f"{h:.1f}s": dict(along_p50=float(np.median(E[:, int(h / DT) - 1, 0])),
                                        along_p95=float(np.percentile(E[:, int(h / DT) - 1, 0], 95)),
                                        lat_p95=float(np.percentile(E[:, int(h / DT) - 1, 1], 95)))
                     for h in (0.3, 0.5, 1.0, 1.5)}
    return out


def gather(seeds):
    with mp.get_context("fork").Pool(4) as pool:
        R = pool.map(collect, [(s, 1500, c) for s in seeds for c in ("nominal", FIELD)])
    D = []
    for i, (d, s) in enumerate(R):
        d[:, 0] += 10000 * i                                    # segment ids unique across runs
        D.append(d)
    return np.concatenate(D)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--identify", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default=DEFAULT)
    ap.add_argument("--adapt-test", action="store_true", help="online adaptation under conditions never identified")
    a = ap.parse_args()
    if a.identify:
        D = gather(range(4))
        M = fit(D)
        M["low"] = os.environ.get("GYRA_LOW", "rl/runs/loco31/best.pt")
        M["base"] = os.environ.get("GYRA_BASE", "classical")
        json.dump(M, open(a.out, "w"), indent=1)
        print(f"{M['samples']} transitions; output-error cost {M['output_error_fit']['cost0']:.1f} -> {M['output_error_fit']['cost']:.1f}; "
              f"yaw: slew {M['w_slew']:.3g}, dead {M['w_dead'] * DT:.2f} s, lag {M['w_tau']:.2f} s; eig A {np.round(np.abs(np.linalg.eigvals(np.array(M['A']))), 3).tolist()}")
    if a.adapt_test:
        import sysid_low as SI
        R = Response(a.out)
        for cond in ("nominal", FIELD, "downhill 8 deg", "uphill 8 deg", "ice mu 0.3", "payload 10 kg"):
            with mp.get_context("fork").Pool(4) as pool:
                Dc = pool.map(collect, [(200 + s, 600, cond) for s in range(4)])
            D = []
            for j, (d, _) in enumerate(Dc):
                d[:, 0] += 10000 * j
                D.append(d)
            r = validate_adaptive(np.concatenate(D), R)
            print(f"{cond:42s} 1.0 s along-track error p50 / p95: fixed {r['fixed'][0]:.3f} / {r['fixed'][1]:.3f} m, "
                  f"adapted {r['adapted'][0]:.3f} / {r['adapted'][1]:.3f} m  ({r['fixed'][2]} predictions)")
    if a.validate:
        D = gather(range(100, 102))
        V = validate(D, Response(a.out))
        for name, r in V.items():
            print(f"{name:9s} " + "  ".join(f"{h}: along p50 {x['along_p50']:.2f} p95 {x['along_p95']:.2f} lat p95 {x['lat_p95']:.2f}"
                                           for h, x in r.items()))
