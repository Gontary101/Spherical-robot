"""
Why does a well-steered robot stall under the safety filter? (plan phase 2 diagnosis)

Runs the privileged teacher (rl/teacher.py) with the filter. Whenever the geodesic distance has not dropped by
0.2 m over 5 s, it asks what the teacher would do WITHOUT the filter and why the filter refuses it: the binding point
of that command's predicted path, the blocking map cell nearest to it, and whether that cell is
  unknown      never observed (blind wedge, behind something)
  ghost        mapped occupied but no true obstacle within 0.10 m of its mean hit point
  real         a true obstacle (the geometry itself is tight for the required margin)
with the deficit (how many cm the command misses by) and the margin components at the binding point.

    python rl/diagnose_stuck.py --episodes 16 "rooms, 1.0 m doors" mixed
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_nav2 import CALM, LOCO, SCEN  # noqa: E402
from local_map import KNOWN_FREE, OCC_TH  # noqa: E402
from nav2_env import Nav2Env, _footprint  # noqa: E402
from teacher import GeoMPC, grid_bilinear  # noqa: E402

LOW = os.environ.get("GYRA_LOW") or os.path.join(os.path.dirname(__file__), "runs/loco31/best.pt")


def poly_dist(P, polys):
    """Exact distance from points (n, 2) to the nearest polygon (0 inside)."""
    d = np.full(len(P), 1e9)
    for poly in polys:
        A, Bq = poly, np.roll(poly, -1, 0)
        AB = Bq - A
        t = np.clip(((P[:, None] - A[None]) * AB[None]).sum(-1) / (AB ** 2).sum(-1)[None], 0, 1)
        dd = np.linalg.norm(P[:, None] - (A[None] + t[..., None] * AB[None]), axis=-1).min(1)
        inside = np.ones(len(P), bool)
        for j in range(len(poly)):
            a, b = poly[j], poly[(j + 1) % len(poly)]
            inside &= (b[0] - a[0]) * (P[:, 1] - a[1]) - (b[1] - a[1]) * (P[:, 0] - a[0]) >= -1e-9
        d = np.minimum(d, np.where(inside, 0.0, dd))
    return d


def odo_to_world(env, q):
    pos, yaw, _ = env._pose()
    dth = yaw - env.loc[2]
    c, s = math.cos(dth), math.sin(dth)
    rel = q - env.loc[:2]
    return pos + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])


def analyse(env):
    pol = GeoMPC()
    env.shield = False
    want = pol.plan(env)
    env.shield = True
    # the least the robot must be allowed to do to make progress the teacher's way: 0.25 m/s in its direction
    want = np.array([0.25 * (np.sign(want[0]) or 1.0), want[1] * min(1.0, 0.25 / max(abs(want[0]), 0.25))])
    f, m = env.filter, env.lmap
    v0, w0 = env._velocity_estimate()
    ok, sl = f.admissible(m, env.loc.copy(), v0, w0, np.array([want]))
    if ok[0]:
        return dict(kind="slow_progress_admissible", deficit=0.0)
    P, ts = f._rollout(env.loc.copy(), v0, w0, np.array([want]))
    M = f._last[0]
    i = int(np.argmin(M))
    p = P[0, min(i, P.shape[1] - 1)]
    blocked = m.L > KNOWN_FREE
    rr, cc = np.nonzero(blocked)
    n = np.maximum(m.HN[rr, cc], 1e-6)
    has = m.HN[rr, cc] > 0.5
    cx = np.where(has, m.HX[rr, cc] / n, (cc + 0.5) * m.cell) + m.origin[0]
    cy = np.where(has, m.HY[rr, cc] / n, (rr + 0.5) * m.cell) + m.origin[1]
    d = np.hypot(cx - p[0], cy - p[1])
    j = int(np.argmin(d))
    q = np.array([cx[j], cy[j]])
    if m.L[rr[j], cc[j]] <= OCC_TH:
        kind = "unknown"
    else:
        # against the exact obstacle polygons: env.clear is a 0.1 m grid whose value at a wall surface is ~5 cm, which
        # made every mapped wall cell look like a "ghost"
        tq = poly_dist(odo_to_world(env, q)[None], [_footprint(o) for o in env.static_obs])[0]
        kind = "ghost" if tq > 0.05 else "real"
    pw = odo_to_world(env, p)
    return dict(kind=kind, deficit=float(-M[i]), t_bind=float(ts[0, min(i, ts.shape[1] - 1)]), want=[float(want[0]), float(want[1])],
                map_clear=float(d[j]), true_clear=float(grid_bilinear(env, env.clear, pw[None])[0]))


def run(args):
    scen, ep = args[:2]
    seed0, deployable = (args[2], True) if len(args) > 2 else (0, False)
    env = Nav2Env(seed=70000 + seed0 + ep, low_ckpt=LOW, level=0.8, adaptive=False,
                  scenario={"nav": CALM | SCEN[scen], "loco": LOCO}, shield=True)
    oa, _ = env.reset()
    pol = GeoMPC(deployable=deployable)
    geo, out = [], []
    while True:
        a = pol(oa, env)
        req = env.loco.cmd.copy()
        (oa, _), _, done, info = env.step(a)
        geo.append(env.geo_prev)
        if done:
            break
        if len(geo) > 50 and env.steps % 25 == 0 and geo[-50] - geo[-1] < 0.2:
            # who holds the robot: the filter (it corrected the navigator's request) or the navigator (it asked for
            # little: a recovery, or its own plan is to stand / turn)
            who = ("filter" if info.get("intervened") else
                   "navigator_recovery" if getattr(pol, "mode", "drive") != "drive" else "navigator")
            rec = analyse(env) | dict(scenario=scen, ep=ep, who=who, mode=str(getattr(pol, "mode", "")),
                                      cmd=[float(x) for x in env.loco.cmd])
            g = getattr(pol, "dbg", None)
            if g is not None and who == "navigator":
                # the chosen candidate against the best admissible one that makes progress (geodesic cost lower than
                # standing still's): which cost term made the navigator prefer what it did
                C, i = g["C"], g["i"]
                stand = int(np.argmin(np.abs(C[:, 0]) + np.abs(C[:, 1])))
                prog = (g["geo"] < g["geo"][stand] - 0.05) & (g["ok"] if g["ok"] is not None else True)
                if prog.any():
                    j = np.flatnonzero(prog)[int(np.argmin(g["cost"][prog]))]
                    rec["choice"] = {k: [float(g[k][i]), float(g[k][j])] for k in ("cost", "geo", "clr", "hit", "head", "track")}
                    rec["choice"]["cmd"] = [C[i].round(2).tolist(), C[j].round(2).tolist()]
                else:
                    rec["choice"] = "no admissible progress"
            out.append(rec)
    return out, dict(scenario=scen, ep=ep, success=bool(info["success"]), timeout=bool(info["timeout"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", default=["rooms, 1.0 m doors", "mixed"])
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--from", dest="src", default=None,
                    help="a pipeline benchmark file (rl/runs/nav2/bench/pipe_*_<set>.json): rerun its timeouts with the "
                         "deployable navigator (seed set from the file name)")
    a = ap.parse_args()
    if a.src:
        import json
        seed0 = 100 if "heldout" in a.src else 0
        jobs = [(e["scenario"], e["ep"], seed0) for e in json.load(open(a.src)) if e.get("failure") == "timeout"]
    else:
        jobs = [(s, e) for s in a.scenarios for e in range(a.episodes)]
    with mp.get_context("fork").Pool(4) as pool:
        R = pool.map(run, jobs, chunksize=1)
    S = sum((r[0] for r in R), [])
    E = [r[1] for r in R]
    print(f"episodes {len(E)}: success {np.mean([e['success'] for e in E]):.0%}, timeouts {np.mean([e['timeout'] for e in E]):.0%}; "
          f"stalled samples {len(S)}")
    print("who holds the robot:", dict(Counter(s.get("who", "?") for s in S)), " navigator modes:",
          dict(Counter(s.get("mode", "") for s in S)))
    print("blocking cause:", dict(Counter(s["kind"] for s in S)))
    X = [s["choice"] for s in S if isinstance(s.get("choice"), dict)]
    print("navigator samples:", len([s for s in S if "choice" in s]), " no admissible progress:",
          sum(s.get("choice") == "no admissible progress" for s in S))
    if X:
        for k in ("cost", "geo", "clr", "hit", "head", "track"):
            d = np.array([x[k][1] - x[k][0] for x in X])
            print(f"  progress candidate minus chosen, {k:5s}: median {np.median(d):+.2f}  mean {d.mean():+.2f}")
    for k in ("unknown", "ghost", "real"):
        X = [s for s in S if s["kind"] == k]
        if X:
            print(f"  {k:8s} deficit median {np.median([x['deficit'] for x in X]) * 100:4.1f} cm, binding at t median "
                  f"{np.median([x['t_bind'] for x in X]):.2f} s, wanted v median {np.median([x['want'][0] for x in X]):.2f}, "
                  f"map clear {np.median([x['map_clear'] for x in X]):.2f} vs true {np.median([x['true_clear'] for x in X]):.2f}")
