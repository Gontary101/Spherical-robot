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
from nav2_env import Nav2Env  # noqa: E402
from teacher import GeoMPC, grid_bilinear  # noqa: E402

LOW = os.path.join(os.path.dirname(__file__), "runs/loco31/best.pt")


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
        tq = grid_bilinear(env, env.clear, odo_to_world(env, q)[None])[0]
        kind = "ghost" if tq > 0.10 else "real"
    pw = odo_to_world(env, p)
    return dict(kind=kind, deficit=float(-M[i]), t_bind=float(ts[0, min(i, ts.shape[1] - 1)]), want=[float(want[0]), float(want[1])],
                map_clear=float(d[j]), true_clear=float(grid_bilinear(env, env.clear, pw[None])[0]))


def run(args):
    scen, ep = args
    env = Nav2Env(seed=70000 + ep, low_ckpt=LOW, level=0.8, adaptive=False, scenario={"nav": CALM | SCEN[scen], "loco": LOCO},
                  shield=True)
    oa, _ = env.reset()
    pol = GeoMPC()
    geo, out = [], []
    while True:
        (oa, _), _, done, info = env.step(pol(oa, env))
        geo.append(env.geo_prev)
        if done:
            break
        if len(geo) > 50 and env.steps % 25 == 0 and geo[-50] - geo[-1] < 0.2:
            out.append(analyse(env) | dict(scenario=scen, ep=ep))
    return out, dict(scenario=scen, ep=ep, success=bool(info["success"]), timeout=bool(info["timeout"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", default=["rooms, 1.0 m doors", "mixed"])
    ap.add_argument("--episodes", type=int, default=16)
    a = ap.parse_args()
    with mp.get_context("fork").Pool(4) as pool:
        R = pool.map(run, [(s, e) for s in a.scenarios for e in range(a.episodes)], chunksize=1)
    S = sum((r[0] for r in R), [])
    E = [r[1] for r in R]
    print(f"episodes {len(E)}: success {np.mean([e['success'] for e in E]):.0%}, timeouts {np.mean([e['timeout'] for e in E]):.0%}; "
          f"stalled samples {len(S)}")
    print("blocking cause:", dict(Counter(s["kind"] for s in S)))
    for k in ("unknown", "ghost", "real"):
        X = [s for s in S if s["kind"] == k]
        if X:
            print(f"  {k:8s} deficit median {np.median([x['deficit'] for x in X]) * 100:4.1f} cm, binding at t median "
                  f"{np.median([x['t_bind'] for x in X]):.2f} s, wanted v median {np.median([x['want'][0] for x in X]):.2f}, "
                  f"map clear {np.median([x['map_clear'] for x in X]):.2f} vs true {np.median([x['true_clear'] for x in X]):.2f}")
