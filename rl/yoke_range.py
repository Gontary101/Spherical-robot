"""
Range of the yoke-to-spine angle and the bob lean actually used by the deployed control (rl/runs/loco31) over the
robustness scenarios: sizes the yoke's mechanical pitch stops (docs/11_build.md).

    python rl/yoke_range.py --episodes 3 --out cad/build/out/yoke_range.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from eval_robust import NOMINAL, SCENARIOS, make_policy  # noqa: E402
from loco2_env import Loco2Env  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(args):
    name, ts_path, n_ep, seed0 = args
    pol = make_policy("v2", ts_path)
    sc = dict(NOMINAL)
    sc.update(SCENARIOS[name])
    P, L, W = [], [], []
    for ep in range(n_ep):
        env = Loco2Env(seed=seed0 + ep, scenario=sc)
        oa, oc = env.reset()
        m, d = env.m, env.d
        qy, qb = m.joint("yoke").qposadr[0], m.joint("bob").qposadr[0]
        vy = m.joint("yoke").dofadr[0]
        while True:
            (oa, oc), r, done, info = env.step(pol(oa))
            P.append(float(d.qpos[qy]))
            L.append(float(d.qpos[qb]))
            W.append(float(d.qvel[vy]))
            if done:
                break
    return name, np.degrees(np.asarray(P)), np.degrees(np.asarray(L)), np.asarray(W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "rl/runs/loco31/actor_final.ts"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "cad/build/out/yoke_range.json"))
    a = ap.parse_args()
    jobs = [(n, a.policy, a.episodes, 3000) for n in SCENARIOS]
    with mp.get_context("spawn").Pool(int(os.environ.get("PROCS", min(len(jobs), os.cpu_count())))) as pool:
        res = pool.map(run, jobs)
    out = {}
    allP = np.concatenate([r[1] for r in res])
    allL = np.concatenate([r[2] for r in res])
    allW = np.concatenate([r[3] for r in res])
    for name, p, l, w in res + [("ALL", allP, allL, allW)]:
        ap_ = np.abs(p)
        hi = ap_ > 90.0
        out[name] = dict(pitch_p99=float(np.percentile(ap_, 99)), pitch_max=float(ap_.max()),
                         lean_p99=float(np.percentile(np.abs(l), 99)), lean_max=float(np.abs(l).max()),
                         rate_max_above_90deg=float(np.abs(w[hi]).max()) if hi.any() else 0.0,
                         outward_rate_max_above_90deg=float(np.max(np.sign(p[hi]) * w[hi])) if hi.any() else 0.0)
        print(f"{name:28s} |yoke-spine| p99 {out[name]['pitch_p99']:6.1f} max {out[name]['pitch_max']:6.1f} deg   "
              f"|lean| p99 {out[name]['lean_p99']:5.1f} max {out[name]['lean_max']:5.1f} deg   "
              f"outward rate above 90 deg max {out[name]['outward_rate_max_above_90deg']:5.2f} rad/s")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
