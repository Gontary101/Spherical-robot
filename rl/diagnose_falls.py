"""
Root-cause analysis of locomotion failures.

For every scenario of the robustness benchmark it runs a controller and, for each fall, records which limit was hit
(pendulum loop-over / roll-over / spine pitch) and what the system was doing in the 2 s before: pendulum angle,
drive-torque saturation, levelling-motor saturation, bob at its stop, commanded vs actual speed, pushes.

    python rl/diagnose_falls.py --policy v2 --v2 rl/runs/loco2/v2_final_run1.ts --episodes 16
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_robust import NOMINAL, SCENARIOS, make_policy  # noqa: E402
from loco2_env import HW_MK2, Loco2Env  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAIL_SCEN = ["flat, 8 m/s", "12 deg incline", "pushes to 350 N", "wind 40 N + gusts", "payload 10 kg off-centre",
             "rough + potholes", "everything at once"]


def run(args):
    name, pol_name, path, n = args
    pol = make_policy(pol_name, path)
    sc = dict(NOMINAL)
    sc.update(SCENARIOS[name])
    falls, eps = [], 0
    pend_all, sat_all, lvl_all = [], [], []
    for ep in range(n):
        env = Loco2Env(seed=5000 + ep, scenario=sc, hw=HW_MK2)
        oa, oc = env.reset()
        hist = []
        eps += 1
        while True:
            (oa, oc), r, done, info = env.step(pol(oa))
            hist.append(env.diag)
            if done:
                break
        H = {k: np.array([h[k] for h in hist]) for k in hist[0] if k != "mode"}
        pend_all.append(np.abs(H["pend"]).max())
        sat_all.append(float((H["sat"] > 0.02).mean()))
        lvl_all.append(float((np.abs(H["level"]) > 14.9).mean()))
        if info["fell"]:
            w = slice(max(0, len(hist) - 100), len(hist))
            falls.append(dict(
                mode=hist[-1]["mode"], t=len(hist) * 0.02,
                pend_max_before=float(np.abs(H["pend"][w][:-5]).max()) if len(hist) > 5 else None,
                first_t_pend_gt_90=float(np.argmax(np.abs(H["pend"]) > np.pi / 2) * 0.02) if (np.abs(H["pend"]) > np.pi / 2).any() else None,
                sat_frac_2s=float((H["sat"][w] > 0.02).mean()), level_sat_frac_2s=float((np.abs(H["level"][w]) > 14.9).mean()),
                bob_stop_frac_2s=float((np.abs(H["bob"][w]) > 0.68).mean()), push_frac_2s=float(H["push"][w].mean()),
                cmd_v=float(H["cmd_v"][w].mean()), v=float(H["v"][w].mean()), cmd_w=float(H["cmd_w"][w].mean()),
                roll_max=float(np.abs(H["roll"][w]).max()), pitch_max=float(np.abs(H["pitch"][w]).max())))
    return name, pol_name, dict(episodes=eps, falls=falls, pend_max_mean=float(np.mean(pend_all)),
                                sat_frac_mean=float(np.mean(sat_all)), level_sat_frac_mean=float(np.mean(lvl_all)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["classical", "v2"])
    ap.add_argument("--v2", default=os.path.join(ROOT, "rl/runs/loco2/v2_final_run1.ts"))
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(ROOT, "rl/runs/loco2/bench/fall_diagnosis.json"))
    a = ap.parse_args()
    jobs = [(s, p, a.v2, a.episodes) for s in FAIL_SCEN for p in a.policies]
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(run, jobs, chunksize=1)
    out = {}
    for s, p, r in res:
        out.setdefault(s, {})[p] = r
        modes = {}
        for f in r["falls"]:
            modes[f["mode"]] = modes.get(f["mode"], 0) + 1
        fl = r["falls"]
        avg = lambda k: np.mean([f[k] for f in fl if f[k] is not None]) if fl else float("nan")  # noqa: E731
        print(f"{s:26s} {p:9s} falls {len(fl):2d}/{r['episodes']} modes {modes} | before fall: pend_max {np.degrees(avg('pend_max_before')):5.0f} deg "
              f"drive-sat {avg('sat_frac_2s'):.2f} level-sat {avg('level_sat_frac_2s'):.2f} bob-stop {avg('bob_stop_frac_2s'):.2f} "
              f"push {avg('push_frac_2s'):.2f} cmd_v {avg('cmd_v'):+.2f} v {avg('v'):+.2f} | all eps: drive-sat {r['sat_frac_mean']:.3f} "
              f"level-sat {r['level_sat_frac_mean']:.3f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
