"""
Hardware study: which mechanical change removes the root cause of the falls?

Root cause (rl/diagnose_falls.py, docs/08 sec. 4): every fall is a sideways roll ending in a pod strike. Roll is
controlled only by the passive twin-crown/low-CoM stiffness (which the pendulum spends when it swings to drive or
hold on a slope) and by the worm-driven bob, which is too slow (rate saturation, phase lag). Candidate fixes are
compared with the SAME classical controller on the SAME fixed-seed scenarios, so the result measures the hardware.

    python rl/hw_study.py --episodes 16
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_robust import NOMINAL, SCENARIOS  # noqa: E402
from loco2_env import HW_MK2, Loco2Env  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
W8 = dict(I=0.030, mass=2.5, tau=8.0, kp=20.0, kd=6.0, kw=0.3)
W15 = dict(I=0.050, mass=3.5, tau=15.0, kp=35.0, kd=10.0, kw=0.5)
VARIANTS = {
    "Mk2 as built": dict(HW_MK2),
    "fast lean 180 deg/s": HW_MK2 | dict(lean_rate=180.0),
    "crown offset 90 mm": HW_MK2 | dict(crown_d=0.090),
    "roll wheel 8 N m": HW_MK2 | dict(wheel=W8),
    "roll wheel 15 N m": HW_MK2 | dict(wheel=W15),
    "crown 90 + lean 180 + PI level": HW_MK2 | dict(crown_d=0.090, lean_rate=180.0, level_pi=True),
    "Mk2.1": HW_MK2 | dict(crown_d=0.090, lean_rate=180.0, level_pi=True),
    "crown 90 + lean 180 + PI + wheel 8": HW_MK2 | dict(crown_d=0.090, lean_rate=180.0, level_pi=True, wheel=W8),
}


def job(args):
    var, scen, n = args
    sc = dict(NOMINAL)
    sc.update(SCENARIOS[scen])
    out = []
    for ep in range(n):
        env = Loco2Env(seed=5000 + ep, scenario=sc, hw=VARIANTS[var])
        env.reset()
        while True:
            _, _, done, info = env.step(np.zeros(3))
            if done:
                e = info["episode"]
                out.append(dict(fell=bool(e["fell"]), ev=float(e["ev_rms"]), tilt=float(e["tilt_rms"])))
                break
    return var, scen, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(ROOT, "rl/runs/loco2/bench/hw_study.json"))
    a = ap.parse_args()
    jobs = [(v, s, a.episodes) for v in VARIANTS for s in SCENARIOS]
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=1)
    T = {}
    for v, s, out in res:
        T.setdefault(v, {})[s] = dict(fall=float(np.mean([o["fell"] for o in out])), ev=float(np.mean([o["ev"] for o in out])),
                                     tilt=float(np.mean([o["tilt"] for o in out])))
    json.dump(T, open(a.out, "w"), indent=1)
    names = list(SCENARIOS)
    print(f"{'scenario (fall %, classical ctrl)':30s} | " + " | ".join(f"{v[:22]:>22s}" for v in VARIANTS))
    for s in names:
        print(f"{s:30s} | " + " | ".join(f"{100 * T[v][s]['fall']:22.0f}" for v in VARIANTS))
    print(f"{'MEAN':30s} | " + " | ".join(f"{100 * np.mean([T[v][s]['fall'] for s in names]):22.1f}" for v in VARIANTS))
    print(f"{'mean spine tilt (deg)':30s} | " + " | ".join(f"{np.mean([T[v][s]['tilt'] for s in names]):22.2f}" for v in VARIANTS))


if __name__ == "__main__":
    main()
