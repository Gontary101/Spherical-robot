"""
Does the deployed control still work on the hardware as it will be built? Runs the deployed locomotion policy
(rl/runs/loco31) through the robustness scenarios with the concept CAD's mass properties (cad/out21) and with the
build CAD's (cad/build/out), same seeds, and compares falls, tracking error, sensor-platform tilt and energy.

    python rl/build_mass_check.py --episodes 3 --out cad/build/out/sim_build_masses.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from eval_robust import NOMINAL, SCENARIOS, make_policy, summarise  # noqa: E402
from loco2_env import Loco2Env  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(args):
    name, cad, ts_path, n_ep, seed0 = args
    pol = make_policy("v2", ts_path)
    sc = dict(NOMINAL)
    sc.update(SCENARIOS[name])
    out = []
    for ep in range(n_ep):
        env = Loco2Env(seed=seed0 + ep, scenario=sc, hw=dict(cad=cad))
        oa, oc = env.reset()
        dist = 0.0
        while True:
            (oa, oc), r, done, info = env.step(pol(oa))
            dist += abs(env._truth()["v"]) * 0.02
            if done:
                e = info["episode"]
                out.append(dict(fell=bool(e["fell"]), ev=float(e["ev_rms"]), ew=float(e["ew_rms"]), tilt=float(e["tilt_rms"]),
                                wh_km=float(e["energy"] / 3600 / max(dist / 1000, 1e-3)), length=int(e["length"])))
                break
    return name, cad, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "rl/runs/loco31/actor_final.ts"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "cad/build/out/sim_build_masses.json"))
    a = ap.parse_args()
    jobs = [(n, cad, a.policy, a.episodes, 2000) for n in SCENARIOS for cad in ("out21", "build/out")]
    with mp.get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(run, jobs)
    table = {}
    for name, cad, rows in res:
        table.setdefault(name, {})[cad] = summarise(rows)
    allrows = {cad: [r for n, c, rows in res if c == cad for r in rows] for cad in ("out21", "build/out")}
    table["ALL"] = {cad: summarise(rows) for cad, rows in allrows.items()}
    json.dump(table, open(a.out, "w"), indent=1)
    print(f"{'scenario':28s} {'fall concept':>12s} {'fall build':>10s} {'ev c/b':>13s} {'tilt rms c/b':>15s} {'Wh/km c/b':>11s}")
    for n, t in table.items():
        c, b = t["out21"], t["build/out"]
        print(f"{n:28s} {c['fall']:12.2f} {b['fall']:10.2f} {c['ev']:6.3f}/{b['ev']:6.3f} "
              f"{c['tilt']:7.4f}/{b['tilt']:6.4f} "
              f"{c['wh_km']:5.0f}/{b['wh_km']:4.0f}")


if __name__ == "__main__":
    main()
