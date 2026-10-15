"""
Actuator duty measured in simulation, used to size the real hardware (docs/11_build.md).

Runs the deployed locomotion policy (rl/runs/loco31) through the robustness scenarios and records, per actuator:
torque / force percentiles, speed percentiles, RMS (thermal) torque and electrical-side power, plus energy per km.

    python rl/actuator_loads.py --episodes 4 --out cad/build/out/actuator_loads.json
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
    rec = {k: [] for k in ("lean_tau", "lean_w", "level_tau", "level_w", "drive_tau", "drive_w")}
    dist, energy, fell = 0.0, 0.0, 0
    for ep in range(n_ep):
        env = Loco2Env(seed=seed0 + ep, scenario=sc)
        oa, oc = env.reset()
        m, d = env.m, env.d
        aid = env.aid
        jq = {n: m.joint(n).dofadr[0] for n in ("tyreL", "tyreR", "yoke", "bob")}
        while True:
            (oa, oc), r, done, info = env.step(pol(oa))
            w_y = d.qvel[jq["yoke"]]
            rec["lean_tau"].append(float(d.actuator_force[aid["lean"]]))
            rec["lean_w"].append(float(d.qvel[jq["bob"]]))
            rec["level_tau"].append(float(d.actuator_force[aid["level"]]))
            rec["level_w"].append(float(w_y))           # spine <-> yoke relative rate (spine stays ~level)
            for s, a in (("tyreL", "driveL"), ("tyreR", "driveR")):
                rec["drive_tau"].append(float(d.actuator_force[aid[a]]))
                rec["drive_w"].append(float(d.qvel[jq[s]] - w_y))
            dist += abs(env._truth()["v"]) * 0.02
            if done:
                e = info["episode"]
                energy += float(e["energy"])
                fell += int(bool(e["fell"]))
                break
    return name, {k: np.asarray(v) for k, v in rec.items()}, dist, energy, fell


def stats(tau, w):
    a = np.abs(tau)
    p = tau * w
    return dict(tau_p50=float(np.percentile(a, 50)), tau_p95=float(np.percentile(a, 95)), tau_p99=float(np.percentile(a, 99)),
                tau_max=float(a.max()), tau_rms=float(np.sqrt(np.mean(tau ** 2))),
                w_p95=float(np.percentile(np.abs(w), 95)), w_max=float(np.abs(w).max()),
                mech_power_p99=float(np.percentile(np.abs(p), 99)), mech_power_mean=float(np.mean(np.abs(p))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "rl/runs/loco31/actor_final.ts"))
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "cad/build/out/actuator_loads.json"))
    a = ap.parse_args()
    jobs = [(n, a.policy, a.episodes, 1000) for n in SCENARIOS]
    with mp.Pool(min(len(jobs), os.cpu_count())) as pool:
        res = pool.map(run, jobs)
    out, allrec = {"scenarios": {}}, {}
    tot_d, tot_e = 0.0, 0.0
    for name, rec, dist, energy, fell in res:
        out["scenarios"][name] = dict(
            lean=stats(rec["lean_tau"], rec["lean_w"]), level=stats(rec["level_tau"], rec["level_w"]),
            drive=stats(rec["drive_tau"], rec["drive_w"]), wh_per_km=energy / 3600 / max(dist / 1000, 1e-3), falls=fell)
        for k, v in rec.items():
            allrec.setdefault(k, []).append(v)
        tot_d += dist
        tot_e += energy
    cat = {k: np.concatenate(v) for k, v in allrec.items()}
    out["all"] = dict(lean=stats(cat["lean_tau"], cat["lean_w"]), level=stats(cat["level_tau"], cat["level_w"]),
                      drive=stats(cat["drive_tau"], cat["drive_w"]), wh_per_km=tot_e / 3600 / max(tot_d / 1000, 1e-3))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for k in ("lean", "level", "drive"):
        s = out["all"][k]
        print(f"{k:6s} |tau| p95 {s['tau_p95']:6.1f}  p99 {s['tau_p99']:6.1f}  max {s['tau_max']:6.1f}  rms {s['tau_rms']:5.1f} N m"
              f"   |w| p95 {s['w_p95']:5.2f} max {s['w_max']:5.2f} rad/s   P99 {s['mech_power_p99']:6.0f} W")
    print(f"energy {out['all']['wh_per_km']:.0f} Wh/km over all scenarios")
    for n, s in out["scenarios"].items():
        print(f"  {n:28s} lean p99 {s['lean']['tau_p99']:6.1f}  level p99 {s['level']['tau_p99']:5.1f}  "
              f"drive p99 {s['drive']['tau_p99']:5.1f}  {s['wh_per_km']:5.0f} Wh/km  falls {s['falls']}")


if __name__ == "__main__":
    main()
