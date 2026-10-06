"""
Low-speed precision of a locomotion controller: what navigation in tight spaces needs from the 50 Hz layer.

    python rl/precision_test.py --policies classical rl/runs/loco3/best.pt rl/runs/loco31/best.pt
Tests (flat ground, nominal robot, 4 seeds each):
  hold      command (0, 0) for 6 s          -> heading drift (deg), displacement (cm)
  spin      command (0, 1.5 rad/s) for 4 s  -> heading error vs the integrated command (deg), max displacement (cm)
  creep     command (0.3 m/s, 0) for 10 s   -> heading drift (deg), lateral deviation (cm)
  stop      1.0 m/s for 2 s, then (0, 0)    -> distance travelled after the stop command (cm)
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import loco2_env as L  # noqa: E402
from eval_robust import NOMINAL  # noqa: E402

L.GEOFENCE = 1e9                 # straight-line tests leave the terrain patch; the geofence is irrelevant here


def controller(name):
    if name == "classical":
        return lambda oa: np.zeros(3)
    from nav2_env import LowLevel
    low = LowLevel(name)
    return lambda oa: low(oa)[0]


def run(pol, prog, T, seed):
    e = L.Loco2Env(seed, scenario=dict(NOMINAL), external_cmd=True)
    e.spawn = (0.0, 0.0, 0.0)
    oa, _ = e.reset()
    for _ in range(50):
        e.cmd = np.zeros(2)
        (oa, _), *_ = e.step(pol(oa))
    p0, psi_ref = e.d.qpos[0:2].copy(), e._truth()["yaw"]
    P, err = [p0], []
    for k in range(int(T / 0.02)):
        e.cmd = np.array(prog(k * 0.02))
        (oa, _), *_ = e.step(pol(oa))
        psi_ref += e.cmd[1] * 0.02
        err.append(math.degrees((e._truth()["yaw"] - psi_ref + math.pi) % (2 * math.pi) - math.pi))
        P.append(e.d.qpos[0:2].copy())
    return np.array(P), np.array(err)


def evaluate(name, seeds=4):
    pol = controller(name)
    R = {k: [] for k in ("hold_heading_deg", "hold_disp_cm", "spin_heading_deg", "spin_disp_cm", "creep_heading_deg",
                         "creep_lateral_cm", "stop_overrun_cm")}
    for s in range(seeds):
        P, e = run(pol, lambda t: (0.0, 0.0), 6.0, s)
        R["hold_heading_deg"].append(abs(e[-1]))
        R["hold_disp_cm"].append(100 * np.linalg.norm(P[-1] - P[0]))
        P, e = run(pol, lambda t: (0.0, 1.5), 4.0, s)
        R["spin_heading_deg"].append(abs(e[-1]))
        R["spin_disp_cm"].append(100 * np.linalg.norm(P - P[0], axis=1).max())
        P, e = run(pol, lambda t: (0.3, 0.0), 10.0, s)
        R["creep_heading_deg"].append(abs(e[-1]))
        R["creep_lateral_cm"].append(100 * np.abs(P[:, 1] - P[0, 1]).max())     # spawn heading is +x
        P, e = run(pol, lambda t: (1.0, 0.0) if t < 2 else (0.0, 0.0), 5.0, s)
        dist = np.linalg.norm(P - P[0], axis=1)
        R["stop_overrun_cm"].append(100 * (dist[100:].max() - dist[100]))
    return {k: float(np.mean(v)) for k, v in R.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["classical", "rl/runs/loco3/best.pt"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {p: evaluate(p) for p in a.policies}
    keys = list(next(iter(res.values())))
    print(f"{'metric':20s} | " + " | ".join(f"{os.path.basename(os.path.dirname(p)) or p:>14s}" for p in a.policies))
    for k in keys:
        print(f"{k:20s} | " + " | ".join(f"{res[p][k]:14.1f}" for p in a.policies))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
