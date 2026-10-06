"""
Robustness benchmark: the same fixed-seed scenarios for every controller.

    python rl/eval_robust.py --policies classical v1 v2 --episodes 12 --out docs/robustness
Controllers
  classical : the hand-tuned cascade (zero residual)
  v1        : the first learned locomotion policy (rl/runs/loco/best.pt; 4-frame MLP, trained on flat ground with
              pushes) through an observation/action adapter, since v2's world uses a wider residual range
  v2        : rl/runs/loco2/actor.ts (or --v2 path): 1 s history encoder + estimator, terrain + disturbances
Each episode is 20 s of the virtual operator's commands (random speed / yaw-rate steps and ramps).
Metrics: fall rate, speed RMSE, yaw-rate RMSE, spine pitch RMS (sensor stability), energy per km.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from gyra2_model import R as R_NOM  # noqa: E402
from loco2_env import FRAME, Loco2Env  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOMINAL = dict(k=0.0, family="flat", terrain_k=0.0, slope=0.0, slope_dir=0.0, mu=1.0, mu2=None, mu_switch_t=8.0, tors=0.015,
               roll_fr=0.005, mscale=np.ones(4), com=np.zeros(3), payload=0.0, payload_pos=np.zeros(3), tyre_r=R_NOM,
               strength=np.ones(2), tau_noise=0.0, bob_slew=np.radians(60), delay=0, push_rate=0.0, push_max=50.0,
               kick_max=0.0, wind=0.0, wind_dir=0.0, gust=0.0, gyro_bias=np.zeros(3), imu_tilt=np.zeros(2), glitch=0.0,
               v_max=3.0)
SCENARIOS = {
    "flat, 3 m/s": {},
    "flat, 8 m/s": dict(v_max=8.0),
    "hills": dict(family="hills", terrain_k=1.0),
    "rough + potholes": dict(family="rough", terrain_k=1.0),
    "curbs / steps 6 cm": dict(family="curbs", terrain_k=1.0),
    "ramps to 15 deg": dict(family="ramps", terrain_k=1.0),
    "12 deg incline": dict(slope=np.radians(12)),
    "ice (mu 0.3)": dict(mu=0.3),
    "asphalt -> ice at 8 s": dict(mu=1.0, mu2=0.3, mu_switch_t=8.0),
    "pushes to 350 N": dict(push_rate=1 / 2.0, push_max=350.0, kick_max=40.0),
    "wind 40 N + gusts": dict(wind=40.0, gust=10.0),
    "payload 10 kg off-centre": dict(payload=10.0, payload_pos=np.array([0.06, 0.12, 0.08])),
    "weak motors (70 %)": dict(strength=np.array([0.7, 0.75])),
    "40 ms latency": dict(delay=2),
    "IMU bias + misalignment": dict(gyro_bias=np.array([0.03, -0.03, 0.03]), imu_tilt=np.radians([1.5, -1.5]), glitch=0.002),
    "everything at once": dict(family="mixed", terrain_k=0.8, mu=0.45, wind=25.0, gust=8.0, push_rate=1 / 3.0, push_max=250.0,
                               kick_max=25.0, payload=6.0, payload_pos=np.array([0.04, -0.1, 0.05]), strength=np.array([0.8, 0.85]),
                               delay=1, gyro_bias=np.array([0.02, 0.02, -0.02]), imu_tilt=np.radians([1.0, 1.0]), tau_noise=0.05),
}


class V1Adapter:
    """Runs the v1 policy inside the v2 world: v1 saw 4 frames whose action fields were in v1 units."""
    def __init__(self):
        from nav_env import NumpyActor
        self.pi = NumpyActor(os.path.join(ROOT, "rl/runs/loco/best.pt"))
        self.scale = np.array([25 / 60, 25 / 60, 0.35 / 0.5])

    def __call__(self, oa):
        h = oa[:-2].reshape(-1, FRAME)[-4:].copy()
        h[:, 12:15] /= self.scale                  # last action back to v1 units
        h[:, 15:18] *= 2.0                         # base action was normalised by 50 N m in v1 (100 in v2)
        a1 = self.pi(np.concatenate([h.reshape(-1), oa[-2:]]).astype(np.float32))
        return np.clip(a1, -1, 1) * self.scale


def make_policy(name, v2_path):
    if name == "classical":
        return lambda oa: np.zeros(3)
    if name == "v1":
        return V1Adapter()
    import torch
    torch.set_num_threads(1)
    ts = torch.jit.load(v2_path)
    return lambda oa: ts(torch.as_tensor(oa[None])).detach().numpy()[0]


def run_scenario(args):
    name, pol_name, v2_path, n_ep, seed0, hw = args
    pol = make_policy(pol_name, v2_path)
    sc = dict(NOMINAL)
    sc.update(SCENARIOS[name])
    out = []
    for ep in range(n_ep):
        env = Loco2Env(seed=seed0 + ep, scenario=sc, hw=hw)
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
    return name, pol_name, out


def summarise(rows):
    f = np.array([r["fell"] for r in rows])
    return dict(fall=float(f.mean()), ev=float(np.mean([r["ev"] for r in rows])), ew=float(np.mean([r["ew"] for r in rows])),
                tilt=float(np.mean([r["tilt"] for r in rows])), wh_km=float(np.median([r["wh_km"] for r in rows])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["classical", "v1", "v2"])
    ap.add_argument("--v2", default=os.path.join(ROOT, "rl/runs/loco2/actor.ts"))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "robustness"))
    ap.add_argument("--hw", default="Mk2.1", help="hardware variant (rl/hw_study.py VARIANTS)")
    a = ap.parse_args()
    from hw_study import VARIANTS
    names = a.scenarios or list(SCENARIOS)
    jobs = [(n, p, a.v2, a.episodes, 5000, VARIANTS[a.hw]) for n in names for p in a.policies]
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(run_scenario, jobs, chunksize=1)
    table = {}
    for n, p, rows in res:
        table.setdefault(n, {})[p] = summarise(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(table, open(a.out + ".json", "w"), indent=1)
    lines = ["| scenario | " + " | ".join(f"{p} fall | {p} v-RMSE | {p} tilt" for p in a.policies) + " |",
             "|---|" + "---|---|---|" * len(a.policies)]
    for n in names:
        cells = []
        for p in a.policies:
            s = table[n][p]
            cells += [f"{100 * s['fall']:.0f} %", f"{s['ev']:.2f}", f"{s['tilt']:.2f}°"]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    open(a.out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
