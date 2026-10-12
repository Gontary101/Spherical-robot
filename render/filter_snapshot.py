"""
Snapshot of the real safety filter for the explainer: drive the deployable MPC in a pole-forest world, then at a
moment with obstacles ahead store the local map, the robot's local pose and speed, and the filter's predicted paths
(reaction + braking) and verdicts for a grid of candidate commands.

    python render/filter_snapshot.py --out media/filter_snapshot.npz
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rl"))
from eval_nav2 import CALM, LOCO, SCEN  # noqa: E402
from nav2_env import Nav2Env  # noqa: E402
from teacher import GeoMPC  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "media", "filter_snapshot.npz"))
    a = ap.parse_args()
    for ep in range(10):
        env = Nav2Env(seed=70100 + ep, low_ckpt=os.path.join(HERE, "..", "rl/runs/loco31/best.pt"), level=0.8, adaptive=False,
                      scenario={"nav": CALM | SCEN["forest (thin poles)"], "loco": dict(LOCO, push_rate=0.0)}, shield=True)
        oa, _ = env.reset()
        nav = GeoMPC(deployable=True)
        for k in range(400):
            (oa, _), _, done, info = env.step(nav(oa, env))
            if done:
                break
            v0, w0 = env._velocity_estimate()
            if k > 40 and v0 > 1.2:
                C = np.array([(v, w) for v in (0.5, 1.0, 1.5, 2.0, 2.5) for w in np.linspace(-1.5, 1.5, 13)])
                ok, sl = env.filter.admissible(env.lmap, env.loc.copy(), v0, w0, C)
                if 0.2 < ok.mean() < 0.8:
                    P, ts = env.filter._rollout(env.loc.copy(), v0, w0, C, env.filter.a0)
                    np.savez_compressed(a.out, L=env.lmap.L, origin=env.lmap.origin, cell=env.lmap.cell, loc=env.loc, v0=v0, w0=w0,
                                        cmds=C, ok=ok, slack=sl, paths=P, times=ts, t_react=env.filter.t_react,
                                        robot_r=env.robot_r, chosen=env.loco.cmd)
                    print(f"episode {ep} step {k}: v0 {v0:.2f}, {ok.sum()}/{len(ok)} admissible -> {a.out}")
                    sys.exit(0)
    print("no suitable moment found")
