"""
Record the command stream the low level actually receives in navigation (docs/12, P3): the navigator's request after
the safety filter and the position hold, one command per 0.1 s, over the benchmark scenarios. Locomotion training
replays windows of it (GYRA_CMD=replay:<file>), so the low level practises exactly what navigation asks of it - near
walls, in doors, among people - instead of a synthetic stand-in.

    python rl/record_cmds.py --episodes 6 --out rl/runs/nav2/bench/cmd_streams.npz      # GYRA_LOW / GYRA_BASE etc.
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_nav2 import CALM, LOCO, SCEN, make  # noqa: E402
from nav2_env import Nav2Env  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(args):
    scen, ep, low = args
    os.environ["GYRA_LOW"] = low
    nav = make("mpc", None)
    env = Nav2Env(seed=90000 + ep, low_ckpt=low, level=0.8, adaptive=False, scenario={"nav": CALM | SCEN[scen], "loco": LOCO},
                  shield=True)
    oa, _ = env.reset()
    rows = []
    while True:
        (oa, _), _, done, info = env.step(nav(oa, env))
        c = env.loco.cmd
        near = float(np.min(env.lmap.clearance(env.loc[None, :2]))) if env.lmap is not None else 99.0
        rows.append((float(c[0]), float(c[1]), near))
        if done:
            break
    return np.array(rows, np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--low", default=os.environ.get("GYRA_LOW") or os.path.join(ROOT, "rl/runs/loco31/best.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "rl/runs/nav2/bench/cmd_streams.npz"))
    a = ap.parse_args()
    jobs = [(s, e, os.path.abspath(a.low)) for s in SCEN for e in range(a.episodes)]
    with mp.get_context("fork").Pool(4) as pool:
        R = pool.map(run, jobs, chunksize=1)
    ep = np.concatenate([np.full(len(r), i, np.int32) for i, r in enumerate(R)])
    D = np.concatenate(R)
    np.savez_compressed(a.out, v=D[:, 0], w=D[:, 1], clearance=D[:, 2], episode=ep)
    sw = np.mean(np.abs(np.diff(D[:, 0])) > 1e-6)
    print(f"{len(D)} commands ({len(D) * 0.1 / 60:.0f} min) from {len(R)} episodes; speed command changes every "
          f"{1 / max(sw, 1e-9) * 0.1:.2f} s; zero speed {np.mean(np.abs(D[:, 0]) < 1e-6):.0%}; "
          f"within 0.6 m of an obstacle {np.mean(D[:, 2] < 0.6):.0%}")
