"""
Train GYRA Mk2 navigation v2 with asymmetric PPO on top of the frozen locomotion v2 policy.

    python rl/train_nav2.py --steps 12e6 --out rl/runs/nav2 --low rl/runs/loco2/best.pt
"""
import argparse
import csv
import os
import sys
import time
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from nav2_env import ACT_OBS, DEPTH_H, DEPTH_W, FAMILIES, LIDAR_FRAMES, N_BEAM, Nav2Env  # noqa: E402
from ppo2 import PPO2, NavActor, PermMirror, rollout2  # noqa: E402
from vec_env import VecEnv  # noqa: E402

N_LOW = ACT_OBS - N_BEAM * LIDAR_FRAMES - DEPTH_W * DEPTH_H


def make_agent(n_critic):
    actor = NavActor(N_BEAM, LIDAR_FRAMES, DEPTH_W, DEPTH_H, N_LOW)
    perm, sg, asg = Nav2Env.mirror_spec()
    return PPO2(actor, ACT_OBS, n_critic, 0, mirror=PermMirror(perm, sg, asg), gamma=0.99, lam=0.95, ent=0.003,
                sym_coef=0.3, lr=3e-4, lr_max=6e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, default=12e6)
    ap.add_argument("--out", default="rl/runs/nav2")
    ap.add_argument("--low", default="rl/runs/loco2/best.pt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ["GYRA_LOW"] = os.path.abspath(a.low)
    env = VecEnv(Nav2Env, n_workers=a.workers, envs_per_worker=a.envs, seed=11)
    oa, oc = env.reset()
    agent = make_agent(oc.shape[1])
    steps = int(agent.load(a.resume).get("steps", 0)) if a.resume else 0
    f = open(os.path.join(a.out, "progress.csv"), "a", newline="")
    wr = None
    t0 = time.time()
    eps = deque(maxlen=400)
    next_mile, it, best = (steps // 1_000_000 + 1) * 1_000_000, 0, -1e9
    while steps < a.steps:
        torch.set_num_threads(1)
        buf, infos, oa, oc = rollout2(env, agent, oa, oc, a.horizon)
        torch.set_num_threads(4)
        st = agent.update(buf)
        steps += a.horizon * env.n
        it += 1
        for inf in infos:
            if "episode" in inf:
                eps.append(inf["episode"])
        levels = np.array([inf["level"] for inf in infos])
        row = dict(iter=it, steps=steps, time_s=round(time.time() - t0, 1), sps=round(a.horizon * env.n * it / (time.time() - t0)),
                   rew=float(buf["raw_rew"].mean()), level_mean=float(levels.mean()), level_p90=float(np.percentile(levels, 90)),
                   std=float(agent.actor.log_std.detach().exp().mean())) | {k: round(v, 5) for k, v in st.items()}
        if eps:
            E = list(eps)
            for k in ("success", "collision", "hit_dynamic", "fell", "timeout", "spl"):
                row[k] = float(np.mean([e[k] for e in E]))
            for fam in FAMILIES:
                sel = [e["success"] for e in E if e["family"] == fam]
                row[f"succ_{fam}"] = float(np.mean(sel)) if sel else np.nan
        if wr is None:
            wr = csv.DictWriter(f, fieldnames=list(row.keys()) + [k for k in ("success", "collision", "hit_dynamic", "fell", "timeout", "spl")
                                                                    + tuple(f"succ_{x}" for x in FAMILIES) if k not in row], extrasaction="ignore")
            if f.tell() == 0:
                wr.writeheader()
        wr.writerow(row)
        f.flush()
        if it % 5 == 0:
            print(f"it {it:4d} {steps/1e6:5.2f}M sps {row['sps']:4d} rew {row['rew']:.3f} lvl {row['level_mean']:.2f}/{row['level_p90']:.2f} "
                  f"succ {row.get('success', 0):.2f} coll {row.get('collision', 0):.2f} (dyn {row.get('hit_dynamic', 0):.2f}) "
                  f"fell {row.get('fell', 0):.2f} to {row.get('timeout', 0):.2f} spl {row.get('spl', 0):.2f} kl {st['kl']:.4f} "
                  f"lr {st['lr']:.1e} std {row['std']:.2f}", flush=True)
            agent.save(os.path.join(a.out, "last.pt"), {"steps": steps})
            score = row["level_mean"] + row.get("success", 0) - row.get("collision", 1)
            if score > best and steps > 1e6:
                best = score
                agent.save(os.path.join(a.out, "best_train.pt"), {"steps": steps})
                agent.export(os.path.join(a.out, "actor.ts"), ACT_OBS)
        if steps >= next_mile:
            agent.save(os.path.join(a.out, f"m{next_mile // 1_000_000:03d}M.pt"), {"steps": steps})
            next_mile += 1_000_000
    agent.save(os.path.join(a.out, "final.pt"), {"steps": steps})
    env.close()


if __name__ == "__main__":
    main()
