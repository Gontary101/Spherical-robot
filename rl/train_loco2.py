"""
Train GYRA Mk2 locomotion v2 (terrain + disturbances, history encoder + estimator, symmetry) with PPO.

    python rl/train_loco2.py --steps 60e6 --out rl/runs/loco2
Logs progress.csv (losses, every reward term, episode metrics, curriculum level, per-terrain fall rates),
saves last.pt every 10 iterations and a milestone every 5M steps (selected later by rl/eval_robust.py), and
exports the deployable TorchScript actor (normaliser folded in).
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import terrain as TR  # noqa: E402
from loco2_env import ACT_OBS, FRAME, HIST, N_ACT, N_CMD, N_EST, Loco2Env  # noqa: E402
from ppo2 import PPO2, Actor2, Mirror, rollout2  # noqa: E402
from vec_env import VecEnv  # noqa: E402


def make_agent(n_critic):
    actor = Actor2(HIST, FRAME, N_CMD, N_EST, N_ACT)
    perm = list(range(FRAME))
    perm[9], perm[10] = 10, 9
    mir = Mirror(HIST, FRAME, Loco2Env.mirror_frame_signs(), perm, (1, -1), (1, -1, -1))
    return PPO2(actor, ACT_OBS, n_critic, N_EST, mirror=mir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, default=60e6)
    ap.add_argument("--out", default="rl/runs/loco2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--resume", default="")
    ap.add_argument("--level0", type=float, default=0.0, help="initial curriculum level of every env (when resuming)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(4)
    env = VecEnv(Loco2Env, n_workers=a.workers, envs_per_worker=a.envs, seed=7)
    env.set_attr(level=a.level0)
    oa, oc = env.reset()
    agent = make_agent(oc.shape[1])
    steps = 0
    if a.resume:
        steps = int(agent.load(a.resume).get("steps", 0))
    f = open(os.path.join(a.out, "progress.csv"), "a", newline="")
    wr = None
    t0 = time.time()
    eps = deque(maxlen=300)
    next_mile = (steps // 5_000_000 + 1) * 5_000_000
    it = 0
    best = -1e9
    while steps < a.steps:
        torch.set_num_threads(1)                 # inference: workers own the cores
        buf, infos, oa, oc = rollout2(env, agent, oa, oc, a.horizon)
        torch.set_num_threads(4)                 # update: workers idle
        st = agent.update(buf)
        steps += a.horizon * env.n
        it += 1
        terms = defaultdict(float)
        for inf in infos:
            for k, v in inf["terms"].items():
                terms[k] += v / len(infos)
            if "episode" in inf:
                eps.append(inf["episode"])
        levels = np.array([inf["level"] for inf in infos])
        row = dict(iter=it, steps=steps, time_s=round(time.time() - t0, 1), sps=round(a.horizon * env.n * it / (time.time() - t0)),
                   rew=float(buf["raw_rew"].mean()), level_mean=float(levels.mean()), level_p90=float(np.percentile(levels, 90)),
                   std=float(agent.actor.log_std.detach().exp().mean()))
        row |= {k: round(v, 5) for k, v in st.items()}
        row |= {f"r_{k}": round(v, 5) for k, v in terms.items()}
        if eps:
            E = list(eps)
            row |= dict(ep_fall=np.mean([e["fell"] for e in E]), ep_track=np.mean([e["track"] for e in E]),
                        ep_ev=np.mean([e["ev_rms"] for e in E]), ep_ew=np.mean([e["ew_rms"] for e in E]),
                        ep_tilt=np.mean([e["tilt_rms"] for e in E]), ep_len=np.mean([e["length"] for e in E]))
            for fam in TR.FAMILIES:
                sel = [e["fell"] for e in E if e["family"] == fam]
                row[f"fall_{fam}"] = np.mean(sel) if sel else np.nan
        if wr is None:              # declare every column up front (episode stats appear only after the first episodes end)
            names = list(row.keys())
            names += [k for k in ("ep_fall", "ep_track", "ep_ev", "ep_ew", "ep_tilt", "ep_len") + tuple(f"fall_{x}" for x in TR.FAMILIES)
                      if k not in names]
            wr = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
            if f.tell() == 0:
                wr.writeheader()
        wr.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        f.flush()
        if it % 10 == 0:
            print(f"it {it:5d} {steps/1e6:6.2f}M sps {row['sps']:5d} rew {row['rew']:.3f} lvl {row['level_mean']:.2f}/"
                  f"{row['level_p90']:.2f} fall {row.get('ep_fall', 0):.3f} track {row.get('ep_track', 0):.3f} "
                  f"tilt {row.get('ep_tilt', 0):.2f} est {st['est_loss']:.3f} sym {st['sym_loss']:.4f} kl {st['kl']:.4f} "
                  f"lr {st['lr']:.1e} std {row['std']:.3f}", flush=True)
            extra = {"steps": steps}
            agent.save(os.path.join(a.out, "last.pt"), extra)
            score = row["level_mean"] - row.get("ep_fall", 1.0)
            if score > best and steps > 2e6:
                best = score
                agent.save(os.path.join(a.out, "best_train.pt"), extra)
                agent.export(os.path.join(a.out, "actor.ts"), ACT_OBS)
        if steps >= next_mile:
            agent.save(os.path.join(a.out, f"m{next_mile // 1_000_000:03d}M.pt"), {"steps": steps})
            next_mile += 5_000_000
    agent.save(os.path.join(a.out, "final.pt"), {"steps": steps})
    agent.export(os.path.join(a.out, "actor_final.ts"), ACT_OBS)
    env.close()


if __name__ == "__main__":
    main()
