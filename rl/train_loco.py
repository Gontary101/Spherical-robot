"""
Train the GYRA Mk2 locomotion (teleoperation) policy with asymmetric PPO.

    python rl/train_loco.py --steps 30e6 --out rl/runs/loco
Writes progress.csv, checkpoints, and the deployable actor (TorchScript + normaliser JSON).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from loco_env import ACT_OBS, CRIT_OBS, N_ACT, LocoEnv  # noqa: E402
from ppo import PPO, rollout  # noqa: E402
from vec_env import VecEnv  # noqa: E402


def export_actor(agent, out):
    agent.actor.eval()
    ts = torch.jit.trace(agent.actor, torch.zeros(1, ACT_OBS))
    ts.save(os.path.join(out, "actor.pt"))
    json.dump(agent.na.state(), open(os.path.join(out, "actor_obs_norm.json"), "w"))
    agent.actor.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, default=30e6)
    ap.add_argument("--out", default="rl/runs/loco")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)
    env = VecEnv(LocoEnv, n_workers=a.workers, envs_per_worker=a.envs, seed=1)
    agent = PPO(ACT_OBS, CRIT_OBS, N_ACT, init_std=0.3, ent=0.0, max_std=0.35, lr=3e-4, lr_max=3e-4)
    v_max, steps0, diff = 4.0, 0, 0.5
    if a.resume:
        extra = agent.load(a.resume)
        v_max, steps0 = extra.get("v_max", 3.0), int(extra.get("steps", 0))
        diff = extra.get("difficulty", 0.5)
    env.set_attr(v_max=v_max, difficulty=diff)
    oa, oc = env.reset()
    f = open(os.path.join(a.out, "progress.csv"), "a", newline="")
    wr = csv.writer(f)
    if f.tell() == 0:
        wr.writerow(["iter", "steps", "time_s", "sps", "mean_rew", "ev", "ew", "fall_rate", "v_max", "kl", "lr", "std"])
    steps, it, t0 = steps0, 0, time.time() - steps0 / 2150.0
    best = -1e9
    hist = []
    last_adv = steps
    while steps < a.steps:
        buf, infos, oa, oc = rollout(env, agent, oa, oc, a.horizon)
        st = agent.update(buf)
        steps += a.horizon * env.n
        it += 1
        ev = float(np.mean([i["ev"] for i in infos]))
        ew = float(np.mean([i["ew"] for i in infos]))
        ends = [i for i in infos if i.get("fell") or i.get("timeout")]
        fall = float(np.mean([i["fell"] for i in ends])) if ends else 0.0
        mr = float(buf["rew"].mean())
        std = float(agent.actor.log_std.clamp(max=agent.actor.max_log_std).exp().mean())
        el = time.time() - t0
        wr.writerow([it, steps, round(el, 1), round(steps / el), round(mr, 4), round(ev, 3), round(ew, 3),
                     round(fall, 3), v_max, round(st["kl"], 4), round(st["lr"], 6), round(std, 3)])
        f.flush()
        # curriculum on the commanded speed range (smoothed over the last 20 iterations)
        hist.append((ev / max(v_max, 3.0) * 3.0, ew, len([i for i in ends if i["fell"]]), len(ends)))
        hist = hist[-20:]
        h_fall = sum(x[2] for x in hist) / max(1, sum(x[3] for x in hist))
        # speed error has a physics floor (gravity-limited acceleration after command jumps), so the
        # schedule is step-based and gated only on the fall rate
        if len(hist) == 20 and steps - last_adv > 1.2e6 and h_fall < 0.2 and (v_max < 8.5 or diff < 1.0):
            last_adv = steps
            v_max = min(8.5, v_max + 1.0)
            diff = min(1.0, diff + 0.125)
            env.set_attr(v_max=v_max, difficulty=diff)
            hist = []
            print(f"curriculum -> v_max {v_max} difficulty {diff}", flush=True)
        if it % 10 == 0:
            print(f"it {it:4d} steps {steps/1e6:6.2f}M sps {steps/el:6.0f} rew {mr:.3f} ev {ev:.2f} ew {ew:.2f} "
                  f"fall {fall:.3f} vmax {v_max} kl {st['kl']:.4f} lr {st['lr']:.1e} std {std:.2f}", flush=True)
            agent.save(os.path.join(a.out, "last.pt"), {"v_max": v_max, "steps": steps, "difficulty": diff})
            score = mr - 2 * h_fall + v_max * 0.05 + diff * 0.2
            if score > best:
                best = score
                agent.save(os.path.join(a.out, "best.pt"), {"v_max": v_max, "steps": steps, "difficulty": diff})
                export_actor(agent, a.out)
    agent.save(os.path.join(a.out, "final.pt"), {"v_max": v_max, "steps": steps, "difficulty": diff})
    export_actor(agent, a.out)
    env.close()


if __name__ == "__main__":
    main()
