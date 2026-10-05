"""
Train the GYRA Mk2 autonomous-navigation policy (asymmetric PPO) on top of the frozen
locomotion policy.

    python rl/train_nav.py --loco rl/runs/loco/best.pt --steps 4e6 --out rl/runs/nav
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
from nav_env import ACT_OBS, CRIT_OBS, NavEnv  # noqa: E402
from ppo import PPO, rollout  # noqa: E402
from vec_env import VecEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, default=4e6)
    ap.add_argument("--out", default="rl/runs/nav")
    ap.add_argument("--loco", default="rl/runs/loco/best.pt")
    ap.add_argument("--low", default="learned")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)
    env = VecEnv(NavEnv, n_workers=a.workers, envs_per_worker=a.envs, seed=3,
                 low_level=a.low, loco_ckpt=os.path.abspath(a.loco))
    agent = PPO(ACT_OBS, CRIT_OBS, 2, hidden=(256, 256, 128), ent=0.005, gamma=0.99, lam=0.95)
    oa, oc = env.reset()
    f = open(os.path.join(a.out, "progress.csv"), "a", newline="")
    wr = csv.writer(f)
    if f.tell() == 0:
        wr.writerow(["iter", "steps", "time_s", "sps", "mean_rew", "success", "collision", "fell", "timeout", "kl", "lr", "std"])
    steps, it, t0, best = 0, 0, time.time(), -1.0
    window = []
    while steps < a.steps:
        buf, infos, oa, oc = rollout(env, agent, oa, oc, a.horizon)
        st = agent.update(buf)
        steps += a.horizon * env.n
        it += 1
        ends = [i for i in infos if i["success"] or i["collision"] or i["fell"] or i["timeout"]]
        window = (window + ends)[-300:]
        rate = lambda k: float(np.mean([e[k] for e in window])) if window else 0.0  # noqa: E731
        el = time.time() - t0
        wr.writerow([it, steps, round(el, 1), round(steps / el), round(float(buf["rew"].mean()), 4), round(rate("success"), 3),
                     round(rate("collision"), 3), round(rate("fell"), 3), round(rate("timeout"), 3), round(st["kl"], 4),
                     round(st["lr"], 6), round(float(agent.actor.log_std.exp().mean()), 3)])
        f.flush()
        if it % 5 == 0:
            print(f"it {it:4d} steps {steps/1e6:5.2f}M sps {steps/el:5.0f} succ {rate('success'):.2f} coll {rate('collision'):.2f} "
                  f"fell {rate('fell'):.2f} tout {rate('timeout'):.2f} kl {st['kl']:.4f} std {agent.actor.log_std.exp().mean():.2f}", flush=True)
            agent.save(os.path.join(a.out, "last.pt"), {"steps": steps})
            if len(window) >= 100 and rate("success") > best:
                best = rate("success")
                agent.save(os.path.join(a.out, "best.pt"), {"steps": steps, "success": best})
                ts = torch.jit.trace(agent.actor.eval(), torch.zeros(1, ACT_OBS))
                ts.save(os.path.join(a.out, "actor.pt"))
                json.dump(agent.na.state(), open(os.path.join(a.out, "actor_obs_norm.json"), "w"))
                agent.actor.train()
    agent.save(os.path.join(a.out, "final.pt"), {"steps": steps})
    env.close()


if __name__ == "__main__":
    main()
