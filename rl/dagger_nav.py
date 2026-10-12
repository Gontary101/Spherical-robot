"""
Phase 2: distil the privileged teacher (rl/teacher.py, 91 % under the shield) into the map-aware navigator by DAgger.

Each round, 4 worker processes drive shielded episodes from the training distribution (curriculum level, all world
families, sensor faults, pedestrians). The executed command is the teacher's with probability beta (decaying to 0),
else the student's; every visited state is labelled with the teacher's command. The student (NavActor with the local
map branch, warm-started from the shield-trained navigator) is fitted by MSE on the aggregated data (FIFO, fp16).
After every round the student is benchmarked on its own rollouts (beta = 0) - that is the number that matters.

    python rl/dagger_nav.py --init rl/runs/nav2s/best.pt --out rl/runs/nav3d --rounds 12
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from nav2_env import ACT_OBS, Nav2Env  # noqa: E402
from teacher import GeoMPC  # noqa: E402
from train_nav2 import make_agent  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOW = os.path.join(ROOT, "rl/runs/loco31/best.pt")


def collect(args):
    seed, ts_path, beta, n_steps, level = args
    torch.set_num_threads(1)
    student = torch.jit.load(ts_path)
    rng = np.random.default_rng(seed)
    env = Nav2Env(seed=seed, low_ckpt=LOW, level=level, adaptive=False, shield=True)
    teacher = GeoMPC()
    oa, _ = env.reset()
    O, A, eps = [], [], []
    for _ in range(n_steps):
        a_t = teacher(oa, env)
        O.append(oa.astype(np.float16))
        A.append(a_t.astype(np.float32))
        if rng.random() < beta:
            a = a_t
        else:
            with torch.no_grad():
                a = student(torch.as_tensor(oa[None], dtype=torch.float32)).numpy()[0]
        (oa, _), _, done, info = env.step(a)
        if done:
            e = info["episode"]
            eps.append(dict(success=bool(e["success"]), collision=bool(e["collision"]), timeout=bool(e["timeout"]),
                            family=e["family"]))
            oa, _ = env.reset()
            teacher.reset()
    return np.array(O), np.array(A), eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(ROOT, "rl/runs/nav2s/best.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "rl/runs/nav3d"))
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=2500, help="env steps per worker per round")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--level", type=float, default=0.9)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = Nav2Env(seed=0, low_ckpt=LOW, shield=True)
    _, oc = env.reset()
    agent = make_agent(len(oc))
    from train_loco2 import load_partial
    load_partial(agent, a.init)
    opt = torch.optim.Adam(agent.actor.parameters(), lr=3e-4)
    O = np.zeros((0, ACT_OBS), np.float16)
    A = np.zeros((0, 2), np.float32)
    log = open(os.path.join(a.out, "dagger.jsonl"), "a")
    t0 = time.time()
    for rnd in range(a.rounds):
        beta = max(0.0, 1.0 - rnd / 4.0) * 0.8                     # 0.8, 0.6, 0.4, 0.2, 0, 0, ...
        ts_path = os.path.join(a.out, "student.ts")
        agent.export(ts_path, ACT_OBS)
        with mp.get_context("fork").Pool(a.workers) as pool:
            R = pool.map(collect, [(1000 * rnd + w, ts_path, beta, a.steps, a.level) for w in range(a.workers)])
        On = np.concatenate([r[0] for r in R])
        An = np.concatenate([r[1] for r in R])
        eps = sum((r[2] for r in R), [])
        O = np.concatenate([O, On])[-a.buffer:]
        A = np.concatenate([A, An])[-a.buffer:]
        # observation normaliser: running statistics over everything seen
        agent.na.update(On.astype(np.float32))
        mean = torch.as_tensor(agent.na.mean, dtype=torch.float32)
        std = torch.as_tensor(np.sqrt(agent.na.var + 1e-8), dtype=torch.float32)
        losses = []
        n = len(O)
        for ep in range(a.epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, 512):
                idx = perm[i:i + 512]
                x = torch.clamp((torch.as_tensor(O[idx].astype(np.float32)) - mean) / std, -agent.na.clip, agent.na.clip)
                y = torch.as_tensor(A[idx])
                mu, _ = agent.actor.mean(x)
                loss = ((mu - y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
        stats = dict(round=rnd, beta=beta, samples=n, loss=float(np.mean(losses[-50:])), time_min=round((time.time() - t0) / 60, 1),
                     episodes=len(eps), success=float(np.mean([e["success"] for e in eps])) if eps else None,
                     collision=float(np.mean([e["collision"] for e in eps])) if eps else None,
                     timeout=float(np.mean([e["timeout"] for e in eps])) if eps else None)
        print(json.dumps(stats), flush=True)
        log.write(json.dumps(stats) + "\n")
        log.flush()
        agent.save(os.path.join(a.out, f"r{rnd:02d}.pt"), {"round": rnd})
        agent.export(os.path.join(a.out, f"r{rnd:02d}.ts"), ACT_OBS)


if __name__ == "__main__":
    main()
