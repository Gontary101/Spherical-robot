"""Subprocess vector environment: W worker processes, each stepping K environments in a loop."""
import multiprocessing as mp

import numpy as np


def _worker(conn, make_env, seeds, env_kwargs):
    envs = [make_env(seed=s, **env_kwargs) for s in seeds]
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            obs = [e.reset() for e in envs]
            conn.send(([o[0] for o in obs], [o[1] for o in obs]))
        elif cmd == "step":
            A, C, R, D, I = [], [], [], [], []
            for e, a in zip(envs, data):
                (oa, oc), r, d, info = e.step(a)
                if d:
                    info["final_obs_critic"] = oc
                    oa, oc = e.reset()
                A.append(oa), C.append(oc), R.append(r), D.append(d), I.append(info)
            conn.send((A, C, R, D, I))
        elif cmd == "set":
            for e in envs:
                for k, v in data.items():
                    setattr(e, k, v)
            conn.send(True)
        elif cmd == "close":
            conn.close()
            return


class VecEnv:
    def __init__(self, make_env, n_workers=4, envs_per_worker=8, seed=0, **env_kwargs):
        ctx = mp.get_context("fork")
        self.pipes, self.procs = [], []
        self.k = envs_per_worker
        for w in range(n_workers):
            a, b = ctx.Pipe()
            seeds = [seed * 10000 + w * 100 + i for i in range(envs_per_worker)]
            p = ctx.Process(target=_worker, args=(b, make_env, seeds, env_kwargs), daemon=True)
            p.start()
            self.pipes.append(a)
            self.procs.append(p)
        self.n = n_workers * envs_per_worker

    def reset(self):
        for p in self.pipes:
            p.send(("reset", None))
        res = [p.recv() for p in self.pipes]
        return np.array(sum([r[0] for r in res], [])), np.array(sum([r[1] for r in res], []))

    def step(self, actions):
        for i, p in enumerate(self.pipes):
            p.send(("step", actions[i * self.k:(i + 1) * self.k]))
        res = [p.recv() for p in self.pipes]
        A = np.array(sum([r[0] for r in res], []))
        C = np.array(sum([r[1] for r in res], []))
        R = np.array(sum([r[2] for r in res], []), dtype=np.float32)
        D = np.array(sum([r[3] for r in res], []))
        infos = sum([r[4] for r in res], [])
        return A, C, R, D, infos

    def set_attr(self, **kw):
        for p in self.pipes:
            p.send(("set", kw))
        [p.recv() for p in self.pipes]

    def close(self):
        for p in self.pipes:
            p.send(("close", None))
        for p in self.procs:
            p.join(timeout=2)
