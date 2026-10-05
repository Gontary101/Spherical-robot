"""
Asymmetric actor-critic PPO (Pinto et al. 2017 style).

The actor sees only deployable observations; the critic sees privileged simulator state.
Only the actor (+ its observation normaliser) is exported for the robot.
"""
import numpy as np
import torch
import torch.nn as nn


class RunningNorm:
    def __init__(self, n, clip=8.0):
        self.mean = np.zeros(n)
        self.var = np.ones(n)
        self.count = 1e-4
        self.clip = clip

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip).astype(np.float32)

    def state(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": float(self.count)}

    def load(self, s):
        self.mean, self.var, self.count = np.array(s["mean"]), np.array(s["var"]), s["count"]


def mlp(i, o, hidden=(256, 256, 128), act=nn.ELU):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, hidden=(256, 256, 128), init_std=0.5):
        super().__init__()
        self.mu = mlp(n_obs, n_act, hidden)
        self.log_std = nn.Parameter(torch.full((n_act,), float(np.log(init_std))))
        with torch.no_grad():
            self.mu[-1].weight.mul_(0.01)
            self.mu[-1].bias.zero_()

    def dist(self, obs):
        return torch.distributions.Normal(self.mu(obs), self.log_std.exp())

    def forward(self, obs):          # deterministic, for export
        return torch.clamp(self.mu(obs), -1, 1)


class PPO:
    def __init__(self, n_actor, n_critic, n_act, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=5,
                 minibatches=4, ent=0.003, vf_coef=1.0, max_grad=1.0, target_kl=0.01, hidden=(256, 256, 128),
                 init_std=0.5):
        self.actor = Actor(n_actor, n_act, hidden, init_std)
        self.critic = mlp(n_critic, 1, hidden)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.lr, self.gamma, self.lam, self.clip = lr, gamma, lam, clip
        self.epochs, self.mb, self.ent, self.vf, self.max_grad, self.target_kl = epochs, minibatches, ent, vf_coef, max_grad, target_kl
        self.na, self.nc = RunningNorm(n_actor), RunningNorm(n_critic)

    @torch.no_grad()
    def act(self, oa, oc, deterministic=False):
        a_in = torch.as_tensor(self.na(oa))
        c_in = torch.as_tensor(self.nc(oc))
        d = self.actor.dist(a_in)
        a = d.mean if deterministic else d.sample()
        return a.numpy(), d.log_prob(a).sum(-1).numpy(), self.critic(c_in).squeeze(-1).numpy()

    @torch.no_grad()
    def value(self, oc):
        return self.critic(torch.as_tensor(self.nc(oc))).squeeze(-1).numpy()

    def update(self, buf):
        T, N = buf["rew"].shape
        adv = np.zeros((T, N), dtype=np.float32)
        last = 0
        for t in reversed(range(T)):
            nonterm = 1.0 - buf["done"][t]
            nv = buf["next_val"][t]
            delta = buf["rew"][t] + self.gamma * nv * buf["boot"][t] - buf["val"][t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        ret = adv + buf["val"]
        flat = lambda x: x.reshape(T * N, *x.shape[2:])  # noqa: E731
        oa, oc = torch.as_tensor(flat(buf["oa_n"])), torch.as_tensor(flat(buf["oc_n"]))
        act, logp_old = torch.as_tensor(flat(buf["act"])), torch.as_tensor(flat(buf["logp"]))
        adv_t, ret_t, val_old = torch.as_tensor(flat(adv)), torch.as_tensor(flat(ret)), torch.as_tensor(flat(buf["val"]))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        n = T * N
        mbs = n // self.mb
        stats = {"pi_loss": 0, "v_loss": 0, "kl": 0, "ent": 0, "clipfrac": 0}
        cnt = 0
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(self.mb):
                idx = perm[i * mbs:(i + 1) * mbs]
                d = self.actor.dist(oa[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                ratio = (logp - logp_old[idx]).exp()
                s1 = ratio * adv_t[idx]
                s2 = ratio.clamp(1 - self.clip, 1 + self.clip) * adv_t[idx]
                pi_loss = -torch.min(s1, s2).mean()
                v = self.critic(oc[idx]).squeeze(-1)
                v_cl = val_old[idx] + (v - val_old[idx]).clamp(-self.clip, self.clip)
                v_loss = torch.max((v - ret_t[idx]) ** 2, (v_cl - ret_t[idx]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pi_loss + self.vf * v_loss - self.ent * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad)
                self.opt.step()
                with torch.no_grad():
                    kl = (logp_old[idx] - logp).mean().item()
                    stats["pi_loss"] += pi_loss.item()
                    stats["v_loss"] += v_loss.item()
                    stats["kl"] += kl
                    stats["ent"] += ent.item()
                    stats["clipfrac"] += ((ratio - 1).abs() > self.clip).float().mean().item()
                    cnt += 1
            # adaptive learning rate on KL
            if stats["kl"] / cnt > 2 * self.target_kl:
                self.lr = max(self.lr / 1.5, 1e-5)
            elif stats["kl"] / cnt < self.target_kl / 2:
                self.lr = min(self.lr * 1.5, 1e-3)
            for g in self.opt.param_groups:
                g["lr"] = self.lr
        return {k: v / cnt for k, v in stats.items()} | {"lr": self.lr}

    def save(self, path, extra=None):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "na": self.na.state(), "nc": self.nc.state(), "extra": extra or {}}, path)

    def load(self, path):
        ck = torch.load(path, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.na.load(ck["na"])
        self.nc.load(ck["nc"])
        return ck.get("extra", {})


def rollout(env, agent, oa, oc, T):
    """Collect T steps from a VecEnv. Handles time-limit bootstrapping via final critic obs."""
    N = env.n
    buf = {k: [] for k in ("oa_n", "oc_n", "act", "logp", "val", "rew", "done", "boot", "next_val")}
    ep_info = []
    for _ in range(T):
        agent.na.update(oa)
        agent.nc.update(oc)
        a, logp, v = agent.act(oa, oc)
        buf["oa_n"].append(agent.na(oa))
        buf["oc_n"].append(agent.nc(oc))
        oa2, oc2, r, d, infos = env.step(np.clip(a, -1, 1))
        nv = agent.value(oc2)
        boot = np.ones(N, dtype=np.float32)
        for i, inf in enumerate(infos):
            if d[i]:
                if inf.get("timeout"):
                    nv[i] = agent.value(inf["final_obs_critic"][None])[0]
                else:
                    boot[i] = 0.0
                    nv[i] = 0.0
            ep_info.append(inf)
        buf["act"].append(a.astype(np.float32))
        buf["logp"].append(logp.astype(np.float32))
        buf["val"].append(v.astype(np.float32))
        buf["rew"].append(r)
        buf["done"].append(d.astype(np.float32))
        buf["boot"].append(boot)
        buf["next_val"].append(nv.astype(np.float32))
        oa, oc = oa2, oc2
    return {k: np.array(v) for k, v in buf.items()}, ep_info, oa, oc
