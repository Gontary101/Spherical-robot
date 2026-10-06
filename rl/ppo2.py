"""
PPO v2: asymmetric actor-critic with a proprioceptive history encoder, a concurrent state estimator and a
left/right symmetry loss.

Actor (deployable):
    history (HIST x FRAME, 1 s) --TCN--> latent z (64)
    z --linear--> estimate of privileged quantities (body velocity, local slope, friction, payload, wind,
                  motor strength); trained by regression on the simulator's true values (Ji et al. 2022)
    [current frame, command, z, estimate] --MLP 256-128-64--> action mean;  state-independent log-std
Critic (training only): MLP 512-256-128 on privileged state + terrain scan.

Losses: clipped PPO surrogate + clipped value loss + estimator MSE + symmetry MSE
(pi(mirror(o)) = mirror(pi(o)), Mittal et al. 2024) - entropy. Adaptive-KL learning rate, running observation
normalisation, return-scale normalisation, time-limit bootstrapping.

Export: one TorchScript module with the observation normaliser folded in (raw sensor vector in, action out).
"""
import numpy as np
import torch
import torch.nn as nn

from ppo import RunningNorm, mlp


class HistoryEncoder(nn.Module):
    def __init__(self, hist, frame, out=64):
        super().__init__()
        self.hist, self.frame = hist, frame
        self.conv = nn.Sequential(
            nn.Conv1d(frame, 32, 5, stride=2), nn.ELU(),
            nn.Conv1d(32, 32, 5, stride=2), nn.ELU(),
            nn.Conv1d(32, 32, 3, stride=2), nn.ELU())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, frame, hist)).numel()
        self.fc = nn.Sequential(nn.Linear(n, out), nn.ELU())

    def forward(self, h):                      # h: (B, HIST, FRAME)
        return self.fc(self.conv(h.transpose(1, 2)).flatten(1))


class Actor2(nn.Module):
    def __init__(self, hist, frame, n_cmd, n_est, n_act, init_std=0.3, min_std=0.05, max_std=0.5):
        super().__init__()
        self.hist, self.frame, self.n_cmd = hist, frame, n_cmd
        self.enc = HistoryEncoder(hist, frame, 64)
        self.est = nn.Linear(64, n_est)
        self.pi = mlp(frame + n_cmd + 64 + n_est, n_act, (256, 128, 64))
        self.log_std = nn.Parameter(torch.full((n_act,), float(np.log(init_std))))
        self.lo, self.hi = float(np.log(min_std)), float(np.log(max_std))
        with torch.no_grad():
            self.pi[-1].weight.mul_(0.01)
            self.pi[-1].bias.zero_()

    def features(self, obs):
        B = obs.shape[0]
        h = obs[:, :self.hist * self.frame].reshape(B, self.hist, self.frame)
        cmd = obs[:, self.hist * self.frame:]
        z = self.enc(h)
        e = self.est(z)
        return torch.cat([h[:, -1], cmd, z, e], 1), e

    def mean(self, obs):
        x, e = self.features(obs)
        return self.pi(x), e

    def dist(self, obs):
        mu, e = self.mean(obs)
        return torch.distributions.Normal(mu, self.log_std.clamp(self.lo, self.hi).exp()), e


class Deployable(nn.Module):
    """Raw observation -> normalise -> actor mean -> clamp. This is what runs on the robot."""
    def __init__(self, actor, mean, std, clip):
        super().__init__()
        self.actor = actor
        self.register_buffer("m", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("s", torch.as_tensor(std, dtype=torch.float32))
        self.clip = clip

    def forward(self, obs):
        x = torch.clamp((obs - self.m) / self.s, -self.clip, self.clip)
        mu, _ = self.actor.mean(x)
        return torch.clamp(mu, -1, 1)


class Mirror:
    """Index permutation + sign flips implementing the left/right mirror of observations and actions."""
    def __init__(self, hist, frame, frame_signs, frame_perm, cmd_signs, act_signs):
        idx, sg = [], []
        for k in range(hist):
            idx += [k * frame + p for p in frame_perm]
            sg += list(frame_signs)
        base = hist * frame
        idx += [base + i for i in range(len(cmd_signs))]
        sg += list(cmd_signs)
        self.idx = torch.as_tensor(idx)
        self.sg = torch.as_tensor(np.array(sg, np.float32))
        self.asg = torch.as_tensor(np.array(act_signs, np.float32))

    def obs(self, o_raw):
        return o_raw[:, self.idx] * self.sg

    def act(self, a):
        return a * self.asg


class PPO2:
    def __init__(self, actor, n_actor, n_critic, n_est, mirror=None, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=5,
                 minibatches=4, ent=0.002, vf_coef=1.0, est_coef=1.0, sym_coef=0.5, max_grad=1.0, target_kl=0.01,
                 lr_max=1e-3, lr_min=1e-5):
        self.actor = actor
        self.critic = mlp(n_critic, 1, (512, 256, 128))
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.lr, self.gamma, self.lam, self.clip = lr, gamma, lam, clip
        self.epochs, self.mb, self.ent, self.vf = epochs, minibatches, ent, vf_coef
        self.est_coef, self.sym_coef, self.max_grad, self.target_kl = est_coef, sym_coef, max_grad, target_kl
        self.lr_max, self.lr_min = lr_max, lr_min
        self.n_est = n_est
        self.mirror = mirror
        self.na, self.nc = RunningNorm(n_actor), RunningNorm(n_critic, min_var=1e-2)
        self.ret_rms = RunningNorm(1)
        self.ret_acc = None

    # -------------------------------------------------------------------- acting
    @torch.no_grad()
    def act(self, oa, oc, deterministic=False):
        d, _ = self.actor.dist(torch.as_tensor(self.na(oa)))
        a = d.mean if deterministic else d.sample()
        v = self.critic(torch.as_tensor(self.nc(oc))).squeeze(-1).numpy()
        return a.numpy(), d.log_prob(a).sum(-1).numpy(), v

    @torch.no_grad()
    def value(self, oc):
        return self.critic(torch.as_tensor(self.nc(oc))).squeeze(-1).numpy()

    def scale_reward(self, r, done):
        """Divide rewards by the running std of the discounted return (keeps value targets O(1))."""
        if self.ret_acc is None:
            self.ret_acc = np.zeros_like(r)
        self.ret_acc = self.ret_acc * self.gamma + r
        self.ret_rms.update(self.ret_acc[:, None])
        self.ret_acc[done] = 0.0
        return r / np.sqrt(self.ret_rms.var[0] + 1e-8)

    # -------------------------------------------------------------------- learning
    def update(self, buf):
        T, N = buf["rew"].shape
        adv = np.zeros((T, N), dtype=np.float32)
        last = 0
        for t in reversed(range(T)):
            nonterm = 1.0 - buf["done"][t]
            delta = buf["rew"][t] + self.gamma * buf["next_val"][t] * buf["boot"][t] - buf["val"][t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        ret = adv + buf["val"]
        flat = lambda x: torch.as_tensor(x.reshape(T * N, *x.shape[2:]))  # noqa: E731
        oa, oc, oraw = flat(buf["oa_n"]), flat(buf["oc_n"]), flat(buf["oa_raw"])
        act, logp_old = flat(buf["act"]), flat(buf["logp"])
        adv_t, ret_t, val_old = flat(adv), flat(ret), flat(buf["val"])
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        est_tgt = flat(buf["oc_raw"])[:, :self.n_est]      # privileged targets, pre-scaled to O(1) by the env
        if self.mirror is not None:
            m_ = torch.as_tensor(self.na.mean, dtype=torch.float32)
            s_ = torch.as_tensor(np.sqrt(self.na.var + 1e-8), dtype=torch.float32)
            oa_mir = torch.clamp((self.mirror.obs(oraw) - m_) / s_, -self.na.clip, self.na.clip)
        n = T * N
        mbs = n // self.mb
        keys = ("pi_loss", "v_loss", "est_loss", "sym_loss", "kl", "ent", "clipfrac")
        stats = {k: 0.0 for k in keys}
        cnt = 0
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            kl_ep = 0.0
            for i in range(self.mb):
                idx = perm[i * mbs:(i + 1) * mbs]
                d, e = self.actor.dist(oa[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                ratio = (logp - logp_old[idx]).exp()
                pi_loss = -torch.min(ratio * adv_t[idx], ratio.clamp(1 - self.clip, 1 + self.clip) * adv_t[idx]).mean()
                v = self.critic(oc[idx]).squeeze(-1)
                v_cl = val_old[idx] + (v - val_old[idx]).clamp(-self.clip, self.clip)
                v_loss = torch.max((v - ret_t[idx]) ** 2, (v_cl - ret_t[idx]) ** 2).mean()
                est_loss = ((e - est_tgt[idx]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pi_loss + self.vf * v_loss + self.est_coef * est_loss - self.ent * ent
                sym_loss = torch.zeros(())
                if self.mirror is not None and self.sym_coef > 0:
                    mu_m, _ = self.actor.mean(oa_mir[idx])
                    sym_loss = ((mu_m - self.mirror.act(d.mean)) ** 2).mean()
                    loss = loss + self.sym_coef * sym_loss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, self.max_grad)
                self.opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - logp_old[idx])).mean().item()     # low-variance KL estimator
                    kl_ep += kl
                    for k, val in zip(keys, (pi_loss, v_loss, est_loss, sym_loss, kl, ent,
                                             ((ratio - 1).abs() > self.clip).float().mean())):
                        stats[k] += float(val)
                    cnt += 1
            kl_ep /= self.mb
            if kl_ep > 2 * self.target_kl:
                self.lr = max(self.lr / 1.5, self.lr_min)
            elif kl_ep < self.target_kl / 2:
                self.lr = min(self.lr * 1.5, self.lr_max)
            for g in self.opt.param_groups:
                g["lr"] = self.lr
            if kl_ep > 4 * self.target_kl:                 # early stop the epoch loop on a KL blow-up
                break
        return {k: v / max(cnt, 1) for k, v in stats.items()} | {"lr": self.lr}

    # -------------------------------------------------------------------- io
    def save(self, path, extra=None):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "opt": self.opt.state_dict(),
                    "na": self.na.state(), "nc": self.nc.state(), "ret": self.ret_rms.state(), "lr": self.lr,
                    "extra": extra or {}}, path)

    def load(self, path, optimizer=True):
        ck = torch.load(path, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        if optimizer and "opt" in ck:
            self.opt.load_state_dict(ck["opt"])
            self.lr = ck.get("lr", self.lr)
        self.na.load(ck["na"])
        self.nc.load(ck["nc"])
        if "ret" in ck:
            self.ret_rms.load(ck["ret"])
        return ck.get("extra", {})

    def export(self, path, n_actor):
        self.actor.eval()
        dep = Deployable(self.actor, self.na.mean, np.sqrt(self.na.var + 1e-8), self.na.clip)
        ts = torch.jit.trace(dep, torch.zeros(1, n_actor))
        ts.save(path)
        self.actor.train()


def rollout2(env, agent, oa, oc, T):
    N = env.n
    keys = ("oa_n", "oa_raw", "oc_n", "oc_raw", "act", "logp", "val", "rew", "raw_rew", "done", "boot", "next_val")
    buf = {k: [] for k in keys}
    infos_all = []
    for _ in range(T):
        agent.na.update(oa)
        agent.nc.update(oc)
        a, logp, v = agent.act(oa, oc)
        buf["oa_n"].append(agent.na(oa))
        buf["oa_raw"].append(oa.astype(np.float32))
        buf["oc_n"].append(agent.nc(oc))
        buf["oc_raw"].append(oc[:, :agent.n_est].astype(np.float32))
        oa2, oc2, r, d, infos = env.step(np.clip(a, -1, 1))
        rs = agent.scale_reward(r, d).astype(np.float32)
        nv = agent.value(oc2)
        boot = np.ones(N, dtype=np.float32)
        for i, inf in enumerate(infos):
            if d[i]:
                if inf.get("timeout"):
                    nv[i] = agent.value(inf["final_obs_critic"][None])[0]
                else:
                    boot[i], nv[i] = 0.0, 0.0
        infos_all += infos
        buf["act"].append(a.astype(np.float32))
        buf["logp"].append(logp.astype(np.float32))
        buf["val"].append(v.astype(np.float32))
        buf["rew"].append(rs)
        buf["raw_rew"].append(r)
        buf["done"].append(d.astype(np.float32))
        buf["boot"].append(boot)
        buf["next_val"].append(nv.astype(np.float32))
        oa, oc = oa2, oc2
    return {k: np.array(v) for k, v in buf.items()}, infos_all, oa, oc
