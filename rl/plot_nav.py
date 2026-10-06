"""Top-down paths of the learned navigator vs the VFH baseline on a few evaluation maps.

    python rl/plot_nav.py  -> media/rl_nav_paths.png
"""
import os
import sys

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from eval_nav import VFHBaseline  # noqa: E402
from nav_env import ARENA, NavEnv, NumpyActor  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"learned": "#2a78d6", "vfh": "#eb6834"}


def run(env, kind, nav, seed):
    env.rng = np.random.default_rng(seed)
    env.loco.rng = np.random.default_rng(seed)
    oa, oc = env.reset()
    path = [env.d.qpos[0:2].copy()]
    while True:
        a = nav(oa) if kind == "learned" else nav(env)
        (oa, oc), r, done, info = env.step(a)
        path.append(env.d.qpos[0:2].copy())
        if done:
            return np.array(path), info


def obstacles(env):
    out = []
    m = env.m
    for gid in env.obs_gid:
        p = m.geom_pos[gid]
        if p[0] > 100:
            continue
        if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            out.append(Circle(p[:2], m.geom_size[gid][0]))
        else:
            sx, sy = m.geom_size[gid][:2]
            q = m.geom_quat[gid]
            yaw = 2 * np.arctan2(q[3], q[0])
            c, s = np.cos(yaw), np.sin(yaw)
            corners = np.array([[sx, sy], [-sx, sy], [-sx, -sy], [sx, -sy]]) @ np.array([[c, s], [-s, c]]) + p[:2]
            out.append(Polygon(corners))
    return out


def main():
    loco = os.path.join(ROOT, "rl/runs/loco/best.pt")
    learned = NumpyActor(os.path.join(ROOT, "rl/runs/nav/best.pt"))
    env = NavEnv(seed=12345, low_level="learned", loco_ckpt=loco)
    seeds = [9003, 9011, 9024, 9037]
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.2 * len(seeds), 4.6), dpi=140)
    fig.patch.set_facecolor(SURF)
    for ax, sd in zip(axes, seeds):
        res = {}
        for kind, nav in (("vfh", VFHBaseline()), ("learned", learned)):
            res[kind] = run(env, kind, nav, sd)
        for patch in obstacles(env):
            patch.set_facecolor("#c9c4bb")
            patch.set_edgecolor("none")
            ax.add_patch(patch)
        for kind, (path, info) in res.items():
            outcome = "goal" if info["success"] else ("collision" if info["collision"] else ("fall" if info["fell"] else "timeout"))
            ax.plot(path[:, 0], path[:, 1], color=COL[kind], lw=2,
                    label=f"{'learned' if kind == 'learned' else 'VFH baseline'}: {outcome}, {len(path) * 0.1:.1f} s")
        ax.plot(*env.start, "o", color=INK, ms=6)
        ax.plot(*env.goal, "*", color=INK, ms=13)
        ax.set_xlim(-ARENA, ARENA)
        ax.set_ylim(-ARENA, ARENA)
        ax.set_aspect("equal")
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.6)
        [s_.set_visible(False) for s_ in ax.spines.values()]
        ax.tick_params(colors=INK2, labelsize=7.5)
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower center", bbox_to_anchor=(0.5, -0.24))
        ax.set_title(f"map {sd}", color=INK, fontsize=10, loc="left")
    fig.suptitle("Autonomous navigation: learned asymmetric-PPO navigator vs VFH baseline (● start, ★ goal)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "rl_nav_paths.png"), facecolor=SURF, bbox_inches="tight")
    print("wrote media/rl_nav_paths.png")


if __name__ == "__main__":
    main()
