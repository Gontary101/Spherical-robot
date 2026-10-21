"""
Navigation failure taxonomy (plan phase 0).

Runs a navigator on the benchmark scenarios and classifies every failure:
  collision   static / pedestrian; robot speed at impact; contact bearing (front / side / rear); at-fault (robot moving
              toward the contact) vs struck while (nearly) stopped; whether the obstacle had been PERCEIVED in the last
              1 s: by the masked LiDAR (beam toward the contact reported it), by the depth camera (inside its field of
              view and range), or not at all - and if not, why (pods' blind wedge, dead LiDAR sector, beyond range)
  timeout     remaining geodesic distance, time spent nearly stopped, heading reversals (dithering)
  fall        locomotion failure
Contacts of the sensor pods count as collisions too (the env counted only the tyres).

    python rl/diagnose_nav.py --nav rl/runs/nav2/actor.ts --episodes 32
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from collections import Counter, deque

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_nav2 import CALM, LOCO, SCEN, make  # noqa: E402
from nav2_env import MAX_R, N_BEAM, Nav2Env  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEED0 = int(os.environ.get("GYRA_SEED0", 0))      # held-out episodes: GYRA_SEED0=100 (the benchmark uses 0..15)


def contact_info(env):
    """(kind, world contact point) of the first robot-obstacle contact, pods included."""
    m, d = env.m, env.d
    robot = set(env.tyre_gid) | set(env.loco.pods)
    for c in d.contact[:d.ncon]:
        a, b = int(c.geom[0]), int(c.geom[1])
        if a in robot or b in robot:
            o = b if a in robot else a
            if o in env.obs_gid:
                return "static", c.pos[:2].copy(), m.geom(o).name
            if o in env.dyn_gid:
                return "pedestrian", c.pos[:2].copy(), m.geom(o).name
    return None, None, None


def run(args):
    scen, path, low, n, shield = args
    os.environ["GYRA_LOW"] = low
    nav = make("nav2", path) if path not in ("vfh", "oracle", "mpc") else make(path, None)
    out = []
    for ep in range(n):
        env = Nav2Env(seed=70000 + SEED0 + ep, low_ckpt=low, level=0.8, adaptive=False, scenario={"nav": CALM | SCEN[scen], "loco": LOCO},
                      shield=shield)
        oa, oc = env.reset()
        hist = deque(maxlen=10)
        maps = deque(maxlen=30)                         # shield: map snapshots of the last 3 s, for "when was it known"
        stopped, revs, last_w = 0, 0, 0.0
        while True:
            a = nav(oa, env)
            (oa, oc), r, done, info = env.step(a)
            pos, yaw, t = env._pose()
            hist.append(dict(pos=pos.copy(), yaw=yaw, meas=env.scans[-1].copy(), true=env._true_scan(pos, yaw),
                             depth_dead=bool(env.P["depth_blackout"]), interv=bool(info.get("intervened", False)),
                             cmd=env.loco.cmd.copy(), v=float(t["v"])))
            if env.shield:
                maps.append((env.lmap.L.copy(), env.lmap.origin.copy(), env.loc.copy(), pos.copy(), yaw,
                             float(env.loco.cmd[0]), float(t["v"])))
            stopped += abs(t["v"]) < 0.1
            if t["w"] * last_w < 0 and abs(t["w"]) > 0.3:
                revs += 1
            last_w = t["w"] if abs(t["w"]) > 0.3 else last_w
            kind, cp, name = contact_info(env)
            if kind is not None and not done:          # a pod contact the env did not count: end it here
                done, info = True, dict(info, collision=True)
            if done:
                break
        rec = dict(scenario=scen, ep=ep, success=bool(info["success"]), time=env.steps * 0.1)
        if info.get("collision") or kind is not None:
            kind, cp, name = contact_info(env)
            pos, yaw, t = env._pose()
            vel = env.d.qvel[0:2]
            if cp is None:
                cp, kind, name = pos + 0.39 * np.array([math.cos(yaw), math.sin(yaw)]), "static", "?"
            rel = cp - pos
            bearing = math.degrees((math.atan2(rel[1], rel[0]) - yaw + math.pi) % (2 * math.pi) - math.pi)
            toward = float(vel @ rel / max(np.linalg.norm(rel), 1e-6))
            # was it perceived during the last second?
            seen_lidar = seen_depth = 0
            why = Counter()
            for h in hist:
                relh = cp - h["pos"]
                dist = float(np.linalg.norm(relh))
                ang = (math.atan2(relh[1], relh[0]) - h["yaw"] + math.pi) % (2 * math.pi) - math.pi
                k = int(round((ang + math.pi) / (2 * math.pi) * N_BEAM)) % N_BEAM
                ks = [(k + j) % N_BEAM for j in (-1, 0, 1)]
                if min(h["meas"][ks]) < min(dist + 0.25, MAX_R - 0.01):
                    seen_lidar += 1
                else:
                    px, py = dist * math.cos(ang), dist * math.sin(ang)
                    if not env.visible(np.array([px]), np.array([py]))[0]:
                        why["blind_wedge"] += 1
                    elif env.lidar_sector is not None:
                        why["dead_sector"] += 1
                    elif min(h["true"][ks]) > dist + 0.25:
                        why["thin_or_low"] += 1
                    else:
                        why["noise_dropout"] += 1
                if abs(ang) < math.radians(45) and 0.35 < dist < MAX_R and not h["depth_dead"]:
                    seen_depth += 1
            perceived = "lidar" if seen_lidar >= 3 else ("depth_only" if seen_depth >= 3 else "unseen")
            if env.shield:
                # shield forensics: did the map hold the obstacle, was the filter braking, what did it command?
                dth = env.loc[2] - yaw
                c_, s_ = math.cos(dth), math.sin(dth)
                cpo = env.loc[:2] + np.array([c_ * rel[0] - s_ * rel[1], s_ * rel[0] + c_ * rel[1]])
                # when did the contact point enter the map (occupied cell within 0.10 m), and when did braking start?
                known = []
                from local_map import OCC_TH
                for L_, org, odo_, pos_, yaw_, cv, vv in maps:
                    d_ = odo_[2] - yaw_
                    rel_ = cp - pos_
                    c2, s2 = math.cos(d_), math.sin(d_)
                    q = odo_[:2] + np.array([c2 * rel_[0] - s2 * rel_[1], s2 * rel_[0] + c2 * rel_[1]])
                    ij = np.floor((q - org) / env.lmap.cell).astype(int)
                    k2 = 2
                    win = L_[max(ij[1] - k2, 0):ij[1] + k2 + 1, max(ij[0] - k2, 0):ij[0] + k2 + 1]
                    known.append(bool(win.size and (win > OCC_TH).any()))
                first_known = next((i for i, k in enumerate(known) if k), None)
                n = len(maps)
                brake = next((i for i in range(n - 1, -1, -1) if maps[i][5] > 0.05), -1) + 1   # last step with cmd v > 0
                rec_sh = dict(known_s_before=(n - first_known) * 0.1 if first_known is not None else 0.0,
                              brake_s_before=(n - brake) * 0.1, v_at_brake=maps[min(brake, n - 1)][6],
                              dist_at_brake=float(np.linalg.norm(cp - maps[min(brake, n - 1)][3])),
                              map_dist=float(env.lmap.clearance(cpo[None])[0]),
                              interv_1s=int(sum(h["interv"] for h in hist)),
                              cmd_last=[round(float(x), 2) for x in hist[-1]["cmd"]],
                              v_hist=[round(h["v"], 2) for h in hist],
                              cmd_v_hist=[round(float(h["cmd"][0]), 2) for h in hist],
                              push=bool(env.loco.push_left > -50))           # a push in the last 1 s
            else:
                rec_sh = {}
            rec |= dict(failure="collision", kind=kind, obstacle=name, speed=float(np.linalg.norm(vel)), toward=toward,
                        at_fault=bool(toward > 0.1 or kind == "static"), bearing=bearing,
                        side="front" if abs(bearing) < 45 else ("rear" if abs(bearing) > 135 else "side"),
                        perceived=perceived, unseen_reason=(why.most_common(1)[0][0] if why else "-")) | rec_sh
        elif info.get("fell"):
            rec |= dict(failure="fall")
        elif info.get("timeout"):
            geo, _ = env._geo_at(env._pose()[0])
            rec |= dict(failure="timeout", geo_left=float(geo), stopped_frac=stopped / env.steps, w_reversals=revs)
        else:
            rec |= dict(failure=None)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", default=os.path.join(ROOT, "rl/runs/nav2/actor.ts"), help="TorchScript actor, or vfh / oracle")
    ap.add_argument("--low", default=os.path.join(ROOT, "rl/runs/loco31/best.pt"))
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--shield", action="store_true")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "rl/runs/nav2/bench/failure_taxonomy.json"))
    a = ap.parse_args()
    jobs = [(s, a.nav, os.path.abspath(a.low), a.episodes, a.shield) for s in (a.scenarios or SCEN)]
    with mp.get_context("fork").Pool(a.procs) as pool:
        recs = sum(pool.map(run, jobs, chunksize=1), [])
    json.dump(recs, open(a.out, "w"), indent=1)
    N = len(recs)
    F = [r for r in recs if r["failure"]]
    print(f"episodes {N}: success {sum(r['success'] for r in recs) / N:.1%}, failures {len(F)}")
    print("failure types:", Counter(r["failure"] for r in F))
    C = [r for r in F if r["failure"] == "collision"]
    if C:
        print(f"collisions {len(C)} ({len(C) / N:.1%} of episodes):")
        for key in ("kind", "side", "perceived", "unseen_reason", "at_fault"):
            print(f"  {key:14s}", dict(Counter(r[key] for r in C)))
        sp = np.array([r["speed"] for r in C])
        print(f"  speed at impact: median {np.median(sp):.2f} m/s, < 0.2 m/s: {np.mean(sp < 0.2):.0%}, > 1 m/s: {np.mean(sp > 1):.0%}")
        print("  by scenario:", dict(Counter(r["scenario"] for r in C)))
        for r in C:
            if "map_dist" in r:
                print(f"   {r['scenario'][:14]:14s} ep {r['ep']:2d} {r['side']:5s} v {r['speed']:.2f} | in map {r['known_s_before']:.1f} s "
                      f"before, braking {r['brake_s_before']:.1f} s before at {r['v_at_brake']:.2f} m/s, {r['dist_at_brake']:.2f} m away | map_dist "
                      f"{r['map_dist']:.2f} interv {r['interv_1s']} push {r['push']} cmd {r['cmd_last']} v {r['v_hist']} "
                      f"cmd_v {r['cmd_v_hist']}")
    T = [r for r in F if r["failure"] == "timeout"]
    if T:
        print(f"timeouts {len(T)}: geodesic left median {np.median([r['geo_left'] for r in T]):.1f} m, "
              f"stopped {np.median([r['stopped_frac'] for r in T]):.0%} of the time, yaw-rate reversals median "
              f"{np.median([r['w_reversals'] for r in T]):.0f}; by scenario {dict(Counter(r['scenario'] for r in T))}")


if __name__ == "__main__":
    main()
