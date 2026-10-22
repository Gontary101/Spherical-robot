"""
One evaluation pipeline per low level (docs/12, P0): a candidate is always judged with its OWN response model,
braking / reach tables and error envelope - stale tables made two earlier comparisons wrong.

    python rl/lowlevel_pipeline.py --name loco31 --low rl/runs/loco31/best.pt --base classical
    python rl/lowlevel_pipeline.py --name loco34 --low rl/runs/loco34/final.pt --base model --tag calm   # + variant env
    python rl/lowlevel_pipeline.py --compare loco31 loco34_calm

Stages (each skipped when its output exists, unless --force): response model (rl/resp_model.py --identify), braking
and reach tables (rl/sysid_low.py --stops), error envelope of the response model's predictions (--arcs-only),
held-out and standard navigation benchmarks (rl/diagnose_nav.py, as-built sensors with lane cameras, 13 scenarios x
16 episodes each). A --tag names a variant: identification is shared, the benchmarks rerun with the current
environment (e.g. a filter switch), which is recorded in the report. --compare pairs two reports episode by episode
and gives the success difference with a paired bootstrap 95 % interval.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
B = os.path.join(ROOT, "rl/runs/nav2/bench")
SETS = {"heldout": "100", "standard": "0"}
PASS_ENV = ("GYRA_", "MPC_", "FILTER_")         # variant switches recorded with a report


def sh(cmd, env, log):
    t = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"failed ({r.returncode}): {' '.join(cmd)} - see {log}")
    return time.time() - t


def run(a):
    tag = f"{a.name}_{a.tag}" if a.tag else a.name
    low = os.path.abspath(a.low)
    env = dict(os.environ, GYRA_LOW=low, GYRA_BASE=a.base, GYRA_SENSORS="asbuilt_lane", GYRA_REACH="1")
    resp, sysid = f"{B}/resp_{a.name}.json", f"{B}/sysid_{a.name}_pipe.json"
    py = sys.executable
    stages = []
    if a.force or not os.path.exists(resp):
        env_id = {k: v for k, v in env.items() if k != "GYRA_RESP"}
        stages.append(("response model", sh([py, "rl/resp_model.py", "--identify", "--validate", "--out", resp], env_id,
                                            f"{B}/pipe_{a.name}_resp.log")))
    env["GYRA_RESP"] = resp
    if a.force or not os.path.exists(sysid):
        src = json.load(open(f"{B}/sysid.json"))              # carries the static fields the tools expect
        json.dump(src, open(sysid, "w"))
        stages.append(("braking + reach tables", sh([py, "rl/sysid_low.py", "--stops", "--out", sysid], env,
                                                    f"{B}/pipe_{a.name}_stops.log")))
        stages.append(("error envelope", sh([py, "rl/sysid_low.py", "--arcs-only", "--out", sysid], env,
                                            f"{B}/pipe_{a.name}_arcs.log")))
    env["GYRA_SYSID"] = sysid
    rep = dict(name=tag, low=os.path.relpath(low, ROOT), base=a.base, resp=os.path.relpath(resp, ROOT),
               sysid=os.path.relpath(sysid, ROOT), env={k: v for k, v in env.items() if k.startswith(PASS_ENV)
                                                         and k not in ("GYRA_LOW", "GYRA_RESP", "GYRA_SYSID")},
               validation=open(f"{B}/pipe_{a.name}_resp.log").read().strip().splitlines()[-2:]
               if os.path.exists(f"{B}/pipe_{a.name}_resp.log") else None, sets={})
    for s, seed0 in SETS.items():
        out = f"{B}/pipe_{tag}_{s}.json"
        if a.force or a.rerun or not os.path.exists(out):
            stages.append((f"{s} benchmark", sh([py, "rl/diagnose_nav.py", "--nav", "mpc", "--shield", "--episodes",
                                                str(a.episodes), "--procs", "4", "--low", low, "--out", out],
                                               dict(env, GYRA_SEED0=seed0), f"{B}/pipe_{tag}_{s}.log")))
        rep["sets"][s] = dict(file=os.path.relpath(out, ROOT), **summary(json.load(open(out))))
    rep["stages_s"] = {k: round(v) for k, v in stages}
    json.dump(rep, open(f"{B}/pipeline_{tag}.json", "w"), indent=1)
    for s, r in rep["sets"].items():
        print(f"{tag:16s} {s:8s} " + fmt(r))


def summary(E):
    n = len(E)
    fail = [e.get("failure") for e in E]
    col = [e for e in E if e.get("failure") == "collision"]
    return dict(episodes=n, success=sum(bool(e["success"]) for e in E) / n, collisions=len(col) / n,
                at_fault=sum(bool(e.get("at_fault")) for e in col), timeouts=fail.count("timeout") / n,
                falls=fail.count("fall") / n)


def fmt(r):
    return (f"success {r['success']:6.1%}  collisions {r['collisions']:5.1%} (at fault {r['at_fault']})  "
            f"timeouts {r['timeouts']:5.1%}  falls {r['falls']:4.1%}  [{r['episodes']}]")


def compare(a_name, b_name, n_boot=4000):
    A, Bn = (json.load(open(f"{B}/pipeline_{n}.json")) for n in (a_name, b_name))
    rng = np.random.default_rng(0)
    allx, ally = [], []
    for s in SETS:
        if s not in A["sets"] or s not in Bn["sets"]:
            continue
        ea = {(e["scenario"], e["ep"]): e for e in json.load(open(os.path.join(ROOT, A["sets"][s]["file"])))}
        eb = {(e["scenario"], e["ep"]): e for e in json.load(open(os.path.join(ROOT, Bn["sets"][s]["file"])))}
        keys = sorted(set(ea) & set(eb))
        x = np.array([float(ea[k]["success"]) for k in keys])
        y = np.array([float(eb[k]["success"]) for k in keys])
        allx.append(x)
        ally.append(y)
        print(f"{s:8s} {a_name}: {fmt(A['sets'][s])}")
        print(f"{'':8s} {b_name}: {fmt(Bn['sets'][s])}")
    x, y = np.concatenate(allx), np.concatenate(ally)
    d = y - x
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"paired success difference {b_name} - {a_name}: {d.mean():+.1%} (95 % interval {lo:+.1%} .. {hi:+.1%}), "
          f"{len(d)} episode pairs; {int((d > 0).sum())} won, {int((d < 0).sum())} lost")
    return d.mean(), lo, hi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--low")
    ap.add_argument("--base", default="classical")
    ap.add_argument("--tag", default="")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="redo every stage")
    ap.add_argument("--rerun", action="store_true", help="redo the benchmarks only")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        run(a)
