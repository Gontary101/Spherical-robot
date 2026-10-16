"""
GYRA Mk2.1 build assembly: builds every subassembly, applies all bolted joints (holes cut, fasteners placed), and
writes the build outputs to cad/build/out/:

    mass_properties.json   per part / per body / robot, same format as the concept CAD (the simulator can load it:
                           sim/gyra2_model.py mass_props(cad="build/out"))
    joints.json            every joint with its chosen screw, grip, engagement and members
    parts.json             every part: body, kind, material / process / part number

    python cad/build/assemble.py            # build + outputs
    python cad/build/checks.py              # verification (interference over poses, joints, assembly sequence)
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from bob import build_bob  # noqa: E402
from lib import apply_joints, combine, mass_props  # noqa: E402
from pod import build_pod  # noqa: E402
from spine import build_spine  # noqa: E402
from tyre import build_tyre_half  # noqa: E402
from yoke import build_yoke  # noqa: E402

OUT = os.path.join(HERE, "out")
# meshing / rolling pairs: contact by design, checked analytically (gear centre distances, wheel preload)
MESH = {frozenset(p) for p in (("pinion_shaft_F", "ring_gear_L"), ("pinion_shaft_B", "ring_gear_R"), ("level_pinion", "level_gear"),
                               ("lean_pinion", "lean_rack"))}


def build_all(verbose=True):
    t0 = time.time()
    parts, joints, threads, bonds = [], [], [], []
    jsub = {}
    for fn, args in ((build_tyre_half, (1,)), (build_tyre_half, (-1,)), (build_pod, (1,)), (build_pod, (-1,)),
                     (build_spine, ()), (build_yoke, ()), (build_bob, ())):
        out = fn(*args)
        P, J = out[0], out[1]
        extra = out[2] if len(out) > 2 else []
        sub = fn.__name__.replace("build_", "") + ("" if not args else ("_L" if args[0] > 0 else "_R"))
        for p in P:
            p.spec.setdefault("sub", sub)
        for j in J:
            jsub[j.name] = sub
        parts += P
        joints += J
        if fn is build_tyre_half:
            bonds += extra
        else:
            threads += extra
        if verbose:
            print(f"  {fn.__name__}{args}: {len(P)} parts, {len(J)} joints  ({time.time() - t0:.0f} s)", flush=True)
    names = [p.name for p in parts]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise RuntimeError(f"duplicate part names: {sorted(dup)}")
    D = {p.name: p for p in parts}
    fasteners = apply_joints(D, joints)
    for f in fasteners:
        jn = f.spec.get("joint")
        f.spec["sub"] = jsub.get(jn, "?")
    parts += fasteners
    if verbose:
        print(f"  joints applied: {len(joints)} joints, {len(fasteners)} fastener parts ({time.time() - t0:.0f} s)", flush=True)
    return parts, joints, threads, bonds


def write_outputs(parts, joints, threads, bonds):
    os.makedirs(OUT, exist_ok=True)
    mp = {p.name: {**mass_props(p), "group": p.body} for p in parts}
    bodies = {b: combine([mp[p.name] for p in parts if p.body == b]) for b in sorted({p.body for p in parts})}
    robot = combine(list(mp.values()))
    json.dump({"robot": robot, "bodies": bodies, "parts": mp}, open(os.path.join(OUT, "mass_properties.json"), "w"), indent=1)
    json.dump([dict(name=j.name, step=j.step, grade=j.grade, **j.info) for j in joints], open(os.path.join(OUT, "joints.json"), "w"), indent=1)
    json.dump([dict(name=p.name, body=p.body, kind=p.kind, mass=round(mp[p.name]["mass"], 5), **p.spec) for p in parts],
              open(os.path.join(OUT, "parts.json"), "w"), indent=1, default=str)
    json.dump(dict(threads=[list(t) for t in threads], bonds=[list(b) for b in bonds]), open(os.path.join(OUT, "connections.json"), "w"), indent=1)
    return robot, bodies


def main():
    parts, joints, threads, bonds = build_all()
    robot, bodies = write_outputs(parts, joints, threads, bonds)
    for b, v in bodies.items():
        print(f"  {b:6s} m={v['mass']:6.2f} kg  com={[round(c, 1) for c in v['com_mm']]}")
    print(f"  ROBOT  m={robot['mass']:6.2f} kg  com={[round(c, 1) for c in robot['com_mm']]}   parts={len(parts)}  joints={len(joints)}")


if __name__ == "__main__":
    main()
