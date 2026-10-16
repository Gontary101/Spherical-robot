"""
Assembly sequence check: every subassembly is moved along the path it really takes into the robot, and at each
sample of the path it must not intersect anything already installed. Run after assemble.py.

    python cad/build/assembly.py [--step NAME]

The sequence (docs/11_build.md has the full procedure with torques):
  T  tyre halves (bench): drum infused with its ring gear co-bonded in the mould; the rim ring goes in through
     the seam opening (its OD 205 mm clears the drum interior everywhere) and is bonded
  1  axle + hub spacer
  2  yoke (bench-built, bob already run onto its track) slides onto the axle (bearings over the axle)
       2a bob onto the track: rotated in about x from beyond the track end (lean stops fitted afterwards)
  3  level bracket + motor onto the axle from +y, slip ring (rotor + housing) and sleeve from -y
  4  bay clamps, trays and electronics, GNSS plate
  5  tyre halves: each slides on seam-first, held 15 mm off-centre toward its pinion (so its ring gear clears the
     drive cassette), then moved radially into mesh
  6  pods: left V-wheels retracted on their 3.5 mm eccentric studs while the pod slides on, then set; right rollers
     clear the land by 0.3 mm and are preloaded on their 1 mm eccentrics
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from cadquery import Vector  # noqa: E402

from assemble import OUT, build_all  # noqa: E402
from checks import allowed_pairs, bb_hit, bbox  # noqa: E402

O = Vector(0, 0, 0)


def moved(p, dv=None, rot=None):
    s = p.shape
    if rot is not None:
        c, ax, ang = rot
        s = s.rotate(Vector(*c), Vector(*c) + Vector(*ax), ang)
    if dv is not None:
        s = s.translate(Vector(*dv))
    return p.copy(s)


def collide(mov, fixed, ok, tol=0.05):
    hits = []
    fb = [(f, bbox(f.shape)) for f in fixed]
    for m in mov:
        mb = bbox(m.shape)
        for f, b in fb:
            if frozenset((m.name, f.name)) in ok or not bb_hit(mb, b):
                continue
            v = m.shape.intersect(f.shape).Volume()
            if v > tol:
                hits.append((m.name, f.name, round(v, 2)))
    return hits


def sel(parts, pred):
    return [p for p in parts if pred(p)]


def run_path(name, mov, fixed, ok, poses, report):
    """poses: list of (dv, rot) applied to the moving set."""
    t0 = time.time()
    worst = []
    for k, (dv, rot) in enumerate(poses):
        M = [moved(p, dv, rot) for p in mov]
        h = collide(M, fixed, ok)
        if h:
            worst.append(dict(sample=k, dv=dv, rot=None if rot is None else rot[2], hits=h[:6]))
    report[name] = worst
    print(f"  {name:38s} {len(mov):4d} moving vs {len(fixed):4d} fixed, {len(poses):3d} samples: "
          f"{'CLEAR' if not worst else str(len(worst)) + ' blocked samples, e.g. ' + str(worst[0]['hits'][:3])}  ({time.time() - t0:.0f} s)",
          flush=True)


def fastener_names(j):
    return [j.name] + [f"{j.name}_{x}" for x in ("washer", "nut", "nwasher", "insert", "pem")]


def groups(parts, joints, threads, bonds, links=()):
    """Subassemblies = connected components of the joint / thread / bond graph, ignoring the axle (everything on the
    spine clamps to it) and the fasteners that join two different named groups."""
    names = {p.name for p in parts}
    par = {n: n for n in names}

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    def u(a, b):
        if a in par and b in par:
            par[f(a)] = f(b)
    for j in joints:
        mem = [m for m in j.info["members"] if m != "axle"]
        for m in mem[1:]:
            u(mem[0], m)
        for fn in fastener_names(j):
            if mem:
                u(fn, mem[0])
    for a, b in list(threads) + [(b[0], b[1]) for b in bonds] + list(links):
        if "axle" not in (a, b):
            u(a, b)
    G = {}
    for p in parts:
        G.setdefault(f(p.name), []).append(p)
    return list(G.values())


def main():
    only = sys.argv[sys.argv.index("--step") + 1] if "--step" in sys.argv else None
    parts, joints, threads, bonds = build_all(verbose=False)
    ok = allowed_pairs(parts, joints, threads)
    D = {p.name: p for p in parts}
    report = {}
    steps = []
    # the drive pin is pressed into the right hanger; the slip-ring housing slides onto it
    G = groups(parts, joints, [t for t in threads if set(t) != {"slipring_housing", "slipring_pin"}], bonds,
               links=[("level_pinion", "level_motor"), ("slipring_rotor", "slip_sleeve"), ("slipring_housing", "slip_sleeve"),
                      ("compute", "compute_standoff0"), ("compute_standoff0", "bay_tray_L"),
                      ("compute_standoff1", "bay_tray_L"), ("compute_standoff2", "bay_tray_L"), ("compute_standoff3", "bay_tray_L")]
               + [(f"wheel_stud_{s}{k}", f"pod_plate_{s}") for s in "LR" for k in range(8)])

    def grp(name):
        return next(g for g in G if name in {p.name for p in g})
    nm = lambda ps: {p.name for p in ps}  # noqa: E731

    # T: rim into the drum through the seam opening (ring gear co-bonded in the mould)
    for s, sg in (("L", 1), ("R", -1)):
        steps.append((f"T rim_{s} through the seam", [D[f"rim_{s}"]], [D[f"drum_{s}"], D[f"ring_gear_{s}"]],
                      [((0, -sg * d, 0), None) for d in range(330, -1, -10)]))

    # 2a: bob onto the track (yoke without lean stops), rotated in about x from 120 deg
    yoke_g = [p for p in parts if p.body == "yoke" and p.name not in nm(grp("slip_sleeve"))]
    bob = [p for p in parts if p.body == "bob"]
    stops = [p for p in yoke_g if p.name.startswith("lean_stop")]
    steps.append(("2a bob run onto the track (stops off)", bob, [p for p in yoke_g if p not in stops],
                  [(None, ((0, 0, 0), (1, 0, 0), a)) for a in range(120, -1, -2)]))

    # 2: yoke + bob + hub spacer (held between the bearings) onto the axle from +y
    yb = yoke_g + bob + [D["hub_spacer"]]
    steps.append(("2 yoke + bob onto the axle from +y", yb, [D["axle"]], [((0, d, 0), None) for d in range(420, -1, -5)]))
    installed = yb + [D["axle"]]

    # 3: level drive from +y, slip-ring group (sleeve + rotor + housing onto the yoke pin) from -y
    lvl = grp("level_bracket")
    steps.append(("3 level drive onto the axle from +y", lvl, list(installed), [((0, d, 0), None) for d in range(330, -1, -5)]))
    installed += lvl
    sr = grp("slip_sleeve")
    steps.append(("3 slip ring + sleeve from -y", sr, list(installed), [((0, -d, 0), None) for d in range(330, -1, -5)]))
    installed += sr

    # 4: bays (left: tray + clamps + compute + GNSS plate; right: tray + clamps + level controller + DC-DC + IMU)
    for s, sg in (("L", 1), ("R", -1)):
        bay = grp(f"bay_tray_{s}")
        steps.append((f"4 bay {s} from {'+' if sg > 0 else '-'}y", bay, list(installed),
                      [((0, sg * d, 0), None) for d in range(200, -1, -5)]))
        installed += bay

    # 5: tyre halves, offset 15 mm toward their pinion, seam first, then radially into mesh
    a = math.radians(S_DRIVE_ANGLE)
    for s, sg in (("L", 1), ("R", -1)):
        u = (sg * math.sin(a), -math.cos(a))
        off = 15.0
        mov = [D[f"drum_{s}"], D[f"ring_gear_{s}"], D[f"rim_{s}"]]
        poses = [((u[0] * off, sg * d, u[1] * off), None) for d in range(600, -1, -5)]
        poses += [((u[0] * off * f, 0, u[1] * off * f), None) for f in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)]
        steps.append((f"5 tyre {s}: offset slide + radial into mesh", mov, list(installed), poses))
    installed += [p for p in parts if p.body in ("tyreL", "tyreR") and not p.name.startswith("tread")]

    # 6: pods slide on (left V-wheels retracted 3.5 mm on their eccentric studs, then set)
    for s, sg in (("L", 1), ("R", -1)):
        pod = grp(f"pod_hub_{s}")
        rmap = {}
        if s == "L":
            for p in pod:
                m = re.match(r"(vwheel|wheel_retainer|wheel_screw)_L(\d)$", p.name)
                if m:
                    ang = math.radians(int(m.group(2)) * 360.0 / 8)
                    rmap[p.name] = (-3.5 * math.cos(ang), 0, -3.5 * math.sin(ang))
        fixed = list(installed)
        samples = list(range(260, -1, -5))
        worst = []
        t0 = time.time()
        for k, d in enumerate(samples):
            M = [moved(p, tuple(v + w for v, w in zip((0, sg * d, 0), rmap.get(p.name, (0, 0, 0)) if d > 0 else (0, 0, 0))))
                 for p in pod]
            h = collide(M, fixed, ok)
            if h:
                worst.append(dict(sample=k, d=d, hits=h[:6]))
        report[f"6 pod {s} (wheels retracted)"] = worst
        if not only or "pod" in only:
            print(f"  6 pod {s} (wheels retracted)               {len(pod):4d} moving vs {len(fixed):4d} fixed, {len(samples):3d} samples: "
                  f"{'CLEAR' if not worst else str(len(worst)) + ' blocked, e.g. ' + str(worst[0]['hits'][:3])}  ({time.time() - t0:.0f} s)",
                  flush=True)
        installed += pod
    left = [p.name for p in parts if p not in installed and not p.name.startswith("tread")]
    print(f"  parts not placed by any step: {left[:20]}", flush=True)
    report["unplaced"] = left

    for name, mov, fixed, poses in steps:
        if only and only not in name:
            continue
        run_path(name, mov, fixed, ok, poses, report)
    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(os.path.join(OUT, "assembly_check.json"), "w"), indent=1)
    n = sum(1 for k, v in report.items() if v and k != "unplaced") + (1 if left else 0)
    print(f"SUMMARY: {len(report) - 1} assembly moves, {n} blocked / unplaced")
    sys.exit(1 if n else 0)


import spec as _S  # noqa: E402
S_DRIVE_ANGLE = _S.DRIVE_ANGLE

if __name__ == "__main__":
    main()
