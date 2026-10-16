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


def main():
    only = sys.argv[sys.argv.index("--step") + 1] if "--step" in sys.argv else None
    parts, joints, threads, bonds = build_all(verbose=False)
    ok = allowed_pairs(parts, joints, threads)
    jn = {j.name: j for j in joints}
    # a fastener belongs to the body / group of its first member
    def is_tyre(p, s):
        return p.body == f"tyre{s}"
    report = {}
    steps = []

    # T: rim into the drum through the seam opening (ring gear co-bonded in the mould)
    for s, sg in (("L", 1), ("R", -1)):
        rim = sel(parts, lambda p: p.name == f"rim_{s}")
        fixed = sel(parts, lambda p: p.name in (f"drum_{s}", f"ring_gear_{s}"))
        steps.append((f"T rim_{s} through the seam", rim, fixed, [((0, -sg * d, 0), None) for d in range(330, -1, -10)]))

    # 2a: bob onto the track (yoke without lean stops), rotated in about x from 120 deg
    yoke_wo_stops = sel(parts, lambda p: p.body == "yoke" and not p.name.startswith("lean_stop"))
    bob = sel(parts, lambda p: p.body == "bob")
    steps.append(("2a bob run onto the track", bob, yoke_wo_stops, [(None, ((0, 0, 0), (1, 0, 0), a)) for a in range(120, -1, -4)]))

    # 2: yoke (with bob) slides onto the axle + spacer from +y
    spine_core = sel(parts, lambda p: p.name in ("axle",))
    yb = sel(parts, lambda p: p.body in ("yoke", "bob"))
    steps.append(("2 yoke onto the axle", yb, spine_core, [((0, d, 0), None) for d in range(420, -1, -10)]))

    # 3: level drive (+y) and slip ring + sleeve (-y) onto the axle, past the installed yoke
    installed = yb + spine_core + sel(parts, lambda p: p.name == "hub_spacer")
    lvl = sel(parts, lambda p: p.name.startswith(("level_bracket", "level_motor", "level_pinion")))
    steps.append(("3 level drive onto the axle", lvl, installed, [((0, d, 0), None) for d in range(300, -1, -10)]))
    sr = sel(parts, lambda p: p.name.startswith(("slipring_rotor", "slip_sleeve")))
    steps.append(("3 slip ring + sleeve onto the axle", sr, installed, [((0, -d, 0), None) for d in range(300, -1, -10)]))

    # 5: tyre halves, offset 15 mm toward their pinion, seam first, then radially into mesh
    spine_all = sel(parts, lambda p: p.body == "spine" and not p.name.startswith(("pod_", "wheel_", "stud_", "vwheel", "roller", "lidar",
                                                                                     "light_ring", "led_pcb", "shield", "cam", "wheel_retainer",
                                                                                     "riser", "pod")))
    base = yb + spine_all
    a = math.radians(S_DRIVE_ANGLE)
    for s, sg in (("L", 1), ("R", -1)):
        u = (sg * math.sin(a), -math.cos(a))
        off = 15.0
        mov = sel(parts, lambda p, s=s: p.name in (f"drum_{s}", f"ring_gear_{s}", f"rim_{s}"))
        poses = [((u[0] * off, sg * d, u[1] * off), None) for d in range(600, -1, -10)]
        poses += [((u[0] * off * f, 0, u[1] * off * f), None) for f in (0.8, 0.6, 0.4, 0.2, 0.0)]
        steps.append((f"5 tyre {s}: offset slide + radial into mesh", mov, base, poses))
    tyres = sel(parts, lambda p: p.body in ("tyreL", "tyreR") and not p.name.startswith("tread"))
    base2 = base + tyres

    # 6: pods slide on; left wheels retracted 3.5 mm (eccentric studs turned) during the slide
    for s, sg in (("L", 1), ("R", -1)):
        pod = sel(parts, lambda p, s=s: p.body == "spine" and (p.name.endswith(f"_{s}") or f"_{s}" in p.name[-4:]) and
                  p.name.startswith(("pod_", "wheel_", "stud_", "vwheel", "roller", "lidar", "light_ring", "led_pcb", "shield", "cam",
                                     "riser")))
        pod += [f for f in parts if f.kind == "fastener" and f.body == "spine" and any(f.name.startswith(pfx) for pfx in (
            f"pod_hub_clamp_{s}", f"pod_hub_bolt_{s}", f"wheel_screw_{s}", f"lidar_screw_{s}", f"riser_bolt_{s}", f"shield_screw_{s}",
            f"cam_holder_screw_{s}", f"cam_screw_{s}", f"lidar_dowel_{s}"))]
        fixed = base2
        retract = []
        for p in pod:
            if p.name.startswith(("vwheel_", "wheel_retainer_", "wheel_screw_")) and s == "L":
                k = int([c for c in p.name if c.isdigit()][-1])
                ang = math.radians(k * 45.0)
                retract.append((p.name, (-3.5 * math.cos(ang), 0, -3.5 * math.sin(ang))))
        rmap = dict(retract)
        def pod_poses(d, rmap=rmap):
            return d
        samples = list(range(260, -1, -10))
        worst = []
        t0 = time.time()
        for k, d in enumerate(samples):
            M = []
            for p in pod:
                dv = Vector(0, sg * d, 0)
                if p.name in rmap and d > 0:
                    dv = dv + Vector(*rmap[p.name])
                M.append(moved(p, (dv.x, dv.y, dv.z)))
            h = collide(M, fixed, ok)
            if h:
                worst.append(dict(sample=k, d=d, hits=h[:6]))
        report[f"6 pod {s} (wheels retracted)"] = worst
        print(f"  6 pod {s} (wheels retracted)               {len(pod):4d} moving vs {len(fixed):4d} fixed, {len(samples):3d} samples: "
              f"{'CLEAR' if not worst else str(len(worst)) + ' blocked, e.g. ' + str(worst[0]['hits'][:3])}  ({time.time() - t0:.0f} s)", flush=True)

    for name, mov, fixed, poses in steps:
        if only and only not in name:
            continue
        run_path(name, mov, fixed, ok, poses, report)
    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(os.path.join(OUT, "assembly_check.json"), "w"), indent=1)
    n = sum(1 for v in report.values() if v)
    print(f"SUMMARY: {len(report)} assembly moves, {n} blocked")
    sys.exit(1 if n else 0)


import spec as _S  # noqa: E402
S_DRIVE_ANGLE = _S.DRIVE_ANGLE

if __name__ == "__main__":
    main()
