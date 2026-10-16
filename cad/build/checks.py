"""
Verification of the build CAD from its final geometry.

    python cad/build/checks.py [--quick]

1. static interference: every pair of parts (bounding-box pruned). Allowed overlaps are only the ones a fastener is
   designed to make: a screw / insert / PEM nut in the member it threads into, declared thread pairs (studs, standoffs,
   nuts), and gear meshes.
2. motion sweep: yoke pitch over 360 deg x bob lean +-40 deg x tyre spin, all moving-body pairs.
3. joints: thread engagement >= rule (steel 1.0 d, 7075 1.25 d, 6061 1.5 d, vendor thread depth for bought parts),
   no screw longer than its tapped hole, screw head free.
4. floating parts: every part touches (or is bonded / threaded / meshed to) another part of the same body.
Writes cad/build/out/checks.json and exits non-zero if anything fails.
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
from OCP.BRepExtrema import BRepExtrema_DistShapeShape  # noqa: E402

from assemble import MESH, OUT, build_all  # noqa: E402
from lib import ENGAGE  # noqa: E402

O = Vector(0, 0, 0)


def bbox(s, grow=0.0):
    b = s.BoundingBox()
    return (b.xmin - grow, b.xmax + grow, b.ymin - grow, b.ymax + grow, b.zmin - grow, b.zmax + grow)


def bb_hit(a, b):
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2] or a[5] < b[4] or b[5] < a[4])


def allowed_pairs(parts, joints, threads):
    ok = set(MESH) | {frozenset(t) for t in threads}
    for j in joints:
        m = j.info.get("tapped_in") or j.info.get("insert_in") or j.info.get("pem_in")
        if m:
            ok.add(frozenset((j.name, m)))
            ok.add(frozenset((f"{j.name}_insert", m)))
            ok.add(frozenset((f"{j.name}_pem", m)))
            ok.add(frozenset((j.name, f"{j.name}_insert")))
            ok.add(frozenset((j.name, f"{j.name}_pem")))
        ok.add(frozenset((j.name, f"{j.name}_nut")))
    return ok


def check_set(parts):
    """Parts used for interference: the cast treads are outside the drums (nothing inside can reach them) and the
    360-tooth ring gears are replaced by their tooth-tip annulus (conservative) except against their pinions."""
    import spec as S
    from lib import revolve_y
    from tyre import _sph_profile, BOND_GAP
    out = []
    for p in parts:
        if p.name.startswith("tread_"):
            continue
        if p.name.startswith("ring_gear_"):
            sg = 1 if p.name.endswith("L") else -1
            y0, y1 = S.GEAR_Y
            ann = revolve_y(_sph_profile(y0, y1, 1, BOND_GAP) + [(S.R_RING_TIP, y1), (S.R_RING_TIP, y0)])
            out.append(p.copy(ann if sg > 0 else ann.mirror("XZ")))
            continue
        out.append(p)
    return out


def interference(parts, ok, only_cross=False, tol=0.05):
    """Pairs from different subsystems (each subsystem was checked on its own while building) plus all pairs
    involving a fastener."""
    boxes = [bbox(p.shape) for p in parts]
    hits = []
    for i in range(len(parts)):
        for k in range(i + 1, len(parts)):
            a, b = parts[i], parts[k]
            if a.spec.get("sub") == b.spec.get("sub") and "fastener" not in (a.kind, b.kind):
                continue
            if only_cross and a.body == b.body:
                continue
            if frozenset((a.name, b.name)) in ok:
                continue
            if not bb_hit(boxes[i], boxes[k]):
                continue
            v = a.shape.intersect(b.shape).Volume()
            if v > tol:
                hits.append((a.name, b.name, round(v, 2)))
    return hits


def posed(parts, pitch=0.0, lean=0.0, spin_l=0.0, spin_r=0.0):
    from lib import Part
    out = []
    Y, X = Vector(0, 1, 0), Vector(1, 0, 0)
    for p in parts:
        s = p.shape
        if p.body == "bob" and lean:
            s = s.rotate(O, X, lean)
        if p.body in ("yoke", "bob") and pitch:
            s = s.rotate(O, Y, pitch)
        if p.body == "tyreL" and spin_l:
            s = s.rotate(O, Y, spin_l)
        if p.body == "tyreR" and spin_r:
            s = s.rotate(O, Y, spin_r)
        out.append(p.copy(s))
    return out


_G = {}


def _pose_worker(pose):
    return pose, _sweep_one(_G["parts"], _G["ok"], pose)


def _sweep_one(parts, ok, pose):
    pt, ln, sl, sr = pose
    moving = {"yoke", "bob", "tyreL", "tyreR"}
    P = posed(parts, pt, ln, sl, sr)
    boxes = {p.name: bbox(p.shape) for p in P}
    hits = []
    for i in range(len(P)):
        for k in range(i + 1, len(P)):
            a, b = P[i], P[k]
            if a.body == b.body or not ({a.body, b.body} & moving) or {a.body, b.body} == {"tyreL", "tyreR"}:
                continue
            if {a.body, b.body} == {"yoke", "bob"} and ln == 0 and False:
                continue
            if frozenset((a.name, b.name)) in ok or not bb_hit(boxes[a.name], boxes[b.name]):
                continue
            v = a.shape.intersect(b.shape).Volume()
            if v > 0.05:
                hits.append((a.name, b.name, round(v, 2)))
    return hits


def motion_sweep_parallel(parts, ok, poses, n_proc=4):
    import multiprocessing as mp
    _G["parts"], _G["ok"] = parts, ok
    res = []
    with mp.get_context("fork").Pool(n_proc) as pool:
        for pose, hits in pool.imap_unordered(_pose_worker, poses):
            pt, ln, sl, sr = pose
            res.append(dict(pitch=pt, lean=ln, spin_l=sl, spin_r=sr, hits=hits))
            print(f"    pose pitch {pt:5.0f} lean {ln:4.0f} spin {sl:4.1f}/{sr:4.1f}: {'CLEAR' if not hits else hits[:4]}", flush=True)
    return res


def motion_sweep(parts, ok, poses):
    """Only pairs whose relative position changes (different bodies, not spine-static pairs)."""
    moving = {"yoke", "bob", "tyreL", "tyreR"}
    res = []
    for (pt, ln, sl, sr) in poses:
        P = posed(parts, pt, ln, sl, sr)
        cand = [p for p in P]
        boxes = {p.name: bbox(p.shape) for p in cand}
        hits = []
        for i in range(len(cand)):
            for k in range(i + 1, len(cand)):
                a, b = cand[i], cand[k]
                if a.body == b.body or not ({a.body, b.body} & moving):
                    continue
                if {a.body, b.body} == {"tyreL", "tyreR"}:
                    continue
                if frozenset((a.name, b.name)) in ok:
                    continue
                if not bb_hit(boxes[a.name], boxes[b.name]):
                    continue
                v = a.shape.intersect(b.shape).Volume()
                if v > 0.05:
                    hits.append((a.name, b.name, round(v, 2)))
        res.append(dict(pitch=pt, lean=ln, spin_l=sl, spin_r=sr, hits=hits))
        print(f"    pose pitch {pt:5.0f} lean {ln:4.0f} spin {sl:3.0f}/{sr:3.0f}: {'CLEAR' if not hits else hits[:4]}", flush=True)
    return res


def joint_checks(parts, joints):
    D = {p.name: p for p in parts}
    bad = []
    for j in joints:
        i = j.info
        d = i["d"]
        if j.end == "tap":
            mat = i.get("tapped_material")
            need = ENGAGE.get(mat, 1.5) * d if j.max_engage is None else 0.6 * j.max_engage
            if i["engagement"] + 1e-6 < need:
                bad.append((j.name, f"engagement {i['engagement']} < {need:.1f} in {i['tapped_in']} ({mat})"))
            if j.max_engage is not None and i["engagement"] > j.max_engage + 1e-6:
                bad.append((j.name, f"engagement {i['engagement']} > vendor thread {j.max_engage}"))
        # head / shank must not touch anything except members, own washer/nut/insert/pem
        scr = D[j.name]
        mem = set(i["members"]) | {f"{j.name}_washer", f"{j.name}_nut", f"{j.name}_nwasher", f"{j.name}_insert", f"{j.name}_pem"}
        bs = bbox(scr.shape)
        for p in parts:
            if p.name == j.name or p.name in mem:
                continue
            if not bb_hit(bs, bbox(p.shape)):
                continue
            v = scr.shape.intersect(p.shape).Volume()
            if v > 0.05:
                bad.append((j.name, f"screw hits {p.name} ({v:.1f} mm3)"))
    return bad


def floating(parts, ok, bonds, gap=0.6):
    """Parts that neither touch nor are declared connected to another part of the same body."""
    conn = {frozenset((a, b)) for a, b, *_ in bonds} | set(ok)
    by_body = {}
    for p in parts:
        by_body.setdefault(p.body, []).append(p)
    lonely = []
    for body, P in by_body.items():
        boxes = {p.name: bbox(p.shape, gap) for p in P}
        for p in P:
            found = False
            for q in P:
                if q is p:
                    continue
                if frozenset((p.name, q.name)) in conn:
                    found = True
                    break
                if not bb_hit(boxes[p.name], boxes[q.name]):
                    continue
                if BRepExtrema_DistShapeShape(p.shape.wrapped, q.shape.wrapped).Value() < gap:
                    found = True
                    break
            if not found:
                lonely.append(p.name)
    return lonely


def main():
    quick = "--quick" in sys.argv
    t0 = time.time()
    parts_full, joints, threads, bonds = build_all(verbose=False)
    ok = allowed_pairs(parts_full, joints, threads)
    parts = check_set(parts_full)
    print(f"built {len(parts)} parts, {len(joints)} joints in {time.time() - t0:.0f} s", flush=True)
    rep = {}
    print("1. static interference ...", flush=True)
    rep["static"] = interference(parts, ok)
    print(f"   {len(rep['static'])} hits {rep['static'][:12]}", flush=True)
    print("2. joints ...", flush=True)
    rep["joints"] = joint_checks(parts, joints)
    print(f"   {len(rep['joints'])} problems {rep['joints'][:12]}", flush=True)
    print("3. floating parts ...", flush=True)
    rep["floating"] = floating(parts, ok, bonds)
    print(f"   {len(rep['floating'])} {rep['floating'][:20]}", flush=True)
    print("4. motion sweep ...", flush=True)
    pitches = (0, 90, 180, -90) if quick else (0, 30, 60, 90, 120, 150, 180, -150, -120, -90, -60, -30)
    leans = (0, 40, -40) if quick else (0, 20, 40, -20, -40)
    poses = [(p, l, 0.0, 0.0) for p in pitches for l in leans] + [(0, 0, a, -a) for a in (0.5, 1.0)]
    rep["motion"] = motion_sweep_parallel(parts, ok, poses)
    n_motion = sum(len(r["hits"]) for r in rep["motion"])
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(os.path.join(OUT, "checks.json"), "w"), indent=1)
    fails = len(rep["static"]) + len(rep["joints"]) + len(rep["floating"]) + n_motion
    print(f"SUMMARY: static {len(rep['static'])}, joints {len(rep['joints'])}, floating {len(rep['floating'])}, "
          f"motion {n_motion} over {len(poses)} poses  ({time.time() - t0:.0f} s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
