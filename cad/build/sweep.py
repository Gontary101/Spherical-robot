"""
Continuous motion clearance: proves that no two parts of bodies that move relative to each other collide anywhere in the
range of motion, not only at sampled poses.

Relative motions (all axes through the origin):
    yoke  vs spine   pitch about y, full turn
    bob   vs yoke    lean about x, +-LEAN (the yoke pitch is common to both)
    bob   vs spine   lean about x, then pitch about y (2-D)
    bob   vs tyres   lean (the tyre parts in the check set are bodies of revolution about y, so pitch and spin drop out)
    yoke / spine vs tyres: invariant for the same reason (covered by the static check)

For each candidate pair (pruned by swept-envelope overlap) the motion box is bisected: at the box centre the exact
minimum distance d between the posed solids is computed (BRepExtrema); no point of the moving part can move further than
sum(rho_i * halfwidth_i) from its centre-pose position anywhere in the box (rho_i = the part's largest distance from
axis i), so d > that bound certifies the whole box clear. Otherwise the box is split until it is certified, or a pose
with real overlap (> 0.05 mm3) is found (a HIT), or the box is narrower than MIN_W while the parts are within 0.05 mm
(a TOUCH, reported).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import time

from cadquery import Vector
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

O = Vector(0, 0, 0)
AX = {"x": Vector(1, 0, 0), "y": Vector(0, 1, 0)}
MIN_W = math.radians(0.05)
TOL_V = 0.05
TOL_D = 0.05


def _bb(s):
    b = s.BoundingBox()
    return (b.xmin, b.xmax, b.ymin, b.ymax, b.zmin, b.zmax)


def _corners(bb):
    return [(x, y, z) for x in bb[0:2] for y in bb[2:4] for z in bb[4:6]]


def _clamp_dist(lo, hi):
    return 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))


def envelope(bb, axis):
    """(axial lo, axial hi, radial lo, radial hi) of the region a box can sweep rotating about axis x or y."""
    i = {"x": 0, "y": 1}[axis]
    a = bb[2 * i: 2 * i + 2]
    others = [k for k in range(3) if k != i]
    p, q = (bb[2 * others[0]: 2 * others[0] + 2], bb[2 * others[1]: 2 * others[1] + 2])
    rmin = math.hypot(_clamp_dist(*p), _clamp_dist(*q))
    rmax = max(math.hypot(u, v) for u in p for v in q)
    return a[0], a[1], rmin, rmax


def shell(bb):
    rmin = math.sqrt(sum(_clamp_dist(bb[2 * k], bb[2 * k + 1]) ** 2 for k in range(3)))
    rmax = max(math.sqrt(x * x + y * y + z * z) for x, y, z in _corners(bb))
    return rmin, rmax


def overlap(a0, a1, b0, b1, g=0.5):
    return not (a1 + g < b0 or b1 + g < a0)


def pose(shape, axes, angles):
    s = shape
    for ax, ang in zip(axes, angles):          # applied in order: lean (x) first, then pitch (y)
        if ang:
            s = s.rotate(O, AX[ax], math.degrees(ang))
    return s


def certify_pair(m_shape, s_shape, axes, box, rhos):
    """-> ('clear', n_eval) | ('hit', angles_deg, volume, n_eval) | ('touch', angles_deg, d, n_eval)"""
    stack = [box]
    n = 0
    worst_touch = None
    while stack:
        b = stack.pop()
        c = [(lo + hi) / 2 for lo, hi in b]
        hw = [(hi - lo) / 2 for lo, hi in b]
        ms = pose(m_shape, axes, c)
        d = BRepExtrema_DistShapeShape(ms.wrapped, s_shape.wrapped).Value()
        n += 1
        bound = sum(r * h for r, h in zip(rhos, hw))
        if d > bound + TOL_D:
            continue
        if d < TOL_D:
            v = ms.intersect(s_shape).Volume()
            if v > TOL_V:
                return ("hit", [round(math.degrees(x), 2) for x in c], round(v, 2), n)
        k = max(range(len(b)), key=lambda i: rhos[i] * hw[i])
        if hw[k] < MIN_W:
            if d < TOL_D and worst_touch is None:
                worst_touch = ([round(math.degrees(x), 2) for x in c], round(d, 3))
            continue
        lo, hi = b[k]
        mid = (lo + hi) / 2
        for half in ((lo, mid), (mid, hi)):
            nb = list(b)
            nb[k] = half
            stack.append(nb)
    if worst_touch:
        return ("touch", worst_touch[0], worst_touch[1], n)
    return ("clear", n)


_W = {}


def _init(shapes):
    _W["s"] = shapes


def _job(task):
    i, k, axes, box, rhos = task
    S = _W["s"]
    t0 = time.time()
    r = certify_pair(S[i][1], S[k][1], axes, box, rhos)
    return S[i][0], S[k][0], axes, r, round(time.time() - t0, 1)


def pairs_for(parts, ok, lean_deg, log=print):
    """Candidate pairs (moving index, static index, axes, box, rho per axis)."""
    lean = math.radians(lean_deg)
    bbs = [_bb(p.shape) for p in parts]
    env = {ax: [envelope(b, ax) for b in bbs] for ax in ("x", "y")}
    sh = [shell(b) for b in bbs]
    by = {}
    for i, p in enumerate(parts):
        by.setdefault(p.body, []).append(i)
    tasks = []

    def add(mi, si, axes, box, rhos):
        if frozenset((parts[mi].name, parts[si].name)) in ok:
            return
        tasks.append((mi, si, axes, box, rhos))

    # yoke vs spine: pitch about y over a full turn
    for mi in by.get("yoke", []):
        a = env["y"][mi]
        for si in by.get("spine", []):
            b = env["y"][si]
            if overlap(a[0], a[1], b[0], b[1]) and overlap(a[2], a[3], b[2], b[3]):
                add(mi, si, ("y",), [(-math.pi, math.pi)], [a[3]])
    # bob vs yoke and bob vs tyres: lean about x
    for mi in by.get("bob", []):
        a = env["x"][mi]
        for body in ("yoke", "tyreL", "tyreR"):
            for si in by.get(body, []):
                b = env["x"][si]
                if overlap(a[0], a[1], b[0], b[1]) and overlap(a[2], a[3], b[2], b[3]):
                    add(mi, si, ("x",), [(-lean, lean)], [a[3]])
    # bob vs spine: lean then pitch
    for mi in by.get("bob", []):
        a = sh[mi]
        for si in by.get("spine", []):
            b = sh[si]
            if overlap(a[0], a[1], b[0], b[1]):
                add(mi, si, ("x", "y"), [(-lean, lean), (-math.pi, math.pi)], [a[1], a[1]])
    log(f"   {len(tasks)} candidate pairs after envelope pruning")
    return tasks


def run(parts, ok, lean_deg=41.5, procs=4, log=print):
    tasks = pairs_for(parts, ok, lean_deg, log)
    # longest first (2-D boxes, big parts)
    tasks.sort(key=lambda t: -(len(t[2]) * 10 + sum(t[4]) / 100))
    shapes = [(p.name, p.shape) for p in parts]
    out = dict(hits=[], touches=[], n_pairs=len(tasks), n_eval=0)
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs, initializer=_init, initargs=(shapes,)) as pool:
        for k, (a, b, axes, r, dt) in enumerate(pool.imap_unordered(_job, tasks, chunksize=4)):
            out["n_eval"] += r[-1]
            if r[0] == "hit":
                out["hits"].append(dict(moving=a, static=b, motion="+".join(axes), at_deg=r[1], volume=r[2]))
                log(f"   HIT   {a} x {b} ({'+'.join(axes)} at {r[1]} deg, {r[2]} mm3)")
            elif r[0] == "touch":
                out["touches"].append(dict(moving=a, static=b, motion="+".join(axes), at_deg=r[1], d=r[2]))
                log(f"   touch {a} x {b} ({'+'.join(axes)} at {r[1]} deg, d {r[2]})")
            if (k + 1) % 200 == 0:
                log(f"   ... {k + 1}/{len(tasks)} pairs, {out['n_eval']} distance evaluations, {time.time() - t0:.0f} s")
    out["seconds"] = round(time.time() - t0)
    return out
