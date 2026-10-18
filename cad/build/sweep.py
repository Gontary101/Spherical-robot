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
with real overlap (> 0.05 mm3) is found (a HIT), or the box is narrower than MIN_W while the parts are within TOL_D (0.3 mm running clearance)
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
TOL_D = 0.3          # required running clearance between parts in relative motion (mm)


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


def sector(bb, axis):
    """Angular interval (deg) that a box covers about axis x or y (angle grows with a positive rotation), or None when
    the box surrounds the axis."""
    if axis == "x":
        u, v = (bb[2], bb[3]), (bb[4], bb[5])          # angle = atan2(z, y)
    else:
        u, v = (bb[4], bb[5]), (bb[0], bb[1])          # angle = atan2(x, z)
    if u[0] <= 0 <= u[1] and v[0] <= 0 <= v[1]:
        return None
    angs = [math.degrees(math.atan2(vv, uu)) for uu in u for vv in v]
    a0 = angs[0]
    rel = [((a - a0 + 180) % 360) - 180 for a in angs]
    return a0 + min(rel), a0 + max(rel)


def rel_intervals(sm, ss, lo, hi, m):
    """Rotations t in [lo, hi] (deg) for which sector sm rotated by t can meet sector ss; list of (a, b) in deg."""
    if sm is None or ss is None:
        return [(lo, hi)]
    a, b = ss[0] - sm[1] - m, ss[1] - sm[0] + m
    if b - a >= 360:
        return [(lo, hi)]
    out = []
    for k in range(-2, 3):
        x, y = max(lo, a + 360 * k), min(hi, b + 360 * k)
        if x < y:
            out.append((x, y))
    return out


def pairs_for(parts, ok, lean_deg, log=print, pitch_deg=180.0, only=None):
    """Candidate (moving index, static index, axes, box (rad), rho per axis) after envelope + angular pruning."""
    bbs = [_bb(p.shape) for p in parts]
    env = {ax: [envelope(b, ax) for b in bbs] for ax in ("x", "y")}
    sec = {ax: [sector(b, ax) for b in bbs] for ax in ("x", "y")}
    by = {}
    for i, p in enumerate(parts):
        by.setdefault(p.body, []).append(i)
    tasks = []
    R = math.radians

    def add(mi, si, axes, box, rhos):
        if only and not (only(parts[mi].name) or only(parts[si].name)):
            return
        if frozenset((parts[mi].name, parts[si].name)) not in ok:
            tasks.append((mi, si, axes, box, rhos))

    # yoke vs spine: pitch about y
    for mi in by.get("yoke", []):
        a = env["y"][mi]
        for si in by.get("spine", []):
            b = env["y"][si]
            if overlap(a[0], a[1], b[0], b[1]) and overlap(a[2], a[3], b[2], b[3]):
                for lo, hi in rel_intervals(sec["y"][mi], sec["y"][si], -pitch_deg, pitch_deg, 1.0):
                    add(mi, si, ("y",), [(R(lo), R(hi))], [a[3]])
    # bob vs yoke and bob vs tyres: lean about x
    for mi in by.get("bob", []):
        a = env["x"][mi]
        for body in ("yoke", "tyreL", "tyreR"):
            for si in by.get(body, []):
                b = env["x"][si]
                if overlap(a[0], a[1], b[0], b[1]) and overlap(a[2], a[3], b[2], b[3]):
                    for lo, hi in rel_intervals(sec["x"][mi], sec["x"][si], -lean_deg, lean_deg, 1.0):
                        add(mi, si, ("x",), [(R(lo), R(hi))], [a[3]])
    # bob vs spine: lean then pitch. Per 1-deg lean sample, the posed part's (y range, distance from y, sector about y)
    step = 1.0
    leans = [-lean_deg + step * k for k in range(int(2 * lean_deg / step) + 1)] + [lean_deg]
    sp = by.get("spine", [])
    for mi in by.get("bob", []):
        rx = env["x"][mi][3]
        g = rx * math.radians(step / 2) + 0.5
        samples = []
        ry_max = 0.0
        for L in leans:
            bb = _bb(parts[mi].shape.rotate(O, AX["x"], L))
            e = envelope(bb, "y")
            ry_max = max(ry_max, e[3])
            samples.append((L, e, sector(bb, "y")))
        for si in sp:
            b = env["y"][si]
            cells = {}
            for L, e, sm in samples:
                if not (overlap(e[0] - g, e[1] + g, b[0], b[1]) and overlap(e[2] - g, e[3] + g, b[2], b[3])):
                    continue
                m = 1.0 + (math.degrees(g / e[2]) if e[2] > g else 180.0)
                for lo, hi in rel_intervals(sm, sec["y"][si], -pitch_deg, pitch_deg, m):
                    for k in range(int((lo + 180) // 10), int(min(hi + 180, 359.999) // 10) + 1):
                        cells.setdefault(k, []).append(L)
            if not cells:
                continue
            ks = sorted(cells)
            runs, cur = [], [ks[0]]
            for k in ks[1:]:
                if k == cur[-1] + 1:
                    cur.append(k)
                else:
                    runs.append(cur)
                    cur = [k]
            runs.append(cur)
            if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == 35:          # wraps through +-180
                runs[0] = [k - 36 for k in runs[-1]] + runs[0]
                runs.pop()
            for run in runs:
                Ls = [L for k in run for L in cells[k % 36]]
                lb = (max(-lean_deg, min(Ls) - step), min(lean_deg, max(Ls) + step))
                pb = (max(-pitch_deg, run[0] * 10 - 180.0), min(pitch_deg, (run[-1] + 1) * 10 - 180.0))
                if pb[0] >= pb[1]:
                    continue
                add(mi, si, ("x", "y"), [(R(lb[0]), R(lb[1])), (R(pb[0]), R(pb[1]))], [rx, ry_max])
    n2 = sum(1 for t in tasks if len(t[2]) == 2)
    log(f"   {len(tasks)} candidate pair-intervals after envelope + angular pruning ({n2} two-axis)")
    return tasks


def run(parts, ok, lean_deg=41.5, procs=4, log=print, pitch_deg=180.0, only=None):
    tasks = pairs_for(parts, ok, lean_deg, log, pitch_deg, only)
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
