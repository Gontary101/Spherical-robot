"""
Build-level CAD kernel for GYRA Mk2.1: parts with manufacturing metadata, ISO fasteners, and bolted joints that
drill their own holes.

A Joint is the single source of a connection: a fastener spec, the seat point and shank direction, and the ordered
stack of parts it clamps. Applying it cuts the right hole in every member (clearance / counterbore / tapped /
heat-set insert / nut), picks a standard screw length and places the fastener (and nut / washer) as parts. The
checks in checks.py then verify every joint from the finished geometry alone (engagement, head seat, nothing hit),
so a hole that does not line up cannot pass silently.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import cadquery as cq
import numpy as np
from cadquery import Vector
from OCP.BRepGProp import BRepGProp
from OCP.BRepIntCurveSurface import BRepIntCurveSurface_Inter
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# densities, kg/mm^3
RHO = {"al": 2.70e-6, "al7075": 2.81e-6, "steel": 7.85e-6, "ss": 7.95e-6, "wnife": 18.0e-6, "pu": 1.15e-6,
       "gfrp": 1.85e-6, "cfrp": 1.55e-6, "pa12": 1.01e-6, "pom": 1.41e-6, "pc": 1.20e-6, "brass": 8.50e-6,
       "nbr": 1.25e-6, "fr4": 1.85e-6, "nylon": 1.14e-6}


@dataclass
class Part:
    """One physical item. kind: custom (made for GYRA) | cots (bought) | fastener | consumable."""
    name: str
    shape: cq.Shape
    body: str                    # kinematic body: spine, tyreL, tyreR, yoke, bob
    render: str                  # render material key (render_colors.RGB)
    rho: str | None = None       # density key for custom parts
    mass: float | None = None    # kg for bought parts (datasheet)
    kind: str = "custom"
    spec: dict = field(default_factory=dict)   # material, process, finish, part number, supplier, notes

    def copy(self, shape=None):
        return Part(self.name, shape if shape is not None else self.shape, self.body, self.render, self.rho, self.mass,
                    self.kind, dict(self.spec))


# --------------------------------------------------------------------------------------------- geometry helpers
def V(*a) -> Vector:
    return Vector(*a)


def unit(v) -> Vector:
    v = Vector(*v) if not isinstance(v, Vector) else v
    return v.normalized()


def cyl(r, p0, p1) -> cq.Shape:
    """Solid cylinder of radius r from point p0 to p1."""
    p0, p1 = Vector(*p0), Vector(*p1)
    d = p1 - p0
    return cq.Solid.makeCylinder(r, d.Length, p0, d.normalized())


def tube(r_in, r_out, p0, p1) -> cq.Shape:
    s = cyl(r_out, p0, p1)
    return s.cut(cyl(r_in, Vector(*p0) - (Vector(*p1) - Vector(*p0)) * 0.01, Vector(*p1) + (Vector(*p1) - Vector(*p0)) * 0.01)) if r_in > 0 else s


def cone(r0, r1, p0, p1) -> cq.Shape:
    p0, p1 = Vector(*p0), Vector(*p1)
    d = p1 - p0
    return cq.Solid.makeCone(r0, r1, d.Length, p0, d.normalized())


def box(x0, x1, y0, y1, z0, z1) -> cq.Shape:
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, Vector(x0, y0, z0))


def plane(origin, xdir, normal) -> cq.Plane:
    return cq.Plane(origin=tuple(origin), xDir=tuple(xdir), normal=tuple(normal))


def plate(pl: cq.Plane, outline, t, holes=(), offset=0.0) -> cq.Shape:
    """Plate whose 2-D outline (list of (u, v), or a closed list of arcs via `outline` callable) lies in plane pl,
    extruded by t along the plane normal (starting `offset` along the normal). holes: (u, v, d) through holes."""
    wp = cq.Workplane(pl).workplane(offset=offset)
    if callable(outline):
        wp = outline(wp)
    else:
        wp = wp.polyline(outline).close()
    s = wp.extrude(t).val()
    for (u, v, d) in holes:
        p = pl.toWorldCoords((u, v)) + pl.zDir * (offset - 1)
        s = s.cut(cyl(d / 2, p, p + pl.zDir * (t + 2)))
    return s


def revolve_y(profile_ry) -> cq.Shape:
    """Closed (r, y) profile revolved about the global y axis."""
    wp = cq.Workplane("XZ").polyline(profile_ry).close().revolve(360, (0, 0, 0), (0, 1, 0))
    return wp.val().rotate(Vector(0, 0, 0), Vector(1, 0, 0), -90)


def revolve_axis(profile_ra, origin, axis, xref) -> cq.Shape:
    """Closed (r, a) profile (r radial, a along the axis) revolved about an arbitrary axis."""
    pl = cq.Plane(origin=tuple(origin), xDir=tuple(xref), normal=tuple(Vector(*axis).cross(Vector(*xref))))
    wp = cq.Workplane(pl).polyline([(r, a) for r, a in profile_ra]).close()
    return wp.revolve(360, (0, 0, 0), (0, 1, 0)).val()


def sphere(r, c=(0, 0, 0)) -> cq.Shape:
    return cq.Solid.makeSphere(r, Vector(*c), angleDegrees1=-90, angleDegrees2=90)


def slab_y(y0, y1, big=2000) -> cq.Shape:
    return box(-big, big, y0, y1, -big, big)


def mirror_y(s):
    return s.mirror("XZ")


def mirror_x(s):
    return s.mirror("YZ")


def rot_y(s, deg, c=(0, 0, 0)):
    return s.rotate(Vector(*c), Vector(c[0], c[1] + 1, c[2]), deg)


def rot_x(s, deg, c=(0, 0, 0)):
    return s.rotate(Vector(*c), Vector(c[0] + 1, c[1], c[2]), deg)


def rot_z(s, deg, c=(0, 0, 0)):
    return s.rotate(Vector(*c), Vector(c[0], c[1], c[2] + 1), deg)


def arc_pts(cx, cz, r, a0, a1, n=24):
    """Points of an arc in a 2-D plane (angles in degrees)."""
    return [(cx + r * math.cos(math.radians(a)), cz + r * math.sin(math.radians(a))) for a in np.linspace(a0, a1, n)]


def circle_pts(cx, cz, r, n=48):
    return [(cx + r * math.cos(t), cz + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def hull2d(pts):
    from scipy.spatial import ConvexHull
    p = np.array(pts)
    return [tuple(p[i]) for i in ConvexHull(p).vertices]


def sector_yz(r0, r1, a0, a1, n=40):
    """Annular sector in the y-z plane, angle from -z toward +y (deg)."""
    a = np.radians(np.linspace(a0, a1, n))
    return [(r1 * math.sin(t), -r1 * math.cos(t)) for t in a] + [(r0 * math.sin(t), -r0 * math.cos(t)) for t in a[::-1]]


def extrude_yz(pts, x0, x1):
    return cq.Workplane("YZ").polyline(pts).close().extrude(x1 - x0).val().translate(Vector(x0, 0, 0))


def extrude_xz(pts, y0, y1):
    return cq.Workplane("XZ").polyline(pts).close().extrude(-(y1 - y0)).val().translate(Vector(0, y0, 0))


def solid(s):
    """Unwrap a Compound holding a single solid (booleans on such compounds misbehave in OCCT)."""
    sols = s.Solids()
    if len(sols) == 1:
        return sols[0]
    return s


def fuse(*shapes):
    s = shapes[0]
    for t in shapes[1:]:
        s = s.fuse(t)
    return s.clean() if hasattr(s, "clean") else s


def compound(shapes):
    return cq.Compound.makeCompound(list(shapes))


def ray_intervals(shape: cq.Shape, p, n, t_max=2000.0):
    """Material intervals [(t0, t1)] of `shape` along the ray p + t n (t from -t_max to t_max)."""
    p, n = Vector(*p), unit(n)
    inter = BRepIntCurveSurface_Inter()
    lin = gp_Lin(gp_Pnt(p.x, p.y, p.z), gp_Dir(n.x, n.y, n.z))
    inter.Init(shape.wrapped, lin, 1e-6)
    ts = []
    while inter.More():
        ts.append(inter.W())
        inter.Next()
    ts = sorted(t for t in ts if -t_max <= t <= t_max)
    # de-duplicate tangent / edge double hits
    out = []
    for t in ts:
        if not out or abs(t - out[-1]) > 1e-4:
            out.append(t)
    return [(out[i], out[i + 1]) for i in range(0, len(out) - 1, 2)]


def mass_props(p: Part) -> dict:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(p.shape.wrapped, props)
    vol = props.Mass()
    c = props.CentreOfMass()
    I = props.MatrixOfInertia()
    m = p.mass if p.mass is not None else vol * RHO[p.rho]
    k = m / vol if vol > 0 else 0.0
    Imat = np.array([[I.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)]) * k * 1e-6
    return {"volume_cm3": vol / 1e3, "mass": m, "com_mm": [c.X(), c.Y(), c.Z()], "I_com_kgm2": Imat.tolist()}


def combine(props):
    m = sum(p["mass"] for p in props)
    c = sum(np.array(p["com_mm"]) * p["mass"] for p in props) / m
    I = np.zeros((3, 3))
    for p in props:
        d = (np.array(p["com_mm"]) - c) * 1e-3
        I += np.array(p["I_com_kgm2"]) + p["mass"] * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return {"mass": m, "com_mm": c.tolist(), "I_com_kgm2": I.tolist()}


# --------------------------------------------------------------------------------------------- ISO fasteners
# ISO 4762 socket head cap screw: d -> (head dia dk, head height k, hex key s)
SHCS = {2.5: (4.5, 2.5, 2.0), 3: (5.5, 3.0, 2.5), 4: (7.0, 4.0, 3.0), 5: (8.5, 5.0, 4.0), 6: (10.0, 6.0, 5.0),
        8: (13.0, 8.0, 6.0), 10: (16.0, 10.0, 8.0)}
# ISO 7380-1 button head: d -> (dk, k, s)
BHCS = {3: (5.7, 1.65, 2.0), 4: (7.6, 2.2, 2.5), 5: (9.5, 2.75, 3.0), 6: (10.5, 3.3, 4.0), 8: (14.0, 4.4, 5.0)}
# ISO 10642 countersunk (90 deg): d -> (dk theoretical, k, s)
FHCS = {3: (6.72, 1.86, 2.0), 4: (8.96, 2.48, 2.5), 5: (11.2, 3.1, 3.0), 6: (13.44, 3.72, 4.0), 8: (17.92, 4.96, 5.0)}
# ISO 4032 hex nut (s across flats, m height); ISO 7040 nyloc (s, h)
NUT = {2.5: (5.0, 2.0), 3: (5.5, 2.4), 4: (7.0, 3.2), 5: (8.0, 4.7), 6: (10.0, 5.2), 8: (13.0, 6.8), 10: (16.0, 8.4)}
NYLOC = {3: (5.5, 4.0), 4: (7.0, 5.0), 5: (8.0, 5.0), 6: (10.0, 6.0), 8: (13.0, 8.0), 10: (16.0, 10.0)}
# ISO 7089 washer: d -> (inner, outer, thickness)
WASHER = {3: (3.2, 7.0, 0.5), 4: (4.3, 9.0, 0.8), 5: (5.3, 10.0, 1.0), 6: (6.4, 12.0, 1.6), 8: (8.4, 16.0, 1.6),
          10: (10.5, 20.0, 2.0)}
# ISO 273 medium clearance, tap drill, thread pitch
CLEAR = {2.5: 2.9, 3: 3.4, 4: 4.5, 5: 5.5, 6: 6.6, 8: 9.0, 10: 11.0}
TAPDRILL = {2.5: 2.05, 3: 2.5, 4: 3.3, 5: 4.2, 6: 5.0, 8: 6.8, 10: 8.5}
PITCH = {2.5: 0.45, 3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5}
# counterbore for ISO 4762 heads (dia, depth)
CBORE = {2.5: (5.0, 2.8), 3: (6.5, 3.4), 4: (8.0, 4.4), 5: (10.0, 5.4), 6: (11.0, 6.4), 8: (15.0, 8.6), 10: (18.0, 10.6)}
# heat-set inserts for thermoplastics (e.g. Ruthex RX-M3x5.7 / RX-M4x8.1): d -> (hole dia, length)
INSERT = {2.5: (3.6, 5.7), 3: (4.0, 5.7), 4: (5.6, 8.1), 5: (6.4, 9.5)}
# standard stocked lengths
LEN = {"SHCS": (6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 110, 120),
       "BHCS": (6, 8, 10, 12, 14, 16, 20, 25, 30, 35, 40),
       "FHCS": (6, 8, 10, 12, 14, 16, 20, 25, 30, 35, 40)}
STD = {"SHCS": "ISO 4762", "BHCS": "ISO 7380-1", "FHCS": "ISO 10642"}
# minimum engagement by tapped-member material, in diameters
ENGAGE = {"steel": 1.0, "ss": 1.0, "wnife": 1.0, "al7075": 1.25, "al": 1.5, "brass": 1.25, "insert": 0.0}
# tightening torque (N m), A2-70 stainless / 8.8 steel into steel/aluminium, dry, ~70 % proof
TORQUE = {"A2-70": {2.5: 0.5, 3: 1.0, 4: 2.3, 5: 4.5, 6: 7.8, 8: 19.0, 10: 37.0},
          "8.8": {2.5: 0.6, 3: 1.2, 4: 2.9, 5: 5.7, 6: 9.9, 8: 24.0, 10: 48.0},
          "12.9": {3: 2.0, 4: 4.6, 5: 9.0, 6: 15.5, 8: 38.0, 10: 75.0}}


def _hex_prism(s, h, p0, n, xref):
    """Hexagon prism (across flats s) from p0 along n by h."""
    n = unit(n)
    pl = cq.Plane(origin=tuple(p0), xDir=tuple(xref), normal=tuple(n))
    return cq.Workplane(pl).polygon(6, s / math.cos(math.pi / 6)).extrude(h).val()


def _perp(n):
    n = unit(n)
    a = Vector(1, 0, 0) if abs(n.x) < 0.9 else Vector(0, 1, 0)
    return n.cross(a).normalized()


def screw_solid(kind, d, L, seat, n):
    """Screw with its head seated at `seat`, shank along n. Head geometry per ISO, shank = nominal diameter."""
    n = unit(n)
    seat = Vector(*seat)
    if kind == "SHCS":
        dk, k, s = SHCS[d]
        head = cyl(dk / 2, seat - n * k, seat).cut(_hex_prism(s, 0.6 * k, seat - n * k - n * 0.01, n, _perp(n)))
    elif kind == "BHCS":
        dk, k, s = BHCS[d]
        head = cyl(dk / 2, seat - n * k, seat).cut(_hex_prism(s, 0.6 * k, seat - n * k - n * 0.01, n, _perp(n)))
    elif kind == "FHCS":   # seat = where the cone meets the surface plane (head top flush)
        dk, k, s = FHCS[d]
        head = cone(dk / 2, d / 2, seat, seat + n * (dk - d) / 2)
        return head.fuse(cyl(d / 2, seat + n * (dk - d) / 2 * 0.99, seat + n * L)).clean()
    else:
        raise ValueError(kind)
    return head.fuse(cyl(d / 2, seat - n * 0.01, seat + n * L)).clean()


def nut_solid(d, seat, n, nyloc=True):
    s, h = (NYLOC if nyloc else NUT)[d]
    return _hex_prism(s, h, seat, n, _perp(n)).cut(cyl(d / 2 * 0.98, Vector(*seat) - unit(n), Vector(*seat) + unit(n) * (h + 1)))


def washer_solid(d, seat, n):
    di, do, t = WASHER[d]
    return tube(di / 2, do / 2, seat, Vector(*seat) + unit(n) * t)


def screw_mass(kind, d, L):
    """Approximate steel mass (kg): head volume + shank."""
    hk = {"SHCS": SHCS, "BHCS": BHCS, "FHCS": FHCS}[kind][d]
    vol = math.pi / 4 * hk[0] ** 2 * hk[1] * (0.75 if kind != "FHCS" else 0.4) + math.pi / 4 * d ** 2 * L * 0.85
    return vol * 7.9e-6


# --------------------------------------------------------------------------------------------- joints
@dataclass
class Joint:
    """A bolted connection. seat: where the head bears (head underside / countersink surface), n: shank direction
    into the stack. members: ordered part names it passes through; the last one takes the thread unless end='nut'.
    end: 'tap' (tapped blind/through hole in the last member), 'insert' (heat-set insert in a printed last member),
         'nut' (all members clearance, nut + washer behind the last), 'press' (dowel pin: no screw).
    head: 'SHCS' | 'BHCS' | 'FHCS'. cbore: sink the SHCS head into the first member. washer: under the head."""
    name: str
    d: float
    seat: tuple
    n: tuple
    members: list
    end: str = "tap"
    head: str = "SHCS"
    cbore: bool = False
    washer: bool = False
    grade: str = "A2-70"
    L: float | None = None          # forced length (else chosen)
    max_engage: float | None = None # thread depth limit of a bought part's tapped hole (mm)
    tool_len: float = 80.0          # straight tool access needed above the head (mm)
    step: str = ""                  # assembly step that installs it
    # filled in by apply()
    info: dict = field(default_factory=dict)


def _member_interval(shape, seat, n):
    """First material interval of `shape` along the axis, measured from the seat (t>0 into the stack)."""
    iv = ray_intervals(shape, seat, n)
    iv = [(a, b) for a, b in iv if b > -50]
    return iv


def apply_joints(parts: dict, joints: list[Joint]) -> list[Part]:
    """Cut holes for every joint into the member parts (in place) and return the new fastener parts."""
    cuts = {}                     # part name -> list of tool solids
    new = []
    for j in joints:
        n = unit(j.n)
        seat = Vector(*j.seat)
        d = j.d
        mem = [parts[m] for m in j.members]
        ivs = []
        for m in mem:
            iv = _member_interval(m.shape, seat, n)
            if not iv:
                raise RuntimeError(f"joint {j.name}: axis misses member {m.name}")
            ivs.append(iv)
        # extent of the clamped stack
        if j.end in ("tap", "insert"):
            clamped, thread_m = mem[:-1], mem[-1]
        else:
            clamped, thread_m = mem, None
        if len(mem) == 2 and mem[0] is mem[1]:
            # clamp across a slit in one part: clearance in the first material interval, thread in the next
            iv_all = sorted(i for i in ivs[0] if i[1] > 0)
            ivs = [[iv_all[0]], [iv_all[1]]]
        t_enter_thread = min(a for a, b in ivs[-1] if b > 0) if thread_m is not None else None
        t_exit_stack = max(b for iv in ivs for a, b in iv)
        washer_t = WASHER[d][2] if j.washer else 0.0
        head_seat = seat - n * washer_t
        # clearance holes through clamped members (whole span of each member along the axis, generous)
        for m, iv in zip(clamped, ivs[:len(clamped)]):
            t0 = min(a for a, b in iv) - 1.0
            t1 = max(b for a, b in iv) + 1.0
            cuts.setdefault(m.name, []).append(cyl(CLEAR[d] / 2, seat + n * t0, seat + n * t1))
        if j.cbore:
            m = mem[0]
            cd, _ = CBORE[d]
            t_surf = min(a for a, b in ivs[0])
            cuts.setdefault(m.name, []).append(cyl(cd / 2, seat + n * (t_surf - 1.0), seat))
        if j.head == "FHCS":
            m = mem[0]
            dk = FHCS[d][0]
            cuts.setdefault(m.name, []).append(cone(dk / 2 + 0.2, CLEAR[d] / 2, seat - n * 0.01, seat + n * ((dk - CLEAR[d]) / 2 + 0.2)))
        mat = thread_m.rho if thread_m is not None else None
        if j.end == "tap":
            eng_min = ENGAGE.get(mat, 1.5) * d
            avail = max(b for a, b in ivs[-1]) - t_enter_thread
            blind = avail > 0 and True
            grip = t_enter_thread
            need = grip + eng_min
            Ls = [L for L in LEN[j.head] if L >= need - 1e-6]
            if j.max_engage is not None:     # vendor thread: longest screw that cannot bottom out
                Ls = [L for L in LEN[j.head] if grip + 0.6 * j.max_engage <= L <= grip + j.max_engage]
                if not Ls and not j.L:
                    raise RuntimeError(f"joint {j.name}: no length fits grip {grip:.1f} within {j.max_engage} mm of thread")
                Ls = Ls[::-1]
            L = j.L or (Ls[0] if Ls else LEN[j.head][-1])
            eng = L - grip
            # tapped depth: engagement + 2 pitches, never deeper than the member
            depth = min(eng + 2 * PITCH[d] + 1.0, avail)
            cuts.setdefault(thread_m.name, []).append(cyl(TAPDRILL[d] / 2, seat + n * (t_enter_thread - 1.0), seat + n * (t_enter_thread + depth)))
            if depth >= avail - 1e-6:
                cuts[thread_m.name].append(cyl(TAPDRILL[d] / 2, seat + n * (t_enter_thread + depth - 0.5), seat + n * (t_enter_thread + avail + 1.0)))
            j.info = dict(L=L, grip=round(grip, 2), engagement=round(eng, 2), tapped_depth=round(depth, 2),
                          tapped_in=thread_m.name, tapped_material=mat, through_tapped=depth >= avail - 1e-6)
        elif j.end == "insert":
            hole_d, ins_L = INSERT[d]
            grip = t_enter_thread
            Ls = [L for L in LEN[j.head] if grip + 0.5 * ins_L <= L <= grip + ins_L + 0.5]
            if not Ls and not j.L:
                raise RuntimeError(f"joint {j.name}: no standard {j.head} M{d:g} length for grip {grip:.1f} + insert {ins_L}")
            L = j.L or Ls[-1]
            cuts.setdefault(thread_m.name, []).append(cyl(hole_d / 2, seat + n * (t_enter_thread - 0.5), seat + n * (t_enter_thread + ins_L + 1.0)))
            ins = tube(d / 2, hole_d / 2 + 0.25, seat + n * t_enter_thread, seat + n * (t_enter_thread + ins_L))
            new.append(Part(f"{j.name}_insert", ins, thread_m.body, "brass", mass=0.6e-3 if d <= 3 else 1.3e-3, kind="fastener",
                            spec=dict(type="heat-set insert", d=d, pn=f"brass heat-set insert M{d:g} x {ins_L}", joint=j.name)))
            j.info = dict(L=L, grip=round(grip, 2), engagement=round(min(L - grip, ins_L), 2), insert_in=thread_m.name)
        elif j.end == "nut":
            nyl = NYLOC[d][1]
            nwt = WASHER[d][2]
            grip = t_exit_stack
            need = grip + nwt + nyl + 2 * PITCH[d]
            Ls = [L for L in LEN[j.head] if L >= need]
            L = j.L or Ls[0]
            wseat = seat + n * t_exit_stack
            new.append(Part(f"{j.name}_nwasher", washer_solid(d, wseat, n), clamped[-1].body, "steel", mass=WASHER[d][1] ** 2 * 0.6e-6 * WASHER[d][2],
                            kind="fastener", spec=dict(type="washer", std="ISO 7089", d=d, joint=j.name)))
            new.append(Part(f"{j.name}_nut", nut_solid(d, wseat + n * nwt, n), clamped[-1].body, "steel",
                            mass=NYLOC[d][0] ** 2 * NYLOC[d][1] * 5.0e-6, kind="fastener",
                            spec=dict(type="nyloc nut", std="ISO 7040", d=d, joint=j.name)))
            j.info = dict(L=L, grip=round(grip, 2), protrusion=round(L - need + 2 * PITCH[d], 2))
        if j.washer:
            new.append(Part(f"{j.name}_washer", washer_solid(d, head_seat, n), mem[0].body, "steel",
                            mass=WASHER[d][1] ** 2 * 0.6e-6 * WASHER[d][2], kind="fastener",
                            spec=dict(type="washer", std="ISO 7089", d=d, joint=j.name)))
        s = screw_solid(j.head, d, j.info["L"] + washer_t, head_seat, n)
        new.append(Part(j.name, s, mem[0].body, "steel", mass=screw_mass(j.head, d, j.info["L"]), kind="fastener",
                        spec=dict(type="screw", std=STD[j.head], head=j.head, d=d, L=j.info["L"], grade=j.grade,
                                  torque_Nm=TORQUE[j.grade][d] if mat not in ("pa12", "insert") else None,
                                  threadlocker="Loctite 243" if mat not in ("pa12",) else None,
                                  joint=j.name, step=j.step)))
        j.info.update(seat=[round(c, 3) for c in seat], n=[round(c, 4) for c in n], d=d, head=j.head, end=j.end,
                      members=list(j.members))
    # one boolean per part
    for name, tools in cuts.items():
        p = parts[name]
        p.shape = p.shape.cut(*tools)
    return new


def cut_dowel_holes(parts: dict, pin: "Part", fit_clear=0.0):
    """Cut the pin's own cylinder (plus clearance) out of each member it sits in."""
    for m in pin.spec["members"]:
        tool = pin.shape if fit_clear == 0 else None
        parts[m].shape = parts[m].shape.cut(tool)


def dowel(name, d, L, p0, n, body, members):
    """ISO 8734 dowel pin (m6), pressed into the first member, slip fit in the others; holes cut by the caller."""
    n = unit(n)
    return Part(name, cyl(d / 2, p0, Vector(*p0) + n * L), body, "steel", mass=math.pi / 4 * d * d * L * 7.85e-6,
                kind="fastener", spec=dict(type="dowel pin", std="ISO 8734", d=d, L=L, members=members))
