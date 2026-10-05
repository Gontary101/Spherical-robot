"""
GYRA Mk1 - parametric CAD model (CadQuery / OpenCascade).

Builds every part of the robot, grouped by kinematic body:

    tyre      - GFRP drum halves, PU tread + lugs, rim V-track rings, equatorial ring gear
    spine     - axle, sensor pods (plates, shields, LiDARs, cameras, rollers), brackets, bays
    yoke      - pendulum pitch body: hanger plates, cradle arc plates, rails, drive motors,
                belts, pinions (rotates about +y by `pitch`)
    bob       - battery + lead ballast carriage (rotates about +x by `bob` on the yoke)
    cmgL/cmgR - CMG gimbal housings (rotate about y through the CMG centre by `gimbal`)
    flyL/flyR - CMG flywheels (inside the gimbal housings)

Usage:
    python cad/gyra_cad.py                 # build, export STEP/STL/GLB + mass properties
    python cad/gyra_cad.py --check         # also run the multi-pose interference check
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import cadquery as cq
import numpy as np
from cadquery import Vector
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

sys.path.insert(0, os.path.dirname(__file__))
from params import *  # noqa: E402,F401,F403

OUT = os.path.join(os.path.dirname(__file__), "out")

# densities in kg/mm^3
RHO = {
    "al": 2.70e-6, "steel": 7.85e-6, "lead": 11.34e-6, "pu": 1.15e-6,
    "gfrp": 1.85e-6, "pa12": 1.01e-6, "brass": 8.50e-6,
    "pom": 1.41e-6, "pa66gf": 1.36e-6, "cfrp": 1.55e-6,
}


@dataclass
class Part:
    name: str
    shape: cq.Shape
    group: str
    material: str              # render material key
    rho: str | None = None     # density key, or None if mass given
    mass: float | None = None  # kg override (COTS parts)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def z_to_y(s: cq.Shape) -> cq.Shape:
    """Rotate a shape built around +z so that its axis becomes +y."""
    return s.rotate(Vector(0, 0, 0), Vector(1, 0, 0), -90)


def revolve_y(profile_ry: list[tuple[float, float]]) -> cq.Shape:
    """Revolve a closed (r, y) profile about the global y axis."""
    wp = cq.Workplane("XZ").polyline(profile_ry).close().revolve(360, (0, 0, 0), (0, 1, 0))
    return z_to_y(wp.val())


def cyl_y(r: float, y0: float, y1: float, x: float = 0.0, z: float = 0.0) -> cq.Shape:
    return cq.Solid.makeCylinder(r, y1 - y0, Vector(x, y0, z), Vector(0, 1, 0))


def tube_y(r_in: float, r_out: float, y0: float, y1: float, x=0.0, z=0.0) -> cq.Shape:
    return cyl_y(r_out, y0, y1, x, z).cut(cyl_y(r_in, y0 - 1, y1 + 1, x, z))


def cyl_dir(r: float, h: float, base: Vector, d: Vector) -> cq.Shape:
    return cq.Solid.makeCylinder(r, h, base, d)


def box(x0, x1, y0, y1, z0, z1) -> cq.Shape:
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, Vector(x0, y0, z0))


def sphere(r: float, c=(0, 0, 0)) -> cq.Shape:
    return cq.Solid.makeSphere(r, Vector(*c), angleDegrees1=-90, angleDegrees2=90)


def slab_y(y0: float, y1: float, big: float = 2000) -> cq.Shape:
    return box(-big, big, y0, y1, -big, big)


def extrude_xz(pts_xz, y0, y1) -> cq.Shape:
    """Polygon in the x-z plane extruded along +y from y0 to y1."""
    wp = cq.Workplane("XZ", origin=(0, 0, 0)).polyline(pts_xz).close().extrude(-(y1 - y0))
    return wp.val().translate(Vector(0, y0, 0))


def extrude_yz(pts_yz, x0, x1) -> cq.Shape:
    """Polygon in the y-z plane extruded along +x from x0 to x1."""
    wp = cq.Workplane("YZ").polyline(pts_yz).close().extrude(x1 - x0)
    return wp.val().translate(Vector(x0, 0, 0))


def sector_yz(r0, r1, a0_deg, a1_deg, n=40):
    """Annular sector in the y-z plane, angle measured from -z towards +y."""
    a = np.radians(np.linspace(a0_deg, a1_deg, n))
    outer = [(r1 * math.sin(t), -r1 * math.cos(t)) for t in a]
    inner = [(r0 * math.sin(t), -r0 * math.cos(t)) for t in a[::-1]]
    return outer + inner


def circle_pts(cx, cz, r, n=48, a0=0.0, a1=2 * math.pi):
    return [(cx + r * math.cos(t), cz + r * math.sin(t)) for t in np.linspace(a0, a1, n, endpoint=False)]


def hull2d(pts):
    from scipy.spatial import ConvexHull
    p = np.array(pts)
    h = ConvexHull(p)
    return [tuple(p[i]) for i in h.vertices]


def gear_profile(z_teeth: int, module: float, internal: bool):
    """Trapezoidal-tooth approximation of an involute gear outline (x-y plane)."""
    rp = module * z_teeth / 2
    if internal:
        r_tip, r_root = rp - module, rp + 1.25 * module
    else:
        r_tip, r_root = rp + module, rp - 1.25 * module
    pts = []
    p = 2 * math.pi / z_teeth
    for k in range(z_teeth):
        a = k * p
        # tooth (material): narrow at tip, wide at root
        pts += [
            (r_root, a - 0.30 * p), (r_tip, a - 0.16 * p),
            (r_tip, a + 0.16 * p), (r_root, a + 0.30 * p),
        ]
    if internal:   # gap between teeth is at the root for internal gears -> swap
        pts = []
        for k in range(z_teeth):
            a = k * p
            pts += [
                (r_tip, a - 0.22 * p), (r_root, a - 0.10 * p),
                (r_root, a + 0.10 * p), (r_tip, a + 0.22 * p),
            ]
    return [(r * math.cos(t), r * math.sin(t)) for r, t in pts]


def gear_y(z_teeth, module, y0, y1, cx=0.0, cz=0.0, internal=False, body_r=None, bore=0.0):
    """Gear with its axis along y, centred at (cx, ., cz)."""
    prof = gear_profile(z_teeth, module, internal)
    teeth = cq.Workplane("XY").polyline(prof).close().extrude(y1 - y0).val()
    if internal:
        disk = cq.Solid.makeCylinder(body_r, y1 - y0, Vector(0, 0, 0), Vector(0, 0, 1))
        g = disk.cut(teeth)
    else:
        g = teeth
        if bore > 0:
            g = g.cut(cq.Solid.makeCylinder(bore, y1 - y0 + 2, Vector(0, 0, -1), Vector(0, 0, 1)))
    g = z_to_y(g)                    # axis now +y, spanning y in [0, y1-y0]
    return g.translate(Vector(cx, y0, cz))


def halfspace(n: Vector, inside: Vector, big: float = 1500.0) -> cq.Shape:
    """Half-space bounded by the plane through the origin with normal n, containing `inside`."""
    n = n.normalized()
    if n.dot(inside) < 0:
        n = -n
    xd = n.cross(Vector(0, 1, 0)) if abs(n.y) < 0.9 else n.cross(Vector(1, 0, 0))
    pl = cq.Plane(origin=(0, 0, 0), xDir=xd.normalized(), normal=n)
    return cq.Workplane(pl).rect(2 * big, 2 * big).extrude(big).val()


def mirror_y(s: cq.Shape) -> cq.Shape:
    return s.mirror("XZ")


def mirror_x(s: cq.Shape) -> cq.Shape:
    return s.mirror("YZ")


# ---------------------------------------------------------------------------
# TYRE
# ---------------------------------------------------------------------------
def build_tyre() -> list[Part]:
    parts = []
    shell = sphere(R_SHELL_OUT).cut(sphere(R_SHELL_IN))
    tread = sphere(R_TREAD_BASE).cut(sphere(R_SHELL_OUT))
    g = TREAD_GROOVE_W / 2
    for side, (y0, y1) in (("L", (0.0, Y_EDGE)), ("R", (-Y_EDGE, 0.0))):
        drum = shell.intersect(slab_y(y0, y1))
        parts.append(Part(f"drum_{side}", drum, "tyre", "gfrp", rho="gfrp"))
        ty0, ty1 = (g, Y_EDGE) if side == "L" else (-Y_EDGE, -g)
        parts.append(Part(f"tread_base_{side}", tread.intersect(slab_y(ty0, ty1)), "tyre", "rubber", rho="pu"))

    # chevron lugs: one ruled block per row, then patterned around the y axis
    lug_shell = sphere(R_OUT).cut(sphere(R_TREAD_BASE - 0.5))
    pitch = 360.0 / N_LUG_PITCH
    lugs_L, lugs_R = [], []
    for (y0, y1, phase) in LUG_ROWS:
        def corner(theta_deg, y, rr):
            t = math.radians(theta_deg)
            return Vector(rr * math.cos(t), y, rr * math.sin(t))

        A, B = corner(0.0, y0, R_OUT), corner(LUG_SKEW_DEG, y1, R_OUT)
        C, D = corner(LUG_ARC_DEG, y0, R_OUT), corner(LUG_ARC_DEG + LUG_SKEW_DEG, y1, R_OUT)
        mid = corner(LUG_ARC_DEG / 2 + LUG_SKEW_DEG / 2, (y0 + y1) / 2, R_OUT)
        blk = slab_y(y0, y1).intersect(halfspace(A.cross(B), mid)).intersect(halfspace(D.cross(C), mid))
        lug = lug_shell.intersect(blk)
        for k in range(N_LUG_PITCH):
            ang = (k + phase) * pitch
            l = lug.rotate(Vector(0, 0, 0), Vector(0, 1, 0), ang)
            lugs_L.append(l)
            # right half: mirrored (chevron), half-pitch staggered
            lugs_R.append(mirror_y(lug).rotate(Vector(0, 0, 0), Vector(0, 1, 0), ang + pitch / 2))
    parts.append(Part("lugs_L", cq.Compound.makeCompound(lugs_L), "tyre", "rubber", rho="pu"))
    parts.append(Part("lugs_R", cq.Compound.makeCompound(lugs_R), "tyre", "rubber", rho="pu"))

    # rim V-track rings (aluminium) bonded inside each tyre edge
    y_in = 215.0
    prof = [
        (RIM_TRACK_R + 6, 229.0), (RIM_TRACK_R, 232.0), (RIM_TRACK_R + 6, 235.0),
        (shell_in_r_at(235.0) - 0.3, 235.0), (shell_in_r_at(y_in) - 0.3, y_in),
        (shell_in_r_at(y_in) - 4, y_in), (RIM_TRACK_R + 14, 229.0),
    ]
    rim = revolve_y(prof)
    parts.append(Part("rim_L", rim, "tyre", "alu", rho="al"))
    parts.append(Part("rim_R", mirror_y(rim), "tyre", "alu", rho="al"))

    # equatorial internal ring gear (joins the two drum halves)
    ring = gear_y(GEAR_Z_RING, GEAR_MODULE, -GEAR_FACE / 2, GEAR_FACE / 2,
                  internal=True, body_r=GEAR_R_BODY - 0.3)
    ring = ring.cut(tube_y(GEAR_R_PITCH + 6, GEAR_R_BODY, -GEAR_FACE / 2 + 4, GEAR_FACE / 2 - 4))  # I-section web
    parts.append(Part("ring_gear", ring, "tyre", "alu_dark", rho="pa66gf"))
    # drum equator flanges
    fl = revolve_y([(280.0, 9.0), (R_SHELL_IN - 0.3, 9.0), (shell_in_r_at(14) - 0.3, 14.0), (280.0, 14.0)])
    parts.append(Part("eq_flange_L", fl, "tyre", "gfrp", rho="gfrp"))
    parts.append(Part("eq_flange_R", mirror_y(fl), "tyre", "gfrp", rho="gfrp"))
    # equator bolts visible in the centre groove
    bolts = []
    for k in range(24):
        a = math.radians(k * 15 + 7.5)
        d = Vector(math.cos(a), 0, math.sin(a))
        bolts.append(cyl_dir(3.5, 12.0, d * (R_SHELL_OUT - 10.0), d))
    parts.append(Part("equator_bolts", cq.Compound.makeCompound(bolts), "tyre", "steel", rho="steel"))
    return parts


# ---------------------------------------------------------------------------
# SPINE (+ pods)
# ---------------------------------------------------------------------------
def _unit(x, y, z):
    n = math.sqrt(x * x + y * y + z * z)
    return Vector(x / n, y / n, z / n)


# (position direction on the dome, optical axis) for the left pod
CAMS = [
    (_unit(0.30, 0.94, 0.16), _unit(0.95, 0.31, 0.0)),    # front, 18 deg toe-out
    (_unit(-0.30, 0.94, 0.16), _unit(-0.95, 0.31, 0.0)),  # rear
    (_unit(0.0, 0.92, -0.39), _unit(0.0, 0.55, -0.83)),   # ground / near-field
]


def build_pod(side: int) -> list[Part]:
    s = "L" if side > 0 else "R"
    P = []

    def my(shape):
        return shape if side > 0 else mirror_y(shape)

    plate = cyl_y(POD_PLATE_R, POD_PLATE_Y0, POD_PLATE_Y1)
    plate = plate.cut(cyl_y(33.5, POD_PLATE_Y0 - 1, POD_PLATE_Y1 + 1))
    for k in range(6):   # lightening pockets between the rollers
        a = math.radians(POD_ROLLER_PHASE_DEG + 30 + k * 60)
        plate = plate.cut(cyl_y(30.0, POD_PLATE_Y0 + 2, POD_PLATE_Y1 + 1, 95 * math.cos(a), 95 * math.sin(a)))
    P.append(Part(f"pod_plate_{s}", my(plate), "spine", "alu_dark", rho="al"))

    # protective shield: spherical cap, 6 mm PA12 wall, with sensor apertures
    cap = sphere(R_OUT - 1.0).cut(sphere(R_OUT - 7.0)).intersect(slab_y(POD_PLATE_Y1, 400))
    cap = cap.cut(cyl_y(36.0, 200, 400))     # LiDAR aperture
    for d, a in CAMS:
        p = d * (R_OUT - 4.0)
        cap = cap.cut(cyl_dir(15.5, 40.0, p - a * 20.0, a))
    P.append(Part(f"pod_shield_{s}", my(cap), "spine", "nylon", rho="pa12"))

    # LiDAR (Livox Mid-360 class): body + optical window band
    lid_body = cyl_y(32.5, POD_PLATE_Y1, POD_PLATE_Y1 + 22).fuse(cyl_y(32.5, POD_PLATE_Y1 + 47, POD_PLATE_Y1 + 58))
    lid_win = cyl_y(31.5, POD_PLATE_Y1 + 22, POD_PLATE_Y1 + 47)
    P.append(Part(f"lidar_body_{s}", my(lid_body), "spine", "lidar_white", mass=0.17))
    P.append(Part(f"lidar_window_{s}", my(lid_win), "spine", "glass", mass=0.095))

    # cameras (global-shutter modules) + lenses
    for i, (d, a) in enumerate(CAMS):
        p = d * (R_OUT - 4.0)
        body = cyl_dir(14.0, 18.0, p - a * 18.0, a)
        lens = cyl_dir(9.0, 4.0, p - a * 1.0, a)
        P.append(Part(f"cam{i}_{s}", my(body), "spine", "motor_black", mass=0.03))
        P.append(Part(f"lens{i}_{s}", my(lens), "spine", "glass", mass=0.005))

    # GNSS antenna radome at the top of the pod
    d = _unit(0, 0.87, 0.49)
    rad = cyl_dir(22.0, 10.0, d * 284.0, d)
    P.append(Part(f"gnss_{s}", my(rad), "spine", "nylon_light", mass=0.06))

    # status light ring at the pod rim
    led = tube_y(POD_PLATE_R - 7, POD_PLATE_R - 1.5, POD_PLATE_Y1, POD_PLATE_Y1 + 3)
    P.append(Part(f"led_ring_{s}", my(led), "spine", "led", mass=0.04))

    # 6 V-rollers riding the rim track (roller just touches the track tip)
    rollers = []
    r_roll = POD_ROLLER_R
    rc = RIM_TRACK_R - r_roll - 0.5
    for k in range(POD_ROLLER_N):
        a = math.radians(POD_ROLLER_PHASE_DEG + k * 360 / POD_ROLLER_N)
        x, z = rc * math.cos(a), rc * math.sin(a)
        rollers.append(cyl_y(r_roll, 226.0, 236.0, x, z))
        rollers.append(cyl_y(5.0, 236.0, POD_PLATE_Y0, x, z))     # standoff pin
    P.append(Part(f"rollers_{s}", my(cq.Compound.makeCompound(rollers)), "spine", "steel", mass=0.30))

    # axle end flange
    fl = cyl_y(45.0, 226.0, POD_PLATE_Y0).cut(cyl_y(AXLE_OD / 2 - 0.2, 220, 240))
    P.append(Part(f"axle_flange_{s}", my(fl), "spine", "alu", rho="al"))

    # outer gimbal bearing block for the CMG on the pod plate's inner face
    blk = cyl_y(20.0, 229.5, POD_PLATE_Y0, 0.0, CMG_Z)
    P.append(Part(f"gimbal_bearing_{s}", my(blk), "spine", "alu", rho="al"))
    return P


def build_spine() -> list[Part]:
    P = []
    axle = tube_y(AXLE_ID / 2, AXLE_OD / 2, -AXLE_HALF + 12, AXLE_HALF - 12)
    P.append(Part("axle", axle, "spine", "carbon", rho="cfrp"))
    P += build_pod(+1)
    P += build_pod(-1)

    # inner spine brackets: hub ring + upper arm to the gimbal actuator (+ lower lobe on +y)
    for side in (+1, -1):
        s = "L" if side > 0 else "R"
        up = [(-40, 0), (-40, CMG_Z + 38), (40, CMG_Z + 38), (40, 0)]
        br = extrude_xz(up, 66.0, 72.0).fuse(cyl_y(45.0, 66.0, 72.0))
        if side > 0:   # lobe that carries the levelling motor
            lx, _, lz = LEVEL_MOTOR_POS
            lobe = extrude_xz(hull2d(circle_pts(0, 0, 30) + circle_pts(lx, lz, 30)), 66.0, 72.0)
            br = br.fuse(lobe)
        br = br.cut(cyl_y(AXLE_OD / 2 - 0.2, 60, 80))
        if side < 0:
            br = mirror_y(br)
        P.append(Part(f"spine_bracket_{s}", br, "spine", "alu", rho="al"))

        # gimbal actuator (CubeMars AK70-10 class), inner side of each CMG
        y0, y1 = 72.0, 72.0 + GIMBAL_ACT_L
        act = cyl_y(GIMBAL_ACT_D / 2, y0, y1, 0.0, CMG_Z)
        P.append(Part(f"gimbal_actuator_{s}", act if side > 0 else mirror_y(act), "spine", "motor_black", mass=0.52))

        # electronics bays below the axle (left: compute, right: power/IMU/radio)
        bay = box(BAY_X[0], BAY_X[1], BAY_Y[0], BAY_Y[1], BAY_Z[0], BAY_Z[1])
        strut = box(-10, 10, 124, 134, BAY_Z[1] - 1, 0).fuse(cyl_y(28, 124, 134))
        strut = strut.cut(cyl_y(AXLE_OD / 2 - 0.2, 100, 150))
        if side < 0:
            bay, strut = mirror_y(bay), mirror_y(strut)
        P.append(Part(f"bay_{s}", bay, "spine", "alu_dark",
                      mass=0.85 if side > 0 else 0.45))
        P.append(Part(f"bay_strut_{s}", strut, "spine", "alu", rho="al"))

    lx, ly, lz = LEVEL_MOTOR_POS
    lm = cyl_y(LEVEL_MOTOR_D / 2, 50.0, 66.0, lx, lz)
    P.append(Part("level_motor", lm, "spine", "motor_black", mass=0.35))
    cs = tube_y(AXLE_OD / 2 + 0.5, CLOCKSPRING_D / 2, *CLOCKSPRING_Y)
    P.append(Part("clockspring", cs, "spine", "orange", mass=0.15))
    return P


# ---------------------------------------------------------------------------
# CMG pair
# ---------------------------------------------------------------------------
def build_cmg(side: int) -> list[Part]:
    s = "L" if side > 0 else "R"
    yc = side * CMG_Y
    P = []
    hr, hh = 58.0, 44.0
    outer = cq.Solid.makeCylinder(hr, hh, Vector(0, yc, CMG_Z - hh / 2), Vector(0, 0, 1))
    inner = cq.Solid.makeCylinder(hr - 4, hh - 8, Vector(0, yc, CMG_Z - hh / 2 + 4), Vector(0, 0, 1))
    housing = outer.cut(inner)
    # cut a window so the flywheel is visible in renders (polycarbonate lid is modelled separately)
    housing = housing.cut(cq.Solid.makeCylinder(hr - 12, 10, Vector(0, yc, CMG_Z + hh / 2 - 5), Vector(0, 0, 1)))
    trunnion_in = cyl_y(12.0, 72.0 + GIMBAL_ACT_L, CMG_Y - hr + 2, 0, CMG_Z)
    trunnion_out = cyl_y(12.0, CMG_Y + hr - 2, 229.5, 0, CMG_Z)
    if side < 0:
        trunnion_in, trunnion_out = mirror_y(trunnion_in), mirror_y(trunnion_out)
    lid = cq.Solid.makeCylinder(hr - 12, 2.0, Vector(0, yc, CMG_Z + hh / 2 - 3), Vector(0, 0, 1))
    spin_motor = cq.Solid.makeCylinder(14.0, 12.0, Vector(0, yc, CMG_Z - hh / 2 + 4), Vector(0, 0, 1))
    P.append(Part(f"cmg_housing_{s}", housing.fuse(trunnion_in).fuse(trunnion_out), f"cmg{s}", "alu", rho="al"))
    P.append(Part(f"cmg_lid_{s}", lid, f"cmg{s}", "polycarb", mass=0.03))
    P.append(Part(f"cmg_spin_motor_{s}", spin_motor, f"cmg{s}", "motor_black", mass=0.15))
    fly = cq.Solid.makeCylinder(FLY_OD / 2, FLY_T, Vector(0, yc, CMG_Z - FLY_T / 2 + 2), Vector(0, 0, 1))
    fly = fly.cut(cq.Solid.makeCylinder(FLY_ID / 2, FLY_T + 2, Vector(0, yc, CMG_Z - FLY_T / 2 + 1), Vector(0, 0, 1)))
    hub = cq.Solid.makeCylinder(FLY_ID / 2, 8.0, Vector(0, yc, CMG_Z - 2), Vector(0, 0, 1))
    hub = hub.cut(cq.Solid.makeCylinder(8.0, 10, Vector(0, yc, CMG_Z - 3), Vector(0, 0, 1)))
    P.append(Part(f"flywheel_{s}", fly.fuse(hub), f"fly{s}", "brass", rho="steel",
                  meta={"spin_axis": [0, 0, 1], "rpm": CMG_RPM, "dir": side}))
    return P


# ---------------------------------------------------------------------------
# YOKE (pendulum pitch body)
# ---------------------------------------------------------------------------
def drive_geometry():
    a = math.radians(DRIVE_ANGLE_DEG)
    pin = (DRIVE_PINION_RC * math.sin(a), -DRIVE_PINION_RC * math.cos(a))
    mot = (DRIVE_MOTOR_RC * math.sin(a), -DRIVE_MOTOR_RC * math.cos(a))
    return pin, mot


def build_yoke() -> list[Part]:
    P = []
    (px, pz), (mx, mz) = drive_geometry()
    hy0, hy1 = HANGER_Y - HANGER_T / 2 - 14, HANGER_Y - HANGER_T / 2 - 8   # 36..42
    # hanger plate outline (x-z), symmetric in x
    plate = cyl_y(48.0, hy0, hy1)
    for sx in (1, -1):
        wing = hull2d(circle_pts(0, 0, 48) + circle_pts(sx * mx, mz, 40) + circle_pts(sx * px, pz, 20))
        plate = plate.fuse(extrude_xz(wing, hy0, hy1))
        # junction lobe down to the arc plate
        plate = plate.fuse(extrude_xz([(sx * 70, -112), (sx * 86, -112), (sx * 86, -178), (sx * 70, -178)], hy0, hy1))
    # lightening holes
    for sx in (1, -1):
        plate = plate.cut(extrude_xz(circle_pts(sx * 95, -60, 22), hy0 - 1, hy1 + 1))
        plate = plate.cut(extrude_xz(circle_pts(sx * 150, -150, 13), hy0 - 1, hy1 + 1))
    plate = plate.cut(cyl_y(31.0, hy0 - 1, hy1 + 1))          # hub bearing seat
    P.append(Part("hanger_L", plate, "yoke", "alu", rho="al"))
    P.append(Part("hanger_R", mirror_y(plate), "yoke", "alu", rho="al"))

    brg = tube_y(AXLE_OD / 2 + 0.2, 31.0, hy0 - 6, hy1)
    P.append(Part("hub_bearing_L", brg, "yoke", "steel", rho="steel"))
    P.append(Part("hub_bearing_R", mirror_y(brg), "yoke", "steel", rho="steel"))

    # levelling ring gear on the +y hanger outer face
    lgear = gear_y(46, 3.0, hy1, hy1 + 6, internal=False, bore=AXLE_OD / 2 + 2)
    lgear = lgear.cut(tube_y(AXLE_OD / 2 + 14, 58.0, hy1 - 1, hy1 + 7))
    P.append(Part("level_gear", lgear, "yoke", "steel", rho="pom"))

    # cradle arc plates (y-z sectors about the x axis) + curved rails + rack
    ax0 = ARC_PLATE_X - ARC_PLATE_T / 2 - 3   # 75
    ax1 = ax0 + ARC_PLATE_T                   # 81
    arc = extrude_yz(sector_yz(ARC_R_IN, ARC_R_OUT, -ARC_HALF_DEG, ARC_HALF_DEG), ax0, ax1)
    P.append(Part("arc_plate_F", arc, "yoke", "alu", rho="al"))
    P.append(Part("arc_plate_B", mirror_x(arc), "yoke", "alu", rho="al"))
    rails = []
    for (r0, r1) in ((ARC_R_IN + 6, ARC_R_IN + 14), (ARC_R_OUT - 14, ARC_R_OUT - 6)):
        rl = extrude_yz(sector_yz(r0, r1, -ARC_HALF_DEG + 2, ARC_HALF_DEG - 2), ax0 - 5, ax0)
        rails += [rl, mirror_x(rl)]
    P.append(Part("bob_rails", cq.Compound.makeCompound(rails), "yoke", "steel", rho="steel"))
    rack = extrude_yz(sector_yz(192, 198, -ARC_HALF_DEG + 4, ARC_HALF_DEG - 4), ax0 - 8, ax0)
    P.append(Part("lean_rack", rack, "yoke", "steel", rho="steel"))

    # drive train: front motor on +y side, rear motor mirrored to -y side
    def train(sign):
        x_m, x_p = sign * mx, sign * px
        mot = cyl_y(DRIVE_MOTOR_D / 2, hy1, hy1 + DRIVE_MOTOR_L, x_m, mz)
        bell = cyl_y(DRIVE_MOTOR_D / 2 - 6, hy1 + DRIVE_MOTOR_L, hy1 + DRIVE_MOTOR_L + 6, x_m, mz)
        p_small = cyl_y(12.0, 12.0, 24.0, x_m, mz)
        p_big = cyl_y(24.0, 12.0, 24.0, x_p, pz)
        shaft_m = cyl_y(4.0, 12.0, hy1, x_m, mz)
        outer = hull2d(circle_pts(x_m, mz, 13.6) + circle_pts(x_p, pz, 25.6))
        inner = hull2d(circle_pts(x_m, mz, 12.1) + circle_pts(x_p, pz, 24.1))
        belt = extrude_xz(outer, 13.0, 23.0).cut(extrude_xz(inner, 12.0, 24.0))
        pinion = gear_y(DRIVE_PINION_Z, GEAR_MODULE, -GEAR_FACE / 2 + 1, GEAR_FACE / 2 - 1, x_p, pz, bore=4)
        shaft_p = cyl_y(6.0, -hy1, hy1, x_p, pz)
        if sign < 0:
            mot, bell, p_small, p_big, shaft_m, belt = (mirror_y(v) for v in (mot, bell, p_small, p_big, shaft_m, belt))
        t = "F" if sign > 0 else "B"
        return [
            Part(f"drive_motor_{t}", mot.fuse(bell), "yoke", "motor_black", mass=0.90),
            Part(f"pulley_m_{t}", p_small.fuse(shaft_m), "yoke", "alu", rho="al"),
            Part(f"pulley_p_{t}", p_big, "yoke", "alu", rho="al"),
            Part(f"belt_{t}", belt, "yoke", "rubber", mass=0.03),
            Part(f"pinion_{t}", pinion, "yoke", "steel", rho="steel"),
            Part(f"pinion_shaft_{t}", shaft_p, "yoke", "steel", rho="steel"),
        ]
    P += train(+1) + train(-1)

    # motor controllers (ODrive S1 class) on the hanger plates near the hub
    for sx, sy in ((1, 1), (-1, -1)):
        esc = box(-30, 30, hy0 - 13, hy0 - 1, -105, -62)   # inner face of the hanger plate
        if sy < 0:
            esc = mirror_y(esc)
        P.append(Part(f"esc_{'L' if sy > 0 else 'R'}", esc, "yoke", "pcb", mass=0.15))
    return P


# ---------------------------------------------------------------------------
# BOB (battery + ballast carriage)
# ---------------------------------------------------------------------------
def build_bob() -> list[Part]:
    P = []
    lead = extrude_yz(sector_yz(203.0, BOB_R_OUT, -BOB_HALF_DEG, BOB_HALF_DEG), -BOB_X + 2, BOB_X - 2)
    batt = extrude_yz(sector_yz(BOB_R_IN + 4, 199.0, -BOB_HALF_DEG + 1.5, BOB_HALF_DEG - 1.5), -BOB_X + 6, 34.0)
    worm = cq.Solid.makeCylinder(15.0, 24.0, Vector(38.0, 0, -178.0), Vector(1, 0, 0))
    sec = sector_yz(BOB_R_IN, BOB_R_OUT, -BOB_HALF_DEG - 2, BOB_HALF_DEG + 2)
    carriage = [extrude_yz(sec, BOB_X, BOB_X + 4), extrude_yz(sec, -BOB_X - 4, -BOB_X)]
    pin = cyl_dir(10.0, 8.0, Vector(BOB_X + 4, 0, -195.0), Vector(1, 0, 0))
    P.append(Part("bob_lead", lead, "bob", "lead", mass=BOB_LEAD_KG))
    P.append(Part("bob_battery", batt, "bob", "battery", mass=BATTERY_KG))
    P.append(Part("bob_worm_motor", worm, "bob", "motor_black", mass=0.35))
    P.append(Part("bob_carriage", cq.Compound.makeCompound(carriage), "bob", "alu_dark", rho="al"))
    P.append(Part("bob_lean_pinion", pin, "bob", "steel", rho="steel"))
    return P


# ---------------------------------------------------------------------------
# assembly / posing
# ---------------------------------------------------------------------------
def build_all() -> list[Part]:
    t0 = time.time()
    parts = build_tyre() + build_spine() + build_cmg(+1) + build_cmg(-1) + build_yoke() + build_bob()
    print(f"built {len(parts)} parts in {time.time() - t0:.1f}s")
    return parts


def pose(parts: list[Part], pitch=0.0, bob=0.0, gimbal=(0.0, 0.0), explode=0.0) -> list[Part]:
    """Return re-located copies. pitch: yoke about +y; bob: about +x; gimbal: (L, R) about y."""
    out = []
    O = Vector(0, 0, 0)
    for p in parts:
        s = p.shape
        g = p.group
        if g == "bob" and bob:
            s = s.rotate(O, Vector(1, 0, 0), bob)
        if g in ("yoke", "bob") and pitch:
            s = s.rotate(O, Vector(0, 1, 0), pitch)
        if g in ("cmgL", "flyL") and gimbal[0]:
            s = s.rotate(Vector(0, CMG_Y, CMG_Z), Vector(0, CMG_Y + 1, CMG_Z), gimbal[0])
        if g in ("cmgR", "flyR") and gimbal[1]:
            s = s.rotate(Vector(0, -CMG_Y, CMG_Z), Vector(0, -CMG_Y + 1, CMG_Z), gimbal[1])
        if explode:
            s = s.translate(explode_offset(p) * explode)
        out.append(Part(p.name, s, p.group, p.material, p.rho, p.mass, p.meta))
    return out


def explode_offset(p: Part) -> Vector:
    n = p.name
    side = 1 if n.endswith("_L") else (-1 if n.endswith("_R") else 0)
    if n.startswith(("drum", "tread_base", "lugs", "rim", "eq_flange")):
        return Vector(0, side * 260, 0)
    if n.startswith(("pod_", "lidar", "cam", "lens", "gnss", "led_ring", "rollers", "axle_flange", "gimbal_bearing")):
        return Vector(0, side * 480, 0)
    if p.group in ("cmgL", "flyL"):
        return Vector(0, 60, 330)
    if p.group in ("cmgR", "flyR"):
        return Vector(0, -60, 330)
    if p.group == "bob":
        return Vector(0, 0, -290)
    if n.startswith(("bay", "gimbal_actuator", "spine_bracket")):
        s = 1 if "_L" in n else -1
        return Vector(0, s * 150, 0)
    if n == "ring_gear" or n == "equator_bolts":
        return Vector(0, 0, 0)
    return Vector(0, 0, 0)


# ---------------------------------------------------------------------------
# mass properties
# ---------------------------------------------------------------------------
def mass_props(p: Part):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(p.shape.wrapped, props)
    vol = props.Mass()                            # mm^3
    c = props.CentreOfMass()
    I = props.MatrixOfInertia()                   # about CoM, density 1 (mm^5)
    if p.mass is not None:
        m = p.mass
    else:
        m = vol * RHO[p.rho]
    k = m / vol if vol > 0 else 0.0
    Imat = np.array([[I.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)]) * k * 1e-6  # kg m^2
    return {"volume_cm3": vol / 1e3, "mass": m, "com_mm": [c.X(), c.Y(), c.Z()], "I_com_kgm2": Imat.tolist()}


def combine(props: list[dict]):
    m = sum(p["mass"] for p in props)
    c = sum(np.array(p["com_mm"]) * p["mass"] for p in props) / m
    I = np.zeros((3, 3))
    for p in props:
        d = (np.array(p["com_mm"]) - c) * 1e-3
        I += np.array(p["I_com_kgm2"]) + p["mass"] * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return {"mass": m, "com_mm": c.tolist(), "I_com_kgm2": I.tolist()}


# ---------------------------------------------------------------------------
# interference check
# ---------------------------------------------------------------------------
ALLOWED = {
    frozenset(("pinion_F", "ring_gear")), frozenset(("pinion_B", "ring_gear")),
    frozenset(("bob_lean_pinion", "lean_rack")),
}


def body_of(p: Part) -> str:
    g = p.group
    return {"flyL": "cmgL", "flyR": "cmgR"}.get(g, g)


def interference(parts: list[Part], tol_mm3=1.0):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    def bb(s):
        b = Bnd_Box()
        BRepBndLib.Add_s(s.wrapped, b)
        return b

    skip = ("lugs", "tread_base", "equator_bolts")
    cand = [p for p in parts if not p.name.startswith(skip)]
    boxes = [bb(p.shape) for p in cand]
    hits = []
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = cand[i], cand[j]
            if body_of(a) == body_of(b):
                continue
            if frozenset((a.name, b.name)) in ALLOWED:
                continue
            if boxes[i].IsOut(boxes[j]):
                continue
            common = a.shape.intersect(b.shape)
            v = common.Volume() if common is not None else 0.0
            if v > tol_mm3:
                hits.append((a.name, b.name, round(v, 1)))
    return hits


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def export(parts: list[Part], tag="", stl=True):
    os.makedirs(os.path.join(OUT, "stl" + tag), exist_ok=True)
    manifest = []
    for p in parts:
        fn = os.path.join(OUT, "stl" + tag, p.name + ".stl")
        if stl:
            fine = p.name.startswith(("lugs", "drum", "tread", "pod_shield", "ring_gear", "pinion", "level_gear"))
            p.shape.exportStl(fn, tolerance=0.25 if fine else 0.4, angularTolerance=0.15)
        e = explode_offset(p)
        manifest.append({"name": p.name, "group": p.group, "material": p.material,
                         "stl": os.path.relpath(fn, OUT), "explode_mm": [e.x, e.y, e.z]})
    return manifest


def export_step(parts: list[Part], path):
    from render_colors import RGB
    asm = cq.Assembly(name="GYRA_Mk1")
    for p in parts:
        r, g, b = RGB.get(p.material, (0.6, 0.6, 0.6))
        asm.add(p.shape, name=p.name, color=cq.Color(r, g, b))
    asm.save(path, exportType="STEP")


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = build_all()
    t0 = time.time()
    mp = {p.name: {**mass_props(p), "group": p.group} for p in parts}
    bodies = {}
    for g in sorted({body_of(p) if not p.group.startswith("fly") else p.group for p in parts}):
        bodies[g] = combine([mp[p.name] for p in parts if p.group == g])
    robot = combine(list(mp.values()))
    with open(os.path.join(OUT, "mass_properties.json"), "w") as f:
        json.dump({"robot": robot, "bodies": bodies, "parts": mp}, f, indent=1)
    print(f"mass properties in {time.time() - t0:.1f}s")
    for g, b in bodies.items():
        print(f"  {g:6s} m={b['mass']:6.2f} kg  com={np.round(b['com_mm'], 1)}")
    print(f"  ROBOT  m={robot['mass']:6.2f} kg  com={np.round(robot['com_mm'], 1)}")

    man = export(parts)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    export_step(parts, os.path.join(OUT, "gyra_mk1.step"))
    print("exported STEP + STL")

    if "--check" in sys.argv:
        quick = "--quick" in sys.argv
        pts_ = (0, 75, -75) if quick else (0, 45, -45, -75, 75)
        gms_ = ((0, 0), (90, -90)) if quick else ((0, 0), (90, -90), (-60, 60))
        poses = [(pt, bb, gm) for pt in pts_ for bb in (0, -30, 30) for gm in gms_]
        report = []
        for (pt, bb, gm) in poses:
            hits = interference(pose(parts, pitch=pt, bob=bb, gimbal=gm))
            report.append({"pitch": pt, "bob": bb, "gimbal": gm, "hits": hits})
            print(f"pose pitch={pt:4} bob={bb:4} gimbal={gm}: {len(hits)} hits {hits[:6] if hits else 'CLEAR'}")
        with open(os.path.join(OUT, "interference_report.json"), "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
