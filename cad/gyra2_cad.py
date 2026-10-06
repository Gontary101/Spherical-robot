"""
GYRA Mk2 - split twin-crown tyre with differential drive, no CMGs, 360-degree pendulum.

What changes vs Mk1 (cad/gyra_cad.py, whose helpers this file reuses):
  * the tyre is two independently driven halves. Each half is Mk1's spherical zone with its
    crown offset outward by D_CROWN, so the robot stands on two contact patches 2*D_CROWN apart
  * each half has its own internal ring gear; the front motor drives the left half, the rear
    motor the right half. Sum of torques -> pendulum swing (drive), difference -> yaw couple
  * a slewing ring at the equator lets the halves turn relative to each other
  * CMGs and gimbal actuators removed; their mass goes into a bigger tungsten + battery bob
  * nothing of the spine sits in the pendulum's swept volume -> the pendulum can swing 360 deg
  * axle, pods and bays move outward by D_CROWN

    python cad/gyra2_cad.py [--check]                 # Mk2   -> cad/out2/
    GYRA_VARIANT=mk21 python cad/gyra2_cad.py         # Mk2.1 -> cad/out21/

Mk2.1 (root-cause fix of the sideways roll-overs, docs/08 sec. 2): crown offset 50 -> 90 mm (wider stance, roll
stiffness independent of the pendulum angle) and the worm-driven bob lean replaced by a back-drivable BLDC + ball-screw
drive (60 -> 180 deg/s, so the lean no longer rate-saturates against the ~2 s rocking mode).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import cadquery as cq
import numpy as np
from cadquery import Vector

sys.path.insert(0, os.path.dirname(__file__))
import gyra_cad as G  # noqa: E402
from gyra_cad import (Part, box, circle_pts, cyl_dir, cyl_y, extrude_xz, extrude_yz, gear_y,  # noqa: E402
                      halfspace, hull2d, mirror_x, mirror_y, revolve_y, sector_yz, slab_y, sphere, tube_y)
from params import (AXLE_HALF, AXLE_ID, AXLE_OD, CLOCKSPRING_D, CLOCKSPRING_Y, DRIVE_MOTOR_D,  # noqa: E402
                    DRIVE_MOTOR_L, DRIVE_PINION_Z, GEAR_MODULE, GEAR_R_BODY, GEAR_R_PITCH, GEAR_Z_RING,
                    LEVEL_MOTOR_D, LEVEL_MOTOR_POS, LUG_ARC_DEG, LUG_SKEW_DEG, N_LUG_PITCH, R_OUT,
                    R_SHELL_IN, R_SHELL_OUT, R_TREAD_BASE, RIM_TRACK_R, Y_EDGE)

VARIANT = os.environ.get("GYRA_VARIANT", "mk2")
OUT2 = os.path.join(os.path.dirname(__file__), "out21" if VARIANT == "mk21" else "out2")

# ---------------------------------------------------------------------------- Mk2 parameters
D_CROWN = 90.0 if VARIANT == "mk21" else 50.0   # crown offset of each half -> patches 2*D_CROWN apart
SEAM_GAP = 2.5                   # half-gap at the equator (slewing ring sits in it)
RING_Y = (8.0, 24.0)             # |y| span of each half's internal ring gear
PULLEY_Y = (26.0, 34.0)          # |y| span of the belt pulleys
ARC_HALF2 = 60.0                 # cradle arc half-angle
ARC_R2 = (145.0, 240.0)          # cradle arc radii
BOB_R2 = (147.0, 238.0)          # bob radii
BOB_HALF2 = 22.0                 # bob angular half-width
BOB_X2 = 62.0
BOB_TRAVEL2 = 40.0               # lateral travel +-40 deg
TUNGSTEN_KG = 11.0               # W-Ni-Fe heavy alloy (17.5 g/cc)
BATTERY2_KG = 3.9                # 13S4P 21700, 936 Wh incl. BMS
LUG_ROWS_U = ((-D_CROWN + 6, 16.0, 0.0), (24.0, 88.0, 0.5), (96.0, 160.0, 0.0), (168.0, 228.0, 0.5))
BAY_Y2 = (172.0, 262.0)


def lin(s: cq.Shape, dy: float) -> cq.Shape:
    return s.translate(Vector(0, dy, 0))


# ---------------------------------------------------------------------------- tyre halves
def build_tyre_half(side: int) -> list[Part]:
    """Built for the left half in its local frame u = y - D_CROWN, then shifted (and mirrored)."""
    s = "L" if side > 0 else "R"
    grp = "tyre" + s
    u0 = -D_CROWN + SEAM_GAP
    P = []
    shell = sphere(R_SHELL_OUT).cut(sphere(R_SHELL_IN)).intersect(slab_y(u0, Y_EDGE))
    tread = sphere(R_TREAD_BASE).cut(sphere(R_SHELL_OUT)).intersect(slab_y(u0 + 3.0, Y_EDGE))
    lug_shell = sphere(R_OUT).cut(sphere(R_TREAD_BASE - 0.5))
    pitch = 360.0 / N_LUG_PITCH
    lugs = []
    for (y0, y1, phase) in LUG_ROWS_U:
        def corner(theta_deg, y, rr):
            t = math.radians(theta_deg)
            return Vector(rr * math.cos(t), y, rr * math.sin(t))
        A, B = corner(0.0, y0, R_OUT), corner(LUG_SKEW_DEG, y1, R_OUT)
        C, D = corner(LUG_ARC_DEG, y0, R_OUT), corner(LUG_ARC_DEG + LUG_SKEW_DEG, y1, R_OUT)
        mid = corner(LUG_ARC_DEG / 2 + LUG_SKEW_DEG / 2, (y0 + y1) / 2, R_OUT)
        blk = slab_y(y0, y1).intersect(halfspace(A.cross(B), mid)).intersect(halfspace(D.cross(C), mid))
        lug = lug_shell.intersect(blk)
        for k in range(N_LUG_PITCH):
            off = 0.0 if side > 0 else pitch / 2
            lugs.append(lug.rotate(Vector(0, 0, 0), Vector(0, 1, 0), (k + phase) * pitch + off))
    lugs = cq.Compound.makeCompound(lugs)
    y_in = 215.0
    rim = revolve_y([
        (RIM_TRACK_R + 6, 229.0), (RIM_TRACK_R, 232.0), (RIM_TRACK_R + 6, 235.0),
        (G.shell_in_r_at(235.0) - 0.3, 235.0), (G.shell_in_r_at(y_in) - 0.3, y_in),
        (G.shell_in_r_at(y_in) - 4, y_in), (RIM_TRACK_R + 14, 229.0)])
    flange = revolve_y([(272.0, u0), (G.shell_in_r_at(u0) - 0.3, u0),
                        (G.shell_in_r_at(u0 + 6) - 0.3, u0 + 6), (272.0, u0 + 6)])
    pieces = [("drum", shell, "gfrp", "gfrp"), ("tread_base", tread, "rubber", "pu"),
              ("lugs", lugs, "rubber", "pu"), ("rim", rim, "alu", "al"), ("eq_flange", flange, "gfrp", "gfrp")]
    for name, shp, mat, rho in pieces:
        shp = lin(shp, D_CROWN)
        P.append(Part(f"{name}_{s}", shp if side > 0 else mirror_y(shp), grp, mat, rho=rho))
    # this half's ring gear (global coordinates, axis = y)
    ring = gear_y(GEAR_Z_RING, GEAR_MODULE, RING_Y[0], RING_Y[1], internal=True, body_r=GEAR_R_BODY - 2.0)
    ring = ring.cut(tube_y(GEAR_R_PITCH + 6, GEAR_R_BODY, RING_Y[0] + 4, RING_Y[1] - 4))
    P.append(Part(f"ring_gear_{s}", ring if side > 0 else mirror_y(ring), grp, "alu_dark", rho="pa66gf"))
    if side > 0:  # equatorial slewing ring (wire-race), carried by the left half
        P.append(Part("eq_slewing_ring", tube_y(272.0, G.shell_in_r_at(D_CROWN) - 1.0, -SEAM_GAP + 0.3, SEAM_GAP - 0.3),
                      grp, "steel", rho="steel"))
    return P


# ---------------------------------------------------------------------------- spine
def build_spine2() -> list[Part]:
    P = []
    half = AXLE_HALF + D_CROWN
    P.append(Part("axle", tube_y(AXLE_ID / 2, AXLE_OD / 2, -half + 12, half - 12), "spine", "carbon", rho="cfrp"))
    for side in (+1, -1):
        for p in G.build_pod(side):
            if p.name.startswith("gimbal_bearing"):
                continue
            p.shape = lin(p.shape, side * D_CROWN)
            P.append(p)
    for side in (+1, -1):
        s = "L" if side > 0 else "R"
        br = cyl_y(45.0, 66.0, 72.0)
        if side > 0:
            lx, _, lz = LEVEL_MOTOR_POS
            br = br.fuse(extrude_xz(hull2d(circle_pts(0, 0, 30) + circle_pts(lx, lz, 30)), 66.0, 72.0))
        br = br.cut(cyl_y(AXLE_OD / 2 - 0.2, 60, 80))
        P.append(Part(f"spine_bracket_{s}", br if side > 0 else mirror_y(br), "spine", "alu", rho="al"))
        bay = box(-50, 50, BAY_Y2[0], BAY_Y2[1], -72, -28)
        strut = box(-10, 10, 164, 174, -29, 0).fuse(cyl_y(28, 164, 174)).cut(cyl_y(AXLE_OD / 2 - 0.2, 140, 190))
        if side < 0:
            bay, strut = mirror_y(bay), mirror_y(strut)
        P.append(Part(f"bay_{s}", bay, "spine", "alu_dark", mass=0.85 if side > 0 else 0.55))
        P.append(Part(f"bay_strut_{s}", strut, "spine", "alu", rho="al"))
    lx, ly, lz = LEVEL_MOTOR_POS
    P.append(Part("level_motor", cyl_y(LEVEL_MOTOR_D / 2, 50.0, 66.0, lx, lz), "spine", "motor_black", mass=0.35))
    P.append(Part("clockspring", tube_y(AXLE_OD / 2 + 0.5, CLOCKSPRING_D / 2, *CLOCKSPRING_Y), "spine", "orange", mass=0.15))
    return P


# ---------------------------------------------------------------------------- yoke
def build_yoke2() -> list[Part]:
    P = []
    (px, pz), (mx, mz) = G.drive_geometry()
    hy0, hy1 = 36.0, 42.0
    plate = cyl_y(48.0, hy0, hy1)
    for sx in (1, -1):
        wing = hull2d(circle_pts(0, 0, 48) + circle_pts(sx * mx, mz, 40) + circle_pts(sx * px, pz, 20))
        plate = plate.fuse(extrude_xz(wing, hy0, hy1))
        plate = plate.fuse(extrude_xz([(sx * 70, -112), (sx * 86, -112), (sx * 86, -172), (sx * 70, -172)], hy0, hy1))
        plate = plate.cut(extrude_xz(circle_pts(sx * 95, -60, 22), hy0 - 1, hy1 + 1))
        plate = plate.cut(extrude_xz(circle_pts(sx * 150, -150, 13), hy0 - 1, hy1 + 1))
    plate = plate.cut(cyl_y(31.0, hy0 - 1, hy1 + 1))
    P += [Part("hanger_L", plate, "yoke", "alu", rho="al"), Part("hanger_R", mirror_y(plate), "yoke", "alu", rho="al")]
    brg = tube_y(AXLE_OD / 2 + 0.2, 31.0, hy0 - 6, hy1)
    P += [Part("hub_bearing_L", brg, "yoke", "steel", rho="steel"), Part("hub_bearing_R", mirror_y(brg), "yoke", "steel", rho="steel")]
    lgear = gear_y(46, 3.0, hy1, hy1 + 6, internal=False, bore=AXLE_OD / 2 + 2).cut(tube_y(AXLE_OD / 2 + 14, 58.0, hy1 - 1, hy1 + 7))
    P.append(Part("level_gear", lgear, "yoke", "steel", rho="pom"))

    ax0, ax1 = 75.0, 81.0
    arc = extrude_yz(sector_yz(ARC_R2[0], ARC_R2[1], -ARC_HALF2, ARC_HALF2), ax0, ax1)
    P += [Part("arc_plate_F", arc, "yoke", "alu", rho="al"), Part("arc_plate_B", mirror_x(arc), "yoke", "alu", rho="al")]
    rails = []
    for (r0, r1) in ((ARC_R2[0] + 6, ARC_R2[0] + 14), (ARC_R2[1] - 14, ARC_R2[1] - 6)):
        rl = extrude_yz(sector_yz(r0, r1, -ARC_HALF2 + 2, ARC_HALF2 - 2), ax0 - 5, ax0)
        rails += [rl, mirror_x(rl)]
    P.append(Part("bob_rails", cq.Compound.makeCompound(rails), "yoke", "steel", rho="steel"))
    P.append(Part("lean_rack", extrude_yz(sector_yz(192, 198, -ARC_HALF2 + 4, ARC_HALF2 - 4), ax0 - 8, ax0), "yoke", "steel", rho="steel"))

    def train(sign):   # front motor -> left half (+y), rear motor -> right half (-y)
        x_m, x_p = sign * mx, sign * px
        mot = cyl_y(DRIVE_MOTOR_D / 2, hy1, hy1 + DRIVE_MOTOR_L, x_m, mz)
        bell = cyl_y(DRIVE_MOTOR_D / 2 - 6, hy1 + DRIVE_MOTOR_L, hy1 + DRIVE_MOTOR_L + 6, x_m, mz)
        p_small = cyl_y(12.0, *PULLEY_Y, x_m, mz)
        p_big = cyl_y(24.0, *PULLEY_Y, x_p, pz)
        shaft_m = cyl_y(4.0, PULLEY_Y[0], hy1, x_m, mz)
        outer = hull2d(circle_pts(x_m, mz, 13.6) + circle_pts(x_p, pz, 25.6))
        inner = hull2d(circle_pts(x_m, mz, 12.1) + circle_pts(x_p, pz, 24.1))
        belt = extrude_xz(outer, PULLEY_Y[0] + 1, PULLEY_Y[1] - 1).cut(extrude_xz(inner, PULLEY_Y[0], PULLEY_Y[1]))
        pinion = gear_y(DRIVE_PINION_Z, GEAR_MODULE, RING_Y[0] + 1, RING_Y[1] - 1, x_p, pz, bore=4)
        shaft_p = cyl_y(6.0, -hy1, hy1, x_p, pz)
        parts = [mot.fuse(bell), p_small.fuse(shaft_m), p_big, belt, pinion]
        if sign < 0:
            parts = [mirror_y(v) for v in parts]
        t = "F" if sign > 0 else "B"
        mot, pm, pb, belt, pinion = parts
        return [Part(f"drive_motor_{t}", mot, "yoke", "motor_black", mass=0.90),
                Part(f"pulley_m_{t}", pm, "yoke", "alu", rho="al"), Part(f"pulley_p_{t}", pb, "yoke", "alu", rho="al"),
                Part(f"belt_{t}", belt, "yoke", "rubber", mass=0.03), Part(f"pinion_{t}", pinion, "yoke", "steel", rho="steel"),
                Part(f"pinion_shaft_{t}", shaft_p, "yoke", "steel", rho="steel")]
    P += train(+1) + train(-1)
    for sy in (1, -1):
        esc = box(-30, 30, 23, 35, -105, -62)
        P.append(Part(f"esc_{'L' if sy > 0 else 'R'}", esc if sy > 0 else mirror_y(esc), "yoke", "pcb", mass=0.15))
    # Mk2: the compute module rides on the pendulum (extra ballast, better m*L)
    P.append(Part("compute_pendulum", box(-45, 45, -20, 20, -140, -110), "yoke", "alu_dark", mass=0.7))
    return P


# ---------------------------------------------------------------------------- bob
def build_bob2() -> list[Part]:
    P = []
    w = extrude_yz(sector_yz(206.0, BOB_R2[1], -BOB_HALF2, BOB_HALF2), -BOB_X2 + 2, BOB_X2 - 2)
    batt = extrude_yz(sector_yz(BOB_R2[0] + 3, 203.0, -BOB_HALF2 + 1.5, BOB_HALF2 - 1.5), -BOB_X2 + 4, 30.0)
    if VARIANT == "mk21":        # BLDC + ball-screw lean drive: back-drivable, 180 deg/s
        worm = cq.Solid.makeCylinder(17.0, 30.0, Vector(30.0, 0, -180.0), Vector(1, 0, 0))
    else:                        # worm drive: self-locking, 60 deg/s
        worm = cq.Solid.makeCylinder(15.0, 24.0, Vector(34.0, 0, -180.0), Vector(1, 0, 0))
    sec = sector_yz(BOB_R2[0], BOB_R2[1], -BOB_HALF2 - 2, BOB_HALF2 + 2)
    carriage = [extrude_yz(sec, BOB_X2, BOB_X2 + 4), extrude_yz(sec, -BOB_X2 - 4, -BOB_X2)]
    pin = cyl_dir(10.0, 8.0, Vector(BOB_X2 + 4, 0, -195.0), Vector(1, 0, 0))
    P.append(Part("bob_tungsten", w, "bob", "lead", mass=TUNGSTEN_KG))
    P.append(Part("bob_battery", batt, "bob", "battery", mass=BATTERY2_KG))
    P.append(Part("bob_lean_drive" if VARIANT == "mk21" else "bob_worm_motor", worm, "bob", "motor_black",
                  mass=0.45 if VARIANT == "mk21" else 0.35))
    P.append(Part("bob_carriage", cq.Compound.makeCompound(carriage), "bob", "alu_dark", rho="al"))
    P.append(Part("bob_lean_pinion", pin, "bob", "steel", rho="steel"))
    return P


def build_all2() -> list[Part]:
    t0 = time.time()
    parts = build_tyre_half(+1) + build_tyre_half(-1) + build_spine2() + build_yoke2() + build_bob2()
    print(f"built {len(parts)} parts in {time.time() - t0:.1f}s")
    return parts


def explode2(p: Part) -> Vector:
    n = p.name
    side = 1 if n.endswith("_L") else (-1 if n.endswith("_R") else 0)
    if p.group in ("tyreL", "tyreR"):
        return Vector(0, (1 if p.group == "tyreL" else -1) * 300, 0)
    if n.startswith(("pod_", "lidar", "cam", "lens", "gnss", "led_ring", "rollers", "axle_flange")):
        return Vector(0, side * 560, 0)
    if p.group == "bob":
        return Vector(0, 0, -330)
    if n.startswith(("bay", "spine_bracket")):
        return Vector(0, side * 160, 0)
    return Vector(0, 0, 0)


ALLOWED2 = {frozenset(("pinion_F", "ring_gear_L")), frozenset(("pinion_B", "ring_gear_R")),
            frozenset(("bob_lean_pinion", "lean_rack")), frozenset(("eq_slewing_ring", "eq_flange_R"))}


def main():
    os.makedirs(OUT2, exist_ok=True)
    parts = build_all2()
    mp = {p.name: {**G.mass_props(p), "group": p.group} for p in parts}
    bodies = {g: G.combine([mp[p.name] for p in parts if p.group == g]) for g in sorted({p.group for p in parts})}
    robot = G.combine(list(mp.values()))
    json.dump({"robot": robot, "bodies": bodies, "parts": mp}, open(os.path.join(OUT2, "mass_properties.json"), "w"), indent=1)
    for g, b in bodies.items():
        print(f"  {g:6s} m={b['mass']:6.2f} kg  com={np.round(b['com_mm'], 1)}")
    print(f"  ROBOT  m={robot['mass']:6.2f} kg  com={np.round(robot['com_mm'], 1)}")
    G.OUT = OUT2
    G.explode_offset = explode2
    man = G.export(parts)
    json.dump(man, open(os.path.join(OUT2, "manifest.json"), "w"), indent=1)
    G.export_step(parts, os.path.join(OUT2, "gyra_mk2.step"))
    print("exported STEP + STL")
    if "--check" in sys.argv:
        G.ALLOWED = ALLOWED2
        report = []
        for pt in (0, 60, -60, 120, -120, 180):
            for bb in (0, -40, 40):
                hits = G.interference(G.pose(parts, pitch=pt, bob=bb))
                report.append({"pitch": pt, "bob": bb, "hits": hits})
                print(f"pose pitch={pt:4} bob={bb:4}: {len(hits)} hits {hits[:5] if hits else 'CLEAR'}", flush=True)
        json.dump(report, open(os.path.join(OUT2, "interference_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
