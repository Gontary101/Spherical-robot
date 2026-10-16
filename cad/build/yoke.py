"""
Yoke (the pendulum frame, turns 360 deg about the axle) and its drive cassettes, battery and power electronics.

    hangers         2 x 6061 8 mm, waterjet + CNC: hub-bearing cartridge, pinion bearing (6001-2RSNR), bars down
                    to the arc plates, lands for the ODrive S1s and the power box. Outlines keep |x| < 89 mm inside
                    r_yz 120 mm on the front half (the lean actuator sweeps from 120.5 mm) and inside 218 mm on the
                    rear half (nothing of the bob comes closer there than the tungsten at 222 mm).
    hub cartridges  6061, 6908-2RS bore 62 H7 + DIN 472 ring
    level gear      7075 m1.5 z96 ring on the left hanger (meshes the spine's 14T pinion)
    arc plates      2 x 6061 6 mm sectors, joined to the hangers by 4 machined angle brackets (between the hangers)
    track rings     2 x 1045 6.35 mm segments, 90 deg V on both edges (DualVee size 2 track profile), on spacers
    lean rack       7075 internal gear segment m1.5 pitch r 195, on the front track ring
    drive cassettes front -> left half, rear -> right half: pinion shaft (42CrMo4, 24T m1.5), 2 x 6001-2RSNR,
                    Gates P32 / P20 HTD 5M, 285-5M-15 belt, D6374 on a slotted outer plate, 3 spacers to the hanger
    battery         2 modules of 6S2P Molicel P50B (12S2P, 432 Wh), outside the arc plates on the hanger bars
    power           2 x ODrive S1 on ACC-HS04 spreaders on the hangers' inner faces; power box (BMS, fuse, anti-spark)
    slip ring       housing turns with the yoke (pin on the right hanger in the tab slot)
"""
import math

import cadquery as cq
from cadquery import Vector

import spec as S
from cots import ball_bearing, circlip_bore, circlip_shaft, d6374, d6374_holes
from lib import (Joint, Part, box, circle_pts, compound, cut_dowel_holes, cyl, dowel, extrude_xz, extrude_yz, fuse,
                 hull2d, mirror_x, mirror_y, rot_z, sector_yz, solid, tube, unit)


def rot180z(s):
    return s.rotate(Vector(0, 0, 0), Vector(0, 0, 1), 180)


def outline_union(pieces, y0, y1):
    """Union of convex 2-D pieces (each a list of (x, z) points, or ('c', x, z, r)) extruded from y0 to y1."""
    sol = None
    for pc in pieces:
        pts = hull2d(circle_pts(pc[1], pc[2], pc[3], 40)) if pc[0] == "c" else hull2d(sum((circle_pts(*c, 40) for c in pc[1]), []))
        e = extrude_xz(pts, min(y0, y1), max(y0, y1))
        sol = e if sol is None else sol.fuse(e)
    return sol.clean()


def drive_geom(sign):
    a = math.radians(S.DRIVE_ANGLE)
    pin = (sign * S.R_PIN_C * math.sin(a), -S.R_PIN_C * math.cos(a))
    mot = (sign * S.R_MOT_C * math.sin(a), -S.R_MOT_C * math.cos(a))
    return pin, mot


def build_yoke():
    P, J = [], []
    H0, H1 = S.HANGER_Y
    # ---------------------------------------------------------------- hangers
    (pxF, pzF), (mxF, mzF) = drive_geom(+1)
    (pxB, pzB), (mxB, mzB) = drive_geom(-1)
    SO_F = [(160.0, -80.0), (225.0, -140.0), (150.0, -215.0)]          # cassette spacers (front cassette)
    SO_B = [(-x, z) for x, z in SO_F]
    MOD_SO = [(105.0, -95.0), (123.0, -95.0), (105.0, -200.0), (123.0, -200.0)]   # battery module spacers (x>0 side)
    common = [("c", 0.0, 0.0, 64.0), ("h", [(0, 0, 36), (114, -62, 18)]), ("h", [(114, -62, 18), (114, -205, 18)]),
              ("h", [(0, 0, 36), (-114, -62, 18)]), ("h", [(-114, -62, 18), (-114, -205, 18)]),
              ("h", [(0, 0, 30), (0, 73, 12)]),                                                   # top lobe (slip-ring pin on R)
              ("h", [(-62, -50, 4), (0, -50, 4), (-62, -118, 4), (0, -118, 4)]),                  # ODrive S1 land
              ("h", [(-58, -118, 4), (-6, -118, 4), (-58, -208, 4), (-6, -208, 4)])]              # power-box land
    def hanger_pieces(sign):
        px, pz = (pxF, pzF) if sign > 0 else (pxB, pzB)
        so = SO_F if sign > 0 else SO_B
        bar = 114.0 * sign
        return common + [("h", [(bar, -125, 18), (px, pz, 24)]), ("h", [(bar, -62, 18), (so[0][0], so[0][1], 12)]),
                         ("h", [(px, pz, 24), (so[1][0], so[1][1], 12)]), ("h", [(bar, -205, 18), (so[2][0], so[2][1], 12)])]
    for sign, s, (ya, yb) in ((+1, "L", (H0, H1)), (-1, "R", (-H1, -H0))):
        h = outline_union(hanger_pieces(sign), ya, yb)
        h = h.cut(cyl(33.0, (0, ya - 1, 0), (0, yb + 1, 0)))                                     # hub cartridge spigot
        px, pz = (pxF, pzF) if sign > 0 else (pxB, pzB)
        h = h.cut(cyl(14.0, (px, ya - 1, pz), (px, yb + 1, pz)))                                 # 6001 bore 28 H7
        P.append(Part(f"hanger_{s}", h, "yoke", "alu", rho="al",
                      spec=dict(material="6061-T6 plate 8 mm", process="waterjet + CNC (bores, tapped holes)", finish="clear anodise")))
    # ---------------------------------------------------------------- hub bearings + cartridges
    B = S.COTS["hub_bearing"]
    for sign, s in ((1, "L"), (-1, "R")):
        yb0, yb1 = sorted((sign * S.HUB_BRG_Y[0], sign * S.HUB_BRG_Y[1]))
        yh_out = sign * H1                                       # hanger outer face
        yb_in, yb_out = S.HUB_BRG_Y
        cart = tube(B["D"] / 2, 33.0 - 0.01, (0, sign * (yb_in - 1.0), 0), (0, yh_out, 0))        # spigot in the hanger
        cart = cart.fuse(tube(B["D"] / 2, 46.0, (0, yh_out, 0), (0, sign * (H1 + 4.5), 0)))       # flange outside
        cart = cart.fuse(tube(28.0, B["D"] / 2 + 0.01, (0, sign * (yb_in - 1.0), 0), (0, sign * yb_in, 0)))  # outer-race shoulder
        cart = cart.cut(tube(B["D"] / 2 - 0.5, B["D"] / 2 + 1.0, (0, sign * yb_out, 0), (0, sign * (yb_out + 2.0), 0)))  # DIN 472 groove
        P.append(Part(f"hub_cartridge_{s}", cart, "yoke", "alu", rho="al",
                      spec=dict(material="6061-T6", process="CNC turn", fit="spigot 66 h7 in hanger, bore 62 H7, DIN 472 groove")))
        P.append(ball_bearing(f"hub_bearing_{s}", (0, sign * (S.HUB_BRG_Y[0] + S.HUB_BRG_Y[1]) / 2, 0), (0, 1, 0), B["d"], B["D"], B["B"],
                              "yoke", B["pn"], B["mass"]))
        P.append(circlip_bore(f"hub_circlip_{s}", (0, sign * S.HUB_BRG_Y[1], 0), (0, sign, 0), B["D"], "yoke"))
        for k in range(4):
            a = math.radians(45 + 90 * k)
            J.append(Joint(f"hub_cartridge_screw_{s}{k}", 4, (40.0 * math.cos(a), sign * (H1 + 4.5), 40.0 * math.sin(a)), (0, -sign, 0),
                           [f"hub_cartridge_{s}", f"hanger_{s}"], step=f"yoke_hanger_{s}_sub"))
    # ---------------------------------------------------------------- level gear (left hanger, outer face)
    from gyra_cad import gear_y
    LG = S.LEVEL_GEAR
    lg = gear_y(LG["z"], LG["m"], LG["y"][0], LG["y"][1], 0.0, 0.0, bore=0.0)
    lg = lg.cut(cyl(50.0, (0, LG["y"][0] - 1, 0), (0, LG["y"][1] + 1, 0)))
    P.append(Part("level_gear", lg, "yoke", "alu_dark", rho="al7075",
                  spec=dict(material="7075-T6", process="hob / wire EDM", finish="hard anodise", gear=f"m{LG['m']} z{LG['z']}, face 10")))
    for k in range(6):
        a = math.radians(60 * k)
        J.append(Joint(f"level_gear_screw{k}", 4, (58.0 * math.cos(a), LG["y"][1], 58.0 * math.sin(a)), (0, -1, 0),
                       ["level_gear", "hanger_L"], cbore=False, step="yoke_hanger_L_sub"))
    # ---------------------------------------------------------------- arc plates + brackets
    x0, x1 = S.ARC_X
    arc = extrude_yz(sector_yz(S.ARC_R[0], S.ARC_R[1], -S.ARC_HALF, S.ARC_HALF), x0, x1)
    for a_ in (20.0, -20.0, 45.0, -45.0):                    # lightening, clear of spacers, rack screws and brackets
        a = math.radians(a_)
        c = (160.0 * math.sin(a), -160.0 * math.cos(a))
        arc = arc.cut(cq.Solid.makeCylinder(14.0, 20.0, Vector(x0 - 5, c[0], c[1]), Vector(1, 0, 0)))
    P.append(Part("arc_plate_F", arc, "yoke", "alu", rho="al", spec=dict(material="6061-T6 plate 6 mm", process="waterjet + CNC")))
    P.append(Part("arc_plate_B", mirror_x(arc), "yoke", "alu", rho="al", spec=dict(material="6061-T6 plate 6 mm", process="waterjet + CNC")))
    for sx, sa in ((1, "F"), (-1, "B")):
        for sy, sh in ((1, "L"), (-1, "R")):
            nm = f"arc_bracket_{sa}{sh}"
            yi = sy * H0                                          # hanger inner face
            leg_a = box(*sorted((sx * x1, sx * (x1 + 6.0))), *sorted((sy * 6.0, yi)), -190.0, -150.0)    # on the arc plate
            leg_b = box(*sorted((sx * x1, sx * (x1 + 36.0))), *sorted((yi, yi - sy * 6.0)), -190.0, -150.0)  # on the hanger
            P.append(Part(nm, leg_a.fuse(leg_b), "yoke", "alu", rho="al",
                          spec=dict(material="6061-T6", process="CNC from 40 x 40 x 6 angle")))
            for k, zz in enumerate((-160.0, -180.0)):
                J.append(Joint(f"{nm}_a{k}", 4, (sx * (x1 + 6.0), sy * 16.0, zz), (-sx, 0, 0), [nm, f"arc_plate_{sa}"],
                               step="yoke_frame"))
                J.append(Joint(f"{nm}_b{k}", 4, (sx * (x1 + 26.0), yi - sy * 6.0, zz), (0, sy, 0), [nm, f"hanger_{sh}"],
                               step="yoke_frame"))
    # ---------------------------------------------------------------- track rings + spacers, lean rack
    tx0, tx1 = S.TRACK_X - S.TRACK_T / 2, S.TRACK_X + S.TRACK_T / 2
    c45 = S.TRACK_T / 2
    prof = [(tx0, S.TRACK_R[0] + c45), (S.TRACK_X, S.TRACK_R[0]), (tx1, S.TRACK_R[0] + c45),
            (tx1, S.TRACK_R[1] - c45), (S.TRACK_X, S.TRACK_R[1]), (tx0, S.TRACK_R[1] - c45)]
    ring = cq.Workplane("XY").polyline(prof).close().revolve(360, (0, 0, 0), (1, 0, 0)).val()
    wedge = extrude_yz(sector_yz(1.0, 400.0, -62.0, 62.0), 0.0, 200.0)
    track = solid(ring.intersect(wedge))
    P.append(Part("track_F", track, "yoke", "steel", rho="steel",
                  spec=dict(material="AISI 1045, edges induction hardened HRC 53 (DualVee track spec)", process="waterjet + CNC V-edges")))
    P.append(Part("track_B", mirror_x(track), "yoke", "steel", rho="steel",
                  spec=dict(material="AISI 1045, edges induction hardened HRC 53 (DualVee track spec)", process="waterjet + CNC V-edges")))
    k_sp = 0
    for sx, sa in ((1, "F"), (-1, "B")):
        for a_ in (0.0, 32.0, -32.0, 55.0, -55.0):
            a = math.radians(a_)
            yy, zz = 183.0 * math.sin(a), -183.0 * math.cos(a)
            nm = f"track_spacer_{sa}{k_sp}"
            P.append(Part(nm, cq.Solid.makeCylinder(7.0, x0 - tx1, Vector(sx * tx1 if sx > 0 else -x0, yy, zz), Vector(1, 0, 0)),
                          "yoke", "alu", rho="al", spec=dict(material="6061", process="turned spacer d14 x 2.8")))
            J.append(Joint(f"track_screw_{sa}{k_sp}", 4, (sx * x1, yy, zz), (-sx, 0, 0), [f"arc_plate_{sa}", nm, f"track_{sa}"],
                           step="yoke_tracks"))
            k_sp += 1
    # lean rack: internal m1.5 segment, pitch r 195, +-52 deg, on the front track via spacers
    rx0, rx1 = S.RACK_X
    from tyre import gear_profile_internal
    zr = int(round(2 * S.RACK_R / S.MODULE))
    pts = gear_profile_internal(zr, S.MODULE)
    sp = cq.Workplane("XY").polyline(pts).close().extrude(rx1 - rx0).val()           # toothed outline about +z
    body = cq.Solid.makeCylinder(S.RACK_R + 11.0, rx1 - rx0, Vector(0, 0, 0), Vector(0, 0, 1))
    rk = body.cut(sp)
    rk = rk.rotate(Vector(0, 0, 0), Vector(0, 1, 0), 90).translate(Vector(rx0, 0, 0))   # axis -> +x
    rk = solid(rk.intersect(extrude_yz(sector_yz(1.0, 400.0, -52.0, 52.0), rx0 - 1, rx1 + 1)))
    P.append(Part("lean_rack", rk, "yoke", "alu_dark", rho="al7075",
                  spec=dict(material="7075-T6", process="wire EDM", finish="hard anodise", gear=f"internal m{S.MODULE} z{zr} segment +-52 deg")))
    for k, a_ in enumerate((14.0, -14.0, 30.0, -30.0, 48.0, -48.0)):
        a = math.radians(a_)
        rr = S.RACK_R + 6.5
        yy, zz = rr * math.sin(a), -rr * math.cos(a)
        for (xa, xb, nm) in ((tx1, x0, f"rack_spacer_o{k}"), (rx1, tx0, f"rack_spacer_i{k}")):
            P.append(Part(nm, cq.Solid.makeCylinder(5.5, xb - xa, Vector(xa, yy, zz), Vector(1, 0, 0)), "yoke", "alu", rho="al",
                          spec=dict(material="6061", process="turned spacer d11")))
        J.append(Joint(f"rack_screw{k}", 4, (x1, yy, zz), (-1, 0, 0),
                       ["arc_plate_F", f"rack_spacer_o{k}", "track_F", f"rack_spacer_i{k}", "lean_rack"], step="yoke_tracks"))

    # ---------------------------------------------------------------- drive cassettes
    for sign, t, half in ((1, "F", "L"), (-1, "B", "R")):
        (px, pz), (mx, mz) = drive_geom(sign)
        sy = sign                                            # front cassette on +y, rear on -y
        hy0, hy1 = sorted((sy * H0, sy * H1))
        oy0, oy1 = sorted((sy * S.OUTER_PLATE_Y[0], sy * S.OUTER_PLATE_Y[1]))
        so = SO_F if sign > 0 else SO_B
        # outer plate (motor slots +-2 mm along the belt line)
        op = outline_union([("h", [(mx, mz, 30.0), (px, pz, 24.0)]), ("h", [(mx, mz, 30.0), (so[0][0], so[0][1], 12.0)]),
                            ("h", [(px, pz, 24.0), (so[1][0], so[1][1], 12.0)]), ("h", [(px, pz, 24.0), (so[2][0], so[2][1], 12.0)])],
                           oy0, oy1)
        op = op.cut(cyl(14.0, (px, oy0 - 1, pz), (px, oy1 + 1, pz)))            # 6001 bore
        u = Vector(px - mx, 0, pz - mz).normalized()
        for d_ in (-2.0, 0.0, 2.0):                                             # shaft clearance slot
            op = op.cut(cyl(8.0, (mx + u.x * d_, oy0 - 1, mz + u.z * d_), (mx + u.x * d_, oy1 + 1, mz + u.z * d_)))
        P.append(Part(f"outer_plate_{t}", op, "yoke", "alu", rho="al",
                      spec=dict(material="6061-T6 plate 8 mm", process="waterjet + CNC", note="motor holes slotted +-2 mm for belt tension")))
        # spacers hanger <-> outer plate
        for k, (sx_, sz_) in enumerate(so):
            ya, yb = (H1, S.OUTER_PLATE_Y[0]) if sy > 0 else (-S.OUTER_PLATE_Y[0], -H1)
            P.append(Part(f"cassette_spacer_{t}{k}", cyl(7.0, (sx_, ya, sz_), (sx_, yb, sz_)), "yoke", "alu", rho="al",
                          spec=dict(material="6061 d14", process="turned, M5 tapped both ends 12 deep")))
            J.append(Joint(f"cassette_spacer_screw_i{t}{k}", 5, (sx_, sy * H0, sz_), (0, sy, 0), [f"hanger_{half}", f"cassette_spacer_{t}{k}"],
                           head="BHCS", step=f"yoke_cassette_{t}"))
            J.append(Joint(f"cassette_spacer_screw_o{t}{k}", 5, (sx_, sy * S.OUTER_PLATE_Y[1], sz_), (0, -sy, 0),
                           [f"outer_plate_{t}", f"cassette_spacer_{t}{k}"], step=f"yoke_cassette_{t}"))
        # pinion shaft
        g0, g1 = S.PINION_FACE
        shaft = gear_y(S.Z_PINION, S.MODULE, g0, g1, px, pz, bore=0.0) if sy > 0 else gear_y(S.Z_PINION, S.MODULE, -g1, -g0, px, pz, bore=0.0)
        shaft = shaft.fuse(cyl(6.0, (px, sy * g1, pz), (px, sy * (S.OUTER_PLATE_Y[1] + 3.0), pz)))
        shaft = shaft.fuse(cyl(8.0, (px, sy * g1, pz), (px, sy * (H0 - 0.2), pz)))          # shoulder against the inner bearing
        P.append(Part(f"pinion_shaft_{t}", shaft, "yoke", "steel", rho="steel",
                      spec=dict(material="42CrMo4 QT + gas nitrided", process="turn, hob 24T m1.5, grind bearing seats",
                                gear="m1.5 z24 face 15, ISO 1328 grade 7", note="DIN 6885 4x4 keyway for the P32, DIN 471 groove")))
        Bs = S.COTS["shaft_bearing"]
        for k, yc in enumerate(((H0 + H1) / 2, (S.OUTER_PLATE_Y[0] + S.OUTER_PLATE_Y[1]) / 2)):
            P.append(ball_bearing(f"pinion_bearing_{t}{k}", (px, sy * yc, pz), (0, 1, 0), Bs["d"], Bs["D"], Bs["B"], "yoke",
                                  Bs["pn"].replace("6001-2RS", "6001-2RSNR (snap ring)"), Bs["mass"]))
        P.append(circlip_shaft(f"pinion_circlip_{t}", (px, sy * (S.OUTER_PLATE_Y[1] + 0.2), pz), (0, sy, 0), 12.0, "yoke"))
        shp = P[[p.name for p in P].index(f"pinion_shaft_{t}")]
        shp.shape = shp.shape.cut(tube(5.45, 6.5, (px, sy * (S.OUTER_PLATE_Y[1] + 0.1), pz), (px, sy * (S.OUTER_PLATE_Y[1] + 1.3), pz)))  # DIN 471 groove
        # pulleys + belt
        C32, C20 = S.COTS["pulley_32"], S.COTS["pulley_20"]
        f0, f1 = S.BELT_Y - 8.0, S.BELT_Y + 8.0                                # flange spacing 16 for the 15 mm belt
        def pulley(cx, cz, pd_od, flange, hub_d, hub_y, face):
            fa, fb = face
            p_ = fuse(cyl(pd_od / 2, (cx, sy * fa, cz), (cx, sy * fb, cz)),
                      cyl(flange / 2, (cx, sy * (fa - 0.8), cz), (cx, sy * fa, cz)),
                      cyl(flange / 2, (cx, sy * fb, cz), (cx, sy * (fb + 0.8), cz)),
                      cyl(hub_d / 2, (cx, sy * hub_y[0], cz), (cx, sy * hub_y[1], cz)))
            return p_
        p32 = pulley(px, pz, C32["od"], C32["flange"], C32["hub_d"], (f0 - 0.8 - (C32["length"] - 17.6), f0 - 0.8), (f0, f1))
        p32 = p32.cut(cyl(6.0, (px, sy * 30, pz), (px, sy * 70, pz)))
        P.append(Part(f"pulley_32_{t}", p32, "yoke", "alu", mass=C32["mass"], kind="cots",
                      spec=dict(pn=C32["pn"], source=C32["source"])))
        p20 = pulley(mx, mz, C20["od"], C20["flange"], C20["hub_d"], (f1 + 0.8, f1 + 0.8 + (C20["length"] - 17.6)), (f0, f1))
        p20 = p20.cut(cyl(5.0, (mx, sy * 40, mz), (mx, sy * 90, mz)))
        P.append(Part(f"pulley_20_{t}", p20, "yoke", "alu", mass=C20["mass"], kind="cots",
                      spec=dict(pn=C20["pn"], source=C20["source"])))
        r1, r2 = C20["pd"] / 2, C32["pd"] / 2
        outer = hull2d(circle_pts(mx, mz, C20["od"] / 2 + 3.2, 72) + circle_pts(px, pz, C32["od"] / 2 + 3.2, 72))
        inner = hull2d(circle_pts(mx, mz, C20["od"] / 2 + 0.05, 72) + circle_pts(px, pz, C32["od"] / 2 + 0.05, 72))   # teeth not modelled
        bya, byb = sorted((sy * (S.BELT_Y - 7.5), sy * (S.BELT_Y + 7.5)))
        belt = extrude_xz(outer, bya, byb).cut(extrude_xz(inner, bya - 1, byb + 1))
        P.append(Part(f"belt_{t}", belt, "yoke", "rubber", mass=0.025, kind="cots", spec=dict(pn=S.COTS["belt"]["pn"])))
        # motor
        mface = sy * S.OUTER_PLATE_Y[1]
        P.append(d6374(f"drive_motor_{t}", (mx, mface, mz), (0, sy, 0), "yoke", xref=(1, 0, 0)))
        opart = P[[p.name for p in P].index(f"outer_plate_{t}")]
        for hp in d6374_holes((mx, mface, mz), (0, sy, 0), (1, 0, 0), pcd=44.0):          # M4 slots +-2 mm
            for d_ in (-2.0, 0.0, 2.0):
                opart.shape = opart.shape.cut(cyl(2.25, (hp.x + u.x * d_, oy0 - 1, hp.z + u.z * d_), (hp.x + u.x * d_, oy1 + 1, hp.z + u.z * d_)))
        for k, hp in enumerate(d6374_holes((mx, mface, mz), (0, sy, 0), (1, 0, 0), pcd=44.0)):
            J.append(Joint(f"drive_motor_screw_{t}{k}", 4, (hp.x, sy * S.OUTER_PLATE_Y[0], hp.z), (0, sy, 0),
                           [f"outer_plate_{t}", f"drive_motor_{t}"], max_engage=5.5, washer=True, precut=(f"outer_plate_{t}",),
                           step=f"yoke_cassette_{t}_sub"))

    # ---------------------------------------------------------------- battery modules (x>0: A on -y; x<0: B on +y)
    Cc = S.CELL
    pitch = 22.6
    for sign, m, half in ((1, "A", "R"), (-1, "B", "L")):
        sy = -sign
        xa, xb = sorted((sign * 102.0, sign * 152.0))
        ya, yb = sorted((sy * 46.0, sy * 124.0))
        za, zb = -218.0, -76.0
        t = 1.5
        shell = box(xa, xb, ya, yb, za, zb).cut(box(xa + t, xb - t, ya + t, yb - t, za + t, zb - t))
        P.append(Part(f"battery_case_{m}", shell, "yoke", "alu_dark", rho="al",
                      spec=dict(material="5052-H32 1.5 mm", process="laser cut + bent + riveted, PEM nuts on the inboard end",
                                note="fish-paper lined; IP54 gasketed lid on the outboard face")))
        cells = []
        for i in range(2):
            for k in range(6):
                cx = xa + t + 2.0 + pitch * (i + 0.5)
                cz = za + t + 3.0 + pitch * (k + 0.5)
                cells.append(cyl(Cc["d"] / 2, (cx, sy * 50.0, cz), (cx, sy * (50.0 + Cc["L"]), cz)))
        P.append(Part(f"battery_cells_{m}", compound(cells), "yoke", "battery", mass=12 * Cc["mass"], kind="cots",
                      spec=dict(pn=f"12 x {Cc['pn']} (6S2P), spot-welded 0.15 mm nickel, PC cell holders")))
        for k, (sx_, sz_) in enumerate(MOD_SO):
            sxp = sign * sx_
            ys0, ys1 = sorted((sy * H1, sy * 46.0))
            P.append(Part(f"battery_spacer_{m}{k}", cyl(5.0, (sxp, ys0, sz_), (sxp, ys1, sz_)), "yoke", "alu", rho="al",
                          spec=dict(material="6061 d10/5.6", process="cut tube spacer, 11 mm")))
            J.append(Joint(f"battery_screw_{m}{k}", 5, (sxp, sy * H0, sz_), (0, sy, 0),
                           [f"hanger_{half}", f"battery_spacer_{m}{k}", f"battery_case_{m}"], end="pem", step=f"yoke_battery_{m}"))

    # ---------------------------------------------------------------- power electronics (rear half, between the hangers)
    Cs = S.COTS["controller"]
    for sy, s, n in ((1, "L", "F"), (-1, "R", "B")):
        yi = sy * H0
        hs = box(-66.0, 4.0, *sorted((yi, yi - sy * 4.0)), -122.0, -52.0)
        brd = box(-64.0, 2.0, *sorted((yi - sy * 4.0, yi - sy * 14.0)), -112.0, -62.0)
        P.append(Part(f"drive_controller_{n}", hs.fuse(brd), "yoke", "pcb", mass=Cs["mass"], kind="cots",
                      spec=dict(pn=Cs["pn"], note=Cs["note"] + f"; drives the {n} motor")))
        for k, (xx, zz) in enumerate(((-61.0, -57.0), (-1.0, -57.0), (-61.0, -117.0), (-1.0, -117.0))):
            J.append(Joint(f"drive_controller_screw_{n}{k}", 4, (xx, yi - sy * 4.0, zz), (0, sy, 0), [f"drive_controller_{n}", f"hanger_{s}"],
                           step=f"yoke_hanger_{s}_sub"))
    pb = box(-58.0, -6.0, -26.0, 26.0, -205.0, -125.0).cut(box(-56.5, -7.5, -24.5, 24.5, -203.5, -126.5))
    P.append(Part("power_box", pb, "yoke", "alu_dark", rho="al",
                  spec=dict(material="5052-H32 1.5 mm", process="laser cut + bent, PEM nuts",
                            note="12S BMS (<= 80 x 50 x 15), 40 A MIDI fuse, XT90-S anti-spark, CAN hub, e-stop relay")))
    P.append(Part("power_box_parts", box(-54.0, -10.0, -12.0, 12.0, -195.0, -135.0), "yoke", "pcb", mass=0.20, kind="cots",
                  spec=dict(pn="BMS 12S 40 A + MIDI fuse + XT90-S + relay (wired subassembly)", verified=False)))
    for sy, s in ((1, "L"), (-1, "R")):
        for k, (xx, zz) in enumerate(((-50.0, -140.0), (-14.0, -140.0), (-50.0, -190.0), (-14.0, -190.0))):
            J.append(Joint(f"power_box_screw_{s}{k}", 4, (xx, sy * H1, zz), (0, -sy, 0), [f"hanger_{s}", "power_box"], end="pem",
                           head="BHCS", step="yoke_frame"))

    # ---------------------------------------------------------------- slip-ring housing (turns with the yoke) + drive pin
    C = S.COTS["slip_ring"]
    ys0, ys1 = -44.5, -44.5 - (C["length"] - C["collar_len"])
    hsg = tube(36.5, C["od"] / 2, (0, ys1, 0), (0, ys0, 0))
    tab = box(-C["tab"]["w"] / 2, C["tab"]["w"] / 2, ys0 - C["tab"]["t"], ys0, C["od"] / 2 - 2.0, C["tab"]["r_end"])
    tab = tab.cut(box(-C["tab"]["slot_w"] / 2, C["tab"]["slot_w"] / 2, ys0 - 10, ys0 + 1, C["tab"]["slot_r"] - 5.0, C["tab"]["r_end"] + 1))
    P.append(Part("slipring_housing", hsg.fuse(tab), "yoke", "motor_black", mass=0.25, kind="cots",
                  spec=dict(pn=C["pn"], source=C["source"], note="stator housing driven by the hanger pin through its tab slot")))
    pin = dowel("slipring_pin", 8.0, 21.0, (0.0, -H0, C["tab"]["slot_r"]), (0, -1, 0), "yoke", ["hanger_R"])
    P.append(pin)
    cut_dowel_holes({p.name: p for p in P}, pin)
    return P, J
