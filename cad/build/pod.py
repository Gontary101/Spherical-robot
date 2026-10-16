"""
Sensor pods (spine body). One per axle end; the left pod locates its tyre half axially and radially on 8 DualVee
V-wheels running on the rim's 90 deg V-track, the right pod carries its half on 8 crowned track rollers on a
cylindrical land (axially floating, so thermal growth between the CFRP axle and the GFRP tyre cannot preload
anything). 4 of the 8 studs per pod are eccentric (0.5 mm) to set the preload.

    pod_hub       6061 split clamp on the axle end, flange bolted to the pod plate (6 x M5)
    pod_plate     6061, 8 mm, CNC
    wheel studs   303 stainless, CNC turned: seat d9.525 g6 (DualVee bore 3/8"), shoulder d14, stem M8x1.25
    lidar_riser   6061, 5 mm, 2 dowels into the Mid-360 locating hole + slot, 4 x M3 into the LiDAR
    light_ring    clear PC, turned; status LED PCB underneath
    pod_shield    PA12 (MJF), 6 mm wall, heat-set inserts in its base flange; screwed from inside the tyre
    cameras       3 global-shutter USB modules behind PC windows, printed holders on inserts
"""
import math

from cadquery import Vector

import spec as S
from lib import (Joint, Part, box, cone, cut_dowel_holes, cyl, dowel, fuse, solid, mirror_y, plate, revolve_y, rot_y, sphere, tube, unit,
                 washer_solid, nut_solid, slab_y)

# DualVee size 2 (W2): OD 1.21", vee dia 1.000", bore 3/8", width .438"  (Bishop-Wisecarver via Applied Industrial)
W2 = dict(od=30.73, vee=25.4, bore=9.525, width=11.13, mass=0.038)
ROLLER = dict(od=35.0, bore=12.0, width=10.0, mass=0.047)          # INA LR201-2RSR
WHEEL_GAP = 0.3          # flank / land clearance modelled (preload is set with the eccentric studs)


def g(u, side):
    """Global y of local pod coordinate u (distance outward from the crown centre)."""
    return side * (S.D + u)


def vwheel(c, side):
    """DualVee W2 wheel at centre c (x, y, z), axis along y."""
    w2 = W2["width"] / 2
    ro, rv = W2["od"] / 2, W2["vee"] / 2
    prof = [(W2["bore"] / 2, -w2), (ro, -w2), (ro, -(ro - rv)), (rv, 0.0), (ro, ro - rv), (ro, w2), (W2["bore"] / 2, w2)]
    s = revolve_y([(r, y) for r, y in prof])
    return s.translate(Vector(*c))


def roller(c):
    w2 = ROLLER["width"] / 2
    ro = ROLLER["od"] / 2
    prof = [(ROLLER["bore"] / 2, -w2), (ro - 0.6, -w2), (ro, 0.0), (ro - 0.6, w2), (ROLLER["bore"] / 2, w2)]
    return revolve_y(prof).translate(Vector(*c))


def build_pod(side):
    s = "L" if side > 0 else "R"
    sg = side
    P, J, T = [], [], []
    ex = Vector(1, 0, 0)
    ny = Vector(0, sg, 0)                      # outward along the axle

    def Y(u):
        return g(u, sg)

    # ---------------------------------------------------------------- hub (clamps the axle end)
    u_ax = S.AXLE_END - S.D                    # axle end, local u (222)
    hub = tube(10.0, 32.0, (0, Y(u_ax - 26), 0), (0, Y(S.U_POD_PLATE[0]), 0))
    hub = hub.fuse(tube(10.0, 56.0, (0, Y(S.U_POD_PLATE[0] - 8), 0), (0, Y(S.U_POD_PLATE[0]), 0)))
    hub = hub.cut(cyl(S.AXLE_OD / 2 + 0.02, (0, Y(u_ax - 27), 0), (0, Y(u_ax), 0)))        # axle bore (H7)
    slot = box(-1.0, 1.0, min(Y(u_ax - 27), Y(u_ax - 2)), max(Y(u_ax - 27), Y(u_ax - 2)), -40, -12)
    hub = hub.cut(slot)                                                                  # clamp slit (bottom)
    lug = box(-11, 11, min(Y(u_ax - 25), Y(u_ax - 3)), max(Y(u_ax - 25), Y(u_ax - 3)), -44, -22)
    hub = hub.fuse(lug).cut(slot)
    P.append(Part(f"pod_hub_{s}", hub, "spine", "alu", rho="al",
                  spec=dict(material="6061-T6", process="CNC mill-turn", finish="clear anodise", fit="bore 40 H7, slit clamp")))
    for k, uu in enumerate((u_ax - 20, u_ax - 8)):          # 2 x M5 across the slit
        J.append(Joint(f"pod_hub_clamp_{s}{k}", 5, (11.0, Y(uu), -35.0), (-1, 0, 0), [f"pod_hub_{s}", f"pod_hub_{s}"],
                       end="tap", step=f"pod_{s}_on_axle"))

    # ---------------------------------------------------------------- pod plate
    u0, u1 = S.U_POD_PLATE
    pp = tube(11.0, 176.0, (0, Y(u0), 0), (0, Y(u1), 0))
    for k in range(S.N_WHEELS):                  # lightening windows between the studs
        a = math.radians(22.5 + k * 45.0)
        c = (95 * math.cos(a), 0, 95 * math.sin(a))
        pp = pp.cut(cyl(26.0, (c[0], Y(u0) - sg, c[2]), (c[0], Y(u1) + sg, c[2])))
    P.append(Part(f"pod_plate_{s}", pp, "spine", "alu_dark", rho="al",
                  spec=dict(material="6061-T6 plate 8 mm", process="waterjet + CNC finish (stud holes reamed 8 H7)", finish="black anodise")))
    for k in range(6):                            # hub flange to plate: 6 x M5 from outside
        a = math.radians(30 + k * 60)
        J.append(Joint(f"pod_hub_bolt_{s}{k}", 5, (44 * math.cos(a), Y(u1), 44 * math.sin(a)), -ny,
                       [f"pod_plate_{s}", f"pod_hub_{s}"], head="FHCS", step=f"pod_{s}_plate"))   # flush under the riser

    # ---------------------------------------------------------------- wheels / rollers on studs
    if side > 0:
        rc = S.R_TRACK - W2["vee"] / 2 - WHEEL_GAP * math.sqrt(2)      # flanks 45 deg: radial offset -> flank gap
    else:
        rc = S.R_TRACK - ROLLER["od"] / 2 - WHEEL_GAP
    stems = []
    for k in range(S.N_WHEELS):
        a = math.radians(k * 45.0)
        x, z = rc * math.cos(a), rc * math.sin(a)
        # eccentric studs: left = all 8 at 3.5 mm (turning a stud 90 deg retracts its wheel 3.5 mm so the pod slides
        # past the V-track at assembly, then the same stud sets the preload); right = 1.0 mm on every other stud
        e = 3.5 if side > 0 else (1.0 if k % 2 else 0.0)
        ecc = e > 0
        tx, tz = -math.sin(a), math.cos(a)
        xs, zs = x - e * tx, z - e * tz                           # stem axis (seat axis = wheel centre)
        stems.append((xs, zs))
        r_sh = 9.5 if e > 1 else 7.0
        uw = S.U_TRACK
        width = W2["width"] if side > 0 else ROLLER["width"]
        seat_d = W2["bore"] if side > 0 else ROLLER["bore"]
        w0, w1 = uw - width / 2, uw + width / 2
        stud = fuse(cyl(seat_d / 2, (x, Y(w0 - 0.0), z), (x, Y(w1), z)),
                    cyl(r_sh, (xs, Y(w1), zs), (xs, Y(u0), zs)),
                    cyl(4.0, (xs, Y(u0), zs), (xs, Y(u1 + 1.6 + 8.0 + 2.5), zs)))
        P.append(Part(f"wheel_stud_{s}{k}", stud, "spine", "steel", rho="ss", kind="custom",
                      spec=dict(material="303 stainless", process="CNC turn", type=f"eccentric {e} mm" if ecc else "concentric",
                                note="seat g6, M8x1.25 stem, axial M5 tapped end for the wheel retainer")))
        pp_hole = cyl(4.02, (xs, Y(u0) - sg, zs), (xs, Y(u1) + sg, zs))                # 8 H7: the stud turns in it
        P[[p.name for p in P].index(f"pod_plate_{s}")].shape = P[[p.name for p in P].index(f"pod_plate_{s}")].shape.cut(pp_hole)
        P.append(Part(f"stud_washer_{s}{k}", washer_solid(8, (xs, Y(u1), zs), ny), "spine", "steel", mass=0.0026, kind="fastener",
                      spec=dict(type="washer", std="ISO 7089", d=8)))
        T.append((f"wheel_stud_{s}{k}", f"stud_nut_{s}{k}"))
        P.append(Part(f"stud_nut_{s}{k}", nut_solid(8, Vector(xs, Y(u1), zs) + ny * 1.6, ny), "spine", "steel", mass=0.0055,
                      kind="fastener", spec=dict(type="nyloc nut", std="ISO 7040", d=8, torque_Nm=12.0)))
        if side > 0:
            P.append(Part(f"vwheel_{s}{k}", vwheel((x, Y(uw), z), sg), "spine", "steel", mass=W2["mass"], kind="cots",
                          spec=dict(pn="Bishop-Wisecarver DualVee W2 (W2X)", note="steel, shielded")))
        else:
            P.append(Part(f"roller_{s}{k}", roller((x, Y(uw), z)), "spine", "steel", mass=ROLLER["mass"], kind="cots",
                          spec=dict(pn="INA LR201-2RSR crowned track roller 12x35x10")))
        ret = cyl(7.5, (x, Y(w0 - 2.0), z), (x, Y(w0), z))        # the joint drills its clearance hole
        P.append(Part(f"wheel_retainer_{s}{k}", ret, "spine", "steel", rho="ss", kind="custom",
                      spec=dict(material="303 stainless", process="turned washer 15 x 2 (clamps the inner race)")))
        J.append(Joint(f"wheel_screw_{s}{k}", 5, (x, Y(w0 - 2.0), z), ny, [f"wheel_retainer_{s}{k}", f"wheel_stud_{s}{k}"],
                       step=f"pod_{s}_wheels", grade="A2-70"))

    # ---------------------------------------------------------------- LiDAR (Livox Mid-360) on a doweled riser
    L = S.COTS["lidar"]
    u_base = S.R_OUT - L["optical_h"]               # optical centre on the sphere surface
    t_riser = u_base - u1
    riser = box(-50, 50, min(Y(u1), Y(u_base)), max(Y(u1), Y(u_base)), -50, 50)
    riser = riser.cut(box(-6, 6, min(Y(u1), Y(u_base)) - 1, max(Y(u1), Y(u_base)) + 1, -50, 0))     # cable groove
    P.append(Part(f"lidar_riser_{s}", riser, "spine", "alu", rho="al",
                  spec=dict(material="6061-T6", process="CNC", finish="black anodise", note="2 x d3 m6 dowels to the LiDAR")))
    # Mid-360 body: 65 x 65 base to h 39.5, window / dome above (the axis is along the axle)
    hb = 39.5
    lid = box(-32.5, 32.5, min(Y(u_base), Y(u_base + hb)), max(Y(u_base), Y(u_base + hb)), -32.5, 32.5)
    lid = lid.fuse(cyl(29.0, (0, Y(u_base + hb), 0), (0, Y(u_base + 52.0), 0)))
    lid = lid.fuse(sphere(29.0, (0, Y(u_base + 52.0), 0)).intersect(slab_y(*sorted((Y(u_base + 52.0), Y(u_base + 60.0))))))
    lid = lid.fuse(cyl(7.0, (32.5, Y(u_base + 14.3), 0), (40.5, Y(u_base + 14.3), 0)))         # M12 connector stub
    P.append(Part(f"lidar_{s}", lid, "spine", "lidar_white", mass=L["mass"], kind="cots",
                  spec=dict(pn=L["pn"], source=L["source"], power_W=L["power_W"], connector=L["connector"])))
    hx, hz = L["hole_rect"][0] / 2, L["hole_rect"][1] / 2
    for k, (px, pz) in enumerate(((hx, hz), (-hx, hz), (-hx, -hz), (hx, -hz))):
        J.append(Joint(f"lidar_screw_{s}{k}", 3, (px, Y(u1 + 3.4), pz), ny, [f"lidar_riser_{s}", f"lidar_{s}"],
                       cbore=True, max_engage=4.6, step=f"pod_{s}_lidar_sub"))     # Mid-360: M3 x 5 deep
    byname = {p.name: p for p in P}
    for k, (px, pz) in enumerate(((0.0, -16.0), (0.0, 23.0))):
        pin = dowel(f"lidar_dowel_{s}{k}", 3.0, 6.0, (px, Y(u_base - 4.2), pz), ny, "spine", [f"lidar_riser_{s}", f"lidar_{s}"])
        P.append(pin)
        cut_dowel_holes(byname, pin)
    for k, (px, pz) in enumerate(((44, 44), (-44, 44), (-44, -44), (44, -44))):
        J.append(Joint(f"riser_bolt_{s}{k}", 4, (px, Y(u_base), pz), -ny, [f"lidar_riser_{s}", f"pod_plate_{s}"],
                       step=f"pod_{s}_lidar"))
    # ---------------------------------------------------------------- light ring + LED PCB, shield
    R_SH = S.R_OUT - 1.0                         # shield outer sphere (1 mm inside the tyre envelope)
    c0 = Vector(0, sg * S.D, 0)
    def ring_rev(prof_ru):                        # (r, u) profile -> solid on this side
        sh = revolve_y([(r, u) for r, u in prof_ru]).translate(Vector(0, S.D, 0))
        return sh if side > 0 else mirror_y(sh)
    u_lr = (u1, u1 + 8.0)                          # light ring 248..256
    r_out = lambda u: math.sqrt(R_SH ** 2 - u * u)
    lr = ring_rev([(146.0, u_lr[0]), (r_out(u_lr[0]), u_lr[0]), (r_out(u_lr[1]), u_lr[1]), (146.0, u_lr[1])])
    lr = lr.cut(ring_rev([(150.0, u_lr[0] - 0.1), (166.0, u_lr[0] - 0.1), (166.0, u_lr[0] + 2.0), (150.0, u_lr[0] + 2.0)]))
    P.append(Part(f"light_ring_{s}", lr, "spine", "glass", rho="pc",
                  spec=dict(material="PC clear (Makrolon 2405)", process="CNC turn + vapour polish", note="diffuses the status LEDs")))
    pcb = ring_rev([(151.0, u_lr[0]), (165.0, u_lr[0]), (165.0, u_lr[0] + 1.6), (151.0, u_lr[0] + 1.6)])
    P.append(Part(f"led_pcb_{s}", pcb, "spine", "pcb", mass=0.03, kind="custom",
                  spec=dict(material="FR4 1.6 mm", process="PCBA: 24 x WS2812B-2020 RGB LEDs, JST-GH 3-pin")))
    u_sh0 = u_lr[1]
    t_sh = 6.0
    shell = sphere(R_SH, (0, sg * S.D, 0)).cut(sphere(R_SH - t_sh, (0, sg * S.D, 0)))
    shield = solid(shell.intersect(slab_y(*sorted((Y(u_sh0), Y(R_SH + 5))))))
    shield = solid(shield.cut(cyl(52.0, (0, Y(u_sh0) - sg, 0), (0, Y(R_SH + 10), 0))))   # LiDAR aperture (FOV to -6 deg)
    # 4 standoff lands (flat bosses) inside the shell at r 120
    SO_R, SO_U, SO_L = 120.0, u1 + 18.0, 18.0
    so_ang = [45.0 + 90.0 * k for k in range(4)]
    for a_ in so_ang:
        a = math.radians(a_)
        cx, cz = SO_R * math.cos(a), SO_R * math.sin(a)
        boss = cyl(7.0, (cx, Y(SO_U + 0.01), cz), (cx, Y(SO_U + 12.0), cz)).intersect(sphere(R_SH - 1.0, (0, sg * S.D, 0)))
        shield = solid(shield.fuse(boss))
    for k in range(S.N_WHEELS):                    # reliefs over the stud nuts at the shield base
        cx, cz = stems[k]
        shield = solid(shield.cut(cyl(9.0, (cx, Y(u0), cz), (cx, Y(u1 + 14.0), cz))))
    # cameras square to the shell (an oblique window would vignette a 120 deg lens or put the board into the wall);
    # placed between the wheel studs, as far outboard-forward / -rearward / -down as the cap allows
    CAMS = [((0.351, 0.925, 0.145), "front"), ((-0.351, 0.925, 0.145), "rear"), ((0.145, 0.925, -0.351), "ground")]
    cam_parts = []
    for i, (dp, nm) in enumerate(CAMS):
        dp = unit(dp)
        dp = Vector(dp.x, sg * dp.y, dp.z)
        at = lambda rho: c0 + dp * rho                                     # point on the camera axis at radius rho
        fx = unit((dp.z, 0.0, -dp.x))                                      # tangential: the holder's long side
        fy = dp.cross(fx)
        # PC window in a flat counterbore at the surface, conical aperture for the 120 deg field
        shield = solid(shield.cut(cyl(16.5, at(297.0), at(R_SH + 3.0))))
        shield = solid(shield.cut(cone(5.5, 14.5, at(290.0), at(297.05))))
        win = cyl(16.3, at(297.0), at(298.5))
        cam_parts.append(Part(f"cam_window_{s}{i}", win, "spine", "glass", rho="pc", spec=dict(
            material="PC 1.5 mm, AR coated", process="laser cut disc d32.6", bond="Dow 732 RTV seal")))
        # Waveshare OV9281 1MP USB Camera (A): 25 x 25 board, 4 x d2 on 21 x 21, depth 11.6 incl. 5.5 lens, 120 deg
        lens = cyl(4.55, at(285.0), at(290.5))
        board = _box_on(at(281.95), dp, fx, fy, 25.0, 25.0, 6.1)
        cam_parts.append(Part(f"camera_{s}{i}", lens.fuse(board), "spine", "motor_black", mass=0.027, kind="cots",
                              spec=dict(pn="Waveshare OV9281 1MP Mono USB Camera (A)", view=nm, fov_deg=120,
                                        source="waveshare.com product page + outline drawing", verified=True)))
        # holder plate (rho 285..290) on two bosses at +-21
        hold = _box_on(at(287.5), dp, fx, fy, 52.0, 30.0, 5.0).cut(cyl(5.5, at(284.0), at(291.0)))
        cam_parts.append(Part(f"cam_holder_{s}{i}", hold, "spine", "nylon", rho="pa12",
                              spec=dict(material="PA12", process="MJF print", note="2 x M3 into shield inserts; camera on 4 x M2 into M2 inserts")))
        for k, du in enumerate((21.0, -21.0)):
            boss = cyl(4.5, at(290.0) + fx * du, at(296.0) + fx * du).intersect(sphere(R_SH - 1.0, (0, sg * S.D, 0)))
            shield = solid(shield.fuse(boss))
            J.append(Joint(f"cam_holder_screw_{s}{i}{k}", 3, tuple(at(285.0) + fx * du), tuple(dp),
                           [f"cam_holder_{s}{i}", f"pod_shield_{s}"], end="insert", step=f"pod_{s}_shield_sub"))
        for k, (du, dv) in enumerate(((10.5, 10.5), (-10.5, 10.5), (-10.5, -10.5), (10.5, -10.5))):
            J.append(Joint(f"cam_screw_{s}{i}{k}", 2, tuple(at(278.9) + fx * du + fy * dv), tuple(dp),
                           [f"camera_{s}{i}", f"cam_holder_{s}{i}"], end="insert", step=f"pod_{s}_shield_sub"))
    P.append(Part(f"pod_shield_{s}", shield, "spine", "nylon", rho="pa12",
                  spec=dict(material="PA12 (HP MJF), dyed black", process="MJF print", finish="bead blast + clear coat",
                            note="4 x M4 button heads on spot faces into the standoffs; serviceable from outside")))
    P += cam_parts
    for k, a_ in enumerate(so_ang):                # standoffs: M4 male-female hex 7 AF x 18, male end into the plate
        a = math.radians(a_)
        cx, cz = SO_R * math.cos(a), SO_R * math.sin(a)
        from lib import _hex_prism
        so = _hex_prism(7.0, SO_L, Vector(cx, Y(u1), cz), ny, Vector(1, 0, 0)).fuse(cyl(2.0, (cx, Y(u1 - 6.0), cz), (cx, Y(u1), cz)))
        T.append((f"shield_standoff_{s}{k}", f"pod_plate_{s}"))
        P.append(Part(f"shield_standoff_{s}{k}", so, "spine", "steel", rho="ss", kind="cots",
                      spec=dict(pn="hex standoff M4 male-female, 7 AF x 18 mm, stainless (e.g. Wurth 971180445 class)")))
        pp = P[[p.name for p in P].index(f"pod_plate_{s}")]
        pp.shape = pp.shape.cut(cyl(1.65, (cx, Y(u1 + 0.5), cz), (cx, Y(u1 - 7.5), cz)))       # M4 tapped, 7 deep
        seat_u = math.sqrt(R_SH ** 2 - (SO_R + 5.0) ** 2) - 0.2
        J.append(Joint(f"shield_screw_{s}{k}", 4, (cx, Y(seat_u), cz), -ny, [f"pod_shield_{s}", f"shield_standoff_{s}{k}"],
                       head="BHCS", cbore=True, max_engage=6.0, step=f"pod_{s}_shield"))
    return P, J, T


def np_lin(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def _perp_to(n):
    n = unit(n)
    a = Vector(0, 0, 1) if abs(n.z) < 0.9 else Vector(1, 0, 0)
    return n.cross(a).normalized()


def _box_on(c, n, xr, yr, w, h, t):
    """Box of w x h (in the xr, yr directions) and thickness t along n, centred at c."""
    from lib import plane as mkplane
    import cadquery as cq
    pl = cq.Plane(origin=tuple(c - n * (t / 2)), xDir=tuple(xr), normal=tuple(n))
    return cq.Workplane(pl).rect(w, h).extrude(t).val()
