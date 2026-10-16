"""
Spine (the levelled, non-rotating body): CFRP axle, the electronics bays, the levelling motor and the slip ring's
stationary core. Everything here sits inside radius 89 mm of the axle wherever the pendulum can sweep past it
(the bob's closest approach at |y| = 200 mm is 97.5 mm), see checks.py for the swept-volume test.

    axle            roll-wrapped CFRP tube 40 x 34 (3 mm wall), 624 mm, ends inside the pod hubs
    hub_spacer      6061 tube between the two 6908 inner races (they are clamped between the level bracket and
                    the slip-ring sleeve)
    level_bracket   6061 split clamp + arm carrying the levelling D6374 (motor face at y 66, pinion 14T m1.5)
    slip_sleeve     6061 split clamp sleeve, OD 50 h7: carries the slip ring core (rotor collar set screws)
    bay clamps      4 x 6061 split clamps with flat tabs; bays are 1.5 mm 5052 trays with PEM nuts
    bays            left: Jetson Orin NX on the reComputer J401 carrier; right: level ODrive S1 on its heat
                    spreader, 48->12 V brick, IMU
    gnss            u-blox ANN-MB1 on a 3 mm plate above the left bay (the tyre is GFRP + PU: RF-transparent)
"""
import math

from cadquery import Vector

import spec as S
from cots import d6374, d6374_holes
from lib import Joint, Part, box, cyl, fuse, tube, unit

AX_R = S.AXLE_OD / 2


def split_clamp(name, y0, y1, r_out, body, lug_side=-1, tabs=(), spec=None):
    """Axle clamp hub (bore 40 H7) with a slit and a lug: lug_side -1 / +1 = below / above the axle, 'x' = at +x
    (horizontal slit, leaves the top and bottom free for mounting tabs)."""
    ya, yb = min(y0, y1), max(y0, y1)
    s = tube(AX_R + 0.02, r_out, (0, ya, 0), (0, yb, 0))
    if lug_side == "x":
        s = s.fuse(box(r_out - 6, r_out + 12, ya, yb, -11, 11)).cut(box(5, r_out + 13, ya - 1, yb + 1, -1.0, 1.0))
    else:
        lz0, lz1 = (r_out - 6, r_out + 12) if lug_side > 0 else (-(r_out + 12), -(r_out - 6))
        s = s.fuse(box(-11, 11, ya, yb, lz0, lz1))
        sz0, sz1 = (5, r_out + 13) if lug_side > 0 else (-(r_out + 13), -5)
        s = s.cut(box(-1.0, 1.0, ya - 1, yb + 1, sz0, sz1))
    return Part(name, s, body, "alu", rho="al", spec=spec or dict(material="6061-T6", process="CNC", finish="clear anodise",
                                                                    fit="bore 40 H7 slit clamp on the CFRP axle"))


def clamp_joints(name, y0, y1, r_out, lug_side=-1, step=""):
    """2 x M5 across the slit of a split_clamp."""
    ya, yb = min(y0, y1), max(y0, y1)
    zc = (r_out + 3) * (1 if lug_side > 0 else -1)
    L = yb - ya
    ys = [ya + L / 2] if L < 16 else [ya + L * 0.25, ya + L * 0.75]
    return [Joint(f"{name}_clamp{k}", 5, (11.0, yy, zc), (-1, 0, 0), [name, name], step=step) for k, yy in enumerate(ys)]


def build_spine():
    P, J = [], []
    # ---------------------------------------------------------------- axle + bearing spacer
    P.append(Part("axle", tube(S.AXLE_ID / 2, AX_R, (0, -S.AXLE_END, 0), (0, S.AXLE_END, 0)), "spine", "carbon", rho="cfrp",
                  spec=dict(material="roll-wrapped CFRP tube 40 x 34, HM fibre, 0/90/+-45", process="cut to 624 mm, ends faced",
                            note="cable slots 10 x 30 at y +-170 and +-296 (deburred, edge-sealed)")))
    P.append(Part("hub_spacer", tube(AX_R + 0.05, 23.0, (0, -S.HUB_BRG_Y[0], 0), (0, S.HUB_BRG_Y[0], 0)), "spine", "alu", rho="al",
                  spec=dict(material="6061-T6 tube", process="CNC turn, faces parallel 0.02", note="clamps the two inner races")))

    # ---------------------------------------------------------------- levelling drive (left)
    y_in = S.HUB_BRG_Y[1]                                    # 37: bracket face against the left inner race
    ym = S.LEVEL_MOTOR_Y                                    # 66: motor face
    zc = S.LEVEL_GEAR["m"] * (S.LEVEL_GEAR["z"] + S.LEVEL_GEAR["pinion"]) / 2      # 82.5 centre distance
    br = split_clamp("level_bracket", y_in + 1.0, ym, 30.0, "spine", lug_side=-1)
    br.shape = br.shape.fuse(tube(AX_R + 0.02, 23.5, (0, y_in, 0), (0, y_in + 1.0, 0)))      # inner-race land
    from lib import hull2d, extrude_xz, circle_pts
    arm = extrude_xz(hull2d(circle_pts(0, 0, 30) + circle_pts(0, zc, 30)), ym - 8.0, ym)
    arm = arm.cut(cyl(AX_R + 0.02, (0, ym - 9, 0), (0, ym + 1, 0))).cut(box(-1.0, 1.0, ym - 9, ym + 1, -45, -5))
    arm = arm.cut(cyl(7.0, (0, ym - 9, zc), (0, ym + 1, zc)))                               # shaft clearance
    br.shape = br.shape.fuse(arm)
    br.spec.update(note="split clamp locates the left inner race; arm carries the levelling motor")
    P.append(br)
    J += clamp_joints("level_bracket", y_in + 1.0, ym - 8.0, 30.0, -1, step="spine_level")
    mot = d6374("level_motor", (0, ym, zc), (0, 1, 0), "spine", xref=(1, 0, 0))
    P.append(mot)
    for k, hp in enumerate(d6374_holes((0, ym, zc), (0, 1, 0), (1, 0, 0), pcd=44.0)):
        J.append(Joint(f"level_motor_screw{k}", 4, (hp.x, ym - 8.0, hp.z), (0, 1, 0), ["level_bracket", "level_motor"],
                       max_engage=5.5, step="spine_level_sub"))
    from gyra_cad import gear_y
    pin = gear_y(S.LEVEL_GEAR["pinion"], S.LEVEL_GEAR["m"], 36.0, 46.0, 0.0, zc, bore=5.0)
    P.append(Part("level_pinion", pin, "spine", "steel", rho="steel", kind="custom",
                  spec=dict(material="42CrMo4 QT", process="hob + bore 10H7, 3 mm keyway, M4 set screw",
                            gear="m1.5 z14 x=+0.3 (no undercut), face 10")))

    # ---------------------------------------------------------------- slip ring core sleeve (right)
    C = S.COTS["slip_ring"]
    y0 = -y_in
    sl = tube(AX_R + 0.02, 23.5, (0, y0 - 1.0, 0), (0, y0, 0))                              # inner-race land
    sl = sl.fuse(tube(AX_R + 0.02, C["bore"] / 2 - 0.02, (0, -126.0, 0), (0, y0 - 1.0, 0)))
    clamp = split_clamp("slip_sleeve_tmp", -126.0, -146.0, 30.0, "spine", lug_side=-1)
    P.append(Part("slip_sleeve", sl.fuse(clamp.shape), "spine", "alu", rho="al",
                  spec=dict(material="6061-T6", process="CNC turn + slit", fit="OD 50 h7 for the slip-ring bore; bore 40 H7",
                            note="locates the right inner race; slip-ring rotor collar set screws bear on it")))
    J += [Joint(f"slip_sleeve_clamp{k}", 5, (11.0, yy, -33.0), (-1, 0, 0), ["slip_sleeve", "slip_sleeve"], step="spine_slipring")
          for k, yy in enumerate((-130.5, -141.5))]
    ys0, ys1 = -40.0, -40.0 - (C["length"] - C["collar_len"])
    core = tube(C["bore"] / 2, 36.0, (0, ys1, 0), (0, ys0, 0)).fuse(
        tube(C["bore"] / 2, C["collar_d"] / 2, (0, ys1 - C["collar_len"], 0), (0, ys1, 0)))
    P.append(Part("slipring_rotor", core, "spine", "motor_black", mass=0.20, kind="cots",
                  spec=dict(pn=C["pn"], source=C["source"], note="rotor (bore + collar, 4 x M4 set screws) held by the spine")))

    # ---------------------------------------------------------------- bays + clamps
    BAY = dict(x=52.0, z0=-35.0, z1=-73.0)
    for side in (1, -1):
        s = "L" if side > 0 else "R"
        for k, (ya, yb) in enumerate(((168.0, 178.0), (266.0, 276.0))):
            nm = f"bay_clamp_{s}{k}"
            tabs = [(BAY["z0"] + 0.0, BAY["z0"] + 3.0, 50.0)]
            if side > 0:
                tabs.append((29.0, 32.0, 45.0))                     # top tab for the GNSS plate (left only)
            cl = split_clamp(nm, side * ya, side * yb, 28.0, "spine", lug_side="x")
            y_lo, y_hi = min(side * ya, side * yb), max(side * ya, side * yb)
            sh = cl.shape.fuse(box(-BAY["x"], BAY["x"], y_lo, y_hi, BAY["z0"], BAY["z0"] + 8.0)).fuse(
                box(-8.0, 8.0, y_lo, y_hi, BAY["z0"] + 7.0, -15.0))
            if side > 0:
                sh = sh.fuse(box(-45.0, 45.0, y_lo, y_hi, 24.0, 32.0)).fuse(box(-8.0, 8.0, y_lo, y_hi, 15.0, 25.0))
            cl.shape = sh.cut(cyl(AX_R + 0.02, (0, y_lo - 1, 0), (0, y_hi + 1, 0))).cut(box(5, 45, y_lo - 1, y_hi + 1, -1.0, 1.0))
            P.append(cl)
            J.append(Joint(f"{nm}_clamp0", 5, (34.0, side * (ya + yb) / 2, 11.0), (0, 0, -1), [nm, nm], step=f"spine_bay_{s}"))
        # tray: 1.5 mm 5052, end flanges onto the clamp tabs
        ya, yb = 179.0, 265.0
        t = 1.5
        outer = box(-BAY["x"], BAY["x"], ya, yb, BAY["z1"], BAY["z0"])
        inner = box(-BAY["x"] + t, BAY["x"] - t, ya + t, yb - t, BAY["z1"] + t, BAY["z0"] + 1)
        tray = outer.cut(inner)
        tray = tray.fuse(box(-BAY["x"], BAY["x"], 168.0, 179.0, BAY["z0"] - t, BAY["z0"])).fuse(
            box(-BAY["x"], BAY["x"], 265.0, 276.0, BAY["z0"] - t, BAY["z0"]))
        if side < 0:
            tray = tray.mirror("XZ")
        P.append(Part(f"bay_tray_{s}", tray, "spine", "alu_dark", rho="al",
                      spec=dict(material="5052-H32 sheet 1.5 mm", process="laser cut + bend, PEM nuts", finish="black powder coat",
                                note="lid = the clamp tabs; cable entry through the axle slots")))
        for k, (yy, xx) in enumerate(((173.0, 35.0), (173.0, -35.0), (271.0, 35.0), (271.0, -35.0))):
            J.append(Joint(f"bay_tray_screw_{s}{k}", 4, (xx, side * yy, BAY["z0"] + 8.0), (0, 0, -1),
                           [f"bay_clamp_{s}{0 if yy < 200 else 1}", f"bay_tray_{s}"], end="pem", step=f"spine_bay_{s}"))
        floor_z = BAY["z1"] + t
        if side > 0:
            Cc = S.COTS["compute"]
            jet = box(-Cc["board"][0] / 2, Cc["board"][0] / 2, 182.0, 182.0 + Cc["board"][1], floor_z + 6.0, floor_z + 6.0 + 30.0)
            P.append(Part("compute", jet, "spine", "pcb", mass=Cc["mass"], kind="cots",
                          spec=dict(pn=Cc["pn"], note=Cc["note"], verified=Cc["verified"])))
            for k, (xx, yy) in enumerate(((44.0, 186.0), (-44.0, 186.0), (44.0, 258.0), (-44.0, 258.0))):
                so = cyl(2.5, (xx, yy, floor_z), (xx, yy, floor_z + 6.0))
                P.append(Part(f"compute_standoff{k}", so, "spine", "brass", mass=0.002, kind="cots",
                              spec=dict(pn="M3 x 6 brass standoff, male-female")))
        else:
            Cs = S.COTS["controller"]
            hs = box(-35.0, 35.0, -(186.0 + 70.0), -186.0, floor_z, floor_z + 4.0)
            P.append(Part("level_controller", hs.fuse(box(-33.0, 33.0, -(196.0 + 50.0), -196.0, floor_z + 4.0, floor_z + 14.0)),
                          "spine", "pcb", mass=Cs["mass"], kind="cots", spec=dict(pn=Cs["pn"], note=Cs["note"])))
            for k, (xx, yy) in enumerate(((30.0, 191.0), (-30.0, 191.0), (30.0, 251.0), (-30.0, 251.0))):
                J.append(Joint(f"level_ctrl_screw{k}", 4, (xx, -yy, floor_z + 4.0), (0, 0, -1), ["level_controller", "bay_tray_R"],
                               end="pem", step="spine_bay_R_sub"))
            Cd = S.COTS["dcdc"]
            # 48 -> 12 V brick standing on edge against the +x wall (baseplate to the wall: heat path)
            xw = BAY["x"] - t
            P.append(Part("dcdc", box(xw - Cd["size"][2], xw, -250.0, -250.0 + Cd["size"][0], floor_z + 1.0, floor_z + 1.0 + Cd["size"][1]),
                          "spine", "pcb", mass=Cd["mass"], kind="cots",
                          spec=dict(pn=Cd["pn"], note="baseplate on a 0.5 mm thermal pad against the tray wall; 4 x M3 baseplate inserts")))
            for k, (yy, zz) in enumerate(((-245.0, floor_z + 6.0), (-204.2, floor_z + 21.4))):
                J.append(Joint(f"dcdc_screw{k}", 3, (BAY["x"], yy, zz), (-1, 0, 0), ["bay_tray_R", "dcdc"], max_engage=3.0,
                               step="spine_bay_R_sub"))
            Ci = S.COTS["imu"]
            P.append(Part("imu", box(-Ci["size"][0] / 2, Ci["size"][0] / 2, -(265.0 - t), -(265.0 - t) + Ci["size"][2],
                                     floor_z + 18.0, floor_z + 18.0 + Ci["size"][1]),
                          "spine", "pcb", mass=Ci["mass"], kind="cots",
                          spec=dict(pn=Ci["pn"], note=Ci["note"] + "; on the tray end wall, x axis along the robot's forward axis")))
            for k, xx in enumerate((-14.0, 14.0)):
                J.append(Joint(f"imu_screw{k}", 2, (xx, -265.0, floor_z + 30.0), (0, 1, 0), ["bay_tray_R", "imu"], max_engage=3.0,
                               step="spine_bay_R_sub"))

    # ---------------------------------------------------------------- GNSS (left, above the axle)
    gp = box(-45.0, 45.0, 168.0, 276.0, 32.0, 35.0)
    P.append(Part("gnss_plate", gp, "spine", "alu", rho="al",
                  spec=dict(material="5052-H32 3 mm", process="laser cut, PEM nuts", note="120 mm ground plane for the antenna")))
    Cg = S.COTS["gnss_antenna"]
    hx, hy = Cg["size"][1] / 2, Cg["size"][0] / 2
    ant = box(-hx, hx, 222.0 - hy + 12.0, 222.0 + hy - 12.0, 35.0, 35.0 + Cg["size"][2]).fuse(
        box(-12.0, 12.0, 222.0 - hy, 222.0 + hy, 35.0, 35.0 + Cg["flange_t"]))          # body + end flanges (holes on 68)
    P.append(Part("gnss_antenna", ant, "spine", "nylon_light", mass=Cg["mass"], kind="cots",
                  spec=dict(pn=Cg["pn"], source=Cg["source"], note="cable coiled and tied to the plate")))
    for k, yy in enumerate((222.0 - 34.0, 222.0 + 34.0)):
        J.append(Joint(f"gnss_antenna_screw{k}", 4, (0.0, yy, 35.0 + Cg["flange_t"]), (0, 0, -1), ["gnss_antenna", "gnss_plate"],
                       end="pem", head="BHCS", step="spine_gnss_sub"))
    for k, yy in enumerate((188.0, 256.0)):
        J.append(Joint(f"gnss_plate_screw{k}", 4, (30.0, 173.0 if k == 0 else 271.0, 35.0), (0, 0, -1),
                       ["gnss_plate", f"bay_clamp_L{k}"], step="spine_gnss"))
        J.append(Joint(f"gnss_plate_screw{k + 2}", 4, (-30.0, 173.0 if k == 0 else 271.0, 35.0), (0, 0, -1),
                       ["gnss_plate", f"bay_clamp_L{k}"], step="spine_gnss"))
    return P, J
