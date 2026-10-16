"""
Bob (lean body: tungsten ballast swung sideways on a curved V-track, about the x axis at the axle centre).

    carriage plates 2 x 6061 6 mm sectors r 126..242, +-24 deg
    tungsten        3 blocks W-Ni-Fe 95 % (18.0 g/cc, ASTM B777 class 3), r 222..246, x +-62, each 4 x M8 class 10.9 flat heads
    V-wheels        4 per side (DualVee W2): inner pair (r 138.9, +-20 deg, eccentric studs) on the track's inner edge,
                    outer pair (r 227.1, +-16 deg, studs threaded into the tungsten) on its outer edge; DIN 6799 E-clips
    lean actuator   CubeMars AK70-10 on the front plate's inner face (8 x M3 countersunk), 40T m1.5 pinion on its
                    output flange meshing the yoke's internal rack (pitch r 195): 6.5 : 1
"""
import math

import cadquery as cq
from cadquery import Vector

import spec as S
from cots import ak70_10, ring_pts
from lib import Joint, Part, compound, cyl, extrude_yz, fuse, mirror_x, sector_yz, solid, tube

W2 = S.COTS["vee_wheel"]


def at(r, phi_deg, x):
    """Point at radius r, angle phi (from straight down toward +y) about the x axis, at x."""
    p = math.radians(phi_deg)
    return Vector(x, r * math.sin(p), -r * math.cos(p))


def vwheel_x(c):
    """DualVee W2 with its axis along x, centred at c."""
    w2 = W2["width"] / 2
    ro, rv = W2["od"] / 2, W2["vee"] / 2
    prof = [(W2["bore"] / 2, -w2), (ro, -w2), (ro, -(ro - rv)), (rv, 0.0), (ro, ro - rv), (ro, w2), (W2["bore"] / 2, w2)]
    s = cq.Workplane("XY").polyline([(a, r) for r, a in prof]).close().revolve(360, (0, 0, 0), (1, 0, 0)).val()
    return s.translate(c)


def build_bob():
    P, J, T = [], [], []                 # T: declared thread pairs (studs in tapped parts, nuts on studs)
    x0, x1 = S.BOB_X
    xt = S.TRACK_X
    plate = extrude_yz(sector_yz(126.0, 242.0, -S.BOB_HALF, S.BOB_HALF), x0, x1)
    # front plate: actuator output bore d44 at (r 165, 0 deg)
    ra = S.RACK_R - S.MODULE * S.LEAN_PINION_Z / 2               # 165
    pf = plate.cut(cyl(22.0, at(ra, 0, x0 - 1), at(ra, 0, x1 + 1)))
    P.append(Part("carriage_F", pf, "bob", "alu_dark", rho="al", spec=dict(material="6061-T6 plate 6 mm", process="waterjet + CNC")))
    P.append(Part("carriage_B", mirror_x(plate), "bob", "alu_dark", rho="al", spec=dict(material="6061-T6 plate 6 mm", process="waterjet + CNC")))
    # tungsten
    r0, r1 = S.W_R
    edges = (-S.BOB_HALF, -8.0, 8.0, S.BOB_HALF)
    for k in range(3):
        a0, a1 = edges[k] + (0.25 if k else 0.0), edges[k + 1] - (0.25 if k < 2 else 0.0)
        blk = extrude_yz(sector_yz(r0, r1, a0, a1), -x0, x0)
        P.append(Part(f"tungsten_{k}", blk, "bob", "lead", rho="wnife",
                      spec=dict(material="W-Ni-Fe 95 % heavy alloy, ASTM B777 class 3 (18.0 g/cc)", process="sintered blank, CNC (EDM optional)",
                                note="M8 tapped 16 deep in both end faces")))
        phis = (-21.0, -11.0) if k == 0 else ((-3.5, 3.5) if k == 1 else (11.0, 21.0))
        for sx, sp in ((1, "F"), (-1, "B")):
            for i, ph in enumerate(phis):
                J.append(Joint(f"tungsten_screw_{sp}{k}{i}", 8, at(239.0, ph, sx * x1), (-sx, 0, 0), [f"carriage_{sp}", f"tungsten_{k}"],
                               head="FHCS", grade="10.9", step="bob_tungsten"))
    # V-wheels on studs
    for sx, sp in ((1, "F"), (-1, "B")):
        for name, rc, phs, into in (("in", S.TRACK_R[0] - W2["vee"] / 2 - 0.42, (-20.0, 20.0), None),
                                    ("out", S.TRACK_R[1] + W2["vee"] / 2 + 0.42, (-16.0, 16.0), "tungsten")):
            for i, ph in enumerate(phs):
                c = at(rc, ph, sx * xt)
                w0, w1 = xt - W2["width"] / 2, xt + W2["width"] / 2
                nm = f"bob_stud_{sp}{name}{i}"
                if into is None:     # nut on the plate's inner face
                    stem = cyl(4.0, at(rc, ph, sx * (x0 - 9.0)), at(rc, ph, sx * x1))
                else:                # threaded into the tungsten
                    stem = cyl(4.0, at(rc, ph, sx * (x0 - 12.0)), at(rc, ph, sx * x1))
                st = fuse(stem, cyl(7.0, at(rc, ph, sx * x1), at(rc, ph, sx * w0)), cyl(W2["bore"] / 2, at(rc, ph, sx * w0), at(rc, ph, sx * (w1 + 1.2))))
                st = st.cut(tube(W2["bore"] / 2 - 0.6, W2["bore"] / 2 + 1, at(rc, ph, sx * (w1 + 0.1)), at(rc, ph, sx * (w1 + 1.0))))   # E-clip groove
                P.append(Part(nm, st, "bob", "steel", rho="ss", spec=dict(material="303 stainless", process="CNC turn",
                              type="eccentric 0.5 mm" if name == "in" else "concentric", note="seat d9.53 g6, DIN 6799 groove, M8 stem")))
                P.append(Part(f"bob_wheel_{sp}{name}{i}", vwheel_x(at(rc, ph, sx * xt)), "bob", "steel", mass=W2["mass"], kind="cots",
                              spec=dict(pn=W2["pn"])))
                eclip = tube(W2["bore"] / 2 - 0.55, W2["bore"] / 2 + 2.2, at(rc, ph, sx * (w1 + 0.1)), at(rc, ph, sx * (w1 + 1.0)))
                P.append(Part(f"bob_eclip_{sp}{name}{i}", eclip, "bob", "steel", mass=0.0005, kind="fastener",
                              spec=dict(type="retaining washer", std="DIN 6799", d=7)))
                plt = P[[p.name for p in P].index(f"carriage_{sp}")]
                plt.shape = plt.shape.cut(cyl(4.2, at(rc, ph, sx * (x0 - 1)), at(rc, ph, sx * (x1 + 1))))
                if into is None:
                    from lib import nut_solid, washer_solid
                    P.append(Part(f"{nm}_washer", washer_solid(8, at(rc, ph, sx * x0), Vector(-sx, 0, 0)), "bob", "steel", mass=0.0026,
                                  kind="fastener", spec=dict(type="washer", std="ISO 7089", d=8)))
                    T.append((nm, f"{nm}_nut"))
                    P.append(Part(f"{nm}_nut", nut_solid(8, at(rc, ph, sx * x0) + Vector(-sx * 1.6, 0, 0), Vector(-sx, 0, 0)), "bob", "steel",
                                  mass=0.0055, kind="fastener", spec=dict(type="nyloc nut", std="ISO 7040", d=8, torque_Nm=12.0)))
                else:
                    tk = 0 if ph < 0 else 2
                    T.append((nm, f"tungsten_{tk}"))
                    tw = P[[p.name for p in P].index(f"tungsten_{tk}")]
                    tw.shape = tw.shape.cut(cyl(3.4, at(rc, ph, sx * (x0 + 0.5)), at(rc, ph, sx * (x0 - 14.0))))   # M8 tapped 14
    # lean actuator + pinion
    P.append(ak70_10("lean_actuator", at(ra, 0, x0), Vector(-1, 0, 0), "bob", xref=(0, 1, 0)))
    for k, hp in enumerate(ring_pts(at(ra, 0, x1), Vector(-1, 0, 0), 41.5, 8, 22.5, xref=(0, 1, 0))):
        J.append(Joint(f"lean_actuator_screw{k}", 3, tuple(hp), (-1, 0, 0), ["carriage_F", "lean_actuator"], head="FHCS",
                       max_engage=9.0, step="bob_actuator"))
    rx0, rx1 = S.RACK_X
    from gyra_cad import gear_y
    pz = S.LEAN_PINION_Z
    g = gear_y(pz, S.MODULE, 0.0, rx1 - rx0, 0.0, 0.0, bore=0.0)                 # along +y at the origin
    g = g.rotate(Vector(0, 0, 0), Vector(0, 0, 1), -90)                           # axis -> +x
    g = g.translate(at(ra, 0, rx0))
    hub = cyl(19.5, at(ra, 0, x0 + 2.05), at(ra, 0, rx0))
    pin = g.fuse(hub)
    P.append(Part("lean_pinion", pin, "bob", "steel", rho="steel", spec=dict(material="42CrMo4 QT", process="hob + turn",
                  gear=f"m{S.MODULE} z{pz} face {rx1 - rx0}", note="pilot d20 H7 onto the AK70-10 output, 6 x M3")))
    for k, hp in enumerate(ring_pts(at(ra, 0, rx1), Vector(-1, 0, 0), 12.5, 6, 0.0, xref=(0, 1, 0))):
        J.append(Joint(f"lean_pinion_screw{k}", 3, tuple(hp - Vector(3.4, 0, 0)), (-1, 0, 0), ["lean_pinion", "lean_actuator"], cbore=True,
                       max_engage=6.5, step="bob_actuator"))
    return P, J, T
