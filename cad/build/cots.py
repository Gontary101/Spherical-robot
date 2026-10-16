"""
Bought-part models at their datasheet dimensions (spec.COTS). Simplified outer geometry, but every mounting face, pilot,
shaft and threaded hole the design depends on is where the vendor drawing puts it. Threaded holes are cut by the joints
that use them (with max_engage = the vendor thread depth).
"""
import math

from cadquery import Vector

import spec as S
import cadquery as cq

from lib import Joint, Part, box, cyl, fuse, tube, unit


def _frame(axis, xref=None):
    a = unit(axis)
    if xref is None:
        xref = Vector(1, 0, 0) if abs(a.x) < 0.9 else Vector(0, 0, 1)
    x = (Vector(*xref) - a * Vector(*xref).dot(a)).normalized()
    return a, x, a.cross(x)


def d6374(name, face, axis, body, xref=None):
    """ODrive D6374: mounting face at `face`, `axis` points from the face into the motor body. Output shaft d10 x 30
    out of the face (along -axis), stator base d48.8 (threaded M4 x 6 on PCD 44 and PCD 30), rotating can d63 from
    +4 to +67, rear boss d29 to +74, encoder shaft d8 to +90 (vendor STEP)."""
    a, x, y = _frame(axis, xref)
    f = Vector(*face)
    m = fuse(cyl(24.4, f, f + a * 6.0), cyl(31.5, f + a * 6.0, f + a * 67.0), cyl(14.5, f + a * 67.0, f + a * 74.0),
             cyl(5.0, f - a * 30.0, f), cyl(4.0, f + a * 74.0, f + a * 90.0))
    return Part(name, m, body, "motor_black", mass=S.COTS["drive_motor"]["mass"], kind="cots",
                spec=dict(pn=S.COTS["drive_motor"]["pn"], source=S.COTS["drive_motor"]["source"],
                          note="output shaft d10 with 3 mm key; 4 mm bullet phase leads; NTC 10k"))


def d6374_holes(face, axis, xref=None, pcd=44.0, start_deg=45.0):
    """Mounting hole centres on the face (4 x M4)."""
    a, x, y = _frame(axis, xref)
    out = []
    for k in range(4):
        t = math.radians(start_deg + 90 * k)
        out.append(Vector(*face) + x * (pcd / 2 * math.cos(t)) + y * (pcd / 2 * math.sin(t)))
    return out


def ak70_10(name, out_face, axis, body, xref=None):
    """CubeMars AK70-10: output face at `out_face`, axis from the output face into the actuator. Housing d89 x 50.25,
    output flange d40 standing 2 mm proud with 6 x M3 on PCD 25, housing face 8 x M3 on PCD 83 (brochure drawing)."""
    a, x, y = _frame(axis, xref)
    f = Vector(*out_face)
    m = fuse(cyl(20.0, f, f + a * 2.0), cyl(44.5, f + a * 2.0, f + a * 50.25))
    return Part(name, m, body, "motor_black", mass=S.COTS["lean_actuator"]["mass"], kind="cots",
                spec=dict(pn=S.COTS["lean_actuator"]["pn"], source=S.COTS["lean_actuator"]["source"],
                          note="integrated driver, CAN; 48 V"))


def ring_pts(c, axis, r, n, start_deg=0.0, xref=None):
    a, x, y = _frame(axis, xref)
    return [Vector(*c) + x * (r * math.cos(math.radians(start_deg + 360.0 * k / n))) + y * (r * math.sin(math.radians(start_deg + 360.0 * k / n)))
            for k in range(n)]


def ball_bearing(name, c, axis, d, D, B, body, pn, mass):
    """Deep-groove bearing centred at c (axial midpoint)."""
    a = unit(axis)
    c = Vector(*c)
    s = tube(d / 2, D / 2, c - a * (B / 2), c + a * (B / 2))
    return Part(name, s, body, "steel", mass=mass, kind="cots", spec=dict(pn=pn))


def circlip_bore(name, c, axis, D, body):
    """DIN 472 retaining ring for bore D (thickness 1.5-2, radial section ~3)."""
    a = unit(axis)
    t = 2.0 if D >= 50 else 1.5
    s = tube(D / 2 - 3.5, D / 2 + 1.0, Vector(*c), Vector(*c) + a * t)
    return Part(name, s, body, "steel", mass=0.004, kind="fastener", spec=dict(type="retaining ring", std="DIN 472", D=D))


def circlip_shaft(name, c, axis, d, body):
    a = unit(axis)
    t = 1.0 if d <= 12 else 1.5
    s = tube(d / 2 - 0.5, d / 2 + 1.8, Vector(*c), Vector(*c) + a * t)
    return Part(name, s, body, "steel", mass=0.001, kind="fastener", spec=dict(type="retaining ring", std="DIN 471", d=d))


def _place(shape, origin, w, sv, a):
    """Map a solid built in local (w, s, a) coordinates to the robot frame."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt, gp_Trsf
    o = Vector(*origin)
    t = gp_Trsf()
    t.SetTransformation(gp_Ax3(gp_Pnt(o.x, o.y, o.z), gp_Dir(a.x, a.y, a.z), gp_Dir(w.x, w.y, w.z)), gp_Ax3())
    return cq.Shape.cast(BRepBuilderAPI_Transform(shape.wrapped, t, True).Shape())


ENC_LB = 80.5          # encoder base plane from the motor mounting face (shaft end at +90 -> 9.5 mm into the AMT sleeve)


def motor_encoder(prefix, face, axis, leg, body, mount, step, slot_dir=None):
    """AMT212B-V-OD on the D6374's rear encoder shaft, carried by a 6061 L-bracket footed on the motor's mount plate.

    face: motor mounting face centre; axis: from the face into the motor; leg: unit vector (perpendicular to axis) to the
    side where the bracket's leg runs past the can (2.5 mm clear of the d63 can) and its foot sits on the plate
    (w 34..52 from the motor axis). slot_dir: belt-tension slot direction of the motor (foot holes slotted +-2 mm to match).
    Returns (parts, joints)."""
    C = S.COTS["encoder"]
    a = unit(axis)
    w = unit(leg)
    sv = a.cross(w)
    Lb = ENC_LB
    foot = box(34.0, 52.0, -10.0, 10.0, 0.0, 3.0)
    legp = box(34.0, 37.0, -10.0, 10.0, 3.0, Lb - 5.0)
    bridge = box(-13.0, 37.0, -14.5, 14.5, Lb - 5.0, Lb).cut(cyl(6.0, (0, 0, Lb - 6), (0, 0, Lb + 1)))
    br = _place(fuse(foot, legp, bridge), face, w, sv, a)
    if slot_dir is not None:
        u = unit(slot_dir)
        for ss in (-6.0, 6.0):
            c = Vector(*face) + w * 45.0 + sv * ss
            for d_ in (-2.0, -1.0, 0.0, 1.0, 2.0):
                cc = c + u * d_
                br = br.cut(cyl(1.7, cc - a * 1.0, cc + a * 4.0))
    parts = [Part(f"{prefix}_enc_bracket", br, body, "alu", rho="al",
                  spec=dict(material="6061-T6", process="CNC from 6 mm plate (or 3 mm bent L + 5 mm bridge)",
                            note="M2.5 tapped through in the bridge; foot holes " + ("slotted +-2 mm with the motor" if slot_dir else "d3.4")))]
    enc = fuse(cyl(15.3, (0, 0, Lb), (0, 0, Lb + C["thick"])),
               box(0.0, 20.0, -C["width"] / 2, C["width"] / 2, Lb, Lb + C["thick"]),
               box(20.0, C["height"] - C["h_top"], -8.0, 8.0, Lb, Lb + C["thick_conn"]))
    enc = enc.cut(cyl(C["sleeve"] / 2, (0, 0, Lb - 1), (0, 0, Lb + C["thick"] + 1)))           # sleeve bore on the shaft
    for ss in (-9.525, 9.525):                                                                  # base screws under the snap-on cover
        enc = enc.cut(cyl(2.5, (0, ss, Lb + 1.5), (0, ss, Lb + 4.5)))
    parts.append(Part(f"{prefix}_encoder", _place(enc, face, w, sv, a), body, "motor_black", mass=C["mass"], kind="cots",
                      spec=dict(pn=C["pn"], source=C["source"], note=C["note"])))
    J = []
    for k, ss in enumerate((-6.0, 6.0)):
        seat = Vector(*face) + w * 45.0 + sv * ss + a * 3.0
        J.append(Joint(f"{prefix}_enc_bracket_screw{k}", 3, tuple(seat), tuple(-a), [f"{prefix}_enc_bracket", mount],
                       precut=(f"{prefix}_enc_bracket",) if slot_dir is not None else (), step=step))
    for k, ss in enumerate((-9.525, 9.525)):
        seat = Vector(*face) + sv * ss + a * (Lb + 1.5)
        J.append(Joint(f"{prefix}_encoder_screw{k}", 2.5, tuple(seat), tuple(-a), [f"{prefix}_encoder", f"{prefix}_enc_bracket"],
                       step=step))
    return parts, J
