"""
Bought-part models at their datasheet dimensions (spec.COTS). Simplified outer geometry, but every mounting face, pilot,
shaft and threaded hole the design depends on is where the vendor drawing puts it. Threaded holes are cut by the joints
that use them (with max_engage = the vendor thread depth).
"""
import math

from cadquery import Vector

import spec as S
from lib import Part, cyl, fuse, tube, unit


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
