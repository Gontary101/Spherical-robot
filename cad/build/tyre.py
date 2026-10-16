"""
Tyre halves. Each half is one rotating body (tyreL / tyreR):

    drum        GFRP (E-glass / epoxy), 3.0 mm, vacuum-infused in a two-piece female mould split on the crown plane
    tread       PU 70 Shore A, cast in place on the drum in a closed mould (tread base + lugs in one pour)
    ring_gear   7075-T6 one-piece internal gear m1.5 z360 (gear shaping or wire EDM), hard-anodised, its back turned
                to the drum's inner sphere; co-bonded: located in the mould on the crown-plane spigot and infused with
                the drum (its OD 278-282 cannot pass the 276.5 mm seam opening afterwards), DP490 fillet as a second bond
    rim         6061-T6 ring bonded into the drum's outer edge; left: 90 deg V-track for the locating V-wheels,
                right: cylindrical land for the floating crowned track rollers

Bonded joints are recorded in BONDS (part pair, adhesive, area) for the BOM and the floating-part check.
"""
import math

import cadquery as cq
import numpy as np
from cadquery import Vector

from lib import Part, box, compound, mirror_y, revolve_y, slab_y, sphere
import spec as S

BOND_GAP = 0.3


def _sph_profile(y0, y1, side, off, n=12):
    """(r, y) points on the drum's inner sphere (offset `off` inward), y0 -> y1."""
    return [(S.drum_r(y, side) - off, y) for y in np.linspace(y0, y1, n)]


def gear_profile_internal(z, m):
    """Internal gear tooth-space outline (x, z) points: trapezoid approximation of the involute (drawing specifies
    involute, alpha 20 deg, x = 0, ISO 1328 class 8)."""
    rp = m * z / 2
    r_tip, r_root = rp - m, rp + 1.25 * m
    p = 2 * math.pi / z
    pts = []
    for k in range(z):
        a = k * p
        pts += [(r_tip, a - 0.22 * p), (r_root, a - 0.10 * p), (r_root, a + 0.10 * p), (r_tip, a + 0.22 * p)]
    return [(r * math.cos(t), r * math.sin(t)) for r, t in pts]


def ring_gear(side):
    """One-piece internal ring gear with a bonded back, left-half coordinates mirrored for the right."""
    y0, y1 = S.GEAR_Y
    # body of revolution: tooth tips inside, back turned to the drum sphere (bond line BOND_GAP)
    body = revolve_y(_sph_profile(y0, y1, 1, BOND_GAP) + [(S.R_RING_TIP, y1), (S.R_RING_TIP, y0)])
    # the toothed outline encloses everything inside the tooth spaces: cutting it leaves the teeth
    pts = gear_profile_internal(S.Z_RING, S.MODULE)
    spaces = cq.Workplane("XY").polyline(pts).close().extrude(y1 - y0 + 0.02).val()
    spaces = spaces.rotate(Vector(0, 0, 0), Vector(1, 0, 0), -90).translate(Vector(0, y0 - 0.01, 0))
    g = body.cut(spaces)
    return g if side > 0 else mirror_y(g)


def lugs(side):
    """Chevron lugs, one ruled block per row patterned around y (as the concept model), cast with the tread."""
    from gyra_cad import halfspace
    LUG_ROWS_U = ((-S.D + 9, 16.0, 0.0), (24.0, 88.0, 0.5), (96.0, 160.0, 0.0), (168.0, 228.0, 0.5))
    N, ARC, SKEW = 36, 5.0, 7.0
    shell = sphere(S.R_OUT).cut(sphere(S.R_TREAD - 0.5))
    pitch = 360.0 / N
    out = []
    for (y0, y1, phase) in LUG_ROWS_U:
        def corner(theta_deg, y, rr):
            t = math.radians(theta_deg)
            return Vector(rr * math.cos(t), y, rr * math.sin(t))
        A, B = corner(0.0, y0, S.R_OUT), corner(SKEW, y1, S.R_OUT)
        C, D_ = corner(ARC, y0, S.R_OUT), corner(ARC + SKEW, y1, S.R_OUT)
        mid = corner(ARC / 2 + SKEW / 2, (y0 + y1) / 2, S.R_OUT)
        blk = slab_y(y0, y1).intersect(halfspace(A.cross(B), mid)).intersect(halfspace(D_.cross(C), mid))
        lug = shell.intersect(blk)
        for k in range(N):
            off = 0.0 if side > 0 else pitch / 2
            out.append(lug.rotate(Vector(0, 0, 0), Vector(0, 1, 0), (k + phase) * pitch + off))
    return compound(out)


def rim(side):
    """Rim ring (left-half coordinates, local u = y - D): spherical bond band u 205..235, radial web, track ring."""
    u_t = S.U_TRACK
    if side > 0:   # 90 deg V-track: apex at R_TRACK, flanks 45 deg to a 10 mm wide ring at r 155
        track = [(S.R_TRACK, u_t), (S.R_TRACK + 5.0, u_t + 5.0), (S.R_TRACK + 9.0, u_t + 5.0)]
        track_back = [(S.R_TRACK + 9.0, u_t - 5.0), (S.R_TRACK + 5.0, u_t - 5.0)]
    else:          # cylindrical land for crowned rollers
        track = [(S.R_TRACK, u_t - 5.0), (S.R_TRACK, u_t + 5.0), (S.R_TRACK + 9.0, u_t + 5.0)]
        track_back = [(S.R_TRACK + 9.0, u_t - 5.0)]
    web_u = (u_t - 2.0, u_t + 2.0)            # 4 mm web
    band = [(math.sqrt(S.R_DRUM_IN ** 2 - u * u) - BOND_GAP, u) for u in np.linspace(205.0, 235.0, 10)]
    band_in = [(r - 3.0, u) for r, u in band[::-1]]
    r_web_end_hi = math.sqrt(S.R_DRUM_IN ** 2 - web_u[1] ** 2) - BOND_GAP - 2.9
    r_web_end_lo = math.sqrt(S.R_DRUM_IN ** 2 - web_u[0] ** 2) - BOND_GAP - 2.9
    # build as union of three simple revolved pieces (robust)
    t = revolve_y(track + track_back)
    w = revolve_y([(S.R_TRACK + 8.5, web_u[0]), (S.R_TRACK + 8.5, web_u[1]), (r_web_end_hi + 0.5, web_u[1]), (r_web_end_lo + 0.5, web_u[0])])
    b = revolve_y(band + band_in)
    r = t.fuse(w).fuse(b).clean().translate(Vector(0, S.D, 0))
    return r if side > 0 else mirror_y(r)


def build_tyre_half(side):
    s = "L" if side > 0 else "R"
    body = "tyre" + s
    c = Vector(0, S.D, 0)
    y_out = S.D + S.U_EDGE
    drum = sphere(S.R_DRUM_OUT, (0, S.D, 0)).cut(sphere(S.R_DRUM_IN, (0, S.D, 0))).intersect(slab_y(S.SEAM, y_out))
    tread = sphere(S.R_TREAD, (0, S.D, 0)).cut(sphere(S.R_DRUM_OUT, (0, S.D, 0))).intersect(slab_y(S.SEAM + 3.0, y_out))
    lg = lugs(+1).translate(c)
    tread = compound([tread, lg])
    if side < 0:
        drum, tread = mirror_y(drum), mirror_y(tread)
        lg_r = lugs(-1).translate(c)
        tread = compound([mirror_y(sphere(S.R_TREAD, (0, S.D, 0)).cut(sphere(S.R_DRUM_OUT, (0, S.D, 0))).intersect(slab_y(S.SEAM + 3.0, y_out))),
                          mirror_y(lg_r)])
    P = [
        Part(f"drum_{s}", drum, body, "gfrp", rho="gfrp",
             spec=dict(material="E-glass / epoxy, quasi-isotropic, 3.0 mm", process="vacuum infusion, 2-piece female mould split at the crown",
                       finish="peel-ply inside at bond zones")),
        Part(f"tread_{s}", tread, body, "rubber", rho="pu",
             spec=dict(material="PU elastomer 70 Shore A (e.g. Smooth-On VytaFlex 70 class)", process="cast in place in a closed 2-piece mould",
                       finish="as cast")),
        Part(f"ring_gear_{s}", ring_gear(side), body, "alu_dark", rho="al7075",
             spec=dict(material="7075-T6 ring blank", process="CNC turn back to drum sphere, cut teeth by gear shaping or wire EDM",
                       finish="hard anodise 25 um", gear=f"internal m{S.MODULE} z{S.Z_RING} alpha 20 deg x=0, ISO 1328 grade 8",
                       bond="3M DP490 to drum, 0.3 mm line, 16 mm band")),
        Part(f"rim_{s}", rim(side), body, "alu", rho="al",
             spec=dict(material="6061-T6", process="CNC turn", finish="clear anodise; track flanks hard anodise",
                       track="90 deg V, apex R150" if side > 0 else "cylindrical land R150 x 10", bond="3M DP490 to drum")),
    ]
    bonds = [(f"ring_gear_{s}", f"drum_{s}", "3M Scotch-Weld DP490", "16 mm band at r 278-282"),
             (f"rim_{s}", f"drum_{s}", "3M Scotch-Weld DP490", "30 mm band at r 170-205"),
             (f"tread_{s}", f"drum_{s}", "cast-in-place PU over primed GFRP", "full outer surface")]
    return P, [], bonds
