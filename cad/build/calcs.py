"""
Engineering checks for the build design (hand calculations, documented formulas). Loads come from the simulator
(rl/actuator_loads.py -> cad/build/out/actuator_loads.json, deployed policy over 16 robustness scenarios) and masses
from the build CAD (cad/build/out/mass_properties.json).

    python cad/build/calcs.py        -> cad/build/out/calcs.json + calcs.md

Allowables are from the cited datasheets; where no datasheet number was available the line says 'verify' and what
with. Load cases: p99 = 99th percentile of simulated operation; peak = simulated maximum / motor limit; bump = 3 g
vertical (curbs, drops) unless stated.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spec as S  # noqa: E402

OUT = os.path.join(HERE, "out")
G = 9.81
LEWIS_Y = {14: 0.277, 16: 0.296, 20: 0.322, 24: 0.337, 40: 0.389, 96: 0.443, "internal": 0.45}


def hertz_spur(Ft, b, d1, u, internal, E1=210e3, E2=72e3, nu=0.3):
    ZE = math.sqrt(1 / (math.pi * ((1 - nu ** 2) / E1 + (1 - nu ** 2) / E2)))
    ZH = 2.49
    k = (u - 1) / u if internal else (u + 1) / u
    return ZE * ZH * math.sqrt(Ft / (b * d1) * k)


def main():
    mp = json.load(open(os.path.join(OUT, "mass_properties.json")))
    al = json.load(open(os.path.join(OUT, "actuator_loads.json")))["all"]
    B = mp["bodies"]
    m_robot = mp["robot"]["mass"]
    m_pend = B["yoke"]["mass"] + B["bob"]["mass"]
    rows = []

    def row(group, item, value, unit, allow=None, ok=None, note=""):
        if ok is None and allow is not None:
            ok = value <= allow
        rows.append(dict(group=group, item=item, value=round(value, 3), unit=unit, allowable=allow, ok=ok, note=note))

    # ---------------------------------------------------------------- drive gear mesh (pinion 24T / ring 360T, m1.5)
    r_ring = S.R_PITCH / 1000
    for case, T in (("p99", al["drive"]["tau_p99"]), ("peak (motor limit 82 N m)", 3.8 * 24 * 0.9)):
        Ft = T / r_ring
        sp = Ft / (15.0 * S.MODULE * LEWIS_Y[24])
        sr = Ft / (16.0 * S.MODULE * LEWIS_Y["internal"])
        sh = hertz_spur(Ft, 15.0, S.MODULE * S.Z_PINION, S.Z_RING / S.Z_PINION, True)
        row("drive gear", f"pinion root bending, {case}", sp, "MPa", 300.0, note="42CrMo4 QT + nitrided, allowable bending ~300 MPa")
        row("drive gear", f"ring root bending, {case}", sr, "MPa", 150.0, note="7075-T6 fatigue ~150 MPa")
        row("drive gear", f"contact stress, {case}", sh, "MPa", 400.0 if "peak" in case else 250.0,
            note="hard-anodised 7075 vs steel; p99 kept under 250 MPa for wear life; peak is rare")
    # axle sag under the pendulum moves the pinions relative to the ring gears
    E = 150e3
    I = math.pi / 64 * (S.AXLE_OD ** 4 - S.AXLE_ID ** 4)
    L = 2 * (S.D + S.U_POD_PLATE[0]) - 40                       # pod-hub to pod-hub support span ~ 620 mm
    for case, gfac in (("1 g", 1.0), ("bump 3 g", 3.0)):
        F = m_pend * G * gfac
        d = F * L ** 3 / (48 * E * I)
        M = F * L / 4
        sig = M * (S.AXLE_OD / 2) / I
        row("axle", f"bending stress, {case}", sig, "MPa", 400.0, note="HM CFRP, allowable well above (UD 0 deg strength > 1 GPa)")
        row("axle", f"mid-span sag, {case}", d, "mm", None, ok=True)
        dr = d * math.cos(math.radians(S.DRIVE_ANGLE))
        row("drive gear", f"pinion pushed into the ring by axle sag, {case}", dr, "mm", 0.35, note="design backlash: centre distance +0.20 mm -> 0.15-0.45 mm operating")
    # ---------------------------------------------------------------- belt
    Tm = 3.8                                                    # motor peak N m
    r20 = S.COTS["pulley_20"]["pd"] / 2000
    row("belt", "tight-side tension at motor peak torque", Tm / r20, "N", None, ok=None,
        note="verify against Gates HTD 5M 15 mm allowable working tension (Design Flex) - 285-5M-15 at 20T; tooth-in-mesh 6.3")
    # ---------------------------------------------------------------- pinion shaft bearings (6001-2RSNR, Cr 5.6 kN)
    Ft = al["drive"]["tau_p99"] / r_ring
    Fr = Ft * math.tan(math.radians(20))
    Fg = math.hypot(Ft, Fr)
    Fb = 2 * (al["drive"]["tau_p99"] / 24 / 0.9 * 1.6) / (S.COTS["pulley_32"]["pd"] / 2000) * 0.6   # belt pull on the P32 (~1.2 x Te)
    yA, yB, yg, yb = (S.HANGER_Y[0] + S.HANGER_Y[1]) / 2, sum(S.OUTER_PLATE_Y) / 2, sum(S.PINION_FACE) / 2, S.BELT_Y
    RB = (Fg * (yg - yA) + Fb * (yb - yA)) / (yB - yA)
    RA = Fg + Fb - RB
    n = al["drive"]["w_p95"] * 15 * 60 / (2 * math.pi)         # pinion rpm at p95 speed
    for nm, R in (("hanger bearing A", abs(RA)), ("outer-plate bearing B", abs(RB))):
        L10h = (S.COTS["shaft_bearing"]["Cr"] / max(R, 1)) ** 3 * 1e6 / (60 * max(n, 1))
        row("bearings", f"6001 {nm}: radial load p99", R, "N", S.COTS["shaft_bearing"]["C0r"], note=f"L10 at p95 speed {n:.0f} rpm: {L10h:,.0f} h")
    # hub bearings 6908 (Cr 15 kN, C0r 10 kN): pendulum on 2 bearings, bump
    F = m_pend * G * 3 / 2
    row("bearings", "6908 hub bearing, 3 g bump (per bearing)", F, "N", S.COTS["hub_bearing"]["C0r"])
    # ---------------------------------------------------------------- support wheels (tyre halves on the pods)
    m_half = B["tyreL"]["mass"]
    W_half = (m_robot * G) / 2                                  # ground reaction per half, standing
    for case, gf in (("1 g", 1.0), ("bump 3 g", 3.0)):
        Fr = W_half * gf
        # moment of the crown reaction about the wheel ring (lever = wheel plane to crown, along y)
        lever = (S.U_TRACK) / 1000
        M = Fr * lever
        r_w = (S.R_TRACK - S.COTS["vee_wheel"]["vee"] / 2) / 1000
        Fax = M / (r_w * 4)                                     # moment shared by ~4 of 8 wheels (axial component)
        Frad = Fr / 2.5                                         # radial shared by ~2.5 wheels near the bottom
        row("support wheels", f"DualVee W2 radial per wheel, {case}", Frad, "N", S.COTS["vee_wheel"]["working_radial_N"])
        row("support wheels", f"DualVee W2 axial per wheel (crown moment), {case}", Fax, "N", S.COTS["vee_wheel"]["working_axial_N"] * (2 if gf > 1 else 1),
            note="bump case against 2 x working axial (short-duration); size 3 (W3, 1701 N axial) is the drop-in upgrade if field data demands")
        row("support wheels", f"LR201 roller radial per roller (floating side), {case}", Frad, "N", S.COTS["track_roller"]["C0r"])
    # ---------------------------------------------------------------- lean drive (AK70-10 + 40T pinion / internal rack r 195)
    ratio = S.RACK_R / (S.MODULE * S.LEAN_PINION_Z / 2)
    for case, T in (("rms", al["lean"]["tau_rms"]), ("p99", al["lean"]["tau_p99"]), ("peak", al["lean"]["tau_max"])):
        Ta = T / ratio / 0.95
        lim = S.COTS["lean_actuator"]["rated_Nm"] if case == "rms" else S.COTS["lean_actuator"]["peak_Nm"]
        row("lean drive", f"AK70-10 output torque, {case}", Ta, "N m", lim)
    Ft = al["lean"]["tau_max"] / (S.RACK_R / 1000)
    row("lean drive", "pinion root bending at peak", Ft / (7.5 * S.MODULE * LEWIS_Y[40]), "MPa", 300.0)
    row("lean drive", "rack root bending at peak", Ft / (7.5 * S.MODULE * LEWIS_Y["internal"]), "MPa", 150.0)
    w_out = al["lean"]["w_max"] * ratio * 60 / (2 * math.pi)
    row("lean drive", "AK70-10 speed at max lean rate", w_out, "rpm", S.COTS["lean_actuator"]["rated_rpm_48V"])
    # ---------------------------------------------------------------- levelling (D6374 + 14/96)
    rl = S.LEVEL_GEAR["z"] / S.LEVEL_GEAR["pinion"]
    Kt = 8.27 / 150
    row("level drive", "D6374 torque at the level limit (15 N m)", 15.0 / rl / 0.95, "N m", 3.8)
    row("level drive", "D6374 rms current", al["level"]["tau_rms"] / rl / 0.95 / Kt, "A", 40.0, note="ODrive S1 40 A with heat spreader")
    # ---------------------------------------------------------------- drive motors thermal
    row("drive", "D6374 rms current (p rms tyre torque)", al["drive"]["tau_rms"] / 24 / 0.9 / Kt, "A", 40.0)
    # ---------------------------------------------------------------- tungsten / module fixings (5 g any direction)
    for nm, m, nb, d in (("tungsten block (3.5 kg) on 4 x M8 12.9", 3.5, 4, 8), ("battery module (1.3 kg) on 4 x M5", 1.3, 4, 5)):
        Fs = m * G * 5 / nb
        As = math.pi / 4 * (d * 0.85) ** 2
        row("fixings", f"{nm}: shear per bolt at 5 g", Fs / As, "MPa", 200.0, note="bolts clamp: friction alone carries it at proper preload")
    # ---------------------------------------------------------------- energy
    wh = 12 * 3.6 * 10
    km = wh * 0.85 / al["wh_per_km"]
    row("battery", "usable range at the simulated mix", km, "km", None, ok=True, note=f"{wh:.0f} Wh, 85 % usable, {al['wh_per_km']:.0f} Wh/km")
    P99 = (2 * al["drive"]["mech_power_p99"] + al["lean"]["mech_power_p99"] + al["level"]["mech_power_p99"]) / 0.8 + 60.0
    row("battery", "pack current at p99 mechanical power (+60 W electronics)", P99 / (12 * 3.6), "A", 120.0,
        note="12S2P P50B: 2 x 60 A continuous; the two drive ODrives are current-limited to 60 A phase each")
    row("slip ring", "spine power through 2 paralleled 10 A rings", 200 / 43.2, "A", 20.0, note="Jetson 25 W + 2 LiDAR 13 W + cameras 6 W + level motor ~150 W peak")
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "calcs.json"), "w"), indent=1)
    lines = ["| group | check | value | allowable | ok | note |", "|---|---|---|---|---|---|"]
    for r in rows:
        okc = "" if r["ok"] is None else ("yes" if r["ok"] else "**NO**")
        lines.append(f"| {r['group']} | {r['item']} | {r['value']:,.3g} {r['unit']} | {'' if r['allowable'] is None else r['allowable']} | {okc} | {r['note']} |")
    open(os.path.join(OUT, "calcs.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    bad = [r for r in rows if r["ok"] is False]
    print(f"\n{len(rows)} checks, {len(bad)} failing")


if __name__ == "__main__":
    main()
