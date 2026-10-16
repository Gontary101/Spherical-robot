"""
Per-step fastener / torque tables for docs/11_build.md, generated from cad/build/out/parts.json (run assemble.py first).

    python cad/build/guide.py      -> cad/build/out/assembly_steps.md
"""
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# build order and titles of the joint steps (the step names are set where each joint is declared)
ORDER = [
    ("bob_tungsten", "B1  bob: tungsten blocks between the carriage plates"),
    ("bob_actuator", "B2  bob: AK70-10 on the front carriage, lean pinion on its output"),
    ("bob_wheels", "B3  bob: V-wheel studs, wheels, E-clips (inner pair eccentric, set at step 2a)"),
    ("yoke_hanger_L_sub", "Y1  left hanger: hub cartridge, level gear, drive controller, pitch-stop strikers"),
    ("yoke_hanger_R_sub", "Y2  right hanger: hub cartridge, drive controller"),
    ("yoke_tracks", "Y3  arc plates: tracks on spacers, lean rack"),
    ("yoke_frame", "Y4  frame: arc brackets join arc plates to hangers, power box"),
    ("yoke_cassette_F_sub", "Y5  front drive cassette: motor, pulleys, belt on the outer plate; encoder bracket + AMT212 (after belt tension)"),
    ("yoke_cassette_F", "Y6  front cassette onto the hanger (spacers), belt tension by motor slots"),
    ("yoke_cassette_B_sub", "Y7  rear drive cassette: motor, pulleys, belt"),
    ("yoke_cassette_B", "Y8  rear cassette onto the hanger"),
    ("yoke_battery_A", "Y9  battery module A (left of the arc plates)"),
    ("yoke_battery_B", "Y10 battery module B"),
    ("spine_level_sub", "S1  level motor onto the level bracket; encoder bracket + AMT212"),
    ("spine_level", "S2  level bracket clamp on the axle (pinion in mesh with the level gear)"),
    ("spine_slipring", "S3  slip-ring sleeve clamp on the axle"),
    ("spine_bay_L_sub", "S4  left bay: GNSS receiver on spacers, Jetson standoffs"),
    ("spine_bay_L", "S4b left bay: clamps on the axle, tray"),
    ("spine_bay_R_sub", "S5  right bay electronics: level controller, DC-DC, IMU"),
    ("spine_bay_R_mezz", "S5b right bay: mezzanine on the controller standoffs, USB hub"),
    ("spine_bay_R", "S6  right bay: clamps on the axle, tray"),
    ("spine_gnss_sub", "S7  GNSS antenna on its plate"),
    ("spine_gnss", "S8  GNSS plate onto the left bay clamps"),
    ("pod_L_wheels", "P1  left pod: V-wheels on eccentric studs"),
    ("pod_L_plate", "P2  left pod: plate onto the hub"),
    ("pod_L_lidar_sub", "P3  left pod: Mid-360 on its riser"),
    ("pod_L_lidar", "P4  left pod: riser onto the plate (dowelled)"),
    ("pod_L_shield_sub", "P5  left pod: cameras into holders, holders into the shield"),
    ("pod_L_shield", "P6  left pod: shield onto the standoffs"),
    ("pod_L_on_axle", "P7  left pod hub clamp on the axle end"),
    ("pod_R_wheels", "P8  right pod: crowned rollers on studs"),
    ("pod_R_plate", "P9  right pod: plate onto the hub"),
    ("pod_R_lidar_sub", "P10 right pod: Mid-360 on its riser"),
    ("pod_R_lidar", "P11 right pod: riser onto the plate"),
    ("pod_R_shield_sub", "P12 right pod: cameras, holders"),
    ("pod_R_shield", "P13 right pod: shield"),
    ("pod_R_on_axle", "P14 right pod hub clamp on the axle end"),
]


def main():
    P = json.load(open(os.path.join(OUT, "parts.json")))
    J = {j["name"]: j for j in json.load(open(os.path.join(OUT, "joints.json")))}
    by = collections.defaultdict(lambda: collections.Counter())
    extra = collections.Counter()
    for p in P:
        if p.get("kind") != "fastener":
            continue
        t = p.get("type")
        if t == "screw":
            j = J.get(p.get("joint", p["name"]), {})
            into = j.get("tapped_in") or j.get("insert_in") or j.get("pem_in") or "nut"
            how = {"tap": "tapped", "insert": "heat-set insert", "pem": "PEM nut", "nut": "nut"}.get(j.get("end"), j.get("end"))
            key = (f"{p['std']} M{p['d']:g} x {p['L']} {p.get('grade', '')}", how,
                   "hand tight (0.5 N m)" if p.get("torque_Nm") is None else f"{p['torque_Nm']} N m",
                   p.get("threadlocker") or "-")
            by[p.get("step", "?")][key] += 1
        elif t in ("nyloc nut",):
            extra[(f"{p['std']} M{p['d']:g} nyloc on wheel studs", f"{p.get('torque_Nm', '')} N m")] += 1
    md = ["| step | fastener | into | torque | threadlocker | qty |", "|---|---|---|---|---|---|"]
    seen = set()
    for st, title in ORDER + [(s, s) for s in sorted(by) if s not in dict(ORDER)]:
        if st not in by or st in seen:
            continue
        seen.add(st)
        first = True
        for (f, how, tq, tl), n in sorted(by[st].items()):
            md.append(f"| {title if first else ''} | {f} | {how} | {tq} | {tl} | {n} |")
            first = False
    for (f, tq), n in extra.items():
        md.append(f"| studs | {f} | - | {tq} | (nyloc) | {n} |")
    tot = sum(sum(c.values()) for c in by.values()) + sum(extra.values())
    md.append(f"\n{tot} threaded fasteners in {len(seen)} steps. Torques: ISO 898-1 / ISO 3506 class, ~70 % of proof load, "
              "lightly oiled; into aluminium the tapped length is >= 1.5 d (6061) / 1.25 d (7075), checked by checks.py.")
    open(os.path.join(OUT, "assembly_steps.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md[:12]))
    print(f"... {len(md)} lines -> out/assembly_steps.md")


if __name__ == "__main__":
    main()
