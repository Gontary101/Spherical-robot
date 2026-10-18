"""
Bill of materials from the build CAD (cad/build/out/parts.json + joints.json, written by assemble.py).

    python cad/build/bom.py   -> cad/build/out/bom.csv, bom.md

Custom parts are listed per drawing (mirrored L/R pairs share a drawing, marked 'opp. hand'); bought parts by part
number; fasteners by standard x size x length with their tightening torque; consumables at the end.
"""
import csv
import json
import os
import re
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

CONSUMABLES = [
    ("3M Scotch-Weld DP490 (black), 50 ml", 4, "ring gear / rim / tread primer bonds"),
    ("Loctite 243 threadlocker, 10 ml", 1, "all steel screws into metal (not into inserts / PEM)"),
    ("Loctite 638 retaining compound, 10 ml", 1, "hub bearing outer races, pinion bearing seats"),
    ("Loctite 648 retaining compound, 5 ml", 1, "pulley bores on the motor / pinion shafts (with keys and set screws)"),
    ("Dow 732 RTV silicone, 90 ml", 1, "camera windows, light ring seam"),
    ("Molykote BR2 Plus grease, 50 g", 1, "drive and level gear meshes, lean rack"),
    ("3M VHB 4910, 25 mm x 3 m", 1, "cable clips, antenna cable"),
    ("Smooth-On VytaFlex 70 (or equivalent 70A PU), 2 gal kit", 1, "tread cast (both halves)"),
    ("E-glass biaxial 600 g/m2 + infusion epoxy, per drum", 2, "drum infusion"),
    ("PU 90A sheet 3 mm", 1, "lean stop pads"),
    ("Kapton tape, fish paper 0.25 mm, 0.15 mm pure nickel strip", 1, "battery modules"),
    ("Cable: 10 AWG silicone (battery, motors), 22 AWG twisted pair (CAN), Cat5e (LiDAR), USB-C cables", 1, "harness (docs/11_build.md wiring table)"),
]


def base_name(n):
    """Drawing name: strip side / index suffixes (_L, _R, _F, _B, digits)."""
    if n.endswith("_enc_bracket"):
        return "drive_enc_bracket" if n.startswith("drive_motor") else "level_enc_bracket"
    if re.match(r"pitch_striker\d$", n):
        return "pitch_striker"
    b = n
    for _ in range(3):
        b = re.sub(r"_(L|R|F|B|A|FL|FR|BL|BR|Fp|Fn|Bp|Bn|o|i|p|n|Fin|Fout|Bin|Bout)?\d*$", "", b)
    return b


MC = json.load(open(os.path.join(HERE, "mcmaster.json")))
EXTRA_HW = [   # hardware that is in the BOM but not modelled as solids
    ("DIN 6885 A parallel key 3x3x14", 3, "P20 pulleys on the drive motors, level pinion"),
    ("DIN 6885 A parallel key 4x4x16", 2, "P32 pulleys on the pinion shafts"),
    ("ISO 4029 cup-point set screw M4x6", 3, "P20 pulleys, level pinion (Loctite 243)"),
    ("ISO 4029 cup-point set screw M5x6", 2, "P32 pulleys (Loctite 243)"),
]


def mc_lookup(p):
    """McMaster-Carr number for a fastener part (cad/build/mcmaster.json, each entry with its source)."""
    t, d = p.get("type", ""), p.get("d")
    key = None
    if t == "screw":
        L = p["L"]
        if p["head"] == "SHCS":
            key = f"ISO 4762 SHCS M{d:g}x{L:g} A2"
        elif p["head"] == "BHCS":
            key = f"ISO 7380 button head M{d:g}x{L:g} A2"
        elif d == 8:
            key = "ISO 10642 flat head M8x14 alloy 12.9"
        else:
            key = f"ISO 10642 flat head M{d:g}x{L:g} A2"
    elif t == "nyloc nut":
        key = f"ISO 7040 / DIN 985 nylon-insert locknut M{d:g} A2"
    elif t == "washer":
        key = f"ISO 7089 / DIN 125 washer M{d:g} A2"
    elif t == "dowel pin":
        key = "ISO 2338 m6 dowel pin 3 x 6 mm (18-8)" if d == 3 else "ISO 2338 / ISO 8734 m6 dowel pin 8 x 20 mm"
    elif t == "retaining ring":
        key = "DIN 471 external retaining ring, 12 mm shaft" if p.get("std") == "DIN 471" else "DIN 472 internal retaining ring, 62 mm bore"
    elif t == "retaining washer":
        key = "DIN 6799 E-ring, 7 mm groove (shaft 8-11 mm)"
    elif t == "self-clinching nut":
        key = "Self-clinching nut M4, 3 mm sheet" if "3.0 mm" in p.get("pn", "") else f"Self-clinching nut M{d:g}, 1.5 mm Al sheet"
    elif t == "heat-set insert":
        key = {2: "Heat-set insert M2 (~4 mm)", 3: "Heat-set insert M3 (~5.7 mm)", 4: "Heat-set insert M4 (~8 mm)"}.get(d)
    e = MC.get(key) if key else None
    if not e:
        series = {"SHCS": "91292A", "BHCS": "92095A", "FHCS": "92125A"}.get(p.get("head")) if t == "screw" else None
        return f"{series} (series - pick M{d:g} x {p['L']:g})" if series else ""
    return e["mcmaster"].replace(" (series)", "") + ("" if e["verified"] else " (series - pick size)")


def main():
    parts = json.load(open(os.path.join(OUT, "parts.json")))
    joints = {j["name"]: j for j in json.load(open(os.path.join(OUT, "joints.json")))}
    custom, cots, fast = OrderedDict(), OrderedDict(), defaultdict(lambda: dict(qty=0, mass=0.0))
    for p in parts:
        k = p["kind"]
        if k == "custom":
            key = base_name(p["name"])
            e = custom.setdefault(key, dict(qty=0, mass=0.0, material=p.get("material", ""), process=p.get("process", ""),
                                            finish=p.get("finish", ""), names=[]))
            e["qty"] += 1
            e["mass"] += p["mass"]
            e["names"].append(p["name"])
        elif k == "cots":
            key = p.get("pn", p["name"])
            e = cots.setdefault(key, dict(qty=0, mass=0.0, names=[], verified=p.get("verified", True), source=p.get("source", "")))
            e["qty"] += 1
            e["mass"] += p["mass"]
            e["names"].append(p["name"])
        else:
            t = p.get("type", "")
            if t == "screw":
                j = joints.get(p["name"], {})
                key = f"{p['std']} {p['head']} M{p['d']:g} x {p['L']:g}, {p['grade']}"
                tq = p.get("torque_Nm")
                fast[key]["torque"] = tq
            elif t in ("washer", "nyloc nut", "retaining ring", "retaining washer"):
                key = f"{p.get('std', '')} {t} M{p.get('d', p.get('D', 0)):g}" if "d" in p else f"{p.get('std', '')} {t} {p.get('D', '')}"
            elif t == "dowel pin":
                key = f"ISO 8734 dowel pin d{p['d']:g} x {p['L']:g} m6"
            else:
                key = p.get("pn", t or p["name"])
            fast[key]["qty"] += 1
            fast[key]["mass"] += p["mass"]
            fast[key]["mc"] = mc_lookup(p)
    rows = []
    for k, e in custom.items():
        rows.append(["custom", k, e["qty"], e["material"], e["process"], e["finish"], round(e["mass"], 4), ", ".join(e["names"][:8])])
    for k, e in cots.items():
        rows.append(["bought", k, e["qty"], "", "verified" if e["verified"] else "CHECK hole pattern before machining", e["source"],
                     round(e["mass"], 4), ", ".join(e["names"][:8])])
    for k, e in sorted(fast.items()):
        rows.append(["fastener", k, e["qty"], "", "" if e.get("torque") is None else f"{e['torque']} N m",
                     f"McMaster {e['mc']}" if e.get("mc") else "", round(e["mass"], 4), ""])
    for k, q, use in EXTRA_HW:
        e = MC.get(k, {})
        rows.append(["fastener", k, q, "", use, f"McMaster {e.get('mcmaster', '').replace(' (series)', '')}" + ("" if e.get("verified") else " (series - pick size)"), "", ""])
    for k, q, use in CONSUMABLES:
        rows.append(["consumable", k, q, "", use, "", "", ""])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "bom.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "item", "qty", "material", "process / torque / status", "finish / source", "mass_kg", "CAD parts"])
        w.writerows(rows)
    md = ["| category | item | qty | material | process / torque | finish / source | mass kg |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| " + " | ".join(str(c) for c in r[:7]) + " |")
    open(os.path.join(OUT, "bom.md"), "w").write("\n".join(md) + "\n")
    n_c = sum(1 for r in rows if r[0] == "custom")
    n_b = sum(1 for r in rows if r[0] == "bought")
    n_f = sum(r[2] for r in rows if r[0] == "fastener")
    print(f"BOM: {n_c} custom drawings, {n_b} bought part numbers, {n_f} fasteners in {sum(1 for r in rows if r[0] == 'fastener')} kinds")


if __name__ == "__main__":
    main()
