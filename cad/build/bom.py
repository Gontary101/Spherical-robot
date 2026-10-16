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
    b = n
    for _ in range(3):
        b = re.sub(r"_(L|R|F|B|A|FL|FR|BL|BR|Fp|Fn|Bp|Bn|o|i|p|n|Fin|Fout|Bin|Bout)?\d*$", "", b)
    return b


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
    rows = []
    for k, e in custom.items():
        rows.append(["custom", k, e["qty"], e["material"], e["process"], e["finish"], round(e["mass"], 4), ", ".join(e["names"][:8])])
    for k, e in cots.items():
        rows.append(["bought", k, e["qty"], "", "verified" if e["verified"] else "CHECK hole pattern before machining", e["source"],
                     round(e["mass"], 4), ", ".join(e["names"][:8])])
    for k, e in sorted(fast.items()):
        rows.append(["fastener", k, e["qty"], "", "" if e.get("torque") is None else f"{e['torque']} N m", "", round(e["mass"], 4), ""])
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
