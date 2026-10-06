"""Embed a GLB + manifest into a single self-contained viewer page.

    python render/build_viewer.py          # Mk1 -> docs/viewer/index.html
    python render/build_viewer.py mk2      # Mk2 -> docs/viewer/mk2.html
"""
import base64
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
which = sys.argv[1] if len(sys.argv) > 1 else "mk1"
CFG = {
    "mk1": dict(cad="out", glb="gyra_mk1.glb", page="index.html", title="GYRA Mk1 Viewer", h1="GYRA MK1",
                sub="600 mm rubber-tyred spherical robot: pendulum drive, levelled sensor spine, scissored CMG pair.",
                tyre="Tyre: spins about the axle", pmax=75, bmax=30,
                pitch_help="The drive motors push against the yoke. It swings forward and gravity rolls the tyre. Hard stop ±75°.",
                bob_help="A worm drive slides the 10.9 kg battery + lead bob along a ±30° arc. That leans the robot, and precession turns it.",
                exp_help="Tyre halves, pods, CMG modules and the bob pull apart along their assembly directions.",
                spine="Spine: axle, pods, CMGs, kept level", bob="Bob: battery + lead, lateral lean",
                figs=[("Diameter", "600 mm"), ("Mass (CAD)", "41.4 kg"), ("CoM below centre", "59 mm"), ("Top speed", "6.4 m/s"),
                      ("Drive ratio", "30 : 1"), ("Rated grade", "10°"), ("Lean torque", "11.2 N·m"), ("CMG torque (pair)", "18.5 N·m"),
                      ("Flywheel h", "2.31 N·m·s @ 10 krpm"), ("Turn radius @ 2 m/s", "5.8 m (sim)"), ("Battery", "468 Wh"),
                      ("Draft afloat", "246 mm")]),
    "mk2": dict(cad="out2", glb="gyra_mk2.glb", page="mk2.html", title="GYRA Mk2 Viewer", h1="GYRA MK2",
                sub="Split twin-crown tyre with differential drive, 360° pendulum, tungsten + 936 Wh bob, learned control.",
                tyre="Tyre halves: each driven by its own motor", pmax=180, bmax=40,
                pitch_help="Both drive motors push against the yoke; their sum swings it and gravity rolls the robot. Nothing blocks it: it can swing a full 360°.",
                bob_help="A worm drive slides the 15.6 kg tungsten + 936 Wh battery bob along a ±40° arc to counter-lean in fast turns. Steering itself comes from the left/right torque difference.",
                exp_help="Tyre halves, pods and the bob pull apart along their assembly directions.",
                spine="Spine: axle, pods, bays, kept level", bob="Bob: battery + tungsten, lateral lean",
                figs=[("Size", "600 × 700 mm"), ("Mass (CAD)", "45.1 kg"), ("CoM below centre", "92 mm"), ("Contact patches", "2, 100 mm apart"),
                      ("Turn in place", "176°/s (sim)"), ("Top stable speed", "8.5 m/s (sim)"), ("Grade", "14° (sim)"),
                      ("Drive ratio", "24 : 1 per half"), ("Battery", "936 Wh"), ("Nav success, 100 maps", "79% (learned)"),
                      ("Falls, randomised tests", "8% vs 67% classical")]),
}[which]
man = json.load(open(os.path.join(ROOT, "cad", CFG["cad"], "manifest.json")))
small = [{"n": e["name"], "g": e["group"], "m": e["material"], "x": [round(v, 1) for v in e["explode_mm"]]} for e in man]
glb = base64.b64encode(open(os.path.join(ROOT, "cad", CFG["cad"], CFG["glb"]), "rb").read()).decode()
html = open(os.path.join(ROOT, "render", "viewer_template.html")).read()
figs = "\n".join(f"        <dt>{k}</dt><dd>{v}</dd>" for k, v in CFG["figs"])
for k, v in (("__TITLE__", CFG["title"]), ("__H1__", CFG["h1"]), ("__SUB__", CFG["sub"]), ("__FIGURES__", figs),
             ("__TYRE_LEGEND__", CFG["tyre"]), ("__PMAX__", str(CFG["pmax"])), ("__BMAX__", str(CFG["bmax"])),
             ("__PITCH_HELP__", CFG["pitch_help"]), ("__BOB_HELP__", CFG["bob_help"]), ("__EXP_HELP__", CFG["exp_help"]),
             ("__SPINE_LEGEND__", CFG["spine"]), ("__BOB_LEGEND__", CFG["bob"]), ("__MANIFEST__", json.dumps(small, separators=(",", ":"))), ("__GLB__", glb)):
    html = html.replace(k, v)
os.makedirs(os.path.join(ROOT, "docs", "viewer"), exist_ok=True)
out = os.path.join(ROOT, "docs", "viewer", CFG["page"])
open(out, "w").write(html)
print("wrote", out, len(html) // 1024, "KiB")
