"""Embed cad/out/gyra_mk1.glb + manifest into a single self-contained viewer page (docs/viewer/index.html)."""
import base64
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
man = json.load(open(os.path.join(ROOT, "cad", "out", "manifest.json")))
small = [{"n": e["name"], "g": e["group"], "m": e["material"], "x": [round(v, 1) for v in e["explode_mm"]]} for e in man]
glb = base64.b64encode(open(os.path.join(ROOT, "cad", "out", "gyra_mk1.glb"), "rb").read()).decode()
html = open(os.path.join(ROOT, "render", "viewer_template.html")).read()
html = html.replace("__MANIFEST__", json.dumps(small, separators=(",", ":"))).replace("__GLB__", glb)
os.makedirs(os.path.join(ROOT, "docs", "viewer"), exist_ok=True)
out = os.path.join(ROOT, "docs", "viewer", "index.html")
open(out, "w").write(html)
print("wrote", out, len(html) // 1024, "KiB")
