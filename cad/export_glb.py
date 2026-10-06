"""Pack the per-part STLs into one glTF binary (cad/out/gyra_mk1.glb) with PBR materials.

Node names = part names, so a viewer can re-group them by kinematic body using
cad/out/manifest.json (group + explode offsets).
"""
import json
import os
import sys

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from render_colors import RGB  # noqa: E402

PBR = {  # material -> (metallic, roughness, alpha)
    "alu": (1.0, 0.3, 1), "alu_dark": (0.9, 0.4, 1), "steel": (1.0, 0.25, 1), "brass": (1.0, 0.25, 1),
    "lead": (0.6, 0.55, 1), "rubber": (0.0, 0.88, 1), "nylon": (0.0, 0.62, 1), "gfrp": (0.0, 0.5, 1), "carbon": (0.2, 0.35, 1), "glass": (0.1, 0.05, 1), "polycarb": (0.0, 0.05, 0.35),
}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "out"          # "out" (Mk1) or "out2" (Mk2)
    out = os.path.join(HERE, tag)
    man = json.load(open(os.path.join(out, "manifest.json")))
    scene = trimesh.Scene()
    for e in man:
        m = trimesh.load(os.path.join(out, e["stl"]), force="mesh")
        m.apply_scale(0.001)
        m.merge_vertices()
        r, g, b = RGB.get(e["material"], (0.6, 0.6, 0.6))
        met, rough, alpha = PBR.get(e["material"], (0.0, 0.55, 1))
        emissive = [1.0, 0.45, 0.05] if e["material"] == "led" else [0, 0, 0]
        m.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(
            name=e["material"], baseColorFactor=[r, g, b, alpha], metallicFactor=met,
            roughnessFactor=rough, emissiveFactor=emissive, alphaMode="BLEND" if alpha < 1 else "OPAQUE"))
        # glTF is y-up: robot (x fwd, y left, z up) -> glTF (x, z, -y)
        T = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
        m.apply_transform(T)
        scene.add_geometry(m, node_name=e["name"], geom_name=e["name"])
    path = os.path.join(out, "gyra_mk1.glb" if tag == "out" else "gyra_mk2.glb")
    scene.export(path)
    print("wrote", path, os.path.getsize(path) // 1024, "KiB")


if __name__ == "__main__":
    main()
