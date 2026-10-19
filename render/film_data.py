"""
Data for the build film (render/build_film.py), from the build CAD:

    python render/film_data.py        -> cad/build/out/film/data.json + cad/build/out/film/mesh/<part>.stl

Per part: body, kind, render material, bounding box and centre (mm, robot frame), the assembly group it travels
with (cad/build/assembly.py's joint-graph groups), and for fasteners the joint they belong to (step, seat, axis).
Meshes are tessellated finer than the release meshes (smooth curved parts on screen).
"""
import json
import math
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD = os.path.join(ROOT, "cad", "build")
sys.path.insert(0, BUILD)
sys.path.insert(0, os.path.join(ROOT, "cad"))

import spec as S  # noqa: E402
from assemble import build_all  # noqa: E402
from assembly import groups  # noqa: E402
from yoke import drive_geom  # noqa: E402

OUT = os.path.join(BUILD, "out", "film")


def main():
    os.makedirs(os.path.join(OUT, "mesh"), exist_ok=True)
    parts, joints, threads, bonds = build_all(verbose=False)
    J = {j.name: j for j in joints}
    links = ([("level_pinion", "level_motor"), ("slipring_rotor", "slip_sleeve"), ("slipring_housing", "slip_sleeve"),
              ("compute", "compute_standoff0")] + [(f"compute_standoff{k}", "bay_tray_L") for k in range(4)]
             + [(f"wheel_stud_{s}{k}", f"pod_plate_{s}") for s in "LR" for k in range(8)]
             + [(f"{w}_{s}{k}", f"wheel_stud_{s}{k}") for s, w0 in (("L", "vwheel"), ("R", "roller")) for k in range(8)
                for w in (w0, "stud_washer")]
             + [(f"lidar_dowel_{s}{k}", f"lidar_riser_{s}") for s in "LR" for k in range(2)]
             + [(f"cam_window_{s}{k}", f"pod_shield_{s}") for s in "LR" for k in range(3)]
             + [(f"light_ring_{s}", f"pod_plate_{s}") for s in "LR"] + [(f"led_pcb_{s}", f"light_ring_{s}") for s in "LR"])
    G = groups(parts, joints, [t for t in threads if set(t) != {"slipring_housing", "slipring_pin"}], bonds, links=links)
    gid = {}
    for k, g in enumerate(G):
        for p in g:
            gid[p.name] = k
    out = {}
    for p in parts:
        bb = p.shape.BoundingBox()
        e = dict(body=p.body, kind=p.kind, render=p.render, group=gid.get(p.name),
                 bmin=[round(bb.xmin, 3), round(bb.ymin, 3), round(bb.zmin, 3)],
                 bmax=[round(bb.xmax, 3), round(bb.ymax, 3), round(bb.zmax, 3)],
                 pn=p.spec.get("pn", ""), material=p.spec.get("material", ""))
        if p.kind == "fastener":
            jn = None
            for cand in (p.name, re.sub(r"_(washer|nut|nwasher|insert|pem)$", "", p.name)):
                if cand in J:
                    jn = cand
                    break
            if jn:
                ji = J[jn].info
                e.update(joint=jn, step=ji.get("step") or J[jn].step, seat=ji["seat"], n=ji["n"], d=ji["d"])
            else:
                e.update(joint=None, step=None, type=p.spec.get("type", ""))
        out[p.name] = e
        tol = 0.03 if p.kind == "fastener" else 0.05
        p.shape.exportStl(os.path.join(OUT, "mesh", f"{p.name}.stl"), tolerance=tol, angularTolerance=0.12)
    (pxF, pzF), (mxF, mzF) = drive_geom(1)
    (pxB, pzB), (mxB, mzB) = drive_geom(-1)
    geo = dict(R_OUT=S.R_OUT, D=S.D, drive=dict(F=dict(px=pxF, pz=pzF, mx=mxF, mz=mzF), B=dict(px=pxB, pz=pzB, mx=mxB, mz=mzB)),
               level_zc=S.LEVEL_GEAR["m"] * (S.LEVEL_GEAR["z"] + S.LEVEL_GEAR["pinion"]) / 2,
               ratio_drive=24.0, ratio_ring=S.Z_RING / S.Z_PINION if hasattr(S, "Z_RING") else 15.0, ratio_belt=1.6,
               ratio_level=S.LEVEL_GEAR["z"] / S.LEVEL_GEAR["pinion"], rack_r=S.RACK_R,
               lean_pinion_r=S.MODULE * S.LEAN_PINION_Z / 2, pitch_stop=S.PITCH_STOP, bob_lean=S.BOB_LEAN,
               drive_angle=S.DRIVE_ANGLE)
    json.dump(dict(parts=out, geo=geo), open(os.path.join(OUT, "data.json"), "w"), indent=0)
    print(f"{len(out)} parts, {len(G)} groups -> {OUT}")


if __name__ == "__main__":
    main()
