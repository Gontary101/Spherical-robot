"""
Blender (Cycles) shots of the GYRA Mk2.1 CAD for the video. Dark studio matching the explainer palette.

    blender -b -P render/robot_shots.py -- --shot title|assembly|xray|roll --out <dir> [--start 0 --end -1]

Frames are written as <dir>/<shot>_<frame>.png (resumable: existing frames are skipped) plus
<dir>/<shot>_<frame>.json: screen positions (pixels) of the label anchors of each part group, for the compositor.
"""
import argparse
import json
import math
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_gyra as RG  # noqa: E402

FPS = 24
SHOTS = dict(title=7.0, assembly=32.0, xray=24.0, roll=6.0)
# label anchors: group -> part-name prefixes (centre of their world bounding boxes)
ANCHORS = {
    "tyre_halves": ("drum_", "lugs_"), "seam": ("eq_slewing_ring", "eq_flange_"), "axle": ("axle",),
    "pods": ("pod_shield_", "pod_plate_"), "lidar": ("lidar_body_", "lidar_window_"), "cameras": ("cam0_", "cam1_", "cam2_"),
    "yoke": ("hanger_", "arc_plate_"), "drive": ("drive_motor_", "belt_", "pinion_"), "ring_gears": ("ring_gear_",),
    "bob": ("bob_tungsten", "bob_battery"), "lean": ("bob_lean_drive", "lean_rack", "bob_lean_pinion"),
    "level": ("level_motor", "level_gear"), "compute": ("compute_pendulum",), "gnss": ("gnss_",),
}
# assembly order (group of the manifest, start time s, duration s)
ASSEMBLY = [("yoke", 2.0, 3.5), ("bob", 6.0, 3.5), ("spine", 10.5, 4.0), ("tyreL", 15.5, 3.5), ("tyreR", 19.5, 3.5)]
HIGHLIGHT = [  # x-ray sequence: (name, part prefixes, rgb, t0, t1)
    ("drive", ("drive_motor_", "pulley_", "belt_", "pinion_", "ring_gear_"), (0.29, 0.84, 0.61), 1.0, 8.0),
    ("level", ("level_motor", "level_gear"), (1.0, 0.71, 0.28), 8.0, 13.5),
    ("lean", ("bob_lean_drive", "lean_rack", "bob_lean_pinion", "bob_rails"), (0.66, 0.55, 1.0), 13.5, 18.5),
    ("sensors", ("lidar_", "cam0_", "cam1_", "cam2_", "lens", "gnss_"), (0.27, 0.78, 1.0), 18.5, 23.5),
]


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--step", type=int, default=1)
    return ap.parse_args(argv)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def studio(sc):
    """Dark navy studio: cyclorama floor, low-key lights, navy world."""
    RG.stage(sc, dark=True)
    fl = bpy.data.objects["floor"]
    m = fl.data.materials[0]
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.035, 0.045, 0.075, 1)
    b.inputs["Roughness"].default_value = 0.35
    bg = sc.world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.012, 0.018, 0.04, 1)
    bg.inputs["Strength"].default_value = 0.6
    # amber accent from behind (the robot's status light colour)
    bpy.ops.object.light_add(type="AREA", location=(-2.4, 0.6, 1.2))
    l = bpy.context.object
    l.data.energy, l.data.size, l.data.color = 160, 2.0, (1.0, 0.68, 0.3)
    l.rotation_euler = (Vector((0, 0, 0)) - Vector(l.location)).to_track_quat("-Z", "Y").to_euler()


def setup(shot, res, samples):
    RG.CAD_OUT = os.path.join(RG.ROOT, "cad", "out21")
    sc = RG.reset()
    W, H = (int(v) for v in res.split("x"))
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces, sc.cycles.diffuse_bounces, sc.cycles.glossy_bounces, sc.cycles.transmission_bounces = 4, 2, 2, 4
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.render.use_persistent_data = True
    objs = RG.load()
    M = RG.materials()
    rg = RG.rig(objs)
    ghost = ("drum", "tread_base", "lugs", "eq_flange", "pod_shield", "pod_plate", "led_ring", "rim") if shot == "xray" else ()
    if shot == "xray":                                        # darker, thinner shell for the dark studio
        nt = M["ghost"].node_tree
        for n in nt.nodes:
            if n.type == "BSDF_PRINCIPLED":
                n.inputs["Base Color"].default_value = (0.2, 0.32, 0.5, 1)
                n.inputs["Emission Color"].default_value = (0.25, 0.45, 0.85, 1)
                n.inputs["Emission Strength"].default_value = 0.04
            if n.type == "MAP_RANGE":
                n.inputs["To Min"].default_value = 0.0
                n.inputs["To Max"].default_value = 0.22
    RG.assign(objs, M, (), ghost)
    studio(sc)
    cam = RG.camera(sc, (1.6, 1.1, 0.4), (0, 0, -0.05), lens=50)
    return sc, objs, rg, cam, M


def highlight_materials(objs):
    """Per highlight group: an emissive copy of each part's material whose strength is animated per frame."""
    hl = {}
    for name, prefixes, rgb, _, _ in HIGHLIGHT:
        mats = []
        for o in objs.values():
            if o.name.startswith(prefixes):
                base = o.data.materials[0]
                m = base.copy()
                m.name = f"hl_{name}_{o.name}"
                b = m.node_tree.nodes.get("Principled BSDF")
                if b is not None:
                    b.inputs["Emission Color"].default_value = (*rgb, 1)
                    b.inputs["Emission Strength"].default_value = 0.0
                    mats.append(b)
                o.data.materials.clear()
                o.data.materials.append(m)
        hl[name] = mats
    return hl


def anchors(sc, cam, objs):
    W, H = sc.render.resolution_x, sc.render.resolution_y
    out = {}
    for key, prefixes in ANCHORS.items():
        pts = [o.matrix_world @ Vector(c) for o in objs.values() if o.name.startswith(prefixes) for c in o.bound_box]
        if not pts:
            continue
        c = sum(pts, Vector((0, 0, 0))) / len(pts)
        p = world_to_camera_view(sc, cam, c)
        if p.z > 0:
            out[key] = [round(p.x * W, 1), round((1 - p.y) * H, 1)]
    return out


def look(cam, loc, target):
    cam.location = loc
    cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()


def pose(shot, t, sc, objs, rg, cam, base_loc, hl):
    if shot == "title":
        a = math.radians(30 + 40 * ease(t / SHOTS["title"]))
        r = 2.75 - 0.25 * ease(t / SHOTS["title"])
        # robot whole and in the upper right third; the title sits lower left
        c, s_ = math.cos(a), math.sin(a)
        target = (0.42 * s_, -0.42 * c, -0.16)
        look(cam, (r * c, r * s_, 0.42 + 0.06 * math.sin(t * 0.5)), target)
        rg["tyre"].rotation_euler = (0, math.radians(-20 * t), 0)
    elif shot == "assembly":
        for o in objs.values():
            g = o["group"]
            prog = 1.0
            for grp, t0, dur in ASSEMBLY:
                if g == grp or (grp == "spine" and g == "spine"):
                    prog = ease((t - t0) / dur)
            e = o["explode"]
            k = 1.7 * (1 - prog)
            if g in ("yoke", "bob"):
                k = 1.0 * (1 - prog)                          # inner parts fly in from below/side, closer
            o.location = (base_loc[o.name][0] + e[0] * k, base_loc[o.name][1] + e[1] * k, base_loc[o.name][2] + e[2] * k
                          - (0.6 * (1 - prog) if g in ("yoke", "bob") else 0.0))
        a = math.radians(25 + 55 * ease(t / SHOTS["assembly"]))
        r = 3.1 - 1.1 * ease((t - 18) / 12.0)
        look(cam, (r * math.cos(a), r * math.sin(a), 0.75 - 0.3 * ease(t / SHOTS["assembly"])), (0, 0, -0.08))
    elif shot == "xray":
        rg["robot"].rotation_euler = (0, 0, math.radians(-15 + 60 * t / SHOTS["xray"]))
        rg["yoke"].rotation_euler = (0, math.radians(32 * math.sin(2 * math.pi * t / 6.0)), 0)
        rg["bob"].rotation_euler = (math.radians(22 * math.sin(2 * math.pi * t / 4.0) * ease((t - 13.5) / 1.0) * (1 - ease((t - 18.5) / 1.0))), 0, 0)
        rg["tyre"].rotation_euler = (0, math.radians(-25 * t), 0)
        look(cam, (1.55, 1.0, 0.42), (0, 0, -0.07))
        for name, _, _, t0, t1 in HIGHLIGHT:
            s = 10.0 * min(ease((t - t0) / 0.6), 1 - ease((t - t1 + 0.6) / 0.6))
            for b in hl.get(name, []):
                b.inputs["Emission Strength"].default_value = s
    elif shot == "roll":
        x = -1.2 + 0.55 * t
        rg["robot"].location = (x, 0, 0)
        rg["tyre"].rotation_euler = (0, x / RG.R, 0)
        rg["yoke"].rotation_euler = (0, math.radians(-28), 0)
        look(cam, (x + 1.35, 1.25, 0.32), (x + 0.1, 0, -0.06))


def main():
    a = args()
    sc, objs, rg, cam, M = setup(a.shot, a.res, a.samples)
    base_loc = {n: tuple(o.location) for n, o in objs.items()}
    hl = highlight_materials(objs) if a.shot == "xray" else {}
    os.makedirs(a.out, exist_ok=True)
    n = int(SHOTS[a.shot] * FPS)
    end = n if a.end < 0 else min(a.end, n)
    for f in range(a.start, end, a.step):
        path = os.path.join(a.out, f"{a.shot}_{f:05d}.png")
        if os.path.exists(path):
            continue
        pose(a.shot, f / FPS, sc, objs, rg, cam, base_loc, hl)
        bpy.context.view_layer.update()
        json.dump(anchors(sc, cam, objs), open(path[:-4] + ".json", "w"))
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print("FRAME", path, flush=True)


if __name__ == "__main__":
    main()
