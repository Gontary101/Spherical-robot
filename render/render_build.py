"""
Renders of the build CAD (every part and fastener) from the per-part meshes written by
`python cad/build/export.py --meshes --meshes-only`.

    blender -b -P render/render_build.py -- --view all --samples 64 --res 1800 1100

Views: internals (tyre shells hidden), yoke (yoke + bob), cassette (front drive cassette close-up), spine (axle,
level drive, bays, pods), bob (close-up), exploded. Output: media/build_<view>.png
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MESH = os.path.join(ROOT, "cad", "build", "out", "release", "mesh")
sys.path.insert(0, os.path.join(ROOT, "cad"))
from render_colors import RGB  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default, n=1, cast=str):
    if name in argv:
        i = argv.index(name)
        v = [cast(x) for x in argv[i + 1:i + 1 + n]]
        return v[0] if n == 1 else v
    return default


VIEW = arg("--view", "all")
SAMPLES = arg("--samples", 64, cast=int)
RES = arg("--res", [1800, 1100], n=2, cast=int)

ROUGH = {"alu": 0.32, "alu_dark": 0.38, "steel": 0.25, "brass": 0.3, "rubber": 0.8, "gfrp": 0.5, "carbon": 0.35, "lead": 0.45,
         "battery": 0.4, "nylon": 0.7, "nylon_light": 0.6, "lidar_white": 0.4, "glass": 0.05, "motor_black": 0.45, "pcb": 0.5}
METAL = {"alu", "alu_dark", "steel", "brass", "lead"}


def material(key):
    name = f"m_{key}"
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    r, g, bl = RGB.get(key, (0.6, 0.6, 0.6))
    b.inputs["Base Color"].default_value = (r, g, bl, 1.0)
    b.inputs["Roughness"].default_value = ROUGH.get(key, 0.45)
    b.inputs["Metallic"].default_value = 0.85 if key in METAL else 0.0
    if key == "glass":
        b.inputs["Transmission Weight"].default_value = 0.9
    return m


def load():
    man = json.load(open(os.path.join(MESH, "manifest.json")))
    objs = {}
    for e in man:
        bpy.ops.wm.stl_import(filepath=os.path.join(MESH, e["file"]))
        o = bpy.context.selected_objects[0]
        o.name = e["name"]
        o.scale = (0.001, 0.001, 0.001)
        o.data.materials.clear()
        o.data.materials.append(material(e["render"]))
        o["body"] = e["body"]
        o["kind"] = e["kind"]
        objs[e["name"]] = o
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.shade_auto_smooth() if hasattr(bpy.ops.object, "shade_auto_smooth") else None
    bpy.ops.object.select_all(action="DESELECT")
    return objs


def scene_setup():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in sc.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
    w = bpy.data.worlds.new("w") if not sc.world else sc.world
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.80, 0.82, 0.85, 1.0)
    bg.inputs["Strength"].default_value = 0.25
    for nm, loc, energy, size in (("key", (1.2, -1.4, 1.6), 140, 1.2), ("fill", (-1.5, -0.6, 0.6), 45, 1.5), ("rim", (0.2, 1.6, 1.0), 80, 1.0)):
        ld = bpy.data.lights.new(nm, "AREA")
        ld.energy, ld.size = energy, size
        lo = bpy.data.objects.new(nm, ld)
        sc.collection.objects.link(lo)
        lo.location = loc
        d = Vector((0, 0, -0.05)) - lo.location
        lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    # floor
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, -0.302))
    fl = bpy.context.active_object
    fl.name = "floor"
    m = bpy.data.materials.new("floor")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.55, 0.56, 0.58, 1)
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
    fl.data.materials.append(m)
    return sc


def camera(sc, loc, target, lens=50):
    cd = bpy.data.cameras.get("cam") or bpy.data.cameras.new("cam")
    cd.lens = lens
    co = bpy.data.objects.get("cam") or bpy.data.objects.new("cam", cd)
    if co.name not in sc.collection.objects:
        sc.collection.objects.link(co)
    co.location = loc
    co.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    sc.camera = co


def show(objs, pred):
    for o in objs.values():
        v = bool(pred(o))
        o.hide_render = not v
        o.hide_viewport = not v


def render(sc, name):
    sc.render.filepath = os.path.join(ROOT, "media", f"build_{name}.png")
    bpy.ops.render.render(write_still=True)
    print("wrote", sc.render.filepath, flush=True)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    objs = load()
    sc = scene_setup()
    shells = ("drum_", "tread_")
    views = {
        "internals": (lambda o: not o.name.startswith(shells) and not o.name.startswith("pod_shield"),
                      (0.95, -1.05, 0.55), (0, 0, -0.04), 50),
        "yoke": (lambda o: o["body"] in ("yoke", "bob"), (0.62, -0.62, 0.22), (0, 0, -0.12), 50),
        "cassette": (lambda o: o["body"] in ("yoke", "bob"), (0.45, 0.42, -0.02), (0.15, 0.1, -0.14), 60),
        "spine": (lambda o: o["body"] == "spine" and not o.name.startswith("pod_shield"), (0.75, -0.8, 0.45), (0, 0, 0), 50),
        "bob": (lambda o: o["body"] == "bob", (0.42, -0.30, -0.05), (0, 0, -0.2), 55),
    }
    todo = list(views) + ["exploded"] if VIEW == "all" else [VIEW]
    for v in todo:
        if v == "exploded":
            off = {"tyreL": (0, 0.25, 0), "tyreR": (0, -0.25, 0), "bob": (0, 0, -0.18)}
            for o in objs.values():
                d = off.get(o["body"], (0, 0, 0))
                if o["body"] == "spine" and ("_L" in o.name[-4:] or o.name.endswith("_L")) and o.name.startswith(("pod", "wheel", "stud", "vwheel", "lidar", "light", "led", "cam", "shield", "riser")):
                    d = (0, 0.45, 0)
                if o["body"] == "spine" and ("_R" in o.name[-4:] or o.name.endswith("_R")) and o.name.startswith(("pod", "wheel", "stud", "roller", "lidar", "light", "led", "cam", "shield", "riser")):
                    d = (0, -0.45, 0)
                o.location = d
            show(objs, lambda o: not o.name.startswith("tread_"))
            bpy.data.objects["floor"].location.z = -0.48
            camera(sc, (1.6, -1.9, 0.75), (0, 0, -0.05), 45)
            render(sc, "exploded")
            for o in objs.values():
                o.location = (0, 0, 0)
            continue
        pred, loc, tgt, lens = views[v]
        show(objs, pred)
        camera(sc, loc, tgt, lens)
        render(sc, v)


main()
