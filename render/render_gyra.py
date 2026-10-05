"""
Blender (4.2, Cycles) render pipeline for GYRA Mk1.

    blender -b -P render/render_gyra.py -- --scene hero --out media/hero.png
    scenes: hero, side, cutaway, xray, exploded, section_front, section_side, top, pose

The CAD exports one STL per part plus cad/out/manifest.json; this script rebuilds the
kinematic hierarchy (yoke -> bob, CMG gimbals) with empties so the model can be posed.
Label anchors (3-D points projected to the image) are written next to the image as JSON
so render/annotate.py can draw call-outs.
"""
import argparse
import json
import math
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAD_OUT = os.path.join(ROOT, "cad", "out")
sys.path.insert(0, os.path.join(ROOT, "cad"))
from render_colors import RGB  # noqa: E402

CMG_Y, CMG_Z = 0.170, 0.122
R = 0.300


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="hero")
    ap.add_argument("--out", default=os.path.join(ROOT, "media", "hero.png"))
    ap.add_argument("--res", default="1600x1000")
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--bob", type=float, default=0.0)
    ap.add_argument("--gimbal", type=float, default=0.0)
    ap.add_argument("--explode", type=float, default=1.0)
    ap.add_argument("--yaw", type=float, default=0.0, help="robot heading for turntables (deg)")
    ap.add_argument("--roll", type=float, default=0.0, help="robot lean (deg)")
    ap.add_argument("--spin", type=float, default=0.0, help="tyre rotation (deg)")
    ap.add_argument("--labels-only", action="store_true")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.use_denoising = True
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.unit_settings.system = "METRIC"
    return sc


def mat(name, rgb, metallic=0.0, rough=0.5, emit=0.0, transmission=0.0, alpha=1.0, ior=1.45, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    b.inputs["IOR"].default_value = ior
    b.inputs["Transmission Weight"].default_value = transmission
    b.inputs["Alpha"].default_value = alpha
    b.inputs["Coat Weight"].default_value = coat
    if emit:
        b.inputs["Emission Color"].default_value = (*rgb, 1)
        b.inputs["Emission Strength"].default_value = emit
    if alpha < 1:
        m.blend_method = "BLEND"
    return m


def rubber_mat():
    m = mat("rubber", RGB["rubber"], rough=0.72)
    nt = m.node_tree
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 900.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], nt.nodes["Principled BSDF"].inputs["Normal"])
    return m


def materials(ghost=False):
    M = {
        "rubber": rubber_mat(),
        "gfrp": mat("gfrp", RGB["gfrp"], rough=0.45),
        "alu": mat("alu", RGB["alu"], metallic=1.0, rough=0.28),
        "alu_dark": mat("alu_dark", RGB["alu_dark"], metallic=0.9, rough=0.35),
        "carbon": mat("carbon", RGB["carbon"], rough=0.25, coat=0.8),
        "steel": mat("steel", RGB["steel"], metallic=1.0, rough=0.22),
        "brass": mat("brass", RGB["brass"], metallic=1.0, rough=0.22),
        "lead": mat("lead", RGB["lead"], metallic=0.7, rough=0.55),
        "battery": mat("battery", RGB["battery"], rough=0.35),
        "nylon": mat("nylon", RGB["nylon"], rough=0.5),
        "nylon_light": mat("nylon_light", RGB["nylon_light"], rough=0.5),
        "lidar_white": mat("lidar_white", RGB["lidar_white"], rough=0.35),
        "glass": mat("glass", RGB["glass"], rough=0.03, coat=1.0),
        "led": mat("led", RGB["led"], emit=12.0),
        "motor_black": mat("motor_black", RGB["motor_black"], metallic=0.4, rough=0.35),
        "pcb": mat("pcb", RGB["pcb"], rough=0.4),
        "orange": mat("orange", RGB["orange"], rough=0.4),
        "polycarb": mat("polycarb", RGB["polycarb"], rough=0.02, transmission=1.0, ior=1.58),
    }
    M["ghost"] = ghost_mat()
    return M


def ghost_mat():
    """Crisp x-ray shell: transparent, more opaque at grazing angles (no refraction blur)."""
    m = bpy.data.materials.new("ghost")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    gl = nt.nodes.new("ShaderNodeBsdfPrincipled")
    gl.inputs["Base Color"].default_value = (0.55, 0.75, 1.0, 1)
    gl.inputs["Roughness"].default_value = 0.25
    gl.inputs["Emission Color"].default_value = (0.35, 0.6, 1.0, 1)
    gl.inputs["Emission Strength"].default_value = 0.08
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["To Min"].default_value = 0.015
    ramp.inputs["To Max"].default_value = 0.42
    nt.links.new(lw.outputs["Facing"], ramp.inputs["Value"])
    nt.links.new(ramp.outputs["Result"], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(gl.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.blend_method = "BLEND"
    return m


def load(scale=0.001):
    man = json.load(open(os.path.join(CAD_OUT, "manifest.json")))
    objs = {}
    for e in man:
        path = os.path.join(CAD_OUT, e["stl"])
        bpy.ops.wm.stl_import(filepath=path)
        o = bpy.context.selected_objects[0]
        o.name = e["name"]
        o.scale = (scale, scale, scale)
        bpy.ops.object.transform_apply(scale=True)
        o["group"] = e["group"]
        o["material"] = e["material"]
        o["explode"] = [v * scale for v in e["explode_mm"]]
        for poly in o.data.polygons:
            poly.use_smooth = False
        objs[e["name"]] = o
    # smooth curved surfaces, keep edges sharper than 30 deg crisp
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs.values():
        o.select_set(True)
    bpy.context.view_layer.objects.active = next(iter(objs.values()))
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(30))
    bpy.ops.object.select_all(action="DESELECT")
    return objs


def empty(name, loc=(0, 0, 0), parent=None):
    e = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(e)
    e.location = loc
    if parent:
        e.parent = parent
    return e


def rig(objs):
    """robot (heading/lean) -> {tyre_spin, spine, yoke -> bob, cmg pivots}."""
    robot = empty("robot")
    tyre = empty("tyre_spin", parent=robot)
    spine = empty("spine", parent=robot)
    yoke = empty("yoke", parent=robot)
    bob = empty("bob", parent=yoke)
    cmgL = empty("cmgL", (0, CMG_Y, CMG_Z), parent=spine)
    cmgR = empty("cmgR", (0, -CMG_Y, CMG_Z), parent=spine)
    par = {"tyre": tyre, "spine": spine, "yoke": yoke, "bob": bob,
           "cmgL": cmgL, "flyL": cmgL, "cmgR": cmgR, "flyR": cmgR}
    bpy.context.view_layer.update()
    for o in objs.values():
        p = par[o["group"]]
        o.parent = p
        o.matrix_parent_inverse = p.matrix_world.inverted()
    return dict(robot=robot, tyre=tyre, spine=spine, yoke=yoke, bob=bob, cmgL=cmgL, cmgR=cmgR)


def assign(objs, M, ghost_groups=(), ghost_names=()):
    for o in objs.values():
        key = o["material"]
        if o["group"] in ghost_groups or o.name.startswith(ghost_names):
            key = "ghost"
        o.data.materials.clear()
        o.data.materials.append(M[key])
        if key == "ghost":
            o.visible_shadow = False


def stage(sc, floor_z=-R, dark=False):
    # cyclorama floor
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, floor_z))
    fl = bpy.context.object
    fl.name = "floor"
    fm = mat("floor", (0.11, 0.115, 0.12) if dark else (0.72, 0.73, 0.75), rough=0.55)
    fl.data.materials.append(fm)
    w = bpy.data.worlds.new("world")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.02, 0.022, 0.025, 1) if dark else (0.55, 0.57, 0.6, 1)
    bg.inputs["Strength"].default_value = 0.35 if dark else 0.30

    def area(name, loc, energy, size, rot_target=(0, 0, 0), color=(1, 1, 1)):
        bpy.ops.object.light_add(type="AREA", location=loc)
        l = bpy.context.object
        l.name = name
        l.data.energy = energy
        l.data.size = size
        l.data.color = color
        d = Vector(rot_target) - Vector(loc)
        l.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        return l

    k = 0.45 if not dark else 0.6
    area("key", (2.2, 1.6, 2.4), 420 * k, 1.6)
    area("fill", (-1.4, 2.6, 1.0), 140 * k, 3.0, color=(0.85, 0.9, 1.0))
    area("rim", (-2.0, -1.6, 1.8), 380 * k, 1.2, color=(1.0, 0.92, 0.85))
    area("front", (2.5, -1.0, 0.4), 90 * k, 2.0)


def camera(sc, loc, target, lens=50, ortho=None, clip_start=0.01, clip_end=100):
    cam = bpy.data.cameras.new("cam")
    cam.lens = lens
    cam.clip_start = clip_start
    cam.clip_end = clip_end
    if ortho:
        cam.type = "ORTHO"
        cam.ortho_scale = ortho
    o = bpy.data.objects.new("cam", cam)
    sc.collection.objects.link(o)
    o.location = loc
    d = Vector(target) - Vector(loc)
    o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    sc.camera = o
    return o


def cut(objs, names_prefix, box_min, box_max):
    """Boolean-difference a box out of selected objects (cutaway)."""
    cx = [(a + b) / 2 for a, b in zip(box_min, box_max)]
    sz = [(b - a) for a, b in zip(box_min, box_max)]
    bpy.ops.mesh.primitive_cube_add(size=1, location=cx)
    c = bpy.context.object
    c.scale = sz
    c.name = "cutter"
    c.hide_render = True
    c.display_type = "WIRE"
    for o in objs.values():
        if o.name.startswith(names_prefix):
            m = o.modifiers.new("cut", "BOOLEAN")
            m.operation = "DIFFERENCE"
            m.object = c
            m.solver = "EXACT"
    return c


LABELS = {  # label -> candidate parts (first one with visible surface wins)
    "tyre": ["lugs_R", "lugs_L", "tread_base_R"],
    "pod": ["pod_shield_L", "pod_shield_R"],
    "lidar": ["lidar_window_L", "lidar_window_R"],
    "cmg": ["flywheel_L", "flywheel_R", "cmg_housing_L", "cmg_housing_R"],
    "gimbal": ["gimbal_actuator_L", "gimbal_actuator_R"],
    "ring": ["ring_gear"],
    "drive": ["drive_motor_B", "drive_motor_F"],
    "pinion": ["pinion_B", "pinion_F", "pulley_p_B"],
    "bob": ["bob_lead"],
    "battery": ["bob_battery"],
    "yoke": ["arc_plate_B", "hanger_L", "hanger_R"],
    "axle": ["axle"],
    "bay": ["bay_L", "bay_R"],
    "level": ["level_motor", "level_gear"],
    "rollers": ["rollers_R", "rollers_L"],
    "camera": ["lens0_L", "lens0_R"],
    "clock": ["clockspring"],
}


def export_labels(sc, objs, out_png):
    """Project, for every label, a point of its part that is actually visible from the camera."""
    import random
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    cam = sc.camera
    origin = cam.matrix_world.translation
    res = (sc.render.resolution_x, sc.render.resolution_y)
    pts = {}
    for k, names in LABELS.items():
        for name in names:
            if name not in objs:
                continue
            o = objs[name]
            ev = o.evaluated_get(dg)
            me = ev.to_mesh()
            verts = [ev.matrix_world @ v.co for v in me.vertices]
            ev.to_mesh_clear()
            random.seed(1)
            if len(verts) > 600:
                verts = random.sample(verts, 600)
            vis = []
            for w in verts:
                d = (w - origin)
                dist = d.length
                o0, dn = origin.copy(), d.normalized()
                for _ in range(4):   # skip the (render-hidden) boolean cutters
                    hit, loc, nrm, idx, hob, _ = sc.ray_cast(dg, o0, dn, distance=dist + 0.01)
                    if not (hit and hob.name.startswith("cutter")):
                        break
                    o0 = loc + dn * 1e-4
                if hit and hob.name == name and (loc - w).length < 0.004:
                    p = world_to_camera_view(sc, cam, w)
                    if 0.02 < p.x < 0.98 and 0.02 < p.y < 0.98:
                        vis.append(Vector((p.x * res[0], (1 - p.y) * res[1], p.z)))
            if len(vis) >= 3:
                c = sum(vis, Vector((0, 0, 0))) / len(vis)
                best = min(vis, key=lambda v: (v.xy - c.xy).length)
                pts[k] = [best.x, best.y, best.z]
                break
    json.dump(pts, open(out_png.replace(".png", "_labels.json"), "w"), indent=1)


def main():
    a = args()
    sc = reset()
    W, H = (int(v) for v in a.res.split("x"))
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.cycles.samples = a.samples
    objs = load()
    M = materials()
    rg = rig(objs)

    rg["yoke"].rotation_euler = (0, math.radians(a.pitch), 0)
    rg["bob"].rotation_euler = (math.radians(a.bob), 0, 0)
    rg["cmgL"].rotation_euler = (0, math.radians(a.gimbal), 0)
    rg["cmgR"].rotation_euler = (0, math.radians(-a.gimbal), 0)
    rg["tyre"].rotation_euler = (0, math.radians(a.spin), 0)
    rg["robot"].rotation_euler = (math.radians(a.roll), 0, math.radians(a.yaw))

    s = a.scene
    dark = s in ("xray",)
    ghost_groups, ghost_names = (), ()
    if s == "xray":
        ghost_names = ("drum", "tread_base", "lugs", "eq_flange", "pod_shield", "pod_plate", "led_ring")
    assign(objs, M, ghost_groups, ghost_names)
    stage(sc, dark=dark)

    if s == "hero":
        camera(sc, (1.62, 1.12, 0.42), (0, 0.02, -0.06), lens=55)
    elif s == "side":
        camera(sc, (0.0, 2.3, 0.10), (0, 0, -0.04), lens=60)
    elif s == "top":
        camera(sc, (0.0, 0.0, 2.6), (0, 0, 0), lens=60)
    elif s == "pose":
        camera(sc, (1.75, 1.05, 0.30), (0, 0.0, -0.05), lens=55)
    elif s == "cutaway":
        cut(objs, ("drum", "tread_base", "lugs", "eq_flange", "rim", "ring_gear", "equator_bolts"),
            (-0.02, -1, -0.10), (1, 1, 1))
        cut(objs, ("pod_shield_L", "pod_plate_L", "led_ring_L", "rollers_L", "lidar"), (-0.02, -1, -0.10), (1, 1, 1))
        camera(sc, (1.55, 1.05, 0.62), (0, 0.0, -0.04), lens=50)
    elif s == "xray":
        camera(sc, (1.50, 1.05, 0.50), (0, 0.0, -0.05), lens=52)
    elif s == "exploded":
        for o in objs.values():
            e = o["explode"]
            o.location = (o.location[0] + e[0] * a.explode, o.location[1] + e[1] * a.explode,
                          o.location[2] + e[2] * a.explode)
        bpy.data.objects["floor"].location.z = -0.66
        camera(sc, (2.45, 1.95, 0.85), (0, 0.0, -0.08), lens=48)
    elif s == "section_front":
        # true half-section at x = 0, orthographic view along -x
        cut(objs, ("",), (0.0, -1, -1), (1, 1, 1))
        camera(sc, (2.0, 0, 0), (0, 0, -0.005), ortho=0.655)
        bpy.data.objects["floor"].hide_render = True
    elif s == "section_side":
        # true section at y = +5 mm (just past the ring-gear mid-plane), view along -y
        cut(objs, ("",), (-1, 0.005, -1), (1, 1, 1))
        camera(sc, (0, 2.0, 0), (0, 0, -0.005), ortho=0.655)
        bpy.data.objects["floor"].hide_render = True
    else:
        raise SystemExit(f"unknown scene {s}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sc.render.filepath = a.out
    export_labels(sc, objs, a.out)
    if a.labels_only:
        return
    bpy.ops.render.render(write_still=True)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
