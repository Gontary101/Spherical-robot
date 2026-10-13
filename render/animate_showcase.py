"""
Blender (Cycles) render of the showcase mission (rl/showcase.py) for the video.

    blender -b -P render/animate_showcase.py -- --mission media/showcase.npz --out <dir> [--step 2 --start 0 --end -1]

The world is built from the geometry read back from the compiled MuJoCo model (as render/animate_mission.py), with
materials chosen by object type. The camera follows a zone-aware schedule with smooth 1.2 s blends between modes:
an establishing view when a zone begins, a low chase over the hills, a higher chase through the doorways, a low
side-tracking shot on the fast straights. Frames are numbered at 24 fps (frame k shows mission time k / 24);
--step 2 renders every other frame (interpolated to 24 fps in compositing). Resumable: existing frames are skipped.
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import animate_mission as AM  # noqa: E402
import render_gyra as RG  # noqa: E402

FPS = 24


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default=os.path.join(RG.ROOT, "media", "showcase.npz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--hq", action="store_true", help="final-quality bounces and noise threshold")
    ap.add_argument("--only", type=int, nargs="*", default=None, help="render just these frames (previews)")
    return ap.parse_args(argv)


def noise_mat(name, c0, c1, scale=4.0, rough=0.85, bump=0.4):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = 10.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*c0, 1)
    ramp.color_ramp.elements[1].color = (*c1, 1)
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
    nt.links.new(n.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = bump
    nt.links.new(n.outputs["Fac"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


def terrain_mat():
    """Dry grass over soil, slope-independent; world-space noise at two scales."""
    m = bpy.data.materials.new("field")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 0.35
    n1.inputs["Detail"].default_value = 6.0
    n2 = nt.nodes.new("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = 9.0
    n2.inputs["Detail"].default_value = 12.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.045, 0.065, 0.025, 1)
    ramp.color_ramp.elements[1].color = (0.19, 0.18, 0.09, 1)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.6
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
    nt.links.new(n1.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], mix.inputs["A"])
    nt.links.new(n2.outputs["Color"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.92
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.25
    nt.links.new(n2.outputs["Fac"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


def build_world(sc, mission):
    sc.world = bpy.data.worlds.new("sky")
    sc.world.use_nodes = True
    nt = sc.world.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(32)
    sky.sun_rotation = math.radians(125)
    sky.sun_intensity = 0.25
    nt.links.new(sky.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = 0.22
    bpy.ops.object.light_add(type="SUN", rotation=(math.radians(58), 0, math.radians(125)))
    sun = bpy.context.object
    sun.data.energy, sun.data.angle, sun.data.color = 2.6, math.radians(2.5), (1.0, 0.95, 0.88)
    import terrain as TR
    h = mission["terrain"].astype(float)
    TR.N, TR.HALF = h.shape[0], float(mission["terrain_half"])
    V, F = TR.mesh(h)
    AM.mesh_object("terrain", V, F, terrain_mat(), smooth=True)
    bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, float(h.min()) - 0.03))
    bpy.context.object.data.materials.append(terrain_mat())
    # indoor floor slab under the building (the terrain is flat there)
    zb = float(TR.height_at(h, np.array([9.0]), np.array([16.0]))[0])    # terrain is flat (offset) under the building
    bpy.ops.mesh.primitive_plane_add(size=1, location=(9.0, 16.0, zb + 0.004))
    slab = bpy.context.object
    slab.scale = (22.0, 10.0, 1)
    slab.data.materials.append(AM.concrete_floor())
    mats = dict(rock=noise_mat("rock", (0.16, 0.15, 0.14), (0.42, 0.40, 0.37), 6.0, 0.9, 0.6),
                block=noise_mat("block", (0.42, 0.42, 0.41), (0.62, 0.62, 0.6), 3.0, 0.8, 0.2),
                plaster=noise_mat("plaster", (0.78, 0.77, 0.74), (0.88, 0.87, 0.84), 2.0, 0.85, 0.05),
                perimeter=AM.concrete_floor(), pole=RG.mat("pole", (0.85, 0.62, 0.1), rough=0.4),
                cone=RG.mat("cone", (0.95, 0.32, 0.05), rough=0.45), crate=noise_mat("crate", (0.25, 0.16, 0.09), (0.45, 0.31, 0.18), 1.5, 0.7, 0.1))
    for o in json.loads(str(mission["static"])):
        n = o["name"]
        key = ("rock" if n.startswith(("rock", "bould")) else "block" if n.startswith("block") else
               "plaster" if n.startswith("bw") else "perimeter" if n.startswith("wall") else "pole" if n.startswith("pole") else
               "cone" if n.startswith("cone") else "crate")
        AM.mesh_object(n, o["verts"], o["faces"], mats[key])


# ------------------------------------------------------------------------------------------------ camera schedule
def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def camera_pose(mode, target, h, t_in_mode, side=1.0):
    back = Vector((math.cos(h), math.sin(h), 0))
    left = Vector((-math.sin(h), math.cos(h), 0))
    if mode == "establish":                                       # high, wide, drifting
        loc = target - back * 6.5 + left * 3.0 * side + Vector((0, 0, 5.5 - 0.4 * t_in_mode))
        look = target + back * 2.5
    elif mode == "low":                                           # hills: low chase shows the ground rolling
        loc = target - back * 3.0 + left * 0.8 + Vector((0, 0, 1.0))
        look = target + back * 2.0 + Vector((0, 0, 0.1))
    elif mode == "high":                                          # doorways: see the doors from above
        loc = target - back * 2.0 + Vector((0, 0, 3.4))
        look = target + back * 1.4
    elif mode == "side":                                          # speed: low side tracking
        loc = target + left * 3.2 * side - back * 0.6 + Vector((0, 0, 0.55))
        look = target + back * 0.9 + Vector((0, 0, 0.15))
    else:                                                         # default drone chase
        loc = target - back * 2.7 + Vector((0, 0, 2.3))
        look = target + back * 1.3
    return loc, look


MODE_BY_ZONE = {"Rolling hills": "low", "Pole forest": "chase", "Doorways": "high", "Open field": "side",
                "Boulder garden": "chase", "Back over the hills": "low", "Final sprint": "side"}


def main():
    a = args()
    RG.CAD_OUT = os.path.join(RG.ROOT, "cad", "out21")
    sc = RG.reset()
    W, H = (int(v) for v in a.res.split("x"))
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.cycles.samples = a.samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces, sc.cycles.diffuse_bounces, sc.cycles.glossy_bounces, sc.cycles.transmission_bounces = 4, 2, 2, 2
    sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.render.use_persistent_data = True
    sc.view_settings.look = "AgX - Base Contrast"
    if a.hq:
        RG.high_quality(sc)
    mission = np.load(a.mission, allow_pickle=True)
    F = mission["frames"]
    hud = json.loads(str(mission["hud"]))
    build_world(sc, mission)
    os.makedirs(a.out, exist_ok=True)
    AM.verify_geometry(sc, mission, os.path.join(a.out, "geometry_check.json"))
    objs = RG.load()
    RG.assign(objs, RG.materials())
    rg = AM.rig2(objs)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens, cam_data.clip_start = 28, 0.05
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    # smoothed robot heading and position for the camera
    yaw = np.unwrap(np.array([2 * math.atan2(f[7], f[4]) for f in F]))
    sm = yaw.copy()
    al = 1 - math.exp(-0.02 / 1.0)
    for i in range(1, len(sm)):
        sm[i] = sm[i - 1] + al * (yaw[i] - sm[i - 1])
    ps = F[:, 1:4].copy()
    be = 1 - math.exp(-0.02 / 0.4)
    for i in range(1, len(ps)):
        ps[i] = ps[i - 1] + be * (F[i, 1:4] - ps[i - 1])
    # zone timeline (from the HUD log at 10 Hz): mode per time with an establishing shot when a zone starts
    ht = np.array([x["t"] for x in hud])
    zones = [x["zone"] for x in hud]
    starts = [0.0] + [ht[i] for i in range(1, len(zones)) if zones[i] != zones[i - 1]]

    def mode_at(t):
        k = max(0, int(np.searchsorted(ht, t)) - 1)
        z = zones[min(k, len(zones) - 1)]
        z0 = max(s for s in starts if s <= t + 1e-9)
        if t - z0 < 3.2:
            return "establish", t - z0
        return MODE_BY_ZONE.get(z, "chase"), t - z0

    T = F[-1, 0] - F[0, 0]
    n = int(T * FPS)
    end = n if a.end < 0 else min(a.end, n)
    frames = a.only if a.only else range(a.start, end, a.step)
    prev_pose = None
    for fr in frames:
        path = os.path.join(a.out, f"mission_{fr:05d}.png")
        if os.path.exists(path):
            continue
        t = fr / FPS
        i = min(int(round(t / 0.02)), len(F) - 1)
        f = F[i]
        r = rg["robot"]
        r.location = Vector(f[1:4])
        r.rotation_quaternion = Quaternion(f[4:8])
        rg["tl"].rotation_euler = (0, f[8], 0)
        rg["tr"].rotation_euler = (0, f[9], 0)
        rg["yoke"].rotation_euler = (0, f[10], 0)
        rg["bob"].rotation_euler = (f[11], 0, 0)
        target = Vector(ps[i]) + Vector((0, 0, 0.02))
        mode, tm = mode_at(t)
        side = 1.0 if int(t // 25) % 2 == 0 else -1.0
        loc, look = camera_pose(mode, target, sm[i], tm, side)
        # blend from the previous mode over 1.2 s after a mode change
        pm, ptm = mode_at(max(t - tm - 1e-3, 0.0))
        if tm < 1.2 and t > 1.3 and pm != mode:
            l0, k0 = camera_pose(pm, target, sm[i], ptm + tm, side)
            w = ease(tm / 1.2)
            loc, look = l0.lerp(loc, w), k0.lerp(look, w)
        cam.location = loc
        cam.rotation_euler = (look - loc).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print("FRAME", fr, mode, flush=True)


if __name__ == "__main__":
    main()
