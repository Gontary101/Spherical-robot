"""
Blender (Cycles) animation of a recorded GYRA Mk2 mission, third-person chase camera.

    blender -b -P render/animate_mission.py -- --mission media/mission.npz --out <dir>/frame_ [--start 1 --end 0]

Every frame poses the real Mk2 CAD meshes from the logged MuJoCo state: spine pose, both
tyre-half angles, pendulum (yoke) angle and bob angle.
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
import render_gyra as RG  # noqa: E402

FPS = 24


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default=os.path.join(RG.ROOT, "media", "mission2.npz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--cad", default="out21")
    return ap.parse_args(argv)


def emission_mat(name, rgb, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Emission Color"].default_value = (*rgb, 1)
    b.inputs["Emission Strength"].default_value = strength
    return m


def concrete_floor():
    m = bpy.data.materials.new("floor_concrete")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    brick = nt.nodes.new("ShaderNodeTexBrick")       # 2 m slab joints
    brick.inputs["Scale"].default_value = 0.5
    brick.inputs["Mortar Size"].default_value = 0.004
    brick.inputs["Color1"].default_value = (0.26, 0.26, 0.255, 1)
    brick.inputs["Color2"].default_value = (0.23, 0.235, 0.23, 1)
    brick.inputs["Mortar"].default_value = (0.12, 0.12, 0.12, 1)
    brick.offset = 0.0
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 8.0
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.35
    nt.links.new(tc.outputs["Object"], brick.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(brick.outputs["Color"], mix.inputs["A"])
    nt.links.new(noise.outputs["Color"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.8
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.15
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def striped_barrel_mat():
    m = bpy.data.materials.new("barrel")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    wave = nt.nodes.new("ShaderNodeMath")
    wave.operation = "SINE"
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 28.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.95, 0.35, 0.04, 1)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (0.92, 0.92, 0.9, 1)
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], mul.inputs[0])
    nt.links.new(mul.outputs[0], wave.inputs[0])
    nt.links.new(wave.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.45
    return m


def mesh_object(name, verts, faces, mat, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
    me.validate(clean_customdata=False)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(mat)
    for poly in me.polygons:
        poly.use_smooth = smooth
    return ob


def ground_mat():
    m = bpy.data.materials.new("ground")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.6
    noise.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.17, 0.17, 0.16, 1)
    ramp.color_ramp.elements[1].color = (0.30, 0.29, 0.27, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.8
    return m


def build_world(sc, mission):
    """Every obstacle, wall, pole, pedestrian and the terrain is built from the geometry read back from the
    compiled MuJoCo model (no primitives, no bevels): what is rendered is what the robot collides with."""
    sc.world = bpy.data.worlds.new("sky")
    sc.world.use_nodes = True
    nt = sc.world.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(38)
    sky.sun_rotation = math.radians(140)
    sky.sun_intensity = 0.2
    nt.links.new(sky.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = 0.18
    bpy.ops.object.light_add(type="SUN", rotation=(math.radians(52), 0, math.radians(140)))
    sun = bpy.context.object
    sun.data.energy = 2.2
    sun.data.angle = math.radians(3)

    gm = ground_mat()
    h = mission["terrain"]
    if h.size:
        import terrain as TR
        TR.N, TR.HALF = h.shape[0], float(mission["terrain_half"])
        V, F = TR.mesh(h.astype(float))
        mesh_object("terrain", V, F, gm, smooth=False)
        z_out = float(h.min()) - 0.02
    else:
        z_out = 0.0                     # flat world: the ground plane below IS the MuJoCo floor plane (z = 0)
    bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, z_out))       # ground beyond the patch
    bpy.context.object.data.materials.append(gm)
    bpy.context.object.name = "outer_ground"

    wall_m = concrete_floor()
    crate_m = RG.mat("crate", (0.16, 0.18, 0.21), rough=0.55)
    barrel_m = striped_barrel_mat()
    pole_m = RG.mat("pole", (0.75, 0.6, 0.12), rough=0.4)
    for o in json.loads(str(mission["static"])):
        if o["wall"]:
            mat = wall_m
        elif o["kind"] == "cylinder":
            r = max(abs(v[0] - o["verts"][0][0]) for v in o["verts"]) / 2
            mat = pole_m if r < 0.16 else barrel_m
        else:
            mat = crate_m
        mesh_object(o["name"], o["verts"], o["faces"], mat)
    peds = []
    dyn = json.loads(str(mission["dyn"]))
    if dyn:
        ped_m = RG.mat("pedestrian", (0.12, 0.35, 0.75), rough=0.5)
        for i in range(int(mission["n_dyn"])):
            ob = mesh_object(f"ped{i}", dyn["verts"], dyn["faces"], ped_m)
            peds.append((ob, np.array(dyn["offset"])))
    return peds


def verify_geometry(sc, mission, report):
    """Re-cast the MuJoCo probe rays against the Blender scene: rendered geometry must equal collision geometry."""
    moving = [o for o in sc.objects if o.name.startswith("ped")]          # probes cover static geometry only
    for o in moving:
        o.hide_viewport = True
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    P = mission["probes"]
    errs, info = [], []
    for p in P:
        o, v, dist = Vector(p[0:3]), Vector(p[3:6]), p[6]
        hit, loc, nrm, idx, ob, mtx = sc.ray_cast(dg, o, v, distance=20.0)
        errs.append(abs((loc - o).length - dist) if hit else float("inf"))
        info.append(dict(o=list(p[0:3]), v=list(p[3:6]), mujoco=float(dist), blender=(loc - o).length if hit else None,
                         obj=ob.name if hit else None))
    errs = np.array(errs)
    worst = [info[i] | {"err": float(errs[i])} for i in np.argsort(-errs)[:12]]
    res = dict(n_rays=int(len(errs)), max_err_m=float(errs.max()), p99_err_m=float(np.percentile(errs, 99)),
               n_over_1mm=int((errs > 1e-3).sum()), worst=worst)
    for o in moving:
        o.hide_viewport = False
    json.dump(res, open(report, "w"), indent=1)
    print("GEOMETRY CHECK (MuJoCo rays vs Blender scene):", {k: v for k, v in res.items() if k != "worst"}, flush=True)
    return res


def waypoint_markers(mission):
    rings = []
    h = mission["terrain"]
    for i, w in enumerate(mission["wps"]):
        z = 0.0
        if h.size:
            import terrain as TR
            z = float(TR.height_at(h.astype(float), w[0], w[1]))
        bpy.ops.mesh.primitive_torus_add(major_radius=0.55, minor_radius=0.025, location=(w[0], w[1], z + 0.03))
        r = bpy.context.object
        r.data.materials.append(emission_mat(f"wp{i}", (0.1, 0.8, 1.0), 6.0))
        bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=1.6, location=(w[0], w[1], z + 0.8))
        pole = bpy.context.object
        pole.data.materials.append(emission_mat(f"wpp{i}", (0.1, 0.8, 1.0), 3.0))
        rings.append((r, pole))
    return rings


def rig2(objs):
    robot = RG.empty("robot")
    tl = RG.empty("tyreL_spin", parent=robot)
    tr = RG.empty("tyreR_spin", parent=robot)
    spine = RG.empty("spine", parent=robot)
    yoke = RG.empty("yoke", parent=robot)
    bob = RG.empty("bob", parent=yoke)
    par = {"tyreL": tl, "tyreR": tr, "spine": spine, "yoke": yoke, "bob": bob}
    bpy.context.view_layer.update()
    for o in objs.values():
        p = par[o["group"]]
        o.parent = p
        o.matrix_parent_inverse = p.matrix_world.inverted()
    for e in (robot,):
        e.rotation_mode = "QUATERNION"
    return dict(robot=robot, tl=tl, tr=tr, yoke=yoke, bob=bob)


def main():
    a = args()
    RG.CAD_OUT = os.path.join(RG.ROOT, "cad", a.cad)                 # Mk2.1 CAD: the hardware the policies run on
    sc = RG.reset()
    W, H = (int(v) for v in a.res.split("x"))
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.cycles.samples = a.samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces = 4
    sc.cycles.diffuse_bounces = 2
    sc.cycles.glossy_bounces = 2
    sc.cycles.transmission_bounces = 2
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.use_persistent_data = True
    sc.view_settings.look = "AgX - Base Contrast"
    sc.render.fps = FPS

    mission = np.load(a.mission)
    F = mission["frames"]            # t, qpos[0:7], qL, qR, qyoke, qbob, v, w, roll, vcmd, wcmd
    reached = list(mission["reached"])
    peds = build_world(sc, mission)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    verify_geometry(sc, mission, os.path.join(os.path.dirname(a.out), "geometry_check.json"))
    rings = waypoint_markers(mission)
    objs = RG.load()
    RG.assign(objs, RG.materials())
    rg = rig2(objs)
    n_dyn = int(mission["n_dyn"])

    cam_data = bpy.data.cameras.new("chase")
    cam_data.lens = 30
    cam_data.clip_start = 0.05
    cam = bpy.data.objects.new("chase", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam

    T = F[-1, 0]
    n_frames = int(T * FPS)
    sc.frame_start = a.start
    sc.frame_end = a.end if a.end else n_frames
    sc.frame_step = a.step

    # smoothed camera heading (low-pass on the robot heading, unwrapped)
    yaw = np.array([2 * math.atan2(f[7], f[4]) for f in F])      # qpos quat (w x y z) at 4..7 -> yaw for a level body
    yaw = np.unwrap(yaw)
    alpha = 1 - math.exp(-0.02 / 0.9)
    sm = np.zeros_like(yaw)
    sm[0] = yaw[0]
    for i in range(1, len(yaw)):
        sm[i] = sm[i - 1] + alpha * (yaw[i] - sm[i - 1])
    pos_s = F[:, 1:3].copy()
    beta = 1 - math.exp(-0.02 / 0.35)
    for i in range(1, len(pos_s)):
        pos_s[i] = pos_s[i - 1] + beta * (F[i, 1:3] - pos_s[i - 1])

    ped_z = [np.full(len(F), z) for z in mission["dyn_z"]]          # mocap body heights, as simulated

    def state(frame):
        t = (frame - 1) / FPS
        i = min(int(round(t / 0.02)), len(F) - 1)
        return i, F[i]

    for frame in range(sc.frame_start, sc.frame_end + 1):
        i, f = state(frame)
        r = rg["robot"]
        r.location = Vector(f[1:4])
        r.rotation_quaternion = Quaternion(f[4:8])
        rg["tl"].rotation_euler = (0, f[8], 0)
        rg["tr"].rotation_euler = (0, f[9], 0)
        rg["yoke"].rotation_euler = (0, f[10], 0)
        rg["bob"].rotation_euler = (f[11], 0, 0)
        for key, ob in (("location", r), ("rotation_quaternion", r)):
            ob.keyframe_insert(key, frame=frame)
        for ob in (rg["tl"], rg["tr"], rg["yoke"], rg["bob"]):
            ob.keyframe_insert("rotation_euler", frame=frame)
        h = sm[i]
        for j, (ob, off) in enumerate(peds):
            ob.location = Vector(f[17 + 2 * j:19 + 2 * j].tolist() + [0.0]) + Vector(off) + Vector((0, 0, ped_z[j][i]))
            ob.keyframe_insert("location", frame=frame)
        target = Vector((pos_s[i, 0], pos_s[i, 1], f[3] + 0.02))
        back = Vector((math.cos(h), math.sin(h), 0))
        cam.location = target - back * 3.1 + Vector((0, 0, 1.25))
        d = (target + back * 0.9) - cam.location
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=frame)
        cam.keyframe_insert("rotation_euler", frame=frame)
        t = (frame - 1) / FPS
        for k, (ring, pole) in enumerate(rings):
            done = k < len(reached) and t >= reached[k]
            active = (k == 0 or t >= reached[k - 1]) and not done
            for ob in (ring, pole):
                ob.hide_render = done
                ob.keyframe_insert("hide_render", frame=frame)
            pole.scale = (1, 1, 1.0 if active else 0.35)
            pole.keyframe_insert("scale", frame=frame)
    # constant interpolation for discrete toggles
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sc.render.filepath = a.out
    sc.render.image_settings.file_format = "PNG"
    print(f"RENDER frames {sc.frame_start}-{sc.frame_end} of {n_frames}", flush=True)
    bpy.ops.render.render(animation=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
