"""
What the build's sensors actually see (Blender, Cycles), from the build CAD (cad/build):

    blender -b -P render/camera_views.py -- --out <dir> [--tag current] [--cams cams.json]

Renders the six pod cameras (Waveshare OV9281, 120 deg diagonal lens, behind their PC windows, poses from
cad/build/pod.py) looking at a test course in front of the robot, and ray-casts the floor coverage of every camera
and of both Livox Mid-360s (360 deg around the axle, -7..+52 deg) against the real tyre: <dir>/<tag>_cam_<name>.png
and <dir>/<tag>_coverage.npz (plotted by render/plot_sensor_coverage.py).

--cams overrides the camera poses (json list of {name, pos_mm, dir}) to evaluate a sensor change.
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_film as BF  # noqa: E402

D_MM = 90.0                      # pod shell sphere centre offset (cad/build spec D)
RHO_LENS = 291.0                 # lens front on the camera axis, radius from the shell centre (pod.py)
CAM_HFOV = 111.5                 # 120 deg diagonal on the 1280 x 800 sensor
CAM_VFOV = 85.0
LIDAR_Y = 0.395                  # Mid-360 optical centre, |y| (m), on the axle line
LIDAR_EL = (-7.0, 52.0)          # Mid-360 vertical field, about its own axis (the axle, pointing outboard)
FLOOR_Z = -0.30


def build_cams():
    """The as-built cameras (cad/build/pod.py CAMS): name, position (m), unit view direction."""
    out = []
    for s, sg in (("L", 1), ("R", -1)):
        for i, (dp, nm) in enumerate([((0.351, 0.925, 0.145), "front"), ((-0.351, 0.925, 0.145), "rear"),
                                      ((0.145, 0.925, -0.351), "ground")]):
            d = Vector((dp[0], sg * dp[1], dp[2])).normalized()
            c0 = Vector((0, sg * D_MM, 0))
            p = (c0 + d * RHO_LENS) * 0.001
            out.append(dict(name=f"{s}{i} {nm}", pos=tuple(p), dir=tuple(d)))
    return out


def course(sc):
    """Test objects: low boxes and a pole on the centreline ahead, a person, boxes to the sides; 0.5 m floor grid."""
    objs = []

    def box(name, x, y, sx, sy, sz, col):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, FLOOR_Z + sz / 2))
        o = bpy.context.object
        o.name = name
        o.scale = (sx, sy, sz)
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*col, 1)
        o.data.materials.append(m)
        o.color = (0, 0, 0, 1)
        objs.append(o)
    orange, red, blue, green = (0.9, 0.35, 0.08), (0.8, 0.1, 0.08), (0.12, 0.3, 0.8), (0.1, 0.6, 0.3)
    box("low box 0.6 m", 0.60, 0.0, 0.15, 0.25, 0.15, orange)
    box("low box 1.2 m", 1.20, 0.0, 0.15, 0.25, 0.15, orange)
    box("low box 2.0 m", 2.00, 0.0, 0.15, 0.25, 0.15, orange)
    box("pole 1.5 m", 1.50, 0.0, 0.08, 0.08, 1.0, red)
    box("person 3 m", 3.00, 0.15, 0.35, 0.45, 1.7, blue)
    box("box left", 1.2, 1.2, 0.3, 0.3, 0.4, green)
    box("box right", 1.2, -1.2, 0.3, 0.3, 0.4, green)
    # floor grid: a checker on the studio floor
    fl = bpy.data.objects["floor"]
    nt = fl.active_material.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ck = nt.nodes.new("ShaderNodeTexChecker")
    ck.inputs["Color1"].default_value = (0.05, 0.065, 0.1, 1)
    ck.inputs["Color2"].default_value = (0.11, 0.13, 0.18, 1)
    nt.links.new(tc.outputs["Object"], ck.inputs["Vector"])
    ck.inputs["Scale"].default_value = 2.0                   # object coords in m: 0.5 m squares
    nt.links.new(ck.outputs["Color"], b.inputs["Base Color"])
    return objs


def coverage(cams, hide):
    """Floor coverage: for every floor cell, which sensors see it (direct line of sight, inside the field)."""
    dg = bpy.context.evaluated_depsgraph_get()
    sc = bpy.context.scene
    xs = np.arange(-3.0, 4.0001, 0.04)
    ys = np.arange(-3.0, 3.0001, 0.04)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, FLOOR_Z + 0.002)], -1)
    for n in hide:
        bpy.data.objects[n].hide_set(True)
        bpy.data.objects[n].hide_viewport = True
    dg.update()
    res = {}

    def visible(origin, inside):
        vis = np.zeros(len(pts), bool)
        o = Vector(origin)
        for k in np.nonzero(inside)[0]:
            p = Vector(pts[k])
            d = p - o
            dist = d.length
            hit, loc, *_ = sc.ray_cast(dg, o, d.normalized(), distance=dist + 0.01)
            vis[k] = (not hit) or (loc - p).length < 0.02 or loc.z < FLOOR_Z + 0.004
        return vis
    for c in cams:
        o, dv = np.array(c["pos"]), np.array(c["dir"])
        rel = pts - o
        rn = rel / np.linalg.norm(rel, axis=1, keepdims=True)
        # inside the rectangular field: angles in the camera frame (up = world z projected)
        right = np.cross(dv, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, dv)
        a = np.degrees(np.arctan2(rn @ right, rn @ dv))
        e = np.degrees(np.arctan2(rn @ up, rn @ dv))
        inside = (rn @ dv > 0) & (np.abs(a) < CAM_HFOV / 2) & (np.abs(e) < CAM_VFOV / 2)
        res["cam " + c["name"]] = visible(o + 0.012 * dv, inside)
    for s, sg in (("L", 1), ("R", -1)):
        o = np.array([0.004, sg * LIDAR_Y, 0.0])
        rel = pts - o
        rn = rel / np.linalg.norm(rel, axis=1, keepdims=True)
        el = np.degrees(np.arcsin(np.clip(rn[:, 1] * sg, -1, 1)))         # elevation about the outboard axle axis
        inside = (el > LIDAR_EL[0]) & (el < LIDAR_EL[1]) & (np.linalg.norm(rel[:, :2], axis=1) < 40.0)
        res["lidar " + s] = visible(o + Vector((0, sg * 0.03, 0)), inside)
    for n in hide:
        bpy.data.objects[n].hide_set(False)
        bpy.data.objects[n].hide_viewport = False
    return xs, ys, res


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="current")
    ap.add_argument("--cams", default="")
    ap.add_argument("--res", default="800x500")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    BF.setup_render(sc, a.res, a.samples)
    floor, lights = BF.studio(sc)
    R = BF.Robot()
    for ob in R.o.values():
        ob.color = (0, 0, 0, 1)                      # wrapper shader: no highlight, no ghost, no wipe, opaque
    bpy.context.view_layer.update()
    test_objs = [o.name for o in course(sc)]
    cams = json.load(open(a.cams)) if a.cams else build_cams()
    json.dump(cams, open(os.path.join(a.out, f"{a.tag}_cams.json"), "w"), indent=1)
    # the camera looks through its window: hide the pod shields, windows, camera bodies and holders
    see_through = [n for n in R.o if n.startswith(("pod_shield_", "cam_window_", "camera_", "cam_holder_"))]
    camd = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", camd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    camd.sensor_fit = "HORIZONTAL"
    camd.angle = math.radians(CAM_HFOV)
    camd.clip_start = 0.005
    if not a.no_render:
        for n in see_through:
            R.o[n].hide_render = True
        for c in cams:
            cam.location = Vector(c["pos"])
            d = Vector(c["dir"])
            cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
            sc.render.filepath = os.path.join(a.out, f"{a.tag}_cam_{c['name'].split()[0]}.png")
            bpy.ops.render.render(write_still=True)
        for n in see_through:
            R.o[n].hide_render = False
    xs, ys, res = coverage(cams, see_through + test_objs)
    np.savez_compressed(os.path.join(a.out, f"{a.tag}_coverage.npz"), xs=xs, ys=ys, names=np.array(list(res)),
                        vis=np.stack([res[k] for k in res]))
    print("coverage written", {k: int(v.sum()) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
