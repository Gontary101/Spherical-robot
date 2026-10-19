"""
The GYRA Mk2.1 build film: Blender (Cycles) frames of the build CAD (cad/build), every part and fastener.

    python render/film_data.py                                    # meshes + data from the CAD (once)
    blender -b -P render/build_film.py -- --out <dir> [--res 1920x1080 --samples 8 --worker 0/2 --preview]
    python render/compose_build_film.py --frames <dir> --out media/gyra_mk21_build_film.mp4

Story (24 fps): title, reveal of the five bodies, exploded view, the full assembly sequence in the verified order
(every fastener goes in along its own joint axis), how it works (drive train, pendulum + level drive, pitch stops,
differential steering, lean, pods + sensors, power + electronics), verification, closing.

Writes <dir>/f_<frame>.png for the frames it renders (segments with step 2 render every other frame; the composer
interpolates) and <dir>/f_<frame>.json for every frame: screen positions of the label anchors, captions, overlays.
Resumable: frames already on disk are skipped. --worker k/n renders every n-th frame starting at k.
"""
import argparse
import json
import math
import os
import re
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FILM = os.path.join(ROOT, "cad", "build", "out", "film")
sys.path.insert(0, os.path.join(ROOT, "cad"))
from render_colors import RGB  # noqa: E402

FPS = 24
MM = 0.001
FLOOR_SPEC = 0.35                                    # floor sheen (Principled specular level)
R_TYRE = 0.300

# accent colours (also used by the composer, by name)
ACC = dict(amber=(1.0, 0.71, 0.28), cyan=(0.27, 0.78, 1.0), green=(0.29, 0.84, 0.61), violet=(0.66, 0.55, 1.0),
           coral=(1.0, 0.42, 0.37))


# ====================================================================================================== helpers
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_io(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ramp(t, t0, t1):
    return ease((t - t0) / max(t1 - t0, 1e-6))


def win(t, t0, t1, f=0.5):
    """1 inside [t0, t1] with eased fades of f seconds at both ends."""
    if t <= t0 or t >= t1:
        return 0.0
    return min(ease((t - t0) / f), ease((t1 - t) / f))


def lerp(a, b, x):
    return a + (b - a) * x


def vlerp(a, b, x):
    return tuple(lerp(p, q, x) for p, q in zip(a, b))


def orbit(target, az, el, dist):
    a, e = math.radians(az), math.radians(el)
    return (target[0] + dist * math.cos(e) * math.cos(a), target[1] + dist * math.cos(e) * math.sin(a),
            target[2] + dist * math.sin(e))


def aim(target, az, shift):
    """Target moved sideways in the camera's frame: shift > 0 puts the subject left of centre."""
    a = math.radians(az)
    right = (-math.sin(a), math.cos(a), 0.0)
    return tuple(c + shift * r for c, r in zip(target, right))


def cam_keys(t, keys):
    """keys: [(t, target, az, el, dist, lens)]: eased interpolation of the orbit parameters between keys."""
    if t <= keys[0][0]:
        k = keys[0]
        return orbit(k[1], *k[2:5]), k[1], k[5]
    for a, b in zip(keys, keys[1:]):
        if t <= b[0]:
            x = ease_io((t - a[0]) / (b[0] - a[0]))
            tg = vlerp(a[1], b[1], x)
            az, el, d, lens = (lerp(a[i], b[i], x) for i in range(2, 6))
            return orbit(tg, az, el, d), tg, lens
    k = keys[-1]
    return orbit(k[1], *k[2:5]), k[1], k[5]


# ====================================================================================================== scene
def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--worker", default="0/1")
    ap.add_argument("--preview", action="store_true", help="every 4th frame, low resolution checks")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--only", default="", help="comma list of segment names")
    ap.add_argument("--frames", default="", help="explicit frame list a,b,c")
    return ap.parse_args(argv)


def setup_render(sc, res, samples):
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    W, H = (int(v) for v in res.split("x"))
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.resolution_percentage = 100
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.max_bounces, sc.cycles.diffuse_bounces, sc.cycles.glossy_bounces = 6, 2, 3
    sc.cycles.transmission_bounces, sc.cycles.transparent_max_bounces = 4, 16
    sc.render.use_persistent_data = True
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"


MAT_SPEC = {  # base colour from render_colors, metallic, roughness, coat
    "alu": (1.0, 0.27, 0.0), "alu_dark": (0.85, 0.36, 0.0), "steel": (1.0, 0.22, 0.0), "brass": (1.0, 0.25, 0.0),
    "rubber": (0.0, 0.78, 0.0), "gfrp": (0.0, 0.42, 0.2), "carbon": (0.0, 0.25, 0.8), "lead": (0.85, 0.36, 0.0),
    "battery": (0.0, 0.35, 0.3), "nylon": (0.0, 0.55, 0.0), "nylon_light": (0.0, 0.5, 0.0), "lidar_white": (0.0, 0.32, 0.3),
    "glass": (0.0, 0.04, 1.0), "motor_black": (0.4, 0.38, 0.2), "pcb": (0.0, 0.42, 0.4),
}


def make_material(key):
    """Principled BSDF of the part's material, wrapped so that the object's colour property drives:
    R = highlight (amber emission), G = ghost (x-ray shell), B = wipe (part revealed along its y, 0 = off),
    A = opacity (fade in / out)."""
    m = bpy.data.materials.new(f"m_{key}")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    r, g, b = RGB.get(key, (0.6, 0.6, 0.6))
    met, rough, coat = MAT_SPEC.get(key, (0.0, 0.45, 0.0))
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1)
    bsdf.inputs["Metallic"].default_value = met
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Coat Weight"].default_value = coat
    if key == "glass":
        bsdf.inputs["Transmission Weight"].default_value = 0.85
    if key == "gfrp":                                         # woven glass under resin: warm ivory, faint weave
        bsdf.inputs["Base Color"].default_value = (0.52, 0.50, 0.45, 1)
        bsdf.inputs["Coat Weight"].default_value = 0.35
        wv = nt.nodes.new("ShaderNodeTexWave")
        wv.inputs["Scale"].default_value = 160.0
        wv.inputs["Distortion"].default_value = 1.5
        bump0 = nt.nodes.new("ShaderNodeBump")
        bump0.inputs["Strength"].default_value = 0.05
        nt.links.new(wv.outputs["Fac"], bump0.inputs["Height"])
        nt.links.new(bump0.outputs["Normal"], bsdf.inputs["Normal"])
    if key == "rubber":                                       # fine grain on the tread
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 900.0
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.1
        nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    info = nt.nodes.new("ShaderNodeObjectInfo")
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(info.outputs["Color"], sep.inputs["Color"])
    # highlight: add an amber emission scaled by R
    emi = nt.nodes.new("ShaderNodeEmission")
    emi.inputs["Color"].default_value = (*ACC["amber"], 1)
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 0.45
    nt.links.new(sep.outputs["Red"], mul.inputs[0])
    nt.links.new(mul.outputs[0], emi.inputs["Strength"])
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(bsdf.outputs[0], add.inputs[0])
    nt.links.new(emi.outputs[0], add.inputs[1])
    # ghost: crisp x-ray look, more opaque at grazing angles
    ghost = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    gl = nt.nodes.new("ShaderNodeBsdfPrincipled")
    gl.inputs["Base Color"].default_value = (0.2, 0.32, 0.5, 1)
    gl.inputs["Roughness"].default_value = 0.3
    gl.inputs["Emission Color"].default_value = (0.25, 0.45, 0.85, 1)
    gl.inputs["Emission Strength"].default_value = 0.05
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 0.16
    nt.links.new(lw.outputs["Facing"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], ghost.inputs["Fac"])
    nt.links.new(tr.outputs[0], ghost.inputs[1])
    nt.links.new(gl.outputs[0], ghost.inputs[2])
    gmix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(sep.outputs["Green"], gmix.inputs["Fac"])
    nt.links.new(add.outputs[0], gmix.inputs[1])
    nt.links.new(ghost.outputs[0], gmix.inputs[2])
    # wipe: visible where generated y < 1.12 B - 0.12 (soft edge), only when B > 0
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sxyz = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sxyz.inputs[0])

    def op(kind, a, b, clamp_=False):
        nd = nt.nodes.new("ShaderNodeMath")
        nd.operation = kind
        nd.use_clamp = clamp_
        for i, v in enumerate((a, b)):
            if isinstance(v, (int, float)):
                nd.inputs[i].default_value = v
            else:
                nt.links.new(v, nd.inputs[i])
        return nd.outputs[0]
    w = op("SUBTRACT", op("MULTIPLY", sep.outputs["Blue"], 9.3333), op("MULTIPLY", sxyz.outputs["Y"], 8.3333), True)
    wipe = op("SUBTRACT", 1.0, op("MULTIPLY", op("GREATER_THAN", sep.outputs["Blue"], 0.0005), op("SUBTRACT", 1.0, w)))
    # opacity
    amix = nt.nodes.new("ShaderNodeMixShader")
    tr2 = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(op("MULTIPLY", info.outputs["Alpha"], wipe), amix.inputs["Fac"])
    nt.links.new(tr2.outputs[0], amix.inputs[1])
    nt.links.new(gmix.outputs[0], amix.inputs[2])
    nt.links.new(amix.outputs[0], out.inputs["Surface"])
    return m


def studio(sc):
    w = bpy.data.worlds.new("world")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.012, 0.018, 0.04, 1)
    bg.inputs["Strength"].default_value = 0.6
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -R_TYRE))
    fl = bpy.context.object
    fl.name = "floor"
    m = bpy.data.materials.new("floor")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.022, 0.03, 0.052, 1)
    b.inputs["Roughness"].default_value = 0.5
    b.inputs["Specular IOR Level"].default_value = FLOOR_SPEC
    # the floor dissolves into the world with distance from the camera: no hard horizon line
    nt = m.node_tree
    cd = nt.nodes.new("ShaderNodeCameraData")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 5.0, 14.0
    mx = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mo = nt.nodes["Material Output"]
    nt.links.new(cd.outputs["View Distance"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], mx.inputs["Fac"])
    lwf = nt.nodes.new("ShaderNodeLayerWeight")
    lwf.inputs["Blend"].default_value = 0.5
    mrf = nt.nodes.new("ShaderNodeMapRange")
    mrf.inputs["From Min"].default_value, mrf.inputs["From Max"].default_value = 0.55, 0.8
    df = nt.nodes.new("ShaderNodeBsdfDiffuse")
    df.inputs["Color"].default_value = (0.022, 0.03, 0.052, 1)
    mxd = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lwf.outputs["Facing"], mrf.inputs["Value"])
    nt.links.new(mrf.outputs["Result"], mxd.inputs["Fac"])
    nt.links.new(b.outputs[0], mxd.inputs[1])
    nt.links.new(df.outputs[0], mxd.inputs[2])
    nt.links.new(mxd.outputs[0], mx.inputs[1])
    nt.links.new(tr.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], mo.inputs["Surface"])
    fl.data.materials.append(m)

    def area(name, loc, energy, size, color=(1, 1, 1)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size, ld.color = energy, size, color
        ld.shape = "DISK"                                     # soft round reflections, no straight edges on the floor
        lo = bpy.data.objects.new(name, ld)
        sc.collection.objects.link(lo)
        lo.location = loc
        lo.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        lo.visible_camera = False                             # light, reflect, but never show up in shot
        return lo

    k = 0.6
    L = [area("key", (2.2, 1.6, 2.4), 420 * k, 1.6), area("fill", (-1.4, 2.6, 1.0), 140 * k, 3.0, (0.85, 0.9, 1.0)),
         area("rim", (-2.0, -1.6, 1.8), 380 * k, 1.2, (1.0, 0.92, 0.85)), area("front", (2.5, -1.0, 0.4), 90 * k, 2.0),
         area("amber", (-2.4, 0.6, 1.2), 70, 2.0, (1.0, 0.68, 0.3))]
    rig = bpy.data.objects.new("lights", None)               # lights follow the robot when it rolls
    sc.collection.objects.link(rig)
    for lo in L:
        lo.parent = rig
    # a one-sided area light leaves the floor behind its plane dark: a straight edge across the floor a few metres
    # out. Only the key (whose plane meets the floor beyond the fade) lights the floor.
    nf = bpy.data.collections.new("not_floor")
    nf.objects.link(fl)
    for lo in L[1:]:
        lo.light_linking.receiver_collection = nf
    for co in nf.collection_objects:
        co.light_linking.link_state = "EXCLUDE"
    # a soft pool on the floor from straight above (horizontal plane: no edge), lighting the floor only
    pool = area("pool", (0.0, 0.0, 3.2), 520, 4.0)
    pool.rotation_euler = (0, 0, 0)
    pool.parent = rig
    of = bpy.data.collections.new("only_floor")
    of.objects.link(fl)
    pool.light_linking.receiver_collection = of
    return fl, rig


def empty(name, parent=None, loc=(0, 0, 0)):
    o = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    if parent is not None:
        o.parent = parent
    return o


class Robot:
    def __init__(self):
        d = json.load(open(os.path.join(FILM, "data.json")))
        self.P, self.geo = d["parts"], d["geo"]
        self.J = {j["name"]: j for j in json.load(open(os.path.join(ROOT, "cad", "build", "out", "joints.json")))}
        sc = bpy.context.scene
        self.root = empty("ROOT")
        self.spine = empty("SPINE", self.root)
        self.yoke = empty("YOKE", self.root)
        self.bob = empty("BOB", self.yoke)
        self.tyre = {"L": empty("TYRE_L", self.root), "R": empty("TYRE_R", self.root)}
        body_parent = {"spine": self.spine, "yoke": self.yoke, "bob": self.bob, "tyreL": self.tyre["L"], "tyreR": self.tyre["R"]}
        mats = {}
        self.o = {}
        for name, e in self.P.items():
            bpy.ops.wm.stl_import(filepath=os.path.join(FILM, "mesh", f"{name}.stl"))
            ob = bpy.context.selected_objects[0]
            ob.name = name
            ob.data.name = name
            ob.scale = (MM, MM, MM)
            key = e["render"]
            if key not in mats:
                mats[key] = make_material(key)
            ob.data.materials.clear()
            ob.data.materials.append(mats[key])
            self.o[name] = ob
        bpy.ops.object.select_all(action="SELECT")
        for ob in self.o.values():
            if ob.type != "MESH":
                ob.select_set(False)
        bpy.context.view_layer.objects.active = next(iter(self.o.values()))
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(32))
        bpy.ops.object.select_all(action="DESELECT")
        for ob in self.o.values():
            ob.color = (0, 0, 0, 1)
        # rotors: parts that turn about their own axis inside their body
        self.rot = {}
        self.parent = {}
        for name, ob in self.o.items():
            par = body_parent[self.P[name]["body"]]
            ax = self.rotor_axis(name)
            if ax is not None:
                c, a, key = ax
                rk = f"ROT_{key}"
                if rk not in self.rot:
                    self.rot[rk] = (empty(rk, par, c), a)
                par = self.rot[rk][0]
            ob.parent = par
            self.parent[name] = par
        bpy.context.view_layer.update()
        for name, ob in self.o.items():
            par = self.parent[name]
            ob.matrix_parent_inverse = par.matrix_world.inverted()
        bpy.context.view_layer.update()
        self.rest_c = {n: self.centre(n) for n in self.P}

    # ---------------------------------------------------------------- geometry helpers (robot frame, metres)
    def centre(self, n):
        e = self.P[n]
        return Vector([(a + b) / 2 * MM for a, b in zip(e["bmin"], e["bmax"])])

    def rotor_axis(self, n):
        """(centre, axis, rotor key) for parts that spin about their own axis."""
        c = self.centre(n)
        Y, X = Vector((0, 1, 0)), Vector((1, 0, 0))
        m = re.match(r"(drive_motor|pulley_20)_([FB])$", n)
        if m:
            g = self.geo["drive"][m.group(2)]
            return Vector((g["mx"] * MM, 0, g["mz"] * MM)), Y, f"motor_{m.group(2)}"
        m = re.match(r"(pinion_shaft|pulley_32)_([FB])$", n)
        if m:
            g = self.geo["drive"][m.group(2)]
            return Vector((g["px"] * MM, 0, g["pz"] * MM)), Y, f"pinion_{m.group(2)}"
        if n in ("level_motor", "level_pinion"):
            return Vector((0, 0, self.geo["level_zc"] * MM)), Y, "level"
        if n == "lean_pinion":
            return Vector((0, c.y, c.z)), X, "lean"
        m = re.match(r"bob_wheel_([FB])(in|out)(\d)$", n)
        if m:
            return Vector((0, c.y, c.z)), X, f"bw_{m.group(1)}{m.group(2)}{m.group(3)}"
        m = re.match(r"(vwheel|roller)_([LR])(\d)$", n)
        if m:
            return Vector((c.x, 0, c.z)), Y, f"pw_{m.group(2)}{m.group(3)}"
        return None

    def world_centre(self, names):
        pts = []
        for n in names:
            n, pt = (n, None) if isinstance(n, str) else n       # (part, point in the rest robot frame, m)
            ob = self.o.get(n)
            if ob is None or ob.hide_render:
                continue
            mw = ob.matrix_world
            pts += [mw @ Vector(c) for c in ob.bound_box] if pt is None else [mw @ Vector(pt)]
        if not pts:
            return None
        return sum(pts, Vector()) / len(pts)


# ====================================================================================================== assembly
def names_like(P, *pats, kind=None):
    rx = [re.compile(p + "$") for p in pats]
    return [n for n, e in P.items() if any(r.match(n) for r in rx) and (kind is None or e["kind"] == kind)]


def fastener_owner(n, P):
    """Parts a non-joint fastener is fitted to (for its timing)."""
    m = re.match(r"(bob_stud_\w+?\d)_(nut|washer)$", n)
    if m:
        return [m.group(1)]
    m = re.match(r"stud_(nut|washer)_([LR]\d)$", n)
    if m:
        return [f"wheel_stud_{m.group(2)}"]
    m = re.match(r"bob_eclip_(\w+)$", n)
    if m:
        return [f"bob_wheel_{m.group(1)}"]
    m = re.match(r"hub_circlip_([LR])$", n)
    if m:
        return [f"hub_cartridge_{m.group(1)}"]
    m = re.match(r"pinion_circlip_([FB])$", n)
    if m:
        return [f"pinion_shaft_{m.group(1)}"]
    m = re.match(r"lidar_dowel_([LR])\d$", n)
    if m:
        return [f"lidar_riser_{m.group(1)}"]
    if n == "slipring_pin":
        return ["hanger_R"]
    return []


class Assembly:
    """The assembly sequence: events (t0, t1, parts, motion) in segment time; every part belongs to one event,
    every fastener to the event of the last part it joins."""

    BENCH = Vector((0, 0.42, 0))                     # yoke + bob are built off the axle, then slid on

    def __init__(self, R):
        self.R = R
        P = R.P
        nf = {n: e for n, e in P.items() if e["kind"] != "fastener"}

        def N(*pats):
            return [n for n in names_like(nf, *pats)]
        E = []

        def ev(key, t0, t1, parts, motion, **kw):
            E.append(dict(key=key, t0=t0, t1=t1, parts=list(parts), motion=motion, **kw))
        # A tyre halves
        ev("drumL", 0.6, 2.4, N("drum_L", "ring_gear_L"), "fade")
        ev("rimL", 2.8, 5.6, N("rim_L"), "slide", off=(0, -0.33, 0))
        ev("treadL", 6.2, 8.0, N("tread_L"), "wipe")
        ev("tyreR", 7.8, 9.6, N("drum_R", "ring_gear_R", "rim_R", "tread_R"), "slide", off=(0, -0.45, 0))
        # B bob (on the bench at its run-on pose)
        ev("carriage", 11.9, 13.8, N("carriage_B"), "slide", off=(-0.10, 0, 0))
        ev("carriage2", 11.9, 13.8, N("carriage_F"), "slide", off=(0.10, 0, 0))
        ev("tungsten", 14.6, 17.6, N("tungsten_[0-2]"), "slide", off=(0, 0, -0.16), stagger=0.4)
        ev("actuator", 20.0, 22.2, N("lean_actuator"), "slide", off=(-0.12, 0, 0))
        ev("pinion", 22.4, 23.8, N("lean_pinion"), "slide", off=(0.08, 0, 0))
        ev("studsF", 24.2, 26.6, N(r"bob_(stud|wheel)_F\w+"), "slide", off=(0.06, 0, 0), stagger=0.15)
        ev("studsB", 24.6, 27.0, N(r"bob_(stud|wheel)_B\w+"), "slide", off=(-0.06, 0, 0), stagger=0.15)
        # C yoke frame
        ev("hangerL", 28.4, 30.2, N("hanger_L"), "slide", off=(0, 0.12, 0))
        ev("hangerR", 28.4, 30.2, N("hanger_R"), "slide", off=(0, -0.12, 0))
        ev("hubL", 30.4, 32.4, N("hub_cartridge_L", "hub_bearing_L", "level_gear", r"pitch_(striker|bumper)\d"), "slide",
           off=(0, 0.08, 0), stagger=0.2)
        ev("hubR", 30.4, 32.4, N("hub_cartridge_R", "hub_bearing_R"), "slide", off=(0, -0.08, 0))
        ev("spacer", 32.0, 33.2, N("hub_spacer"), "slide", off=(0, -0.10, 0))
        ev("ctrl", 31.0, 33.0, N(r"drive_controller_[FB]"), "slide", off=(0, 0, -0.12))
        ev("arcF", 33.2, 36.6, N("arc_plate_F", "track_F", r"track_spacer_F\d", "lean_rack", r"rack_spacer_[io]\d"), "slide",
           off=(0.15, 0, 0))
        ev("arcB", 33.6, 37.0, N("arc_plate_B", "track_B", r"track_spacer_B\d"), "slide", off=(-0.15, 0, 0))
        ev("brackets", 37.2, 39.0, N(r"arc_bracket_\w+"), "slide", off=(0, 0, -0.08), stagger=0.15)
        ev("powerbox", 38.2, 40.0, N("power_box", "power_box_parts"), "slide", off=(0, 0, -0.14))
        ev("runon", 40.4, 44.0, [], "runon", a0=120.0)
        ev("stops", 44.2, 45.8, N(r"lean_stop_\w+"), "slide", off=(0, 0, 0), side_x=0.05)
        # D drive cassettes
        for side, sy, t in (("F", 1, 46.2), ("B", -1, 54.4)):
            fast = side == "B"
            k = 0.45 if fast else 1.0
            ev(f"shaft{side}", t, t + 2.2 * k, N(f"pinion_shaft_{side}", rf"pinion_bearing_{side}\d"), "slide", off=(0, sy * 0.10, 0))
            ev(f"spacers{side}", t + 2.4 * k, t + 3.8 * k, N(rf"cassette_spacer_{side}\d"), "slide", off=(0, sy * 0.06, 0), stagger=0.1)
            ev(f"p32{side}", t + 4.0 * k, t + 5.4 * k, N(f"pulley_32_{side}", f"belt_{side}"), "slide", off=(0, sy * 0.07, 0))
            ev(f"motor{side}", t + 5.6 * k, t + 7.6 * k, N(f"outer_plate_{side}", f"drive_motor_{side}", f"pulley_20_{side}"), "slide",
               off=(0, sy * 0.12, 0), prebuilt=True)
            ev(f"enc{side}", t + 7.8 * k, t + 9.2 * k, N(f"drive_motor_{side}_enc_bracket", f"drive_motor_{side}_encoder"), "slide",
               off=(0, sy * 0.06, 0), prebuilt=True)
        # E batteries
        ev("batA", 59.6, 62.0, N("battery_case_A", "battery_cells_A", r"battery_spacer_A\d"), "slide", off=(0, -0.14, 0))
        ev("batB", 60.0, 62.4, N("battery_case_B", "battery_cells_B", r"battery_spacer_B\d"), "slide", off=(0, 0.14, 0))
        # F onto the axle
        ev("axle", 63.0, 64.6, N("axle"), "slide", off=(0, -0.25, 0))
        ev("slide_on", 64.8, 69.4, [], "bench")
        # G level drive + slip ring
        ev("level", 69.8, 72.6, N("level_bracket", "level_motor", "level_pinion", "level_motor_enc_bracket", "level_motor_encoder"),
           "slide", off=(0, 0.33, 0), prebuilt=True)
        ev("slipring", 70.4, 73.2, N("slip_sleeve", "slipring_rotor", "slipring_housing"), "slide", off=(0, -0.33, 0), prebuilt=True)
        # H bays
        bayL = [n for n, e in nf.items() if e["group"] == nf["bay_tray_L"]["group"]] + N("gnss_plate", "gnss_antenna", "gnss_receiver",
                                                                                          r"gnss_receiver_spacer\d")
        bayR = [n for n, e in nf.items() if e["group"] == nf["bay_tray_R"]["group"]]
        ev("bayL", 76.0, 79.2, sorted(set(bayL)), "slide", off=(0, 0.22, 0), prebuilt=True)
        ev("bayR", 76.6, 79.8, sorted(set(bayR)), "slide", off=(0, -0.22, 0), prebuilt=True)
        # I tyre halves on: offset slide toward the pinion, then radially into mesh
        ev("tyresL", 82.0, 88.0, [], "tyre", side="L")
        ev("tyresR", 85.0, 91.0, [], "tyre", side="R")
        # J pods
        podL = [n for n, e in nf.items() if e["group"] == nf["pod_hub_L"]["group"]]
        podR = [n for n, e in nf.items() if e["group"] == nf["pod_hub_R"]["group"]]
        ev("podL", 92.0, 96.4, podL, "pod", side="L", off=(0, 0.28, 0), prebuilt=True)
        ev("podR", 94.4, 98.8, podR, "pod", side="R", off=(0, -0.28, 0), prebuilt=True)
        self.E = E
        # where each part enters
        self.of = {}
        for k, e in enumerate(E):
            for n in e["parts"]:
                self.of.setdefault(n, k)
        placed = set(self.of)
        missing = [n for n, e in nf.items() if n not in placed]
        if missing:
            raise RuntimeError(f"parts with no assembly event: {missing}")
        # fasteners: the event of the last member they join
        self.fast = {}
        self.ride = {}
        for n, e in P.items():
            if e["kind"] != "fastener":
                continue
            mem = []
            if e.get("joint") and e["joint"] in R.J:
                mem = [m for m in R.J[e["joint"]]["members"] if m != "axle" or True]
            mem = mem or fastener_owner(n, P)
            ks = [self.of[m] for m in mem if m in self.of]
            if not ks:
                raise RuntimeError(f"fastener {n}: no owner")
            k = max(ks)
            self.fast[n] = k
            if E[k].get("prebuilt") and all(self.of.get(m) == k for m in mem if m in self.of):
                self.ride[n] = [m for m in mem if m in self.of][0]          # fitted on the bench: rides along
        # stagger fasteners inside their event
        byk = {}
        for n, k in self.fast.items():
            if n not in self.ride:
                byk.setdefault(k, []).append(n)
        self.ftime = {}
        for k, ns in byk.items():
            e = E[k]
            ns.sort(key=lambda n: (R.P[n].get("joint") or n))
            t0 = e["t0"] + 0.55 * (e["t1"] - e["t0"])
            t1 = e["t1"] + 0.9
            span = max(t1 - t0 - 0.7, 0.1)
            for i, n in enumerate(ns):
                s = t0 + span * (i / max(len(ns) - 1, 1))
                self.ftime[n] = (s, s + 0.7)
        self.end = 106.0

    def axis(self, n):
        e = self.R.P[n]
        if e.get("n"):
            return Vector(e["n"]).normalized()
        b = [q - p for p, q in zip(e["bmin"], e["bmax"])]
        k = b.index(min(b))
        a = Vector((0, 0, 0))
        a[k] = 1.0
        c = self.R.centre(n)
        if a.dot(c) > 0:                                  # come in from the outside
            a = -a
        return a

    def state(self, t, S):
        """Fill the state for assembly time t."""
        R = self.R
        S["bench"] = self.BENCH * (1 - ramp(t, 64.8, 69.4))
        S["bob_rot"] = 120.0
        for k, e in enumerate(self.E):
            if e["motion"] == "runon":
                S["bob_rot"] = 120.0 * (1 - ease_io((t - e["t0"]) / (e["t1"] - e["t0"])))
        hl = {}
        for k, e in enumerate(self.E):
            t0, t1 = e["t0"], e["t1"]
            if e["motion"] in ("runon", "bench"):
                continue
            if e["motion"] == "tyre":
                s = e["side"]
                sg = 1 if s == "L" else -1
                a = math.radians(R.geo["drive_angle"])
                u = Vector((sg * math.sin(a), 0, -math.cos(a)))
                x = (t - t0) / (t1 - t0)
                slide = 1 - ease_io(clamp(x / 0.72))
                radial = 1 - ease_io(clamp((x - 0.74) / 0.26))
                off = Vector((0, sg * 0.60 * slide, 0)) + u * (0.015 * radial)
                if t >= t0:
                    for n in names_like(R.P, f"(drum|ring_gear|rim|tread)_{s}"):
                        S["off"][n] = off
                continue
            n_parts = e["parts"]
            st = e.get("stagger", 0.0)
            for i, n in enumerate(n_parts):
                tt0 = t0 + st * i
                tt1 = t1 + st * i
                x = (t - tt0) / max(tt1 - tt0, 1e-6)
                if t < tt0:
                    S["alpha"][n] = 0.0
                    continue
                p = ease_io(clamp(x))
                if e["motion"] == "fade":                # x-ray shell first, then solid
                    S["alpha"][n] = ease(clamp(x / 0.3))
                    S["ghost"][n] = 1 - ease_io(clamp((x - 0.3) / 0.7))
                    S["off"][n] = Vector()
                elif e["motion"] == "wipe":              # cast on: revealed across the part
                    S["alpha"][n] = 1.0
                    if x < 1:
                        S["wipe"][n] = max(ease_io(clamp(x)), 0.001)
                    # lifted 1.8 mm off the drum while partly transparent (its bonded face is coincident with the
                    # drum's), settled once opaque
                    S["scale"][n] = 1 + 0.006 * (1 - ease_io(clamp((x - 1) / 0.15)))
                    S["off"][n] = Vector()
                else:
                    off = Vector(e.get("off", (0, 0, 0)))
                    if "side_x" in e:
                        off = Vector((math.copysign(e["side_x"], R.rest_c[n].x), 0, 0))
                    if e["motion"] == "pod":                 # V-wheels retracted on their eccentrics while sliding
                        pass
                    S["off"][n] = off * (1 - p)
                    S["alpha"][n] = ease(clamp(x / 0.18))
                if x < 1.25:
                    hl[n] = 0.35 * (1 - ease(clamp((x - 0.5) / 0.5)))
        # pods: left V-wheels retracted 3.5 mm while sliding, then set
        for e in self.E:
            if e["motion"] == "pod" and e["side"] == "L":
                x = (t - e["t0"]) / (e["t1"] - e["t0"])
                ret = 1 - ease(clamp((x - 0.85) / 0.15))
                for n in names_like(R.P, r"(vwheel|wheel_retainer|wheel_screw)_L\d"):
                    c = R.rest_c[n]
                    rr = Vector((c.x, 0, c.z))
                    if rr.length > 1e-6:
                        S["off"][n] = S["off"].get(n, Vector()) - rr.normalized() * 0.0035 * ret
        # fasteners fitted on the bench ride with their part
        for n, m in self.ride.items():
            S["alpha"][n] = S["alpha"].get(m, 1.0)
            if m in S["off"]:
                S["off"][n] = S["off"][m]
        # fasteners
        for n, (f0, f1) in self.ftime.items():
            k = self.fast[n]
            e = self.E[k]
            base = Vector()
            # a fastener rides with its event's motion
            owner_parts = e["parts"]
            if owner_parts:
                base = S["off"].get(owner_parts[0], Vector()) if e["motion"] in ("pod",) else Vector()
            if t < f0:
                S["alpha"][n] = 0.0
                continue
            x = clamp((t - f0) / (f1 - f0))
            p = ease_io(x)
            a = self.axis(n)
            dist = 0.035 if R.P[n].get("type") != "self-clinching nut" and not n.endswith(("_insert", "_pem")) else 0.012
            S["off"][n] = base - a * dist * (1 - p)
            S["alpha"][n] = ease(clamp(x / 0.25))
            if R.P[n].get("joint") == n:                      # the screw itself turns in
                S["screws"][n] = (a, Vector(R.P[n]["seat"]) * MM, (1 - p) * 4 * 2 * math.pi)
            if x < 1:
                hl[n] = 0.35
        S["hl"] = hl


# ====================================================================================================== timeline
class Film:
    def __init__(self, R):
        self.R = R
        self.A = Assembly(R)
        segs = [("title", 9.0, 2), ("reveal", 12.0, 2), ("explode", 18.0, 2), ("assembly", self.A.end, 2),
                ("drive", 16.0, 1), ("pendulum", 16.0, 2), ("stops", 8.0, 2), ("steer", 10.0, 1), ("lean", 14.0, 1),
                ("pods", 14.0, 1), ("power", 14.0, 2), ("verify", 10.0, 2), ("closing", 9.0, 2)]
        self.segs = []
        f = 0
        for name, dur, step in segs:
            n = int(round(dur * FPS))
            self.segs.append(dict(name=name, f0=f, n=n, step=step, dur=dur))
            f += n
        self.nframes = f
        self.gc = R.geo

    def seg_of(self, f):
        for s in self.segs:
            if s["f0"] <= f < s["f0"] + s["n"]:
                return s
        return self.segs[-1]

    # ---------------------------------------------------------------- state
    def blank(self):
        return dict(root=Vector(), yaw=0.0, lift=0.0, pitch=0.0, lean=0.0, spin={"L": 0.0, "R": 0.0}, bench=Vector(),
                    bob_rot=0.0, off={}, alpha={}, ghost={}, wipe={}, scale={}, hl={}, screws={}, labels=[], caption=None,
                    title=None, cards=[], readouts=[], fov=0.0)

    def state(self, f):
        s = self.seg_of(f)
        t = (f - s["f0"]) / FPS
        S = self.blank()
        S["spin"] = {"L": 0.0, "R": 0.0}
        S["spin_parts"] = {}
        getattr(self, "seg_" + s["name"])(t, S, s["dur"])
        S["segment"], S["t"] = s["name"], t
        return S

    # shells
    SHELLS = r"(drum|tread)_[LR]"

    def ghost_shells(self, S, g, pods=False, rims=False):
        pats = [self.SHELLS] + ([r"(rim|ring_gear)_[LR]"] if rims else []) \
            + ([r"pod_shield_[LR]", r"cam_window_[LR]\d", r"light_ring_[LR]"] if pods else [])
        for n in names_like(self.R.P, *pats):
            S["ghost"][n] = max(S["ghost"].get(n, 0.0), g)

    POD_L = r"(pod_hub|pod_plate|wheel_stud|stud_washer|stud_nut|vwheel|wheel_retainer|lidar_riser|lidar|lidar_dowel|light_ring|led_pcb|pod_shield|cam_window|camera|cam_holder|shield_standoff)_L\d*"

    def ghost_pod(self, S, g, side="L"):
        rx = re.compile(self.POD_L.replace("_L", "_" + side) + "$")
        for n, e in self.R.P.items():
            if rx.match(n) or (e["kind"] == "fastener" and e.get("step", "") and e["step"].startswith(f"pod_{side}")):
                S["ghost"][n] = g

    def hide_pod(self, S, side):
        rx = re.compile(self.POD_L.replace("_L", "_" + side) + "$")
        for n, e in self.R.P.items():
            if rx.match(n) or (e["kind"] == "fastener" and (e.get("step") or "").startswith(f"pod_{side}")):
                S["alpha"][n] = 0.0

    def label(self, S, key, title, sub, anchor, color, a, side="auto"):
        if a > 0.003:
            S["labels"].append(dict(key=key, title=title, sub=sub, anchor=anchor, color=color, alpha=a, side=side))

    # ---------------------------------------------------------------- segments
    def seg_title(self, t, S, dur):
        S["yaw"] = math.radians(-22 + 18 * ease(t / dur))
        S["cam"] = cam_keys(t, [(0, aim((0, 0.02, -0.03), 26, -0.32), 26, 9, 2.7, 50), (dur, aim((0, 0.02, -0.03), 34, -0.28), 34, 12, 2.45, 50)])
        S["title"] = dict(text="GYRA Mk2.1", sub="The build: every part, every bolt", a=win(t, 1.2, dur - 0.6, 1.0))

    def seg_reveal(self, t, S, dur):
        S["yaw"] = math.radians(-4)
        S["cam"] = cam_keys(t, [(0, aim((0, 0.02, -0.03), 34, -0.28), 34, 12, 2.45, 50), (dur, (0, 0, -0.06), 58, 19, 2.3, 48)])
        g = ramp(t, 1.5, 4.5)
        self.ghost_shells(S, g, pods=True)
        A = [("tyre", "Tyre halves", "GFRP drums, PU tread, 7075 ring gears", ["drum_L"], "cyan", 4.8),
             ("pods", "Sensor pods", "LiDAR, 3 cameras, V-wheels on the rims", ["pod_plate_R"], "cyan", 5.6),
             ("spine", "Spine", "carbon axle, bays, level drive", ["bay_tray_L"], "green", 6.4),
             ("yoke", "Yoke: the pendulum", "hangers, drive cassettes, batteries", ["hanger_R"], "amber", 7.2),
             ("bob", "Bob: the lean mass", "10 kg tungsten on V-wheels", ["tungsten_1"], "violet", 8.0)]
        for k, ti, sub, anc, col, t0 in A:
            self.label(S, k, ti, sub, anc, col, win(t, t0, dur - 0.3, 0.5))
        S["caption"] = dict(kicker="OVERVIEW", title="Five bodies on one axle", sub="659 parts: 157 made, 91 bought, 411 fasteners",
                            a=win(t, 1.0, dur - 0.3, 0.6))

    def explode_offsets(self, S, x):
        R = self.R
        for n, e in R.P.items():
            c = R.rest_c[n]
            b = e["body"]
            off = Vector()
            if b in ("tyreL", "tyreR"):
                off = Vector((0, 0.34 if b == "tyreL" else -0.34, 0))
            elif b == "bob":
                off = Vector((0, 0, -0.26))
            elif b == "spine":
                if re.match(r"(pod_|wheel_|stud_|vwheel|roller|lidar|light_ring|led_pcb|cam|shield)", n):
                    off = Vector((0, math.copysign(0.62, c.y), 0))
                elif re.match(r"(bay_|compute|level_controller|dcdc|imu|usb_hub|mezz|gnss)", n) and abs(c.y) > 0.15:
                    off = Vector((0, math.copysign(0.10, c.y), -0.16))
                elif n.startswith("level_"):
                    off = Vector((0, 0.10, 0.08))
                elif n.startswith(("slip", )):
                    off = Vector((0, -0.12, 0))
            elif b == "yoke":
                if re.match(r"(outer_plate|drive_motor|pulley_20|belt|pulley_32)_([FB])", n) or "enc" in n:
                    off = Vector((0, math.copysign(0.10, c.y), 0))
                elif n.startswith("battery"):
                    off = Vector((0, math.copysign(0.10, c.y), 0))
            if e["kind"] == "fastener":
                a = self.A.axis(n)
                off = off - a * 0.03
            S["off"][n] = off * x

    def seg_explode(self, t, S, dur):
        S["yaw"] = math.radians(-4)
        x = ease_io(t / 6.0) * (1 - ease_io((t - 12.0) / 6.0))
        S["lift"] = 0.32 * ease_io(clamp(t / 3.0)) * (1 - ease_io((t - 14.5) / 3.5))
        self.explode_offsets(S, x)
        z = S["lift"]
        # targets relative to the lifted robot: the camera rises with it
        loc, tgt, lens = cam_keys(t, [(0, (0, 0, -0.06), 58, 19, 2.3, 48), (6, (0, 0, -0.18), 112, 18, 3.25, 42),
                                      (12, (0, 0, -0.18), 150, 22, 3.3, 42), (dur, (0, 0, -0.02), 205, 15, 2.6, 48)])
        S["cam"] = (tuple(c + d for c, d in zip(loc, (0, 0, z))), tuple(c + d for c, d in zip(tgt, (0, 0, z))), lens)
        # the floor's sheen eases off while the robot is lifted (the key light's reflection is no longer covered)
        S["floor_spec"] = FLOOR_SPEC * (1 - 0.85 * z / 0.32)
        A = [("tyre", "Tyre half (x2)", "drum + ring gear + rim, bonded; PU tread", ["drum_L"], "cyan"),
             ("pod", "Sensor pod (x2)", "8 V-wheels or rollers carry the tyre", ["pod_plate_L"], "cyan"),
             ("yoke", "Yoke", "2 drive cassettes, level gear, 432 Wh battery", ["hanger_L"], "amber"),
             ("bob", "Bob", "tungsten, AK70-10 lean actuator, 8 V-wheels", ["tungsten_1"], "violet"),
             ("spine", "Spine", "axle, level drive, slip ring, 2 bays", ["axle"], "green")]
        for i, (k, ti, sub, anc, col) in enumerate(A):
            self.label(S, k, ti, sub, anc, col, win(t, 6.2 + 0.35 * i, 11.6, 0.4))
        S["caption"] = dict(kicker="EXPLODED VIEW", title="Every part, every fastener", sub="411 screws, nuts, inserts, dowels and clips, each on its own axis",
                            a=win(t, 0.6, dur - 0.5, 0.6))

    def seg_assembly(self, t, S, dur):
        R, A = self.R, self.A
        S["yaw"] = 0.0
        S["lift"] = 0.30 * (1 - ease_io((t - 102.0) / 3.5))
        S["alpha"] = {}
        A.state(t, S)
        S["off"]["hub_spacer"] = S["off"].get("hub_spacer", Vector()) + S["bench"]
        # tyres hidden between their build (A) and their fitting (I)
        hide_ty = ramp(t, 10.7, 11.5) * (1 - (1.0 if t >= 82.0 else 0.0))
        for n in names_like(R.P, r"(drum|ring_gear|rim|tread)_[LR]"):
            if 11.0 <= t < 82.0:
                S["alpha"][n] = S["alpha"].get(n, 1.0) * (1 - hide_ty)
            if t >= 85.0 and n.endswith("_R") and t < A.E[-3]["t0"]:
                pass
        if t >= 82.0:
            for n in names_like(R.P, r"(drum|ring_gear|rim|tread)_L"):
                S["alpha"][n] = 1.0
        if t >= 85.0:
            for n in names_like(R.P, r"(drum|ring_gear|rim|tread)_R"):
                S["alpha"][n] = 1.0
        elif 82.0 <= t < 85.0:
            for n in names_like(R.P, r"(drum|ring_gear|rim|tread)_R"):
                S["alpha"][n] = 0.0
        # ghost the tyre shells while the drive inside is shown being meshed
        g = win(t, 85.5, 92.0, 0.8)
        self.ghost_shells(S, g)
        b = A.BENCH * (1 - ramp(t, 64.8, 69.4))
        z = S["lift"]

        def yk(p):                     # point in the yoke frame (bench) -> world, metres
            return (p[0] + b.x, p[1] + b.y, p[2] + b.z + z)

        def sp(p):
            return (p[0], p[1], p[2] + z)
        gF, gB = self.gc["drive"]["F"], self.gc["drive"]["B"]
        bob_c = (0, 0.22 * math.sin(math.radians(120)), -0.22 * -math.cos(math.radians(120)) * -1)
        bob_bench = yk((0, 0.19, 0.11))
        keys = [(0.0, sp((0, 0.16, 0)), -62, 20, 1.6, 50), (5.6, sp((0, 0.16, 0)), -40, 24, 1.45, 50),
                (8.0, sp((0, 0.17, 0)), 40, 16, 1.6, 50), (10.8, sp((0, 0, 0)), 40, 14, 2.1, 50),
                (12.0, bob_bench, 35, 22, 1.0, 50), (19.0, bob_bench, 60, 18, 0.85, 55), (27.0, bob_bench, 20, 26, 0.9, 55),
                (29.0, yk((0, 0.06, -0.02)), 35, 20, 1.85, 50), (36.0, yk((0, 0.06, -0.03)), 62, 16, 1.8, 50),
                (40.0, yk((0, 0, -0.10)), 15, 6, 1.35, 50), (44.5, yk((0, 0, -0.10)), 30, 12, 1.25, 50),
                (46.5, yk((gF["mx"] * MM, 0.12, gF["mz"] * MM)), 50, 10, 0.78, 50),
                (53.5, yk((gF["mx"] * MM, 0.10, gF["mz"] * MM)), 32, 18, 0.72, 50),
                (55.0, yk((0, 0, -0.10)), -40, 18, 1.35, 50), (58.0, yk((0, 0, -0.10)), -60, 16, 1.35, 50),
                (62.0, yk((0, 0, -0.10)), -20, 14, 1.5, 50),
                (64.5, sp((0, 0.2, -0.05)), 28, 14, 1.75, 45), (69.5, sp((0, 0.0, -0.04)), 40, 15, 1.45, 45),
                (73.5, sp((0, 0, 0)), 55, 22, 1.4, 45), (80.0, sp((0, 0, 0)), 70, 20, 1.45, 45),
                (82.0, sp((0, 0.0, 0)), 18, 14, 2.5, 45), (86.5, sp((0.05, 0.05, -0.04)), 22, 10, 1.9, 45),
                (89.0, sp((0.12, 0.1, -0.08)), 35, 6, 1.2, 50),
                (91.5, sp((0, 0, 0)), 50, 12, 2.2, 45), (98.5, sp((0, 0, 0)), 75, 14, 2.4, 45),
                (dur, (0, 0, -0.04), 50, 12, 2.3, 50)]
        S["cam"] = cam_keys(t, keys)
        steps = [(0.0, 11.6, "1", "Tyre halves", "drum infused with its ring gear; rim through the seam; tread cast on"),
                 (12.0, 27.8, "2", "Bob", "tungsten between the carriages, AK70-10 + pinion, V-wheels on studs"),
                 (28.0, 40.2, "3", "Yoke frame", "hangers, hub bearings, level gear, arc plates, tracks, lean rack"),
                 (40.2, 46.0, "4", "Bob runs onto the tracks", "rolled in from beyond the track end, then the lean stops"),
                 (46.0, 59.4, "5", "Drive cassettes", "pinion shaft, belt, D6374 on slotted holes, AMT212 encoder"),
                 (59.4, 62.8, "6", "Battery modules", "two 6S2P Molicel P50B packs, 432 Wh"),
                 (63.0, 69.6, "7", "Onto the axle", "the yoke + bob slide onto the carbon axle"),
                 (69.6, 75.8, "8", "Level drive and slip ring", "bracket + motor from the left, slip ring from the right"),
                 (75.8, 81.8, "9", "Bays", "Jetson, GNSS receiver + antenna, IMU, USB hub, level controller"),
                 (81.8, 91.8, "10", "Tyre halves", "slid on 15 mm off-centre, then moved radially into mesh"),
                 (91.8, 101.6, "11", "Sensor pods", "left V-wheels retracted on their eccentrics, then set"),
                 (101.6, dur, "12", "Done", "12 of 12 assembly moves verified clear")]
        for t0, t1, k, ti, sub in steps:
            a = win(t, t0, t1, 0.4)
            if a > 0:
                S["caption"] = dict(kicker=f"STEP {k} / 12", title=ti, sub=sub, a=a)
        L = [(19.5, 23.5, "w", "Tungsten W-Ni-Fe", "3 blocks, 10 kg, 12 x M8 class 10.9", ["tungsten_1"], "violet"),
             (22.5, 27.0, "ak", "CubeMars AK70-10", "lean actuator, 40T pinion", ["lean_actuator"], "violet"),
             (30.6, 34.5, "hub", "Hub cartridge + 6908", "the yoke turns on the axle", ["hub_cartridge_L"], "amber"),
             (31.2, 35.5, "lg", "Level gear 96T", "the level drive's ring", ["level_gear"], "amber"),
             (35.0, 39.5, "tr", "Hardened V-tracks", "the bob's rails", ["track_F"], "amber"),
             (41.0, 44.0, "ro", "V-wheels engage", "", ["bob_wheel_Fout0"], "violet"),
             (49.5, 53.0, "belt", "HTD 5M belt", "20T : 32T", ["belt_F"], "green"),
             (52.0, 55.0, "mot", "ODrive D6374", "150 KV, 48 V", ["drive_motor_F"], "green"),
             (53.6, 55.5, "enc", "AMT212 encoder", "absolute, RS485", ["drive_motor_F_encoder"], "green"),
             (71.0, 75.0, "lvl", "Level motor", "14T pinion on the 96T gear", ["level_motor"], "amber"),
             (71.6, 75.5, "sr", "Moflon slip ring", "power, CAN, e-stop", ["slipring_housing"], "cyan"),
             (87.5, 91.0, "mesh", "24T pinion in the 360T ring", "backlash 0.15-0.45 mm", ["pinion_shaft_F"], "green")]
        for t0, t1, k, ti, sub, anc, col in L:
            self.label(S, k, ti, sub, anc, col, win(t, t0, t1, 0.4))
        S["bob_lean_rot"] = S.get("bob_rot", 0.0)

    # ---- how it works
    def roll(self, S, x, pitch=0.0):
        S["root"] = Vector((x, 0, 0))
        th = x / R_TYRE
        S["spin"] = {"L": th, "R": th}
        S["pitch"] = pitch

    def seg_drive(self, t, S, dur):
        v = 0.08
        x = -0.55 + v * t
        self.roll(S, x, pitch=math.radians(-8))
        self.ghost_shells(S, ramp(t, 0.2, 1.8), pods=True)
        g = self.gc["drive"]["F"]
        tg = (x + g["mx"] * MM * 0.6 + g["px"] * MM * 0.4, 0.10, (g["mz"] * MM + g["pz"] * MM) / 2)
        S["cam"] = cam_keys(t, [(0, (x, 0, -0.05), 40, 14, 1.6, 45), (3.5, tg, 48, 12, 0.85, 50), (dur, (tg[0] + v * 0, 0.10, tg[2]), 30, 18, 0.78, 50)])
        S["cam_follow"] = v
        A = [(4.0, "mot", "D6374 motor", "24 x the tyre speed", ["drive_motor_F"], "green"),
             (5.2, "belt", "Belt 20T : 32T", "x 1.6", ["belt_F"], "green"),
             (6.4, "pin", "24T pinion", "on the yoke", ["pinion_shaft_F"], "green"),
             (7.6, "ring", "360T ring gear", "bonded in the tyre half: x 15", ["ring_gear_L"], "green"),
             (8.8, "enc", "AMT212 encoder", "commutation + position", ["drive_motor_F_encoder"], "green")]
        for t0, k, ti, sub, anc, col in A:
            self.label(S, k, ti, sub, anc, col, win(t, t0, dur - 0.4, 0.4))
        S["caption"] = dict(kicker="HOW IT WORKS", title="Drive: 24 : 1 per tyre half", sub="motor, belt and pinion ride on the yoke; the ring gear turns the tyre",
                            a=win(t, 0.5, dur - 0.4, 0.6))
        S["readouts"].append(dict(text="1.6 x 15 = 24 : 1", a=win(t, 9.5, dur - 0.4, 0.5), pos="tr"))

    def seg_pendulum(self, t, S, dur):
        # accelerate forward: the drive torque swings the pendulum forward; the level drive holds the spine level
        p = -math.radians(55) * math.sin(math.pi * clamp(t / 9.0)) ** 2 if t < 9.0 else -math.radians(30) * math.sin(math.pi * clamp((t - 9.0) / 6.0)) ** 2
        acc = 0.05
        x = -0.4 + 0.5 * acc * t * t if t < 8 else -0.4 + 0.5 * acc * 64 + acc * 8 * (t - 8)
        x = min(x, 1.4)
        self.roll(S, x, pitch=p)
        self.ghost_shells(S, 1.0, pods=True, rims=True)
        self.hide_pod(S, "L")
        S["cam"] = cam_keys(t, [(0, (x, 0.05, -0.07), 80, 8, 1.7, 45), (dur, (x, 0.05, -0.07), 86, 10, 1.7, 45)])
        S["cam_track_x"] = True
        self.label(S, "yoke", "Yoke swings with the drive torque", "its weight is the reaction: up to m g L", ["arc_plate_F"], "amber", win(t, 1.5, 8.5, 0.5))
        self.label(S, "spine", "Spine stays level", "level drive 14T : 96T on the left hanger", ["level_motor"], "green", win(t, 3.0, dur - 0.4, 0.5))
        S["readouts"].append(dict(text=f"yoke {abs(math.degrees(p)):4.0f} deg", a=win(t, 0.8, dur - 0.4, 0.4), pos="tr", mono=True))
        S["caption"] = dict(kicker="HOW IT WORKS", title="Pendulum drive, level spine", sub="sum of the two drive torques swings the yoke; the sensors stay level",
                            a=win(t, 0.5, dur - 0.4, 0.6))

    def seg_stops(self, t, S, dur):
        ps = self.gc["pitch_stop"]
        p = math.radians(ps) * (ease_io(t / 3.0) if t < 4.5 else 1 - ease_io((t - 4.5) / 3.0))
        p = min(p, math.radians(ps))
        S["pitch"] = -p
        keep = re.compile(r"(hanger_L|pitch_striker\d|pitch_bumper\d|level_bracket|level_gear|hub_cartridge_L|hub_bearing_L)$")
        for n, e in self.R.P.items():
            if keep.match(n):
                continue
            if not re.match(r"(level_bracket_clamp|hub_cartridge_screw_L|level_gear_screw|pitch_striker_screw|hub_circlip_L)", n):
                S["alpha"][n] = 0.0
        for n in ("pitch_bumper0", "pitch_bumper1", "pitch_striker0", "pitch_striker1"):
            S["hl"][n] = 0.5
        S["root"] = Vector((0, 0, 0))
        S["cam"] = cam_keys(t, [(0, (0.08, 0.05, 0.03), 76, 12, 1.32, 50), (dur, (0.08, 0.05, 0.03), 98, 9, 1.2, 50)])
        self.label(S, "hg", "Yoke hanger", "pitches with the pendulum", [("hanger_L", (0.121, 0.031, -0.17))], "amber",
                   win(t, 1.0, dur - 0.4, 0.4))
        self.label(S, "lb", "Level bracket", "carries the spine: stays level", [("level_bracket", (0.018, 0.066, 0.06))], "green", win(t, 1.6, dur - 0.4, 0.4))
        self.label(S, "st", "Pitch stop", "PU-sleeved striker meets the level-bracket arm", ["pitch_bumper0"], "coral",
                   win(t, 2.6, dur - 0.4, 0.4))
        S["readouts"].append(dict(text=f"yoke {math.degrees(p):5.1f} deg", a=win(t, 0.4, dur - 0.3, 0.3), pos="tr", mono=True))
        S["caption"] = dict(kicker="HOW IT WORKS", title=f"Stops at +-{ps:.1f} deg", sub="the deployed control uses at most 99.5 deg; first collision beyond 104 deg",
                            a=win(t, 0.4, dur - 0.3, 0.5))

    def seg_steer(self, t, S, dur):
        yaw = math.radians(110) * ease_io(t / dur)
        S["yaw"] = yaw
        d = yaw * 0.09 / R_TYRE * 2.0
        S["spin"] = {"L": d, "R": -d}
        self.ghost_shells(S, 0.0)
        S["cam"] = cam_keys(t, [(0, (0, 0, -0.08), 30, 42, 2.2, 45), (dur, (0, 0, -0.08), 40, 46, 2.1, 45)])
        S["caption"] = dict(kicker="HOW IT WORKS", title="Differential steering", sub="equal torques drive; a torque difference turns the robot on the spot",
                            a=win(t, 0.4, dur - 0.3, 0.5))
        S["readouts"].append(dict(text="left +  /  right -", a=win(t, 1.0, dur - 0.4, 0.4), pos="tr"))

    def seg_lean(self, t, S, dur):
        L = math.radians(40) * math.sin(2 * math.pi * t / 7.0) * ramp(t, 0.5, 1.5) * (1 - ramp(t, dur - 1.5, dur - 0.3))
        S["lean"] = L
        self.ghost_shells(S, 1.0, pods=True)
        S["cam"] = cam_keys(t, [(0, (0, 0, -0.12), 2, 8, 1.55, 45), (6.0, (0.05, 0, -0.17), 12, 10, 0.95, 50),
                                (dur, (0.05, 0, -0.17), -8, 14, 0.9, 50)])
        A = [(2.0, "ak", "AK70-10 actuator", "on the front carriage", ["lean_actuator"], "violet"),
             (3.2, "pin", "40T pinion", "in an internal rack r 195: 6.5 : 1", ["lean_pinion"], "violet"),
             (4.4, "vw", "DualVee V-wheels", "on induction-hardened 1045 tracks", ["bob_wheel_Fout1"], "violet"),
             (5.6, "ls", "Lean stops", "PU-faced, catch the carriage at 42 deg", ["lean_stop_Fp"], "coral")]
        for t0, k, ti, sub, anc, col in A:
            self.label(S, k, ti, sub, anc, col, win(t, t0, dur - 0.4, 0.4))
        S["readouts"].append(dict(text=f"lean {math.degrees(L):+5.1f} deg", a=win(t, 0.6, dur - 0.3, 0.3), pos="tr", mono=True))
        S["caption"] = dict(kicker="HOW IT WORKS", title="Lean: 10 kg sideways", sub="the bob rolls on its own tracks inside the yoke, +-40 deg",
                            a=win(t, 0.4, dur - 0.3, 0.5))

    def seg_pods(self, t, S, dur):
        th = -0.35 * t
        S["spin"] = {"L": th, "R": th}
        self.ghost_shells(S, 1.0, pods=True)
        S["lift"] = 0.0
        if t < 7.0:                                           # the inner face: wheels riding on the rim
            S["cam"] = cam_keys(t, [(0, (0, 0.27, -0.01), 22, 20, 1.0, 45), (7.0, (0, 0.26, -0.02), 36, 16, 0.82, 50)])
        else:
            S["cam"] = cam_keys(t - 7.0, [(0, (0, -0.27, -0.01), -22, 20, 1.0, 45), (7.0, (0, -0.26, -0.02), -36, 16, 0.82, 50)])
        S["fov"] = win(t, 1.0, 6.6, 0.6)
        A = [(0.8, 6.6, "li", "Livox Mid-360", "360 x 59 deg LiDAR", ["lidar_L"], "cyan"),
             (1.6, 6.6, "cm", "3 cameras per pod", "global shutter, square to the shell", ["camera_L0"], "cyan"),
             (2.4, 6.6, "vw", "8 DualVee wheels", "on the rim's V-track: locate the left half", ["vwheel_L2"], "cyan"),
             (7.8, dur - 0.3, "ro", "8 crowned rollers", "on a plain land: the right half floats axially", ["roller_R2"], "cyan"),
             (8.6, dur - 0.3, "ec", "Eccentric studs", "set the preload; 3.5 mm retract for assembly", ["wheel_stud_R3"], "cyan")]
        for t0, t1, k, ti, sub, anc, col in A:
            self.label(S, k, ti, sub, anc, col, win(t, t0, t1, 0.4))
        S["caption"] = dict(kicker="HOW IT WORKS", title="Sensor pods carry the tyres", sub="no seam bearing: each half rides on its own pod",
                            a=win(t, 0.4, dur - 0.3, 0.5))

    POWER = r"(battery_cells_[AB]|power_box_parts|drive_controller_[FB]|level_controller|compute|gnss_receiver|imu|usb_hub|dcdc|slipring_housing|slipring_rotor)"

    def seg_power(self, t, S, dur):
        R = self.R
        S["yaw"] = math.radians(15 * t / dur)
        g = ramp(t, 0.3, 1.8)
        hot = set(names_like(R.P, self.POWER))
        for n, e in R.P.items():
            if n in hot:
                S["hl"][n] = 0.25 * g
            elif e["body"] in ("tyreL", "tyreR") or re.match(r"(pod_shield|cam_window|light_ring)", n) or e["kind"] == "fastener":
                S["ghost"][n] = g
            else:
                S["ghost"][n] = 0.65 * g
        S["cam"] = cam_keys(t, [(0, (0, 0, -0.05), 30, 22, 1.85, 45), (dur, (0, 0, -0.05), 55, 26, 1.7, 45)])
        A = [(2.0, "bat", "Battery 12S2P", "Molicel P50B, 432 Wh, 23 km", ["battery_cells_A"], "cyan"),
             (2.8, "pb", "Power box", "BMS, 40 A fuse, anti-spark, e-stop relay", ["power_box_parts"], "coral"),
             (3.6, "s1", "ODrive S1 x 2", "drive motors, on heat spreaders", ["drive_controller_F"], "green"),
             (4.4, "sr", "Slip ring", "48 V, CAN and e-stop to the spine", ["slipring_housing"], "amber"),
             (5.2, "jet", "Jetson Orin NX", "left bay; GNSS receiver above the axle", ["compute"], "violet"),
             (6.0, "lc", "Level S1, DC-DC, IMU, USB hub", "right bay", ["level_controller"], "violet")]
        for t0, k, ti, sub, anc, col in A:
            self.label(S, k, ti, sub, anc, col, win(t, t0, dur - 0.4, 0.4))
        S["caption"] = dict(kicker="HOW IT WORKS", title="Power and electronics", sub="yoke: battery, power, drives; spine: compute and sensors",
                            a=win(t, 0.4, dur - 0.3, 0.5))

    def seg_verify(self, t, S, dur):
        S["yaw"] = math.radians(-20 + 35 * t / dur)
        S["cam"] = cam_keys(t, [(0, aim((0, 0, -0.03), 25, 0.5), 25, 12, 2.5, 50), (dur, aim((0, 0, -0.03), 35, 0.5), 35, 14, 2.4, 50)])
        rows = [("659", "parts modelled, 411 of them fasteners"), ("0", "static interferences"),
                ("0", "collisions over the full range of motion (continuous proof)"), ("12 / 12", "assembly moves clear"),
                ("42 / 42", "engineering checks"), ("0", "falls: deployed control on the build masses")]
        S["cards"].append(dict(title="Verified before a single part is cut", rows=rows, a=win(t, 0.6, dur - 0.4, 0.6)))

    def seg_closing(self, t, S, dur):
        v = 0.22
        x = -0.6 + v * t
        self.roll(S, x, pitch=math.radians(-12))
        S["cam"] = cam_keys(t, [(0, aim((x - 0.1, 0, -0.05), 115, -0.45), 115, 9, 2.4, 50), (dur, aim((x - 0.1, 0, -0.05), 100, -0.45), 100, 10, 2.5, 50)])
        S["cam_track_x"] = True
        S["title"] = dict(text="GYRA Mk2.1", sub="ready to build: docs/11_build.md", a=win(t, 2.0, dur - 0.2, 1.0))


def make_fov(R):
    """Livox Mid-360 field of view: 360 deg x (-7 .. +52 deg) zone of a 0.75 m sphere around the left LiDAR."""
    import bmesh
    me = bpy.data.meshes.new("fov")
    bm = bmesh.new()
    rad, n_az, n_el = 0.55, 96, 12
    rows = []
    for i in range(n_el + 1):
        el = math.radians(-7 + 59 * i / n_el)
        rows.append([bm.verts.new((rad * math.cos(el) * math.cos(2 * math.pi * k / n_az), rad * math.cos(el) * math.sin(2 * math.pi * k / n_az),
                                   rad * math.sin(el))) for k in range(n_az)])
    for i in range(n_el):
        for k in range(n_az):
            bm.faces.new((rows[i][k], rows[i][(k + 1) % n_az], rows[i + 1][(k + 1) % n_az], rows[i + 1][k]))
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("fov", me)
    bpy.context.scene.collection.objects.link(ob)
    m = bpy.data.materials.new("fov")
    m.use_nodes = True
    nt = m.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*ACC["cyan"], 1)
    em.inputs["Strength"].default_value = 1.6
    info = nt.nodes.new("ShaderNodeObjectInfo")
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = 0.015
    mr.inputs["To Max"].default_value = 0.35
    nt.links.new(lw.outputs["Facing"], mr.inputs["Value"])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(info.outputs["Alpha"], mul.inputs[0])
    nt.links.new(mr.outputs["Result"], mul.inputs[1])
    nt.links.new(mul.outputs[0], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    ob.data.materials.append(m)
    c = R.centre("lidar_L")
    ob.parent = R.spine
    ob.location = (c.x, c.y, c.z)
    ob.color = (0, 0, 0, 0)
    ob.hide_render = True
    ob.visible_shadow = False
    return ob


# ====================================================================================================== apply + render
def apply(R, S, sc, cam, floor, lights):
    root = R.root
    fb = floor.active_material.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"]
    fs = S.get("floor_spec", FLOOR_SPEC)
    if abs(fb.default_value - fs) > 1e-6:
        fb.default_value = fs
    z = S.get("lift", 0.0)
    root.location = S["root"] + Vector((0, 0, z))
    root.rotation_euler = (0, 0, S.get("yaw", 0.0))
    R.yoke.location = S.get("bench", Vector())
    R.yoke.rotation_euler = (0, S.get("pitch", 0.0), 0)
    bob_rot = S.get("bob_rot", 0.0)
    R.bob.rotation_euler = (math.radians(bob_rot) + S.get("lean", 0.0), 0, 0)
    for s in "LR":
        R.tyre[s].rotation_euler = (0, S["spin"][s], 0)
    lights.location = (S["root"].x, 0, 0)
    # rotors
    p, L = S.get("pitch", 0.0), S.get("lean", 0.0) + math.radians(bob_rot)
    g = R.geo
    ang = {"motor_F": 24.0 * (S["spin"]["L"] - p), "pinion_F": 15.0 * (S["spin"]["L"] - p),
           "motor_B": 24.0 * (S["spin"]["R"] - p), "pinion_B": 15.0 * (S["spin"]["R"] - p),
           "level": -g["ratio_level"] * p, "lean": -(g["rack_r"] / g["lean_pinion_r"]) * L}
    for rk, (emp, a) in R.rot.items():
        key = rk[4:]
        if key in ang:
            th = ang[key]
        elif key.startswith("bw_"):
            th = (-11.94 if "in" in key else 16.88) * L
        elif key.startswith("pw_"):
            side = key[3]
            th = (12.8 if side == "L" else 9.57) * S["spin"][side]
        else:
            th = 0.0
        emp.rotation_euler = (th, 0, 0) if a.x > 0.5 else (0, th, 0)
    # parts
    alpha, off, ghost, hl, spins = S["alpha"], S["off"], S["ghost"], S["hl"], S.get("spin_parts", {})
    sp = S.get("spin", {})
    for n, ob in R.o.items():
        a = alpha.get(n, 1.0)
        ob.hide_render = a <= 0.002
        ob.color = (hl.get(n, 0.0), ghost.get(n, 0.0), S.get("wipe", {}).get(n, 0.0), a)
        o = off.get(n)
        M = Matrix.Translation(o) if o is not None else Matrix.Identity(4)
        if n in S.get("scale", {}):                     # about the tyre centre (robot origin)
            M = M @ Matrix.Scale(S["scale"][n], 4)
        sp_ = S.get("spin_screw", {}).get(n)
        ob.matrix_basis = M
    for n, (axis, seat, th) in S.get("screws", {}).items():
        ob = R.o[n]
        ob.matrix_basis = ob.matrix_basis @ Matrix.Translation(seat) @ Matrix.Rotation(th, 4, axis) @ Matrix.Translation(-seat)
    fov = bpy.data.objects.get("fov")
    if fov is not None:
        fa = S.get("fov", 0.0)
        fov.color = (0, 0, 0, fa)
        fov.hide_render = fa <= 0.002
    # camera
    loc, tgt, lens = S["cam"]
    loc, tgt = Vector(loc), Vector(tgt)
    if S.get("cam_track_x"):
        pass
    cam.location = loc
    cam.rotation_euler = (tgt - loc).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = lens
    S["_tgt"] = tgt


def anchors(R, S, sc, cam):
    W, H = sc.render.resolution_x, sc.render.resolution_y
    out = []
    for lb in S["labels"]:
        c = R.world_centre(lb["anchor"])
        if c is None:
            continue
        p = world_to_camera_view(sc, cam, c)
        if p.z <= 0:
            continue
        d = dict(lb)
        d["xy"] = [round(p.x / 1.0, 5), round(1 - p.y, 5)]
        d.pop("anchor")
        out.append(d)
    fov = None
    if S.get("fov", 0) > 0:
        c = R.world_centre(["lidar_L"])
        if c is not None:
            p = world_to_camera_view(sc, cam, c)
            fov = dict(a=S["fov"], xy=[p.x, 1 - p.y])
    return out, fov


def main():
    a = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    res = "480x270" if a.preview else a.res
    setup_render(sc, res, 2 if a.preview else a.samples)
    floor, lights = studio(sc)
    R = Robot()
    make_fov(R)
    F = Film(R)
    camd = bpy.data.cameras.new("cam")
    camd.clip_start, camd.clip_end = 0.01, 100
    cam = bpy.data.objects.new("cam", camd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    os.makedirs(a.out, exist_ok=True)
    k, n = (int(v) for v in a.worker.split("/"))
    only = set(a.only.split(",")) if a.only else None
    json.dump(dict(fps=FPS, nframes=F.nframes, segs=F.segs, res=res, acc=ACC), open(os.path.join(a.out, "film.json"), "w"), indent=1)
    frames = [int(x) for x in a.frames.split(",")] if a.frames else range(F.nframes)
    todo = []
    for f in frames:
        s = F.seg_of(f)
        if only and s["name"] not in only:
            continue
        step = 4 if a.preview else s["step"]
        rel = f - s["f0"]
        render_it = (rel % step == 0) or (rel == s["n"] - 1)
        todo.append((f, render_it))
    i = 0
    for f, render_it in todo:
        S = F.state(f)
        if S.get("bob_rot") is None:
            S["bob_rot"] = 0.0
        jp = os.path.join(a.out, f"f_{f:05d}.json")
        ip = os.path.join(a.out, f"f_{f:05d}.png")
        mine = (i % n == k)
        if render_it:
            i += 1
        need_img = render_it and mine and not a.json_only and not os.path.exists(ip)
        need_js = not os.path.exists(jp) and (mine or a.json_only or not render_it)
        if not (need_img or need_js):
            continue
        apply(R, S, sc, cam, floor, lights)
        bpy.context.view_layer.update()
        labels, fov = anchors(R, S, sc, cam)
        js = dict(frame=f, seg=S["segment"], t=round(S["t"], 4), rendered=render_it, labels=labels, fov=fov,
                  caption=S.get("caption"), title=S.get("title"), cards=S.get("cards"), readouts=S.get("readouts"))
        if need_img:
            sc.render.filepath = ip
            bpy.ops.render.render(write_still=True)
            print(f"frame {f} ({S['segment']} {S['t']:.2f}s)", flush=True)
        if need_js or need_img:
            json.dump(js, open(jp, "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
