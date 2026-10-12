"""
Final composition of the GYRA Mk2.1 video: robot shots with animated callouts, explainer scenes, cards, and the
field mission with its HUD, streamed through ffmpeg at 1920 x 1080, 24 fps (no intermediate files). Poppins only,
upright weights.

    python render/compose_final.py --frames <dir from render/render_video.sh> --explainer <dir> --out media/gyra_mk21_film.mp4
"""
import argparse
import glob
import json
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FPS, W, H = 24, 1920, 1080
FD = os.path.join(HERE, "fonts")
F = {w: os.path.join(FD, f"Poppins-{w}.ttf") for w in ("Light", "Regular", "Medium", "SemiBold", "Bold")}
_fc = {}


def font(w, size):
    k = (w, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(F[w], size)
    return _fc[k]


INK, MUTED, AMBER, CYAN, CORAL, GREEN, VIOLET = (236, 239, 247), (154, 163, 189), (255, 181, 71), (70, 199, 255), (255, 107, 94), (75, 214, 155), (168, 139, 255)
BG0, BG1 = (10, 15, 30), (20, 28, 53)
XFADE = 12


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def win(t, t0, t1, fade=0.5):
    if t < t0 or t > t1:
        return 0.0
    return min(ease((t - t0) / fade), ease((t1 - t) / fade))


def gradient():
    y = np.linspace(0, 1, H)[:, None, None]
    a, b = np.array(BG0)[None, None], np.array(BG1)[None, None]
    img = (a * (1 - y) + b * y) * np.ones((1, W, 1))
    yy, xx = np.mgrid[0:H, 0:W]
    v = np.clip(((xx / W - 0.5) ** 2 / 0.35 + (yy / H - 0.45) ** 2 / 0.6), 0, 1) * 0.35
    return Image.fromarray((img * (1 - v[..., None])).astype(np.uint8))


BG = None


def text(d, xy, s, w, size, color, alpha=1.0, anchor="la"):
    if alpha <= 0.002:
        return
    d.text(xy, s, font=font(w, size), fill=(*color, int(255 * alpha)), anchor=anchor)


def overlay(base, draw_fn):
    """Draw with alpha onto an RGBA layer and composite onto base (RGB)."""
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer), layer)
    out = base.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def panel(d, box, alpha, radius=18, fill=(14, 20, 38), outline=(58, 68, 102)):
    if alpha > 0:
        d.rounded_rectangle(box, radius=radius, fill=(*fill, int(200 * alpha)), outline=(*outline, int(220 * alpha)), width=2)


# ------------------------------------------------------------------------------------------------ frame sources
def ffmpeg_frames(inputs, vf, n_max=None):
    """Yield RGB frames (H x W x 3 uint8) from an ffmpeg decode."""
    cmd = ["ffmpeg", "-v", "error"] + inputs + ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        n += 1
        if n_max and n >= n_max:
            break
    p.stdout.close()
    p.wait()


INTERP = "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,scale=1920:1080:flags=lanczos"


def seq12(pattern):
    """PNG sequence rendered at every other 24 fps frame -> 24 fps (motion-compensated interpolation), 1080p."""
    return ffmpeg_frames(["-framerate", "12", "-pattern_type", "glob", "-i", pattern], INTERP)


def anchors_at(dirn, shot, k):
    """Label anchors (720p json) for output frame k, interpolated between rendered (even) frames, scaled to 1080p."""
    def load(f):
        p = os.path.join(dirn, f"{shot}_{f:05d}.json")
        return json.load(open(p)) if os.path.exists(p) else None
    a0, a1 = load(k - k % 2), load(k - k % 2 + 2)
    if a0 is None:
        return {}
    if a1 is None or k % 2 == 0:
        return {n: (v[0] * 1.5, v[1] * 1.5) for n, v in a0.items()}
    return {n: ((v[0] + a1.get(n, v)[0]) * 0.75, (v[1] + a1.get(n, v)[1]) * 0.75) for n, v in a0.items()}


def callout(d, anchor, label, sub, color, alpha, side=None):
    """Dot on the part, leader line, and a label panel to its side."""
    if alpha <= 0 or anchor is None:
        return
    ax, ay = anchor
    side = side or (1 if ax < W * 0.52 else -1)
    lx = ax + side * 230
    ly = ay - 120
    tw = max(font("SemiBold", 30).getlength(label), font("Regular", 22).getlength(sub)) + 40
    bx0 = lx if side > 0 else lx - tw
    d.ellipse((ax - 9, ay - 9, ax + 9, ay + 9), fill=(*color, int(255 * alpha)))
    d.ellipse((ax - 18, ay - 18, ax + 18, ay + 18), outline=(*color, int(160 * alpha)), width=3)
    d.line([(ax, ay), (lx, ly + 40)], fill=(*color, int(230 * alpha)), width=3)
    panel(d, (bx0, ly - 6, bx0 + tw, ly + 86), alpha)
    text(d, (bx0 + 20, ly + 6), label, "SemiBold", 30, color, alpha)
    text(d, (bx0 + 20, ly + 48), sub, "Regular", 22, INK, alpha)


def lower_third(d, kicker, title, sub, alpha, y=860):
    if alpha <= 0:
        return
    text(d, (120, y), kicker.upper(), "SemiBold", 22, AMBER, alpha)
    text(d, (120, y + 34), title, "SemiBold", 52, INK, alpha)
    if sub:
        text(d, (120, y + 104), sub, "Regular", 28, MUTED, alpha)


def scrim(img, strength=0.55, top=False):
    """Darken the lower (or upper) part of a frame for legible text."""
    a = np.asarray(img).astype(np.float32)
    y = np.linspace(0, 1, H)[:, None, None]
    g = np.clip((y - 0.55) / 0.45, 0, 1) if not top else np.clip((0.35 - y) / 0.35, 0, 1)
    return Image.fromarray((a * (1 - strength * g)).astype(np.uint8))


# ------------------------------------------------------------------------------------------------ segments
def seg_title(fr):
    src = seq12(os.path.join(fr, "robot", "title_*.png"))
    for k, f in enumerate(src):
        t = k / FPS
        im = scrim(Image.fromarray(f), 0.6)

        def dr(d, l):
            a = ease((t - 0.8) / 1.0)
            text(d, (120, 760), "GYRA Mk2.1", "SemiBold", 110, INK, a)
            text(d, (124, 900), "An autonomous spherical field robot: design, physics, control", "Regular", 34, MUTED, ease((t - 1.6) / 1.0))
            text(d, (124, 960), "Simulated in MuJoCo  ·  rendered in Blender  ·  every number from the model", "Regular", 24, AMBER, ease((t - 2.4) / 1.0))
        yield overlay(im, dr)


ASM = [  # (t0, t1, anchor, label, sub, colour)
    (2.0, 6.0, "yoke", "Pendulum yoke", "two drive motors, belts and pinions", GREEN),
    (6.0, 10.0, "bob", "Battery bob", "936 Wh + tungsten ballast, 15.7 kg, hangs low", VIOLET),
    (10.5, 15.0, "lidar", "Spine and sensor pods", "carbon axle · 3-D LiDAR · 6 cameras · GNSS", CYAN),
    (15.5, 19.3, "ring_gears", "Tyre half", "GFRP drum, cast tread, internal ring gear", AMBER),
    (19.5, 23.5, "seam", "Second half", "joined by a wire-race slewing ring at the seam", AMBER),
]


def seg_assembly(fr):
    src = seq12(os.path.join(fr, "robot", "assembly_*.png"))
    d_robot = os.path.join(fr, "robot")
    for k, f in enumerate(src):
        t = k / FPS
        A = anchors_at(d_robot, "assembly", k)

        def dr(d, l):
            lower_third(d, "How it is built", "79 parts, five groups", None, win(t, 0.3, 2.2, 0.4), y=90)
            for t0, t1, key, lab, sub, col in ASM:
                callout(d, A.get(key), lab, sub, col, win(t, t0 + 0.4, t1, 0.4))
            a = ease((t - 24.5) / 0.8)
            for i, (lab, val) in enumerate((("width", "0.79 m"), ("height", "0.60 m"), ("mass", "45.8 kg"), ("battery", "936 Wh"))):
                x = 120 + i * 300
                panel(d, (x, 900, x + 270, 1000), a)
                text(d, (x + 22, 914), lab, "Medium", 22, MUTED, a)
                text(d, (x + 22, 946), val, "SemiBold", 38, AMBER, a)
        yield overlay(Image.fromarray(f), dr)


XR = [(1.0, 8.0, "drive", "Drive", "motor → belt → pinion → ring gear, one per tyre half", GREEN),
      (8.0, 13.5, "level", "Levelling motor", "holds the sensor spine level against the pendulum", AMBER),
      (13.5, 18.5, "lean", "Lean drive", "swings the battery bob sideways, 180 °/s", VIOLET),
      (18.5, 23.5, "lidar", "Sensors", "two Mid-360 LiDARs on the axle line, six cameras", CYAN)]


def seg_xray(fr):
    src = seq12(os.path.join(fr, "robot", "xray_*.png"))
    d_robot = os.path.join(fr, "robot")
    for k, f in enumerate(src):
        t = k / FPS
        A = anchors_at(d_robot, "xray", k)

        def dr(d, l):
            lower_third(d, "How the parts connect", "Everything hangs from one axle", None, win(t, 0.2, 24, 0.5), y=90)
            for t0, t1, key, lab, sub, col in XR:
                callout(d, A.get(key), lab, sub, col, win(t, t0 + 0.3, t1, 0.4))
        yield overlay(Image.fromarray(f), dr)


def seg_roll(fr):
    src = seq12(os.path.join(fr, "robot", "roll_*.png"))
    for k, f in enumerate(src):
        t = k / FPS
        im = scrim(Image.fromarray(f), 0.5)

        def dr(d, l):
            lower_third(d, "Principle", "The pendulum drives, the spine looks", "the tyre rolls around both", win(t, 0.4, 6.0, 0.5))
        yield overlay(im, dr)


def seg_png(dirn, scene):
    files = sorted(glob.glob(os.path.join(dirn, f"{scene}_*.png")))
    for p in files:
        yield Image.open(p).convert("RGB")


def seg_card(title, sub, lines, dur, kicker=None):
    global BG
    BG = BG or gradient()
    for k in range(int(dur * FPS)):
        t = k / FPS

        def dr(d, l):
            a = ease((t - 0.2) / 0.8)
            if kicker:
                text(d, (120, 360), kicker.upper(), "SemiBold", 24, AMBER, a)
            text(d, (120, 400), title, "SemiBold", 76, INK, a)
            if sub:
                text(d, (124, 510), sub, "Regular", 32, MUTED, ease((t - 0.6) / 0.8))
            for i, s in enumerate(lines):
                text(d, (124, 620 + i * 54), s, "Regular", 30, INK, ease((t - 1.2 - 0.5 * i) / 0.8))
        yield overlay(BG.copy(), dr)


ZONE = {"Rolling hills": "slopes up to 8° · boulders to weave around", "Pole forest": "poles 12–24 cm thick · gaps from 1.1 m",
        "Doorways": "three rooms · 1.1 m doors for a 0.79 m wide robot", "Open field": "long straights at up to 3 m/s",
        "Boulder garden": "blocks and boulders, tight lines", "Back over the hills": "the same slopes, the other way",
        "Final sprint": "across the open field to the finish"}
FAST = ("Doorways",)                                   # shown at 2x


def mission_hud():
    """Returns draw(img, t, fast) -> img with the HUD of mission time t (s from the start)."""
    M = np.load(os.path.join(ROOT, "media", "showcase.npz"), allow_pickle=True)
    Fm = M["frames"]
    hud = json.loads(str(M["hud"]))
    t0 = Fm[0, 0]
    ht = np.array([h["t"] for h in hud]) - t0
    occ, free = M["map_occ"], M["map_free"]
    wps = M["wps"]
    reached = np.array(M["reached"]) - t0
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Fm[:, 1:3], axis=0), axis=1))]
    ft = Fm[:, 0] - t0
    zone_start = {}
    for i, h in enumerate(hud):
        zone_start.setdefault(h["zone"], ht[i])

    def zone_at(t):
        return hud[min(int(np.searchsorted(ht, t)), len(hud) - 1)]["zone"]

    def draw(im, t, fast):
        i = min(int(np.searchsorted(ht, t)), len(hud) - 1)
        h = hud[i]
        j = min(int(np.searchsorted(ft, t)), len(ft) - 1)

        def dr(d, l):
            tz = t - zone_start.get(h["zone"], 0)
            a = win(tz, 0.3, 4.2 if not fast else 7.0, 0.5)
            if a > 0:
                w_ = max(font("SemiBold", 54).getlength(h["zone"]), font("Regular", 26).getlength(ZONE.get(h["zone"], ""))) + 60
                panel(d, (90, 70, 90 + w_, 220), a)
                text(d, (120, 88), h["zone"], "SemiBold", 54, INK, a)
                text(d, (122, 162), ZONE.get(h["zone"], ""), "Regular", 26, MUTED, a)
            panel(d, (90, 836, 700, 1010), 1.0)
            text(d, (120, 852), "SPEED", "SemiBold", 20, AMBER)
            text(d, (120, 878), f"{abs(h['v']):3.1f}", "SemiBold", 80, INK)
            text(d, (262, 930), "m/s", "Medium", 26, MUTED)
            frac = min(abs(h["v"]) / 3.5, 1.0)
            d.rounded_rectangle((120, 984, 330, 994), radius=5, fill=(58, 68, 102, 255))
            if frac > 0.02:
                d.rounded_rectangle((120, 984, 120 + 210 * frac, 994), radius=5, fill=(*AMBER, 255))
            text(d, (390, 852), "PENDULUM", "SemiBold", 20, VIOLET)
            text(d, (390, 878), f"{math.degrees(h['pend']):+4.0f}°", "SemiBold", 40, INK)
            text(d, (390, 932), "SENSOR SPINE PITCH", "SemiBold", 20, CYAN)
            text(d, (390, 958), f"{math.degrees(h['pitch']):+5.2f}°", "SemiBold", 40, INK)
            nwp = int((reached <= t).sum())
            panel(d, (1430, 70, 1830, 200), 1.0)
            text(d, (1460, 86), "WAYPOINT", "SemiBold", 20, AMBER)
            text(d, (1460, 112), f"{min(nwp + 1, len(wps))} / {len(wps)}", "SemiBold", 44, INK)
            text(d, (1660, 86), "DRIVEN", "SemiBold", 20, AMBER)
            text(d, (1660, 112), f"{dist[j]:.0f} m", "SemiBold", 44, INK)
            text(d, (1460, 168), f"mission time {t:5.1f} s" + ("   ·   2× speed" if fast else ""), "Regular", 20, MUTED)
            mx, my, ms = 1530, 660, 300
            panel(d, (mx - 22, my - 58, mx + ms + 22, my + ms + 58), 1.0)
            m_o = occ[min(i, len(occ) - 1)].astype(np.float32) / 255
            m_f = free[min(i, len(free) - 1)].astype(np.float32) / 255
            rgb = np.zeros((64, 64, 4), np.uint8)
            rgb[...] = (24, 30, 52, 215)
            rgb[m_f > 0.5] = (50, 64, 104, 235)
            rgb[m_o > 0.25] = (236, 239, 247, 255)
            # egocentric crop: rows = robot-frame y (left), cols = x (forward) -> show forward up, left left
            mp = Image.fromarray(rgb, "RGBA").rotate(90, expand=True).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mp = mp.resize((ms, ms), Image.Resampling.NEAREST)
            l.alpha_composite(mp, (mx, my))
            d.rounded_rectangle((mx - 2, my - 2, mx + ms + 2, my + ms + 2), radius=8, outline=(*AMBER, 200), width=2)
            cx, cy = mx + ms / 2, my + ms / 2
            d.polygon([(cx, cy - 16), (cx - 10, cy + 10), (cx + 10, cy + 10)], fill=(*AMBER, 255))
            text(d, (mx, my - 42), "ROBOT'S OWN MAP  ·  12.8 m", "SemiBold", 18, AMBER)
            if h["interv"]:
                d.ellipse((mx, my + ms + 18, mx + 14, my + ms + 32), fill=(*CORAL, 255))
                text(d, (mx + 24, my + ms + 13), "safety filter adjusting", "Medium", 18, CORAL)
        return overlay(im, dr)
    return draw, zone_at


def seg_mission(fr):
    draw, zone_at = mission_hud()
    src = seq12(os.path.join(fr, "mission", "mission_*.png"))
    for s, f in enumerate(src):
        t = s / FPS
        fast = zone_at(t) in FAST
        if fast and s % 2:                             # 2x through the doorways: drop every other frame
            continue
        yield draw(Image.fromarray(f), t, fast)


def segments(a):
    ex = a.explainer
    return [
        ("title", seg_title(a.frames)),
        ("assembly", seg_assembly(a.frames)),
        ("xray", seg_xray(a.frames)),
        ("roll", seg_roll(a.frames)),
        ("move", seg_png(ex, "move")),
        ("steer", seg_png(ex, "steer")),
        ("level", seg_png(ex, "level")),
        ("arch", seg_png(ex, "arch")),
        ("filter", seg_png(ex, "filter")),
        ("card_field", seg_card("In the field", "One continuous run, no human input",
                                ["238 m  ·  17 waypoints  ·  hills, poles, doorways, open field",
                                 "Wind on  ·  every command checked by the safety filter"], 6.0, kicker="Part two")),
        ("mission", seg_mission(a.frames)),
        ("card_end", seg_card("GYRA Mk2.1", "Learned locomotion, model-based navigation, a safety layer that can always stop",
                              a.results, 9.0, kicker="Results")),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--explainer", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "media", "gyra_mk21_film.mp4"))
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--only", default=None, help="comma-separated segment names (preview)")
    ap.add_argument("--results", default=None, help="json list of result lines for the end card")
    a = ap.parse_args()
    a.results = json.loads(a.results) if a.results else ["Field run: 238 m, 17 waypoints, no collisions"]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    held = []                                          # previous segment's last XFADE frames: they fade into this one
    total = 0
    only = set(a.only.split(",")) if a.only else None
    for name, gen in segments(a):
        if only and name not in only:
            continue
        buf, n = [], 0
        for img in gen:
            arr = np.asarray(img.convert("RGB"), np.uint8)
            if n < len(held):
                w = (n + 1) / (len(held) + 1)
                arr = (held[n].astype(np.float32) * (1 - w) + arr.astype(np.float32) * w).astype(np.uint8)
            buf.append(arr)
            if len(buf) > XFADE:
                enc.stdin.write(buf.pop(0).tobytes())
                total += 1
            n += 1
        held = buf
        print(f"segment {name}: {n} frames", flush=True)
    for arr in held:
        enc.stdin.write(arr.tobytes())
        total += 1
    enc.stdin.close()
    enc.wait()
    print(f"wrote {a.out}: {total} frames, {total / FPS:.1f} s")


if __name__ == "__main__":
    main()
