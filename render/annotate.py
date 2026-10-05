"""Draw call-out labels on a render using the *_labels.json written by render_gyra.py.

    python render/annotate.py media/gyra_cutaway.png media/gyra_cutaway_annotated.png [keys...]
"""
import json
import sys

from PIL import Image, ImageDraw, ImageFont

TEXT = {
    "tyre": "Rubber tyre: cast-PU chevron tread on a GFRP drum",
    "pod": "Sensor pod (non-rotating, on the axle)",
    "lidar": "Livox Mid-360 LiDAR",
    "cmg": "CMG flywheel 10 krpm (x2, counter-rotating)",
    "gimbal": "Gimbal actuator (scissored pair)",
    "ring": "Equatorial ring gear m1.5 x 360T",
    "drive": "Drive motor 6374 (x2) + 2:1 belt",
    "pinion": "24T pinion (x2, anti-backlash pair)",
    "bob": "Lean bob: 8.4 kg lead",
    "battery": "13S2P battery, 468 Wh",
    "yoke": "Pendulum yoke + curved lean rails",
    "axle": "CFRP spine axle (kept level)",
    "bay": "Compute bay (Jetson Orin NX)",
    "level": "Spine levelling motor",
    "rollers": "Rim V-rollers + V-ring seal",
    "camera": "Front stereo camera (0.56 m baseline)",
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def main():
    src, dst = sys.argv[1], sys.argv[2]
    keys = sys.argv[3:] or list(TEXT)
    im = Image.open(src).convert("RGB")
    W, H = im.size
    pts = json.load(open(src.replace(".png", "_labels.json")))
    s = W / 1920
    f = ImageFont.truetype(FONT, int(25 * s))
    dr = ImageDraw.Draw(im, "RGBA")
    items = [(k, pts[k]) for k in keys if k in pts and pts[k][2] > 0 and 0 < pts[k][0] < W and 0 < pts[k][1] < H]
    cx = W * 0.5
    for side in ("L", "R"):
        grp = sorted([it for it in items if (it[1][0] < cx) == (side == "L")], key=lambda it: it[1][1])
        if not grp:
            continue
        gap = 64 * s
        ys = []
        top = 90 * s
        for _, p in grp:
            y = max(p[1], (ys[-1] + gap) if ys else top)
            ys.append(y)
        overflow = ys[-1] - (H - 60 * s)
        if overflow > 0:
            ys = [y - overflow for y in ys]
        for (k, p), y in zip(grp, ys):
            txt = TEXT[k]
            tw = dr.textlength(txt, font=f)
            if side == "L":
                tx = 36 * s
                anchor_x = tx + tw + 14 * s
            else:
                tx = W - 36 * s - tw
                anchor_x = tx - 14 * s
            col = (255, 122, 26, 255)
            dr.line([(p[0], p[1]), (anchor_x, y)], fill=(20, 20, 22, 230), width=max(2, int(2.4 * s)))
            r = 7 * s
            dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=col, outline=(20, 20, 22, 255), width=2)
            pad = 8 * s
            dr.rounded_rectangle([tx - pad, y - 18 * s, tx + tw + pad, y + 18 * s], radius=8 * s, fill=(255, 255, 255, 225),
                                 outline=(20, 20, 22, 60))
            dr.text((tx, y), txt, font=f, fill=(15, 15, 18, 255), anchor="lm")
    fb = ImageFont.truetype(FONT_B, int(34 * s))
    dr.text((36 * s, 36 * s), "GYRA Mk1", font=fb, fill=(15, 15, 18, 255), anchor="lm")
    im.save(dst)
    print("wrote", dst)


if __name__ == "__main__":
    main()
