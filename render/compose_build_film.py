"""
Composition of the GYRA Mk2.1 build film: Blender frames from render/build_film.py + overlays (labels with leader
lines, step captions, titles, readouts, the verification card) -> H.264, 24 fps. Frames that a step-2 segment did
not render are synthesised from their neighbours by bidirectional optical flow (OpenCV DIS). Poppins only, upright
weights (same palette as the earlier film).

    python render/compose_build_film.py --frames <dir> --out media/gyra_mk21_build_film.mp4 [--crf 16]
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
FD = os.path.join(HERE, "fonts")
FPS = 24
INK, MUTED = (236, 239, 247), (154, 163, 189)
PANEL = (12, 18, 38)
COL = dict(amber=(255, 181, 71), cyan=(70, 199, 255), green=(75, 214, 155), violet=(168, 139, 255), coral=(255, 107, 94))
CUTS_FADE = 0.42                                         # s, fade through dark at hard cuts
CUT_AFTER = {"explode", "assembly", "drive", "pendulum", "stops", "steer", "lean", "pods", "power", "verify"}
_fc = {}


def font(w, size):
    k = (w, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(os.path.join(FD, f"Poppins-{w}.ttf"), size)
    return _fc[k]


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Comp:
    def __init__(self, frames, W, H):
        self.dir = frames
        self.W, self.H = W, H
        self.s = H / 1080.0
        meta = json.load(open(os.path.join(frames, "film.json")))
        self.N = meta["nframes"]
        self.segs = meta["segs"]
        self.rendered = sorted(int(os.path.basename(p)[2:7]) for p in glob.glob(os.path.join(frames, "f_*.png")))
        self.rset = set(self.rendered)
        self.cache = {}
        self.dis = None
        self.vign = self.vignette()

    # ---------------------------------------------------------------- images
    def load(self, f):
        if f not in self.cache:
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
            im = Image.open(os.path.join(self.dir, f"f_{f:05d}.png")).convert("RGB")
            if im.size != (self.W, self.H):
                im = im.resize((self.W, self.H), Image.Resampling.LANCZOS)
            self.cache[f] = np.asarray(im)
        return self.cache[f]

    def seg(self, f):
        for s in self.segs:
            if s["f0"] <= f < s["f0"] + s["n"]:
                return s
        return self.segs[-1]

    def neighbours(self, f):
        s = self.seg(f)
        lo, hi = s["f0"], s["f0"] + s["n"] - 1
        a = max([r for r in self.rendered if lo <= r <= f], default=None)
        b = min([r for r in self.rendered if f <= r <= hi], default=None)
        return a, b

    def interp(self, A, B, w):
        import cv2
        if self.dis is None:
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        ga, gb = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY), cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
        fab = self.dis.calc(ga, gb, None)
        fba = self.dis.calc(gb, ga, None)
        h, w_ = ga.shape
        yy, xx = np.mgrid[0:h, 0:w_].astype(np.float32)
        wa = cv2.remap(A, xx - w * fab[..., 0], yy - w * fab[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(B, xx - (1 - w) * fba[..., 0], yy - (1 - w) * fba[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return ((1 - w) * wa.astype(np.float32) + w * wb.astype(np.float32)).clip(0, 255).astype(np.uint8)

    def base(self, f):
        if f in self.rset:
            return self.load(f)
        a, b = self.neighbours(f)
        if a is None and b is None:
            return np.zeros((self.H, self.W, 3), np.uint8)
        if a is None or b is None or a == b:
            return self.load(a if a is not None else b)
        return self.interp(self.load(a), self.load(b), (f - a) / (b - a))

    def vignette(self):
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        v = np.clip(((xx / self.W - 0.5) ** 2 / 0.55 + (yy / self.H - 0.5) ** 2 / 0.75), 0, 1) * 0.28
        return (1 - v)[..., None].astype(np.float32)

    # ---------------------------------------------------------------- overlays (layout in 1080p units)
    def P(self, v):
        return v * self.s

    def txt(self, d, xy, s, w, size, color, a=1.0, anchor="la"):
        if a <= 0.003 or not s:
            return
        d.text((self.P(xy[0]), self.P(xy[1])), s, font=font(w, max(1, round(size * self.s))), fill=(*color, int(255 * a)), anchor=anchor)

    def tw(self, s, w, size):
        return font(w, max(1, round(size * self.s))).getlength(s) / self.s

    def labels(self, d, labels, top_right=60):
        placed = []
        for lb in labels:
            a = lb["alpha"]
            col = COL.get(lb["color"], INK)
            ax, ay = lb["xy"][0] * 1920, lb["xy"][1] * 1080
            tw = max(self.tw(lb["title"], "SemiBold", 28), self.tw(lb.get("sub", ""), "Regular", 20)) + 36
            th = 86 if lb.get("sub") else 54
            right = ax >= 1920 * 0.5                         # callout columns at the sides, away from the subject
            px = 1920 - 70 - tw if right else 70
            py = ay - th / 2
            py = min(max(py, top_right if right else 60), 1080 - 280 - th)      # right column clears a readout pill
            for (qx, qy, qw, qh) in placed:                 # keep panels apart (stable order)
                if px < qx + qw + 12 and qx < px + tw + 12 and py < qy + qh + 12 and qy < py + th + 12:
                    py = qy + qh + 14 if py >= qy else qy - th - 14
            placed.append((px, py, tw, th))
            # leader + dot
            ex = px if right else px + tw
            ey = py + th / 2
            d.line([(self.P(ax), self.P(ay)), (self.P(ex), self.P(ey))], fill=(*col, int(200 * a)), width=max(1, round(2 * self.s)))
            r = 6
            d.ellipse([self.P(ax - r - 4), self.P(ay - r - 4), self.P(ax + r + 4), self.P(ay + r + 4)], outline=(*col, int(170 * a)),
                      width=max(1, round(2 * self.s)))
            d.ellipse([self.P(ax - r + 1), self.P(ay - r + 1), self.P(ax + r - 1), self.P(ay + r - 1)], fill=(*col, int(255 * a)))
            d.rounded_rectangle([self.P(px), self.P(py), self.P(px + tw), self.P(py + th)], radius=self.P(12), fill=(*PANEL, int(215 * a)),
                                outline=(*col, int(90 * a)), width=max(1, round(1.5 * self.s)))
            self.txt(d, (px + 18, py + 12), lb["title"], "SemiBold", 28, col, a)
            self.txt(d, (px + 18, py + 52), lb.get("sub", ""), "Regular", 20, MUTED, a)

    def caption(self, d, c):
        a = c["a"]
        x, y = 92, 1080 - 92
        self.txt(d, (x + 22, y - 112), c.get("kicker", ""), "SemiBold", 20, COL["amber"], a)
        self.txt(d, (x + 22, y - 86), c["title"], "SemiBold", 42, INK, a)
        self.txt(d, (x + 22, y - 30), c.get("sub", ""), "Regular", 24, MUTED, a)
        d.rounded_rectangle([self.P(x), self.P(y - 110), self.P(x + 5), self.P(y + 2)], radius=self.P(3), fill=(*COL["amber"], int(255 * a)))

    def title(self, d, c):
        a = c["a"]
        x, y = 120, 690
        self.txt(d, (x, y), c["text"], "SemiBold", 104, INK, a)
        self.txt(d, (x + 4, y + 132), c.get("sub", ""), "Regular", 36, MUTED, a)
        d.rounded_rectangle([self.P(x + 4), self.P(y - 26), self.P(x + 124), self.P(y - 20)], radius=self.P(3), fill=(*COL["amber"], int(255 * a)))

    def readout(self, d, r):
        a = r["a"]
        s = r["text"]
        w = self.tw(s, "Medium", 30) + 44
        x, y = 1920 - 80 - w, 70
        d.rounded_rectangle([self.P(x), self.P(y), self.P(x + w), self.P(y + 58)], radius=self.P(29), fill=(*PANEL, int(200 * a)),
                            outline=(*COL["amber"], int(120 * a)), width=max(1, round(1.5 * self.s)))
        self.txt(d, (x + 22, y + 11), s, "Medium", 30, INK, a)

    def card(self, d, c):
        a = c["a"]
        rows = c["rows"]
        w, h = 1040, 150 + 74 * len(rows)
        x, y = (1920 - w) / 2 + 360, (1080 - h) / 2
        d.rounded_rectangle([self.P(x), self.P(y), self.P(x + w), self.P(y + h)], radius=self.P(20), fill=(*PANEL, int(222 * a)),
                            outline=(*COL["amber"], int(90 * a)), width=max(1, round(1.5 * self.s)))
        self.txt(d, (x + 48, y + 40), c["title"], "SemiBold", 36, INK, a)
        for i, (num, txt_) in enumerate(rows):
            yy = y + 120 + 74 * i
            self.txt(d, (x + 48, yy), num, "SemiBold", 40, COL["green"] if num.startswith("0") else COL["amber"], a)
            self.txt(d, (x + 240, yy + 8), txt_, "Regular", 26, INK, a)

    # ---------------------------------------------------------------- frame
    def fade(self, f):
        """Brightness factor for fades through dark at hard cuts, start and end."""
        t = f / FPS
        k = 1.0
        k = min(k, ease(t / 1.0))
        k = min(k, ease((self.N - 1 - f) / FPS / 1.2))
        for s in self.segs:
            if s["name"] in CUT_AFTER:
                tb = (s["f0"] + s["n"]) / FPS
                k = min(k, ease(abs(t - tb) / CUTS_FADE) if abs(t - tb) < CUTS_FADE else 1.0)
        return k

    def frame(self, f):
        img = self.base(f).astype(np.float32) * self.vign
        js = json.load(open(os.path.join(self.dir, f"f_{f:05d}.json")))
        im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        c = js.get("caption")
        if c and c["a"] > 0.003:                          # soft shade behind the caption
            sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
            sd = ImageDraw.Draw(sh)
            sd.rectangle([0, self.P(1080 - 300), self.P(1150), self.P(1080)], fill=(4, 8, 20, int(150 * c["a"])))
            sh = sh.filter(ImageFilter.GaussianBlur(self.P(80)))
            layer.alpha_composite(sh)
        tr = any(r.get("pos", "tr") == "tr" for r in (js.get("readouts") or []))     # for the whole shot: no jump
        self.labels(d, js.get("labels") or [], top_right=150 if tr else 60)
        if c and c["a"] > 0.003:
            self.caption(d, c)
        if js.get("title") and js["title"]["a"] > 0.003:
            sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
            ImageDraw.Draw(sh).rectangle([0, self.P(560), self.P(1100), self.P(940)], fill=(4, 8, 20, int(150 * js["title"]["a"])))
            layer.alpha_composite(sh.filter(ImageFilter.GaussianBlur(self.P(90))))
            self.title(d, js["title"])
        for r in js.get("readouts") or []:
            if r["a"] > 0.003:
                self.readout(d, r)
        for cd in js.get("cards") or []:
            if cd["a"] > 0.003:
                self.card(d, cd)
        im = im.convert("RGBA")
        im.alpha_composite(layer)
        out = np.asarray(im.convert("RGB")).astype(np.float32)
        k = self.fade(f)
        bg = np.array([10, 15, 30], np.float32)
        return (out * k + bg * (1 - k)).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "media", "gyra_mk21_build_film.mp4"))
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--stills", default="", help="comma list of frames to write as PNG instead of encoding")
    a = ap.parse_args()
    W, H = (int(v) for v in a.res.split("x"))
    C = Comp(a.frames, W, H)
    if a.stills:
        for f in (int(x) for x in a.stills.split(",")):
            Image.fromarray(C.frame(f)).save(os.path.join(os.path.dirname(a.out) or ".", f"still_{f:05d}.png"))
        return
    end = C.N if a.end < 0 else a.end
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(a.start, end):
        p.stdin.write(C.frame(f).tobytes())
        if f % 240 == 0:
            print(f"composed {f}/{end}", flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
