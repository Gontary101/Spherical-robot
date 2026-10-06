"""
Overlay a telemetry HUD + live LiDAR minimap on rendered mission frames and encode an H.264 MP4.

    python render/compose_video.py --frames <dir> --mission media/mission.npz --out media/gyra_mk2_mission.mp4
Frame files are f_NNNN.png; frame N shows simulation time (N-1)/24 s. Gaps in the numbering
(rendered with a frame step) play back faster and are labelled with a speed badge.
"""
import argparse
import glob
import json
import math
import os
import re

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 24
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
ACCENT = (255, 138, 51)
CYAN = (40, 200, 255)
GREEN = (60, 210, 120)


def rot_box(cx, cy, sx, sy, quat):
    w, x, y, z = quat
    yaw = 2 * math.atan2(z, w)
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + px * c - py * s, cy + px * s + py * c) for px, py in ((sx, sy), (-sx, sy), (-sx, -sy), (sx, -sy))]


class HUD:
    def __init__(self, mission, W, H):
        self.F = mission["frames"]
        self.scans = mission["scans"]
        self.beam = mission["beam_ang"]
        self.wps = mission["wps"]
        self.reached = list(mission["reached"])
        self.obs = json.loads(str(mission["obstacles"]))
        self.W, self.H = W, H
        s = W / 960
        self.s = s
        self.f_title = ImageFont.truetype(FONT_B, int(19 * s))
        self.f_sub = ImageFont.truetype(FONT, int(11.5 * s))
        self.f_lab = ImageFont.truetype(FONT, int(11 * s))
        self.f_val = ImageFont.truetype(FONT_M, int(17 * s))
        self.f_badge = ImageFont.truetype(FONT_B, int(16 * s))
        # minimap geometry
        self.mm = int(210 * s)
        self.mx0, self.my0 = W - self.mm - int(16 * s), int(16 * s)
        self.scale = self.mm / 15.0

    def w2m(self, x, y):
        return (self.mx0 + self.mm / 2 + x * self.scale, self.my0 + self.mm / 2 - y * self.scale)

    def idx(self, t):
        return min(int(round(t / 0.02)), len(self.F) - 1)

    def draw(self, im, t, speedup):
        d = ImageDraw.Draw(im, "RGBA")
        s = self.s
        i = self.idx(t)
        f = self.F[i]
        # ---- title panel
        d.rounded_rectangle([12 * s, 12 * s, 548 * s, 66 * s], radius=10 * s, fill=(12, 14, 18, 170))
        d.text((24 * s, 22 * s), "GYRA Mk2 · autonomous mission", font=self.f_title, fill=(245, 245, 245, 255))
        d.text((24 * s, 46 * s), "learned navigator (pod LiDAR + stereo depth + odometry) → learned locomotion · MuJoCo",
               font=self.f_sub, fill=(200, 205, 212, 255))
        # ---- telemetry panel
        k = sum(1 for r in self.reached if t >= r)
        wp_txt = f"{min(k + 1, len(self.wps))}/{len(self.wps)}" if k < len(self.wps) else "done"
        rows = [("speed", f"{abs(f[12]):4.2f} m/s"), ("yaw rate", f"{math.degrees(f[13]):+5.0f} °/s"),
                ("command", f"{f[15]:4.1f} m/s {math.degrees(f[16]):+4.0f} °/s"), ("waypoint", wp_txt),
                ("time", f"{t:5.1f} s")]
        y0 = self.H - (len(rows) * 24 + 22) * s
        d.rounded_rectangle([12 * s, y0, 262 * s, self.H - 12 * s], radius=10 * s, fill=(12, 14, 18, 170))
        for j, (lab, val) in enumerate(rows):
            yy = y0 + (12 + j * 24) * s
            d.text((24 * s, yy + 3 * s), lab.upper(), font=self.f_lab, fill=(160, 166, 175, 255))
            d.text((112 * s, yy), val, font=self.f_val, fill=(245, 245, 245, 255))
        # ---- minimap
        x0, y0m, mm = self.mx0, self.my0, self.mm
        d.rounded_rectangle([x0 - 6 * s, y0m - 6 * s, x0 + mm + 6 * s, y0m + mm + 22 * s], radius=10 * s, fill=(12, 14, 18, 185))
        for o in self.obs:
            p, sz = o["pos"], o["size"]
            if o["type"] == "cylinder":
                cx, cy = self.w2m(p[0], p[1])
                r = sz[0] * self.scale
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(150, 150, 150, 255))
            else:
                pts = [self.w2m(*q) for q in rot_box(p[0], p[1], sz[0], sz[1], o["quat"])]
                d.polygon(pts, fill=(120, 124, 130, 255) if o["wall"] else (150, 150, 150, 255))
        # path so far
        path = [self.w2m(*self.F[j, 1:3]) for j in range(0, i + 1, 5)]
        if len(path) > 1:
            d.line(path, fill=(90, 160, 255, 220), width=max(1, int(2 * s)))
        # waypoints
        for j, w in enumerate(self.wps):
            cx, cy = self.w2m(*w)
            done = j < len(self.reached) and t >= self.reached[j]
            active = j == k
            col = GREEN if done else (CYAN if active else (110, 115, 125))
            r = (6 + (2 * math.sin(t * 6) if active else 0)) * s
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=max(1, int(2 * s)))
            d.text((cx + 7 * s, cy - 14 * s), str(j + 1), font=self.f_lab, fill=col)
        # LiDAR returns from the latest navigation step (masked by the tyre blind wedge)
        si = np.searchsorted(self.scans[:, 0], t) - 1
        yaw = 2 * math.atan2(f[7], f[4])
        if si >= 0:
            sc = self.scans[si, 1:]
            hit = sc < 7.9
            for a, r in zip(self.beam[hit], sc[hit]):
                px, py = f[1] + r * math.cos(a + yaw), f[2] + r * math.sin(a + yaw)
                cx, cy = self.w2m(px, py)
                d.ellipse([cx - 1.8 * s, cy - 1.8 * s, cx + 1.8 * s, cy + 1.8 * s], fill=(255, 70, 60, 255))
        # robot
        cx, cy = self.w2m(f[1], f[2])
        tri = [(cx + 9 * s * math.cos(-yaw), cy + 9 * s * math.sin(-yaw)),
               (cx + 6 * s * math.cos(-yaw + 2.5), cy + 6 * s * math.sin(-yaw + 2.5)),
               (cx + 6 * s * math.cos(-yaw - 2.5), cy + 6 * s * math.sin(-yaw - 2.5))]
        d.polygon(tri, fill=ACCENT)
        d.text((x0, y0m + mm + 4 * s), "map · LiDAR returns (red)", font=self.f_lab, fill=(190, 195, 202, 255))
        if speedup > 1:
            txt = f"{speedup}× speed"
            tw = d.textlength(txt, font=self.f_badge)
            d.rounded_rectangle([self.W - tw - 40 * s, self.H - 50 * s, self.W - 12 * s, self.H - 14 * s], radius=8 * s,
                                fill=(255, 138, 51, 230))
            d.text((self.W - tw - 26 * s, self.H - 43 * s), txt, font=self.f_badge, fill=(20, 20, 20, 255))
        return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--mission", default="media/mission.npz")
    ap.add_argument("--out", default="media/gyra_mk2_mission.mp4")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds to hold the last frame")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.frames, "f_*.png")), key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[0]))
    nums = [int(re.findall(r"(\d+)", os.path.basename(p))[0]) for p in files]
    mission = np.load(a.mission)
    W, H = Image.open(files[0]).size
    hud = HUD(mission, W, H)
    w = imageio_ffmpeg.write_frames(a.out, (W, H), fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    w.send(None)
    last = None
    for j, (p, n) in enumerate(zip(files, nums)):
        step = (nums[j + 1] - n) if j + 1 < len(nums) else 1
        im = hud.draw(Image.open(p).convert("RGB"), (n - 1) / FPS, step)
        last = np.asarray(im)
        w.send(last.tobytes())
    for _ in range(int(a.hold * FPS)):
        w.send(last.tobytes())
    w.close()
    print("wrote", a.out, len(files), "frames")


if __name__ == "__main__":
    main()
