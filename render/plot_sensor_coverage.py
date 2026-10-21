"""
Sensor-coverage sheet from render/camera_views.py output: the six pod-camera views and a top-down map of the floor
each sensor group can see past the tyre.

    python render/plot_sensor_coverage.py --dir <dir> --tag current --out <png> [--title ...]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for w in ("Regular", "Medium", "SemiBold"):
    font_manager.fontManager.addfont(os.path.join(HERE, "fonts", f"Poppins-{w}.ttf"))
plt.rcParams.update({"font.family": "Poppins", "font.size": 10, "axes.edgecolor": "#c9c8c0", "axes.labelcolor": "#4a4a45",
                     "xtick.color": "#6b6b64", "ytick.color": "#6b6b64", "axes.titleweight": "medium"})

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#1f1f1c", "#4a4a45", "#8a8a82"
BOTH, LIDAR, CAM, NONE = "#2a78d6", "#eb6834", "#1baf7a", "#e4e3dc"      # categorical slots 1-3 + neutral
CAMS = [("L0", "left pod, front"), ("L1", "left pod, rear"), ("L2", "left pod, ground"),
        ("R0", "right pod, front"), ("R1", "right pod, rear"), ("R2", "right pod, ground")]
COURSE = [("low box", 0.60, 0.0, 0.15, 0.25), ("low box", 1.20, 0.0, 0.15, 0.25), ("low box", 2.00, 0.0, 0.15, 0.25),
          ("pole", 1.50, 0.0, 0.08, 0.08), ("person", 3.00, 0.15, 0.35, 0.45), ("box", 1.2, 1.2, 0.3, 0.3),
          ("box", 1.2, -1.2, 0.3, 0.3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", default="current")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="What the build's sensors see")
    ap.add_argument("--sub", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    z = np.load(os.path.join(a.dir, f"{a.tag}_coverage.npz"))
    xs, ys, names, vis = z["xs"], z["ys"], list(z["names"]), z["vis"]
    cam = np.zeros(vis.shape[1], bool)
    lid = np.zeros(vis.shape[1], bool)
    for n, v in zip(names, vis):
        if n.startswith("cam"):
            cam |= v
        else:
            lid |= v
    cls = np.where(cam & lid, 0, np.where(lid, 1, np.where(cam, 2, 3))).reshape(len(ys), len(xs))
    # blind area ahead of the tyre: |y| < 0.4, 0.3 < x < 3
    gx, gy = np.meshgrid(xs, ys)
    ahead = (gx > 0.3) & (gx < 3.0) & (np.abs(gy) < 0.4)
    blind_ahead = (cls == 3) & ahead
    first_seen = np.nan
    cen = np.abs(ys) < 0.021
    row = cls[cen][0]
    seen = np.nonzero((xs > 0.3) & (row != 3))[0]
    if len(seen):
        first_seen = xs[seen[0]]

    fig = plt.figure(figsize=(15, 13.2), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 2.35], hspace=0.28, wspace=0.06, left=0.04, right=0.97, top=0.9, bottom=0.04)
    fig.text(0.04, 0.965, a.title, fontsize=20, fontweight="semibold", color=INK)
    fig.text(0.04, 0.938, a.sub or "Six pod cameras (OV9281, 120 deg) and two Livox Mid-360s, as placed in the build CAD. "
             "Test course: low boxes 0.6 / 1.2 / 2.0 m ahead, a pole at 1.5 m, a person at 3 m, boxes to each side.",
             fontsize=11, color=INK2)
    cams = CAMS
    cj = os.path.join(a.dir, f"{a.tag}_cams.json")
    if os.path.exists(cj):
        import json
        side_name = {"L": "left pod", "R": "right pod"}
        cams = [(c["name"].split()[0], f"{side_name[c['name'][0]]}, {' '.join(c['name'].split()[1:])}") for c in json.load(open(cj))]
    for i, (k, lab) in enumerate(cams):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        p = os.path.join(a.dir, f"{a.tag}_cam_{k}.png")
        if os.path.exists(p):
            ax.imshow(plt.imread(p))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{k}  {lab}", loc="left", fontsize=11.5, color=INK, pad=6)
    ax = fig.add_subplot(gs[2, :2])
    cmap = ListedColormap([BOTH, LIDAR, CAM, NONE])
    ax.imshow(cls, origin="lower", extent=(xs[0] - 0.02, xs[-1] + 0.02, ys[0] - 0.02, ys[-1] + 0.02), cmap=cmap,
              vmin=-0.5, vmax=3.5, interpolation="nearest")
    # robot: tyre pair (top view) and pods
    ax.add_patch(FancyBboxPatch((-0.3, -0.386), 0.6, 0.772, boxstyle="round,pad=0,rounding_size=0.12", fc="#3a3a36", ec=SURFACE, lw=2))
    for sg in (1, -1):
        ax.add_patch(Rectangle((-0.155, sg * 0.386 - (0.03 if sg > 0 else 0)), 0.31, 0.03, fc="#9a9a92", ec=SURFACE, lw=1))
    ax.annotate("", xy=(0.62, 0), xytext=(0.12, 0), arrowprops=dict(arrowstyle="-|>", color=SURFACE, lw=2))
    ax.text(0, -0.55, "robot (top view)", ha="center", va="top", fontsize=9.5, color=INK2)
    for nm, x, y, sx, sy in COURSE:
        ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, fc="none", ec=INK, lw=1.3))
    ax.text(0.6, -0.3, "low boxes 0.6 / 1.2 / 2.0 m", fontsize=9, color=INK, va="top")
    ax.text(1.5, 0.2, "pole", fontsize=9, color=INK, ha="center")
    ax.text(3.0, 0.48, "person", fontsize=9, color=INK, ha="center")
    for r in (1, 2, 3):
        t = np.linspace(-np.pi / 2, np.pi / 2, 200)
        ax.plot(r * np.cos(t), r * np.sin(t), color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.text(r * np.cos(0.55), r * np.sin(0.55) + 0.05, f"{r} m", fontsize=9, color=MUTED)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("ahead (m)")
    ax.set_ylabel("left (m)")
    ax.set_title("Floor the sensors can see past the tyre", loc="left", fontsize=13, color=INK, pad=8)
    side = fig.add_subplot(gs[2, 2])
    side.axis("off")
    side.legend(handles=[Patch(fc=BOTH, label="camera + LiDAR"), Patch(fc=LIDAR, label="LiDAR only"),
                         Patch(fc=CAM, label="camera only"), Patch(fc=NONE, ec="#c9c8c0", label="not seen")],
                loc="upper left", frameon=False, fontsize=11.5, title="floor seen by", title_fontsize=11.5,
                alignment="left")
    fs = f"{first_seen:.1f} m" if np.isfinite(first_seen) else "beyond 4 m"
    lines = [("Straight ahead, the floor is first seen at", INK2, 11), (fs, INK, 22),
             ("Unseen floor in the lane 0.3–3 m ahead", INK2, 11), (f"{blind_ahead.sum() * 0.04 ** 2:.2f} m²", INK, 22)]
    yy = 0.66
    for t, c, f in lines:
        side.text(0.02, yy, t, transform=side.transAxes, color=c, fontsize=f, va="top",
                  fontweight="semibold" if f > 15 else "normal")
        yy -= 0.06 if f < 15 else 0.1
    if a.note:
        import textwrap
        side.text(0.02, yy - 0.02, "\n".join(textwrap.wrap(a.note, 42)), transform=side.transAxes, color=INK2,
                  fontsize=10.5, va="top", linespacing=1.4)
    fig.savefig(a.out, dpi=110, facecolor=SURFACE)
    print(a.out, "first seen ahead", first_seen, "blind m2", blind_ahead.sum() * 0.04 ** 2)


if __name__ == "__main__":
    main()
