"""
Explainer animations for the GYRA Mk2.1 video (physics + control), drawn with matplotlib in Poppins (upright
weights only). 1920 x 1080, 24 fps, one PNG per frame.

    python render/explainer.py --scene move --out <dir>          # scenes: move steer level arch filter
    python render/explainer.py --scene move --out <dir> --only 120  # a single frame (preview)

Every number on screen comes from the CAD mass properties or the simulation:
  pendulum m*L = 15.69 kg * 0.205 m (bob) + 6.71 kg * 0.128 m (yoke) = 4.07 kg m  ->  m g L = 39.9 N m
  tyre radius 0.30 m, robot 45.8 kg, crown offset 90 mm (contact patches 180 mm apart)
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FONT_DIR = os.path.join(HERE, "fonts")
for f in os.listdir(FONT_DIR):
    if f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONT_DIR, f))
# Poppins, upright only; Greek letters (Poppins has none) fall back to the upright DejaVu Sans
_WEIGHT = dict(Light=300, Regular=400, Medium=500, SemiBold=600, Bold=700)
FP = {w: font_manager.FontProperties(family=["Poppins", "DejaVu Sans"], weight=n, style="normal") for w, n in _WEIGHT.items()}

FPS = 24
W, H = 1920, 1080
DPI = 100                                             # 100 -> 1920x1080; --scale 2 -> 3840x2160 (same layout)
# palette (dark navy studio, amber = GYRA's status light ring)
BG0, BG1 = "#0A0F1E", "#141C35"
INK, MUTED, FAINT = "#ECEFF7", "#9AA3BD", "#3A4466"
AMBER, CYAN, CORAL, GREEN, VIOLET = "#FFB547", "#46C7FF", "#FF6B5E", "#4BD69B", "#A88BFF"
TYRE, STEEL, BOB = "#2B3352", "#7C87A8", "#C9CFE0"

# physical constants (CAD / simulation)
R_TYRE, M_ROBOT, MGL = 0.30, 45.8, 39.9
M_BOB, L_BOB = 15.69, 0.205
CROWN_D = 0.090


# ------------------------------------------------------------------------------------------------ helpers
def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def win(t, t0, t1, fade=0.5):
    """Opacity of an element visible in [t0, t1] with fades."""
    if t < t0 or t > t1:
        return 0.0
    return min(ease((t - t0) / fade), ease((t1 - t) / fade))


def figure():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=DPI)
    bg = fig.add_axes([0, 0, 1, 1])
    g = np.linspace(0, 1, 256)[:, None] * np.ones((1, 4))
    c0, c1 = np.array(matplotlib.colors.to_rgb(BG0)), np.array(matplotlib.colors.to_rgb(BG1))
    img = (c0[None, None] * (1 - g[..., None]) + c1[None, None] * g[..., None])[:, :1].repeat(4, 1)
    bg.imshow(img, aspect="auto", extent=[0, 1, 0, 1], origin="lower")
    # soft vignette
    yy, xx = np.mgrid[0:1:108j, 0:1:192j]
    v = np.clip(((xx - 0.5) ** 2 / 0.35 + (yy - 0.55) ** 2 / 0.6), 0, 1) * 0.35
    bg.imshow(np.dstack([np.zeros_like(v)] * 3 + [v]), aspect="auto", extent=[0, 1, 0, 1], origin="lower")
    bg.set_axis_off()
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    return fig, bg


def text(ax, x, y, s, size, weight="Regular", color=INK, alpha=1.0, ha="left", va="baseline", **kw):
    if alpha <= 0.001:
        return None
    return ax.text(x, y, s, fontproperties=FP[weight], fontsize=size, color=color, alpha=alpha, ha=ha, va=va,
                   transform=kw.pop("transform", ax.transAxes), **kw)


def header(ax, t, kicker, title, sub=None, t_in=0.0):
    a = ease((t - t_in) / 0.8)
    dy = (1 - a) * 0.012
    text(ax, 0.0625, 0.905 - dy, kicker.upper(), 15, "SemiBold", AMBER, a)
    text(ax, 0.0625, 0.845 - dy, title, 40, "SemiBold", INK, a)
    if sub:
        text(ax, 0.0625, 0.80 - dy, sub, 19, "Regular", MUTED, a)


def caption(ax, t, items, y=0.085):
    """items: [(t0, t1, text)] - the caption line at the bottom, cross-fading."""
    for t0, t1, s in items:
        a = win(t, t0, t1, 0.4)
        if a > 0:
            text(ax, 0.0625, y, s, 22, "Regular", INK, a)


def chip(ax, x, y, label, value, a, color=AMBER, w=0.17):
    if a <= 0:
        return
    ax.add_patch(FancyBboxPatch((x, y), w, 0.085, boxstyle="round,pad=0.004,rounding_size=0.012", transform=ax.transAxes,
                                facecolor="#1B2442", edgecolor=FAINT, lw=1.2, alpha=a))
    text(ax, x + 0.012, y + 0.052, label, 14, "Medium", MUTED, a)
    text(ax, x + 0.012, y + 0.016, value, 24, "SemiBold", color, a)


def arrow(ax, p0, p1, color, lw=4, a=1.0, ms=26, style="-|>", rad=0.0, z=5):
    if a <= 0:
        return
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, alpha=a,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def curved_torque(ax, c, r, a0, a1, color, a=1.0, lw=4, z=6):
    """Circular arrow (torque) around c from angle a0 to a1 (deg)."""
    if a <= 0:
        return
    th = np.radians(np.linspace(a0, a1, 40))
    ax.plot(c[0] + r * np.cos(th[:-2]), c[1] + r * np.sin(th[:-2]), color=color, lw=lw, alpha=a, zorder=z, solid_capstyle="round")
    arrow(ax, (c[0] + r * np.cos(th[-3]), c[1] + r * np.sin(th[-3])), (c[0] + r * np.cos(th[-1]), c[1] + r * np.sin(th[-1])),
          color, lw=lw, a=a, ms=22, z=z)


def scene_axes(fig, rect, xlim, ylim):
    ax = fig.add_axes(rect)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def draw_side_robot(ax, x, theta, phi, level_err=0.0, a=1.0, bob_alpha=1.0, show_spine=True):
    """Side view at ground contact (x, 0): tyre (radius R), axle at (x, R), pendulum at angle theta (rad, positive =
    swung forward), tread marks rotated by phi, sensor spine (pods) drawn level (tilted by level_err)."""
    c = (x, R_TYRE)
    ax.add_patch(Circle(c, R_TYRE, facecolor=TYRE, edgecolor=STEEL, lw=2.5, alpha=a, zorder=2))
    for k in range(12):                                    # tread lugs
        ang = phi + k * np.pi / 6
        p0 = (c[0] + 0.27 * math.cos(ang), c[1] + 0.27 * math.sin(ang))
        p1 = (c[0] + 0.30 * math.cos(ang), c[1] + 0.30 * math.sin(ang))
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#56618A", lw=3, alpha=a, zorder=3)
    # pendulum: yoke arm + bob, hanging down when theta = 0, swinging forward (+x) for positive theta
    bx, by = c[0] + L_BOB * math.sin(theta), c[1] - L_BOB * math.cos(theta)
    ax.plot([c[0], bx], [c[1], by], color=STEEL, lw=6, alpha=a * bob_alpha, zorder=4, solid_capstyle="round")
    ax.add_patch(Circle((bx, by), 0.062, facecolor=BOB, edgecolor="#E9EDF7", lw=1.5, alpha=a * bob_alpha, zorder=5))
    if show_spine:                                         # spine with pod (sensor platform), level
        th = level_err
        L = 0.17
        p0 = (c[0] - L * math.cos(th), c[1] - L * math.sin(th))
        p1 = (c[0] + L * math.cos(th), c[1] + L * math.sin(th))
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=AMBER, lw=7, alpha=a, zorder=6, solid_capstyle="round")
        ax.add_patch(Circle((p1[0], p1[1] + 0.035), 0.028, facecolor="#20283F", edgecolor=AMBER, lw=2, alpha=a, zorder=7))
    ax.add_patch(Circle(c, 0.022, facecolor=INK, edgecolor="none", alpha=a, zorder=8))
    return c, (bx, by)


def ground(ax, x0, x1, a=1.0, offset=0.0):
    """Ground line with hatching that scrolls by -offset (the camera follows the robot)."""
    ax.plot([x0, x1], [0, 0], color="#4A557A", lw=2.5, alpha=a, zorder=1)
    for x in np.arange(x0 - 0.2, x1 + 0.2, 0.1) - (offset % 0.1):
        ax.plot([x, x - 0.04], [0, -0.04], color="#2C3557", lw=1.5, alpha=a, zorder=1)


def pendulum_motion(t, t_on=6.0, t_off=27.0, a_max=1.0):
    """Acceleration profile and the resulting pendulum angle and distance (theta balances the push + rolling loss)."""
    def acc(u):
        return a_max * ease((u - t_on) / 1.5) * (1 - ease((u - t_off) / 1.5))
    tt = np.linspace(0, max(t, 1e-3), 300)
    av = np.array([acc(u) for u in tt])
    v = np.cumsum(av) * (tt[1] - tt[0])
    xs = float(np.sum(v) * (tt[1] - tt[0]))
    a = acc(t)
    theta = math.asin(min(1.0, (M_ROBOT * a + 5.0 * min(1.0, float(v[-1]))) * R_TYRE / MGL))
    return a, float(v[-1]), xs, theta


# ------------------------------------------------------------------------------------------------ scene: move
def scene_move(t, fig, bg):
    """Pendulum drive: torque pair, swing, gravity balance, force; 30 s."""
    header(bg, t, "How it moves", "A pendulum inside a tyre", "Two motors on the hanging pendulum turn the tyre around it")
    ax = scene_axes(fig, [0.0, 0.12, 0.6, 0.66], (-0.95, 0.95), (-0.25, 0.95))
    acc, v, xs, theta = pendulum_motion(t)
    ground(ax, -0.95, 0.95, offset=xs)
    c, b = draw_side_robot(ax, 0.0, theta, -xs / R_TYRE)
    a_tq = win(t, 8.0, 30, 0.6)
    if a_tq > 0:                                           # motor torque on the tyre, reaction on the pendulum
        curved_torque(ax, c, 0.39, 150, 30, GREEN, a_tq)
        text(ax, c[0] + 0.13, c[1] + 0.43, "τ  on the tyre", 18, "Medium", GREEN, a_tq, transform=ax.transData)
        ar = a_tq * win(t, 8.0, 15.0, 0.6)
        curved_torque(ax, c, 0.12, -150, -40, CORAL, ar, lw=3.5)
        text(ax, c[0] - 0.62, c[1] - 0.3, "reaction  −τ\nswings the pendulum", 16, "Medium", CORAL, ar, transform=ax.transData,
             va="center")
    a_g = win(t, 14, 30, 0.6)
    if a_g > 0:                                            # gravity on the bob, lever arm, angle
        arrow(ax, (b[0], b[1] - 0.065), (b[0], b[1] - 0.2), VIOLET, lw=4, a=a_g)
        text(ax, b[0] + 0.03, b[1] - 0.18, "m·g", 18, "Medium", VIOLET, a_g, transform=ax.transData)
        ax.plot([c[0], c[0]], [c[1], c[1] - 0.26], color=MUTED, lw=1.5, alpha=a_g, ls=(0, (3, 3)), zorder=9)
        ax.add_patch(Wedge(c, 0.15, -90, -90 + math.degrees(theta), width=0.006, color=MUTED, alpha=a_g, zorder=9))
        text(ax, c[0] + 0.035, c[1] - 0.12, "θ", 18, "Medium", INK, a_g, transform=ax.transData)
    a_f = win(t, 18, 30, 0.6)
    if a_f > 0:                                            # forward force at the contact
        arrow(ax, (0.0, 0.0), (0.38, 0.0), AMBER, lw=5, a=a_f)
        text(ax, 0.40, 0.025, "F = τ / R", 18, "Medium", AMBER, a_f, transform=ax.transData)
    # right panel
    eq = [(8.5, "The motors ride on the pendulum", "and drive the tyre."),
          (11.5, "Their reaction swings", "the pendulum forward."),
          (15, "Gravity holds it there:", "τ = m·g·L·sin θ"),
          (19, "The tyre pushes the ground:", "F = τ / R")]
    for i, (t0, s1, s2) in enumerate(eq):
        a = ease((t - t0) / 0.6)
        y = 0.70 - i * 0.105
        text(bg, 0.62, y, s1, 20, "Regular", MUTED if i < 2 else INK, a)
        text(bg, 0.62, y - 0.042, s2, 20 if i < 2 else 24, "Regular" if i < 2 else "SemiBold",
             MUTED if i < 2 else (AMBER if i == 3 else VIOLET), a)
    text(bg, 0.62, 0.235, f"θ = {math.degrees(theta):4.1f}°     a = {acc:3.1f} m/s²", 22, "Medium", INK, win(t, 15, 30, 0.6))
    chip(bg, 0.62, 0.09, "pendulum  m·g·L", f"{MGL:.1f} N·m", ease((t - 22) / 0.6), w=0.115)
    chip(bg, 0.745, 0.09, "max push (θ = 90°)", f"{MGL / R_TYRE:.0f} N", ease((t - 23) / 0.6), w=0.115)
    chip(bg, 0.87, 0.09, "static climb", "17°", ease((t - 24) / 0.6), color=CYAN, w=0.085)


# ------------------------------------------------------------------------------------------------ scene: steer
def _rot(pts, psi, pos):
    c, s_ = math.cos(psi), math.sin(psi)
    return np.c_[pos[0] + c * pts[:, 0] - s_ * pts[:, 1], pos[1] + s_ * pts[:, 0] + c * pts[:, 1]]


def _rounded_rect(x0, x1, y0, y1, r, n=8):
    pts = []
    for cx, cy, a0 in ((x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270)):
        th = np.radians(np.linspace(a0, a0 + 90, n))
        pts += list(zip(cx + r * np.cos(th), cy + r * np.sin(th)))
    return np.array(pts)


def draw_top_robot(ax, pos, psi, a=1.0):
    """Top view: two tyre halves (each 0.60 long, ~0.30 wide), sensor pods outside them, contact patches 180 mm apart."""
    for sgn in (1, -1):
        half = _rounded_rect(-0.30, 0.30, 0.02, 0.325, 0.14)
        half[:, 1] *= sgn
        ax.add_patch(Polygon(_rot(half, psi, pos), closed=True, facecolor=TYRE, edgecolor=STEEL, lw=2, alpha=a, zorder=3))
        for xg in np.linspace(-0.22, 0.22, 7):                        # tread lugs
            seg = np.array([[xg, 0.06], [xg + 0.03, 0.29]]) * [1, sgn]
            pr = _rot(seg, psi, pos)
            ax.plot(pr[:, 0], pr[:, 1], color="#56618A", lw=2, alpha=a, zorder=4)
        pod = _rounded_rect(-0.17, 0.17, 0.335, 0.393, 0.025)
        pod[:, 1] *= sgn
        ax.add_patch(Polygon(_rot(pod, psi, pos), closed=True, facecolor="#20283F", edgecolor=AMBER, lw=2, alpha=a, zorder=4))
    for sgn in (1, -1):
        cp = _rot(np.array([[0.0, sgn * CROWN_D]]), psi, pos)[0]
        ax.add_patch(Circle(cp, 0.028, facecolor=INK, alpha=a, zorder=6))
    nose = _rot(np.array([[0.42, 0.0], [0.34, 0.05], [0.34, -0.05]]), psi, pos)
    ax.add_patch(Polygon(nose, closed=True, facecolor=AMBER, alpha=a, zorder=6))


def scene_steer(t, fig, bg):
    """Differential drive + counter-lean; 26 s."""
    header(bg, t, "How it steers", "Two tyre halves, two motors", "Same torque goes straight; opposite torque spins it in place")
    ax = scene_axes(fig, [0.0, 0.1, 0.55, 0.68], (-1.0, 1.0), (-0.85, 0.95))
    # phase A: straight (0-9 s), phase B: spin (9-17 s), phase C: drive a curve (17-26 s)
    if t < 9:
        psi, pos = math.pi / 2, (0.0, -0.45 + 0.9 * ((max(t - 2, 0) * 0.12) % 1.0) if t > 2 else -0.45)
        fl = fr = 1.0 if t > 2 else 0.0
    elif t < 17:
        psi, pos = math.pi / 2 + 1.6 * ease((t - 10) / 6.5) * math.pi, (0.0, 0.0)
        fl, fr = (-1.0, 1.0) if t > 9.8 else (0.0, 0.0)
    else:
        u = ease((t - 17.5) / 8.0)
        psi, pos = math.pi / 2 + 1.2 * u, (-0.55 + 0.55 * math.cos(1.2 * u), 0.55 * math.sin(1.2 * u) - 0.3)
        pos = (0.55 * (1 - math.cos(1.2 * u)) - 0.3, 0.55 * math.sin(1.2 * u) - 0.35)
        fl, fr = 0.6, 1.0
    draw_top_robot(ax, pos, psi)
    for sgn, F in ((1, fl), (-1, fr)):                    # forces at the contact patches (left = +y body side)
        if abs(F) < 1e-3:
            continue
        cp = _rot(np.array([[0.0, sgn * CROWN_D * 1.0]]), psi, pos)[0]
        d = np.array([math.cos(psi), math.sin(psi)]) * 0.35 * F
        arrow(ax, cp, cp + d, GREEN if F > 0 else CORAL, lw=4, a=1.0, ms=22, z=8)
    lab = [(2, 9, "Equal torque on both halves  →  forward"), (9.8, 17, "Opposite torques  →  a pure yaw couple"),
           (17.5, 26, "Unequal torques  →  a curve")]
    for t0, t1, s_ in lab:
        text(bg, 0.0625, 0.13, s_, 22, "Medium", INK, win(t, t0, t1, 0.4))
    # right panel text
    rows = [(3, "Sum of the torques", "swings the pendulum: drive"),
            (10, "Difference of the torques", "cancels inside the pendulum"),
            (12, "Force difference × offset turns it", "contact patches 90 mm either side of centre")]
    for i, (t0, a1, a2) in enumerate(rows):
        a = min(ease((t - t0) / 0.6), 1 - ease((t - 17.0) / 0.5))
        text(bg, 0.6, 0.70 - i * 0.105, a1, 21, "SemiBold" if i == 2 else "Regular", AMBER if i == 2 else INK, a)
        text(bg, 0.6, 0.70 - i * 0.105 - 0.042, a2, 19, "Regular", MUTED, a)
    chip(bg, 0.6, 0.335, "turn in place (sim)", "258 °/s", min(ease((t - 14) / 0.6), 1 - ease((t - 17.0) / 0.5)), w=0.13)
    chip(bg, 0.745, 0.335, "drift per turn", "1 mm", min(ease((t - 15) / 0.6), 1 - ease((t - 17.0) / 0.5)), w=0.11, color=CYAN)
    # counter-lean, front view (phase C)
    a_c = win(t, 17.5, 26, 0.6)
    if a_c > 0:
        ax2 = scene_axes(fig, [0.6, 0.12, 0.34, 0.44], (-0.55, 0.55), (-0.05, 0.72))
        phi = math.radians(28) * ease((t - 18.5) / 1.5)
        th = np.linspace(0, 2 * np.pi, 400)
        for sgn in (1, -1):                               # twin crowns: spheres offset +-90 mm, clipped at the width
            ys = sgn * CROWN_D + 0.3 * np.cos(th)
            zs = 0.3 + 0.3 * np.sin(th)
            keep = (sgn * ys >= 0.012) & (np.abs(ys) <= 0.325)
            pts = np.c_[ys[keep], zs[keep]]
            pts = pts[np.argsort(np.arctan2(pts[:, 1] - 0.3, pts[:, 0] - sgn * 0.16))]
            ax2.add_patch(Polygon(pts, closed=True, facecolor=TYRE, edgecolor=STEEL, lw=2.5, alpha=a_c, zorder=2))
            ax2.add_patch(Rectangle((sgn * 0.335 - (0.058 if sgn < 0 else 0), 0.14), 0.058, 0.32, facecolor="#20283F",
                                    edgecolor=AMBER, lw=2, alpha=a_c, zorder=3))
        ax2.plot([-0.55, 0.55], [0, 0], color="#4A557A", lw=2, alpha=a_c)
        bx, bz = -L_BOB * math.sin(phi), 0.3 - L_BOB * math.cos(phi)
        ax2.plot([0, bx], [0.3, bz], color=STEEL, lw=5, alpha=a_c, zorder=6)
        ax2.add_patch(Circle((bx, bz), 0.055, facecolor=BOB, alpha=a_c, zorder=7))
        ax2.plot([-0.335, 0.335], [0.3, 0.3], color=AMBER, lw=4, alpha=a_c, zorder=5)
        if phi > 0.05:
            ax2.add_patch(Wedge((0, 0.3), 0.12, 270 - math.degrees(phi), 270, width=0.005, color=INK, alpha=a_c, zorder=8))
            text(ax2, -0.1, 0.1, "φ", 18, "Medium", INK, a_c, transform=ax2.transData)
        arrow(ax2, (0.0, 0.62), (-0.3, 0.62), CYAN, lw=3, a=a_c * ease((t - 19) / 0.6), ms=18)
        text(ax2, -0.3, 0.66, "turn centre", 14, "Medium", CYAN, a_c * ease((t - 19) / 0.6), transform=ax2.transData)
        text(bg, 0.6, 0.70, "In a curve the battery bob leans inward,", 21, "Regular", INK, a_c)
        text(bg, 0.6, 0.658, "like a cyclist; the twin crowns keep the tyre upright", 19, "Regular", MUTED, a_c)
        text(bg, 0.6, 0.6, "tan φ ≈ v·ω / g   ·   lean drive 180 °/s", 21, "SemiBold", AMBER, a_c)


# ------------------------------------------------------------------------------------------------ scene: level
_SHOW = None


def showcase_data():
    global _SHOW
    if _SHOW is None:
        d = np.load(os.path.join(ROOT, "media", "showcase.npz"), allow_pickle=True)
        Hh = json.loads(str(d["hud"]))
        _SHOW = dict(t=np.array([h["t"] for h in Hh]), pend=np.degrees([h["pend"] for h in Hh]),
                     pitch=np.degrees([h["pitch"] for h in Hh]), roll=np.degrees([h["roll"] for h in Hh]),
                     v=np.array([h["v"] for h in Hh]))
    return _SHOW


def scene_level(t, fig, bg):
    """The sensor spine stays level while the pendulum swings: levelling loop + recorded data; 28 s."""
    header(bg, t, "How it stays level", "A level sensor platform on a swinging pendulum",
           "The spine carries the LiDARs and cameras; a levelling motor holds it against the pendulum")
    ax = scene_axes(fig, [0.0, 0.14, 0.42, 0.6], (-0.62, 0.62), (-0.2, 0.95))
    _, v, xs, theta = pendulum_motion(t, 3.0, 9.5, 1.3)
    theta += 0.25 * math.sin(2.2 * t) * ease((t - 11) / 1.0)          # later: rough ground keeps it swinging
    ground(ax, -0.62, 0.62, offset=xs)
    a_ghost = win(t, 4.5, 10.5, 0.6)
    c, b = draw_side_robot(ax, 0.0, theta, -xs / R_TYRE)
    if a_ghost > 0:                                                   # what the spine would do if it were dragged along
        L = 0.17
        ax.plot([-L * math.cos(theta), L * math.cos(theta)], [R_TYRE - L * math.sin(theta), R_TYRE + L * math.sin(theta)],
                color=CORAL, lw=4, ls=(0, (4, 3)), alpha=a_ghost, zorder=9)
        text(ax, -0.6, 0.86, "without levelling: tilts with the pendulum", 15, "Medium", CORAL, a_ghost, transform=ax.transData)
    text(ax, -0.6, 0.78, "levelling motor holds the spine", 15, "Medium", AMBER, win(t, 6.5, 28, 0.6), transform=ax.transData)
    # control law
    text(bg, 0.0625, 0.135, "u = Kp·φ + Kd·dφ/dt + Ki·∫φ dt", 24, "SemiBold", AMBER, ease((t - 8) / 0.6))
    text(bg, 0.0625, 0.095, "φ = spine pitch from the IMU, 50 Hz; the integral removes any steady tilt", 18, "Regular", MUTED,
         ease((t - 8.5) / 0.6))
    # recorded data, hills section of the showcase mission
    D = showcase_data()
    sel = (D["t"] > 4.0) & (D["t"] < 34.0)
    tt, pend, pitch = D["t"][sel] - 4.0, D["pend"][sel], D["pitch"][sel]
    reveal = ease((t - 11) / 12.0)
    n = max(2, int(len(tt) * reveal))
    a_p = ease((t - 10.5) / 0.6)
    if a_p > 0:
        for k, (rect, y, lo, hi, col, lab) in enumerate([([0.47, 0.44, 0.47, 0.24], pend, -120, 100, VIOLET, "pendulum angle (°)"),
                                                         ([0.47, 0.15, 0.47, 0.22], pitch, -2, 2, AMBER, "sensor spine pitch (°)")]):
            axp = fig.add_axes(rect)
            axp.set_facecolor((0, 0, 0, 0))
            axp.set_xlim(0, 30)
            axp.set_ylim(lo, hi)
            for sp in axp.spines.values():
                sp.set_color(FAINT)
            axp.tick_params(colors=MUTED, labelsize=12)
            for lb in axp.get_xticklabels() + axp.get_yticklabels():
                lb.set_fontproperties(FP["Regular"])
                lb.set_fontsize(12)
            axp.grid(color=FAINT, lw=0.6, alpha=0.6)
            axp.plot(tt[:n], y[:n], color=col, lw=2.2, alpha=a_p)
            axp.patch.set_alpha(0)
            text(axp, 0.01, 1.04, lab, 16, "Medium", col, a_p)
            if k == 1:
                text(axp, 0.99, -0.2, "time on the hills (s) — recorded in the field mission below", 13, "Regular", MUTED, a_p, ha="right")
        a_r = ease((t - 21) / 0.6)
        text(bg, 0.47, 0.735, f"pendulum swings {np.percentile(np.abs(pend), 99):.0f}°  ·  spine stays within "
             f"{np.percentile(np.abs(pitch), 99):.2f}°", 20, "SemiBold", INK, a_r)


# ------------------------------------------------------------------------------------------------ scene: arch
def box(ax, x, y, w, h, title, lines, color, a):
    if a <= 0:
        return
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.014", transform=ax.transAxes,
                                facecolor="#18213D", edgecolor=color, lw=2, alpha=a))
    text(ax, x + 0.012, y + h - 0.038, title, 17, "SemiBold", color, a)
    for i, ln in enumerate(lines):
        text(ax, x + 0.012, y + h - 0.075 - i * 0.031, ln, 13.5, "Regular", INK, a)


def scene_arch(t, fig, bg):
    """Autonomy and control stack; 26 s."""
    header(bg, t, "How it decides", "From photons to motor torque", "Every layer runs on the robot (Jetson Orin NX on the pendulum)")
    B = [  # (t_in, x, y, w, h, title, lines, color)
        (1.0, 0.0625, 0.50, 0.15, 0.2, "Sensors", ["2 × 3-D LiDAR (pods)", "6 cameras + depth", "IMU · encoders · GNSS"], CYAN),
        (2.5, 0.235, 0.50, 0.165, 0.2, "Where am I", ["LiDAR-inertial odometry", "localised on the site map", "scan matching fallback"], CYAN),
        (4.0, 0.42, 0.50, 0.165, 0.2, "What is around", ["5 cm occupancy map", "people tracked + predicted", "dust & ghost filtering"], CYAN),
        (5.5, 0.605, 0.50, 0.15, 0.2, "Navigator", ["model-predictive control", "221 candidate commands", "1.5 s ahead · 10 Hz"], AMBER),
        (7.0, 0.775, 0.50, 0.16, 0.2, "Safety filter", ["simulates reaction + stop", "measured error margins", "looks around when blind"], CORAL),
        (9.0, 0.605, 0.18, 0.33, 0.2, "Locomotion policy  ·  50 Hz", ["reinforcement learning (PPO) on top of", "a classical cascade: speed → pendulum angle →",
                                                                         "torque · yaw rate → torque difference · lean"], GREEN),
        (11.0, 0.29, 0.18, 0.28, 0.2, "Actuators", ["2 drive motors (one per tyre half)", "levelling motor (spine)", "lean drive (battery bob)"], VIOLET),
    ]
    for t0, x, y, w, h, ttl, lines, col in B:
        box(bg, x, y, w, h, ttl, lines, col, ease((t - t0) / 0.6))
    arrows = [(1.6, (0.2125, 0.6), (0.235, 0.6)), (3.1, (0.4, 0.6), (0.42, 0.6)), (4.6, (0.585, 0.6), (0.605, 0.6)),
              (6.1, (0.755, 0.6), (0.775, 0.6)), (8.0, (0.855, 0.5), (0.855, 0.38)), (10.0, (0.605, 0.28), (0.57, 0.28))]
    for t0, p0, p1 in arrows:
        arrow(bg, p0, p1, MUTED, lw=2.5, a=ease((t - t0) / 0.5), ms=18)
        bg.patches[-1].set_transform(bg.transAxes) if bg.patches else None
    text(bg, 0.86, 0.43, "safe (v, ω)", 15, "Medium", CORAL, ease((t - 8.2) / 0.5))
    notes = [(13, "Learning is used where it beats hand design: balancing the pendulum on rough ground, wind and pushes."),
             (16.5, "Model-based layers decide where to go and guarantee every command can still be stopped."),
             (20, "Benchmark: 88.9 % of 208 test missions reach the goal; 0.5 % of missions end in a contact the robot caused.")]
    for i, (t0, s_) in enumerate(notes):
        text(bg, 0.0625, 0.12 - i * 0.0 + (0.0 if i == 0 else 0.0), "", 1)
    for i, (t0, s_) in enumerate(notes):
        a = win(t, t0, 26 if i == 2 else notes[i + 1][0] if i + 1 < len(notes) else 26, 0.5) if i < 2 else ease((t - t0) / 0.6)
        text(bg, 0.0625, 0.085, s_, 19, "Regular", INK, a)


# ------------------------------------------------------------------------------------------------ scene: filter
_SNAP = None


def scene_filter(t, fig, bg):
    """The real safety filter at one moment in a pole forest: every candidate is simulated and checked; 22 s."""
    global _SNAP
    if _SNAP is None:
        _SNAP = dict(np.load(os.path.join(ROOT, "media", "filter_snapshot.npz")))
    S_ = _SNAP
    header(bg, t, "The safety filter", "Every command must be stoppable",
           "Real data: the robot's own map and the filter's predictions, 1.3 m/s through a pole forest")
    loc = S_["loc"]
    ax = scene_axes(fig, [0.03, 0.06, 0.55, 0.7], (-3.0, 3.0), (-1.4, 3.8))
    # map in the robot frame (robot at origin, heading +y)
    L, org, cell = S_["L"], S_["origin"], float(S_["cell"])
    n = L.shape[0]
    xs = org[0] + (np.arange(n) + 0.5) * cell
    X, Y = np.meshgrid(xs, xs)
    c, s_ = math.cos(math.pi / 2 - loc[2]), math.sin(math.pi / 2 - loc[2])
    RX, RY = c * (X - loc[0]) - s_ * (Y - loc[1]), s_ * (X - loc[0]) + c * (Y - loc[1])
    sel = (np.abs(RX) < 3.2) & (RY > -1.6) & (RY < 4.0)
    occ, unk = sel & (L > 0.6), sel & (L > -0.3) & (L <= 0.6)
    a_m = ease((t - 0.5) / 1.0)
    ax.scatter(RX[unk], RY[unk], s=3, c="#2A3253", marker="s", alpha=a_m, lw=0)
    ax.scatter(RX[occ], RY[occ], s=6, c="#D8DDEA", marker="s", alpha=a_m, lw=0)
    ax.add_patch(Circle((0, 0), float(S_["robot_r"]), facecolor="none", edgecolor=AMBER, lw=2.5, alpha=a_m, zorder=6))
    ax.add_patch(Polygon([[0, 0.45], [-0.08, 0.32], [0.08, 0.32]], closed=True, facecolor=AMBER, alpha=a_m, zorder=6))
    P, ok = S_["paths"], S_["ok"].astype(bool)
    order = np.argsort(np.abs(S_["cmds"][:, 1]) + 0.01 * S_["cmds"][:, 0])
    shown = int(len(order) * ease((t - 3.0) / 7.0))
    for i in order[:shown]:
        pr = P[i] - loc[:2]
        px, py = c * pr[:, 0] - s_ * pr[:, 1], s_ * pr[:, 0] + c * pr[:, 1]
        ax.plot(px, py, color=GREEN if ok[i] else CORAL, lw=1.6 if ok[i] else 1.1, alpha=0.85 if ok[i] else 0.45, zorder=4)
    a_b = ease((t - 11) / 0.8)
    if a_b > 0:                                       # one admissible path with its footprint along the stop
        i = order[np.flatnonzero(ok[order])[len(np.flatnonzero(ok[order])) // 2]]
        pr = P[i] - loc[:2]
        px, py = c * pr[:, 0] - s_ * pr[:, 1], s_ * pr[:, 0] + c * pr[:, 1]
        for k in range(0, len(px), 3):
            ax.add_patch(Circle((px[k], py[k]), float(S_["robot_r"]), facecolor=GREEN, edgecolor="none", alpha=0.07 * a_b, zorder=3))
        ax.plot(px, py, color=AMBER, lw=3.5, alpha=a_b, zorder=7)
    rows = [(3.2, "Each candidate command is simulated:", "0.32 s of reaction, then the worst-case stop."),
            (7.0, "Rejected if the footprint, grown by the", "measured error, touches an obstacle or unseen space."),
            (11.0, "The navigator may only pick from the", "green set; the stop is always still possible."),
            (15.0, "When only unseen space blocks the way,", "the robot turns in place to look.")]
    for k, (t0, a1, a2) in enumerate(rows):
        a = ease((t - t0) / 0.6)
        text(bg, 0.59, 0.66 - k * 0.12, a1, 19, "Regular", INK, a)
        text(bg, 0.59, 0.66 - k * 0.12 - 0.04, a2, 19, "Regular", MUTED, a)
    a_l = ease((t - 9.5) / 0.6)
    text(bg, 0.59, 0.12, f"{ok.sum()} admissible", 24, "SemiBold", GREEN, a_l)
    text(bg, 0.73, 0.12, f"{(~ok).sum()} rejected", 24, "SemiBold", CORAL, a_l)


# ------------------------------------------------------------------------------------------------ driver
SCENES = dict(move=(scene_move, 30.0), steer=(scene_steer, 26.0), level=(scene_level, 28.0), arch=(scene_arch, 26.0),
              filter=(scene_filter, 22.0))


def render(scene, out, only=None):
    fn, dur = SCENES[scene]
    os.makedirs(out, exist_ok=True)
    frames = [only] if only is not None else range(int(dur * FPS))
    for f in frames:
        path = os.path.join(out, f"{scene}_{f:05d}.png")
        if only is None and os.path.exists(path):
            continue
        fig, bg = figure()
        fn(f / FPS, fig, bg)
        fig.savefig(path, dpi=DPI, facecolor=BG0)
        plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="output size x 1920x1080 (2 = 4K)")
    a = ap.parse_args()
    DPI = round(100 * a.scale)
    render(a.scene, a.out, a.only)
