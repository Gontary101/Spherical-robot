"""
Render one navigation episode with the deployed stack to an MP4 (MuJoCo offscreen, chase camera, 25 fps).

    MUJOCO_GL=osmesa python rl/render_episode.py "rooms, 1.0 m doors" 8 --seed0 100 --out media/clip_door.mp4

The overlay shows the scenario, time, speed, the navigator's mode (drive / door crossing / recovery) and whether the
safety filter corrected the command. Needs an offscreen OpenGL (MUJOCO_GL=egl or osmesa) and ffmpeg.
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("GYRA_SENSORS", "asbuilt_lane")
sys.path.insert(0, os.path.dirname(__file__))
import mujoco  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from eval_nav2 import CALM, LOCO, SCEN, SCEN_EXTRA  # noqa: E402
from nav2_env import Nav2Env  # noqa: E402
from teacher import GeoMPC  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FONT = os.path.join(ROOT, "render/fonts/Poppins-Bold.ttf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("ep", type=int)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--low", default=os.path.join(ROOT, "rl/runs/loco31/best.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=540)
    ap.add_argument("--max-s", type=float, default=60.0)
    a = ap.parse_args()
    sd = dict(SCEN.get(a.scenario) or SCEN_EXTRA[a.scenario])
    env = Nav2Env(seed=70000 + a.seed0 + a.ep, low_ckpt=a.low, level=0.8, adaptive=False,
                  scenario={"nav": CALM | {k: v for k, v in sd.items() if k != "loco"}, "loco": LOCO | sd.get("loco", {})},
                  shield=True)
    oa, _ = env.reset()
    nav = GeoMPC(deployable=True)
    m, d = env.m, env.d
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, a.w), max(m.vis.global_.offheight, a.h)
    r = mujoco.Renderer(m, a.h, a.w)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1                                 # the robot's geoms live in group 3
    m.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
    m.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = 4.2, -42.0
    font = ImageFont.truetype(FONT, 20) if os.path.exists(FONT) else ImageFont.load_default()
    small = ImageFont.truetype(FONT, 16) if os.path.exists(FONT) else font
    tmp = a.out + ".frames"
    os.makedirs(tmp, exist_ok=True)
    st = dict(n=0, k=0, az=None, mode="drive", interv=False, info={})

    lo = env.loco
    step0 = lo.step

    def step(act):                                       # render every other low-level step: 25 fps of sim time
        out = step0(act)
        st["k"] += 1
        if st["k"] % 2 == 0:
            frame()
        return out
    lo.step = step

    def frame():
        t = lo._truth()
        pos = d.qpos[0:3]
        az = math.degrees(t["yaw"]) + 180.0
        if st["az"] is None:
            st["az"] = az
        da = (az - st["az"] + 180.0) % 360.0 - 180.0
        st["az"] += 0.04 * da                            # slow chase: the camera turns behind the robot
        cam.azimuth = st["az"]
        cam.lookat[:] = [pos[0], pos[1], pos[2] + 0.2]
        r.update_scene(d, cam, opt)
        img = Image.fromarray(r.render())
        g = ImageDraw.Draw(img, "RGBA")
        g.rectangle([0, 0, a.w, 38], fill=(0, 0, 0, 150))
        g.text((14, 7), f"GYRA Mk2.1  |  {a.scenario}", font=font, fill=(255, 255, 255))
        mode = st["mode"].replace("_", " ")
        txt = f"t {d.time:5.1f} s   v {t['v']:+.2f} m/s   {mode}" + ("   filter corrected" if st["interv"] else "")
        g.rectangle([0, a.h - 32, a.w, a.h], fill=(0, 0, 0, 150))
        g.text((14, a.h - 28), txt, font=small, fill=(255, 220, 120) if st["interv"] else (255, 255, 255))
        img.save(os.path.join(tmp, f"{st['n']:05d}.png"))
        st["n"] += 1

    while d.time < a.max_s:
        (oa, _), _, done, info = env.step(nav(oa, env))
        st["mode"], st["interv"] = nav.mode, bool(info.get("intervened", False))
        if done:
            break
    res = "success" if info.get("success") else "collision" if info.get("collision") else "timeout" if info.get("timeout") else "end"
    for _ in range(25):                                  # hold the last frame 1 s
        frame()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25", "-i", os.path.join(tmp, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", a.out], check=True)
    subprocess.run(["rm", "-rf", tmp])
    print(f"{a.scenario} ep {a.ep}: {res} after {d.time:.1f} s -> {a.out} ({st['n']} frames)")


if __name__ == "__main__":
    main()
