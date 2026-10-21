"""
As-built camera tables for the navigation simulator (rl/nav2_env.py, GYRA_SENSORS=asbuilt / asbuilt_lane).

render/camera_views.py ray-casts, against the build CAD's meshes, which floor points each pod camera sees past the
tyre. This keeps the floor points between R_MIN and R_MAX (a 0.1 m grid, the camera that sees each one) and the camera
positions, so the simulator can cast each camera's real sight lines into its world:

    python rl/sensor_tables.py --cov <dir>/<tag>_coverage.npz --cams <dir>/<tag>_cams.json --out rl/sensors/<name>.npz
"""
import argparse
import json

import numpy as np

R_MIN, R_MAX, SUB = 0.30, 2.5, 3          # cameras: floor / obstacle segmentation out to 2.5 m; every 3rd
                                           # point of the 0.04 m coverage grid (0.12 m; the env jitters by +-0.06 m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", required=True)
    ap.add_argument("--cams", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.cov)
    xs, ys, names, vis = z["xs"], z["ys"], list(z["names"]), z["vis"]
    cams = json.load(open(a.cams))
    gx, gy = np.meshgrid(xs, ys)
    ix, iy = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)))
    gx, gy = gx.ravel(), gy.ravel()
    r = np.hypot(gx, gy)
    on_grid = ((ix % SUB == 0) & (iy % SUB == 0)).ravel()
    sel = on_grid & (r > R_MIN) & (r < R_MAX)
    cam_rows = [i for i, n in enumerate(names) if n.startswith("cam")]
    cam_idx = np.full(len(gx), -1)
    for k, row in enumerate(cam_rows):               # first camera (in table order) that sees the point
        cam_idx = np.where((cam_idx < 0) & vis[row], k, cam_idx)
    keep = sel & (cam_idx >= 0)
    pos = np.array([c["pos"] for c in cams])
    np.savez_compressed(a.out, pts=np.c_[gx[keep], gy[keep]], cam=cam_idx[keep], cam_pos=pos,
                        names=np.array([c["name"] for c in cams]))
    print(a.out, int(keep.sum()), "floor points of", int(sel.sum()), "in", R_MIN, "-", R_MAX, "m")


if __name__ == "__main__":
    main()
