"""
GYRA Mk2 - MuJoCo model generated from cad/out2/mass_properties.json.

    spine (free)                          axle + pods + bays (IMU lives here)
      tyreL  hinge +y   sphere r=.30 centred (0, +D, 0)   left half, crown offset outward
      tyreR  hinge +y   sphere r=.30 centred (0, -D, 0)   right half
      yoke   hinge +y   (unlimited, 360 deg)               pendulum pitch body, 2 drive motors
        bob  hinge +x   (+-40 deg)                         tungsten + 936 Wh battery

Actuators: driveL = torque between yoke and tyreL, driveR = torque between yoke and tyreR,
level = spine<->yoke levelling motor, lean = worm-driven bob position servo.
Motor torque-speed limits are applied by the controller / env (see MotorModel).
"""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MP2 = json.load(open(os.path.join(ROOT, "cad", "out2", "mass_properties.json")))
R = 0.300
D = 0.050
POD_PROFILE = json.load(open(os.path.join(ROOT, "cad", "out2", "pod_profile.json")))  # CAD pod radius per |y| slice (m)

# contact groups: robot tyres (1) collide with world (2) only
TYRE_CT, TYRE_CA = 1, 2
WORLD_CT, WORLD_CA = 2, 1


def inertial(body, scale_mass=1.0, com_shift=(0, 0, 0)):
    b = MP2["bodies"][body]
    c = np.array(b["com_mm"]) / 1000.0 + np.array(com_shift)
    I = np.array(b["I_com_kgm2"]) * scale_mass
    fi = f"{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} {I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}"
    return f'<inertial pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" mass="{b["mass"] * scale_mass:.5f}" fullinertia="{fi}"/>'


class MotorModel:
    """6374 150 KV at 48 V through 24:1 (pinion 15:1 x belt 1.6:1), 90 % efficient."""
    GEAR = 24.0
    TAU_STALL = 3.8 * 24.0 * 0.9            # N m at the tyre (60 A current limit)
    W_NL = 150 * 48 * 2 * np.pi / 60 / 24.0  # tyre no-load speed relative to the yoke, rad/s

    @classmethod
    def limit(cls, tau, w_rel, strength=1.0):
        """Clip a commanded tyre torque to the motor's torque-speed envelope."""
        t_max = cls.TAU_STALL * strength
        same = np.sign(tau) == np.sign(w_rel)
        avail = np.where(same, t_max * np.clip(1 - np.abs(w_rel) / cls.W_NL, 0, 1), t_max)
        return np.clip(tau, -avail, avail)

    @classmethod
    def limit_s(cls, tau, w_rel, strength=1.0):
        """Scalar fast path of limit()."""
        t_max = cls.TAU_STALL * strength
        if (tau > 0) == (w_rel > 0) and tau != 0 and w_rel != 0:
            f = 1 - abs(w_rel) / cls.W_NL
            avail = t_max * (0.0 if f < 0 else (1.0 if f > 1 else f))
        else:
            avail = t_max
        return -avail if tau < -avail else (avail if tau > avail else tau)


def obstacle_vertices(typ, size, n_seg=64):
    """Convex vertex set of an obstacle in its own frame (the single source for collision AND rendering).
    cylinder: size = (radius, half-height), n_seg-sided prism.  box: size = half-extents (x, y, z)."""
    if typ == "cylinder":
        r, h = size[0], size[1]
        a = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        ring = np.stack([r * np.cos(a), r * np.sin(a)], 1)
        return np.concatenate([np.c_[ring, np.full(n_seg, -h)], np.c_[ring, np.full(n_seg, h)]])
    sx, sy, sz = size[:3]
    return np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], dtype=float)


def build_xml2(slope_deg=0.0, friction=1.0, torsional=0.015, rolling=0.005, mass_scale=None,
               com_shift=(0, 0, 0), obstacles=(), lidar=False, timestep=0.001, arena=None, mesh_obstacles=(),
               terrain=None, tyre_r=None, payload=None, gravity=None,
               mocap_obstacles=(), crown_d=None, roll_wheel=None, pods=True):
    """mass_scale: dict body->scale (domain randomisation). obstacles: list of (type, pos, size) primitives.
    mesh_obstacles: list of dicts {name, verts (local Nx3), pos, quat}: convex mesh geoms whose vertices are
    used verbatim for collision, ray-casting and rendering.
    terrain: (hfield asset xml, z offset) from sim/terrain.py; the data is written into m.hfield_data after compiling.
    tyre_r: tyre crown radius (wear / manufacturing randomisation). payload: (mass kg, (x, y, z) m) bolted to the
    spine. gravity: explicit gravity vector (overrides slope_deg).
    crown_d: lateral offset of each tyre-half sphere centre (default D = 50 mm); pods and tyre masses move with it.
    roll_wheel: dict(I=kg m^2 about the roll axis, mass=kg, tau=N m) adds a reaction wheel on the spine (hinge about x).
    pods: collision geometry of the two sensor pods (stacked cylinders following the CAD radial profile, |y| 258-353 mm). A pod touching the ground is a
    "pod strike" - with the tyre-half spheres truncated at the pod openings this is what really stops a roll."""
    ms = mass_scale or {}
    g = 9.81
    s = np.radians(slope_deg)
    gx, gz = -g * np.sin(s), -g * np.cos(s)
    if gravity is not None:
        gx, gy, gz = gravity
    else:
        gy = 0.0
    Rt = R if tyre_r is None else tyre_r
    Dc = D if crown_d is None else crown_d
    dD = Dc - D
    pod_xml, pod_asset = "", ""
    if pods:                      # stacked cylinders following the CAD radial profile of each pod (6 mm slices)
        for side, sg in (("L", 1), ("R", -1)):
            for j, (y0, y1, r) in enumerate(POD_PROFILE[side]):
                pod_xml += (f'<geom name="pod{side}{j}" type="cylinder" fromto="0 {sg * (y0 + dD):.4f} 0 0 {sg * (y1 + dD):.4f} 0" '
                            f'size="{r:.4f}" contype="{TYRE_CT}" conaffinity="{TYRE_CA}" condim="3" rgba=".9 .5 .1 .3" group="3" mass="0"/>\n')
    wheel_xml = ""
    if roll_wheel:
        I, mw = roll_wheel["I"], roll_wheel["mass"]
        wheel_xml = (f'<body name="rollwheel" pos="0 0 -0.04"><joint name="rollwheel" type="hinge" axis="1 0 0" damping="0.0005"/>'
                     f'<inertial pos="0 0 0" mass="{mw:.4f}" diaginertia="{I:.5f} {I / 2:.5f} {I / 2:.5f}"/></body>')
    obs_xml = ""
    if terrain is not None:
        asset_t, z0 = terrain
        obs_xml += (f'<geom name="terrain" type="hfield" hfield="terrain" pos="0 0 {z0:.5f}" contype="{WORLD_CT}" '
                    f'conaffinity="{WORLD_CA}" friction="{friction} {torsional} {rolling}" condim="6" rgba=".4 .42 .38 1"/>\n')
    for i, (typ, pos, size) in enumerate(obstacles):
        obs_xml += (f'<geom name="obs{i}" type="{typ}" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" '
                    f'size="{" ".join(f"{v:.3f}" for v in size)}" contype="{WORLD_CT}" conaffinity="{WORLD_CA}" '
                    f'rgba=".55 .45 .35 1" group="1"/>\n')
    if arena:
        L = arena
        for i, (x, y, sx, sy) in enumerate([(L, 0, .1, L), (-L, 0, .1, L), (0, L, L, .1), (0, -L, L, .1)]):
            obs_xml += (f'<geom name="wall{i}" type="box" pos="{x} {y} .4" size="{sx} {sy} .4" '
                        f'contype="{WORLD_CT}" conaffinity="{WORLD_CA}" rgba=".6 .6 .65 1" group="1"/>\n')
    asset_xml = ""
    for o in mesh_obstacles:
        v = " ".join(f"{c:.6f}" for c in np.asarray(o["verts"]).ravel())
        asset_xml += f'<mesh name="m_{o["name"]}" vertex="{v}"/>\n'
        p, q = o["pos"], o["quat"]
        obs_xml += (f'<geom name="{o["name"]}" type="mesh" mesh="m_{o["name"]}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
                    f'quat="{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}" contype="{WORLD_CT}" conaffinity="{WORLD_CA}" '
                    f'rgba=".55 .45 .35 1" group="1"/>\n')
    for o in mocap_obstacles:                  # moving obstacles (pedestrians, carts): mocap bodies, mesh geoms
        v = " ".join(f"{c:.6f}" for c in np.asarray(o["verts"]).ravel())
        asset_xml += f'<mesh name="m_{o["name"]}" vertex="{v}"/>\n'
        p = o["pos"]
        obs_xml += (f'<body name="{o["name"]}" mocap="true" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}">'
                    f'<geom name="{o["name"]}" type="mesh" mesh="m_{o["name"]}" contype="{WORLD_CT}" conaffinity="{WORLD_CA}" '
                    f'rgba=".2 .5 .9 1" group="1"/></body>\n')
    sens = ""
    return f"""
<mujoco model="gyra_mk2">
  <compiler angle="radian" inertiafromgeom="false"/>
  <asset>
    {pod_asset}
    {asset_t if terrain is not None else ""}
    {asset_xml}
  </asset>
  <option timestep="{timestep}" integrator="implicitfast" gravity="{gx:.5f} {gy:.5f} {gz:.5f}"/>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="100 100 0.1" pos="0 0 {(terrain[1] - 0.02) if terrain is not None else 0:.4f}" contype="{WORLD_CT}" conaffinity="{WORLD_CA}"
          friction="{friction} {torsional} {rolling}" condim="6" rgba=".35 .37 .4 1"/>
    {obs_xml}
    <body name="spine" pos="0 0 {Rt + 0.001}">
      <freejoint name="root"/>
      {inertial("spine", ms.get("spine", 1.0), com_shift)}
      <site name="imu" pos="0 0 0"/>
      {f'<body name="payload" pos="{payload[1][0]:.4f} {payload[1][1]:.4f} {payload[1][2]:.4f}"><inertial pos="0 0 0" mass="{payload[0]:.4f}" diaginertia="{0.004 * payload[0]:.5f} {0.004 * payload[0]:.5f} {0.004 * payload[0]:.5f}"/></body>' if payload else ""}
      <site name="podL" pos="0 {0.285 + Dc} 0.0"/>
      <site name="podR" pos="0 {-0.285 - Dc} 0.0"/>
      {pod_xml}
      {wheel_xml}
      <body name="tyreL">
        <joint name="tyreL" type="hinge" axis="0 1 0" damping="0.01"/>
        {inertial("tyreL", ms.get("tyre", 1.0), (0, dD, 0))}
        <geom name="tyreL" type="sphere" size="{Rt}" pos="0 {Dc} 0" contype="{TYRE_CT}" conaffinity="{TYRE_CA}"
              condim="6" friction="{friction} {torsional} {rolling}" rgba=".06 .06 .06 1" group="3"/>
      </body>
      <body name="tyreR">
        <joint name="tyreR" type="hinge" axis="0 1 0" damping="0.01"/>
        {inertial("tyreR", ms.get("tyre", 1.0), (0, -dD, 0))}
        <geom name="tyreR" type="sphere" size="{Rt}" pos="0 {-Dc} 0" contype="{TYRE_CT}" conaffinity="{TYRE_CA}"
              condim="6" friction="{friction} {torsional} {rolling}" rgba=".1 .1 .1 1" group="3"/>
      </body>
      <body name="yoke">
        <joint name="yoke" type="hinge" axis="0 1 0" damping="0.05"/>
        {inertial("yoke", ms.get("yoke", 1.0))}
        <body name="bob">
          <joint name="bob" type="hinge" axis="1 0 0" range="-0.698 0.698" limited="true"/>
          {inertial("bob", ms.get("bob", 1.0))}
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="driveL"><joint joint="tyreL" coef="1"/><joint joint="yoke" coef="-1"/></fixed>
    <fixed name="driveR"><joint joint="tyreR" coef="1"/><joint joint="yoke" coef="-1"/></fixed>
  </tendon>
  <actuator>
    <motor name="driveL" tendon="driveL" ctrlrange="-90 90" ctrllimited="true"/>
    <motor name="driveR" tendon="driveR" ctrlrange="-90 90" ctrllimited="true"/>
    <motor name="level" joint="yoke" ctrlrange="-15 15" ctrllimited="true"/>
    <position name="lean" joint="bob" kp="1500" kv="60" ctrlrange="-0.698 0.698" forcerange="-120 120"/>
    {f'<motor name="rollwheel" joint="rollwheel" ctrlrange="-{roll_wheel["tau"]} {roll_wheel["tau"]}" ctrllimited="true"/>' if roll_wheel else ""}
  </actuator>
  {sens}
</mujoco>
"""


if __name__ == "__main__":
    import time
    import mujoco
    m = mujoco.MjModel.from_xml_string(build_xml2())
    d = mujoco.MjData(m)
    print("dofs", m.nv, "mass", round(float(sum(m.body_mass)), 2))
    t0 = time.time()
    n = 20000
    for _ in range(n):
        mujoco.mj_step(m, d)
    print(f"{n / (time.time() - t0):.0f} physics steps/s (dt={m.opt.timestep})")
