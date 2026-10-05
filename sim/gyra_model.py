"""
GYRA Mk1 - MuJoCo model generated from the CAD mass properties.

Kinematic tree (all hinge joints, frames at the sphere centre unless noted):

    spine (free joint)               axle + pods + brackets + bays + gimbal actuators
      tyre      hinge +y             rolling shell (sphere collision geom, r = 0.30 m)
      yoke      hinge +y             pendulum pitch body (drive motors react on it)
        bob     hinge +x             battery + lead, lateral lean
      cmgL      hinge +y @ (0, +.17, .122)   gimbal housing
        flyL    hinge +z             flywheel, spins +z
      cmgR      hinge +y @ (0, -.17, .122)
        flyR    hinge +z             flywheel, spins -z

Actuators:
    drive   fixed tendon (tyre - yoke): torque between yoke and tyre  (2 x 6374 BLDC, 30:1)
    level   yoke hinge torque (reaction on the spine)                 (levelling motor)
    lean    stiff position servo on the bob hinge                     (self-locking worm)
    gimL/R  gimbal velocity servos                                    (AK70-10 class)
    spinL/R flywheel velocity servos                                  (spin motors)
"""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MP = json.load(open(os.path.join(ROOT, "cad", "out", "mass_properties.json")))
R = 0.300
CMG_Y, CMG_Z = 0.170, 0.122


def inertial(body, frame_origin=(0, 0, 0)):
    b = MP["bodies"][body]
    c = np.array(b["com_mm"]) / 1000.0 - np.array(frame_origin)
    I = np.array(b["I_com_kgm2"])
    fi = f"{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} {I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}"
    return f'<inertial pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" mass="{b["mass"]:.5f}" fullinertia="{fi}"/>'


def build_xml(slope_deg=0.0, torsional=0.021, rolling=0.006, ground_friction=1.0,
              obstacle=None, noslip=0):
    g = 9.81
    s = np.radians(slope_deg)
    gx, gz = -g * np.sin(s), -g * np.cos(s)   # slope climbed in +x == gravity tilted backwards
    obst = ""
    if obstacle:
        x, h = obstacle
        obst = f'<geom name="step" type="box" pos="{x + 2.0} 0 {h / 2}" size="2 3 {h / 2}" rgba=".5 .5 .5 1"/>'
    return f"""
<mujoco model="gyra_mk1">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="0.0005" integrator="implicitfast" gravity="{gx:.5f} 0 {gz:.5f}" noslip_iterations="{noslip}">
    <flag energy="enable"/>
  </option>
  <default>
    <joint armature="0.0" damping="0.0"/>
    <geom contype="0" conaffinity="0" rgba=".8 .8 .8 1"/>
  </default>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="200 200 0.1" contype="1" conaffinity="1"
          friction="{ground_friction} {torsional} {rolling}" condim="6" rgba=".35 .37 .4 1"/>
    {obst}
    <body name="spine" pos="0 0 {R + 0.0005}">
      <freejoint name="root"/>
      {inertial("spine")}
      <site name="imu" pos="0 0 0"/>
      <body name="tyre">
        <joint name="tyre" type="hinge" axis="0 1 0" damping="0.02"/>
        {inertial("tyre")}
        <geom name="tyre" type="sphere" size="{R}" contype="1" conaffinity="1" condim="6"
              friction="{ground_friction} {torsional} {rolling}" rgba=".05 .05 .05 1"/>
      </body>
      <body name="yoke">
        <joint name="yoke" type="hinge" axis="0 1 0" damping="0.05" range="-1.309 1.309" limited="true"/>
        {inertial("yoke")}
        <body name="bob">
          <joint name="bob" type="hinge" axis="1 0 0" range="-0.5236 0.5236" limited="true"/>
          {inertial("bob")}
        </body>
      </body>
      <body name="cmgL" pos="0 {CMG_Y} {CMG_Z}">
        <joint name="gimL" type="hinge" axis="0 1 0" damping="0.01"/>
        {inertial("cmgL", (0, CMG_Y, CMG_Z))}
        <body name="flyL">
          <joint name="flyL" type="hinge" axis="0 0 1"/>
          {inertial("flyL", (0, CMG_Y, CMG_Z))}
        </body>
      </body>
      <body name="cmgR" pos="0 {-CMG_Y} {CMG_Z}">
        <joint name="gimR" type="hinge" axis="0 1 0" damping="0.01"/>
        {inertial("cmgR", (0, -CMG_Y, CMG_Z))}
        <body name="flyR">
          <joint name="flyR" type="hinge" axis="0 0 1"/>
          {inertial("flyR", (0, -CMG_Y, CMG_Z))}
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="drive"><joint joint="tyre" coef="1"/><joint joint="yoke" coef="-1"/></fixed>
  </tendon>
  <actuator>
    <motor name="drive" tendon="drive" ctrlrange="-32 32" ctrllimited="true"/>
    <motor name="level" joint="yoke" ctrlrange="-12 12" ctrllimited="true"/>
    <position name="lean" joint="bob" kp="800" kv="40" ctrlrange="-0.5236 0.5236" forcerange="-60 60"/>
    <velocity name="gimL" joint="gimL" kv="6" forcerange="-12 12"/>
    <velocity name="gimR" joint="gimR" kv="6" forcerange="-12 12"/>
    <velocity name="spinL" joint="flyL" kv="0.02" forcerange="-0.4 0.4"/>
    <velocity name="spinR" joint="flyR" kv="0.02" forcerange="-0.4 0.4"/>
  </actuator>
  <sensor>
    <framequat name="q" objtype="body" objname="spine"/>
    <frameangvel name="w" objtype="body" objname="spine"/>
    <framelinvel name="v" objtype="body" objname="spine"/>
  </sensor>
</mujoco>
"""


if __name__ == "__main__":
    import mujoco
    m = mujoco.MjModel.from_xml_string(build_xml())
    print("bodies", m.nbody, "dofs", m.nv, "total mass", round(float(sum(m.body_mass)), 3))
