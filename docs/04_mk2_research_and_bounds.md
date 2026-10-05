# Toward GYRA Mk2: what can get 10× better, and what physics won't allow

The goal for Mk2 is "orders of magnitude better at every aspect". This document
separates the metrics where 10× or more is physically reachable from those where a
hard physical ceiling exists, so the design effort goes where it pays.

## 1. The ceilings every internal-drive sphere hits

A closed sphere gets **no traction force except through its own shell**. Every
propulsive torque comes from an internal mass the shell pushes against. With M the
total mass, m_p and L the pendulum mass and lever arm, R the radius and θ the pendulum
swing:

| quantity | bound | Mk1 value | best realistic |
|---|---|---|---|
| grade | sin α ≤ (m_p / M)(L / R) sin θ | 13.6° (10° rated) | m_p/M → 0.6, L/R → 0.72 ⇒ **~26°** |
| acceleration | a ≤ g (m_p L)/(M_eff R) | 1.9 m/s² | ~4 m/s² |
| top speed | motor and stability, not traction | 6.4 m/s | 10-12 m/s (RotunBot ran 10 m/s) |
| static step | h = R(1 − cos α) | 8 mm | ~30 mm (kinetic: ~0.4 R) |

**Grade and acceleration cannot improve 10×.** Even an all-pendulum robot with
L = R reaches only sin α = 1. The realistic gain is about 2×, from moving everything
heavy into the pendulum (battery, compute, electronics) and pushing the pendulum's
centre of mass outward.

## 2. Where 10× *is* available

| metric | why Mk1 is weak | Mk2 lever | expected gain |
|---|---|---|---|
| **yaw rate at standstill** | CMG yaw mode is impulse-limited (2h per sweep), ~9°/s average | **split tyre, differential drive**: continuous yaw torque 2·F·d, limited only by pivot friction | **> 10×** (target > 120°/s) |
| **turn radius at speed** | lean precession: ρ = J v²/(τ R), so 48 m at 6 m/s | differential yaw torque + bob counter-lean; ρ limited by tip-over v²/(g·e/h) | **~5-8×** |
| **roll wobble** | single contact point, gyroscopic coupling | **twin-crown tyre**: two contact patches 120 mm apart give passive roll stiffness M·g·d | wobble mode suppressed without CMGs |
| **mass spent on stabilisation** | 5.9 kg of CMGs + 1 kg gimbal motors | removed: the twin crown handles roll and the differential handles yaw | −7 kg, all of it turned into battery ballast |
| **endurance** | 468 Wh | the battery *is* the ballast, so a 3× bigger pack costs no extra mass | **~3×** (≈ 13 h at 3 m/s) |
| **autonomy** | hand-tuned PD/PI loops, no navigation | learned locomotion + navigation policies (asymmetric PPO) | new capability |
| **sensor stability** | already 0.75° vs ~46° for an RT-G-style pod | keep the levelled spine | ≈ 60× vs RT-G (kept) |

## 3. The split-tyre idea in detail

Prior art. Split-hemisphere differential steering is an established family of
spherical robots: a Chinese patent on a *hemispherical differential retractable
spherical robot* (CN103387016B), and research robots with "two individually driven
semi-spheres". A 2026 *Actuators* paper (MDPI 15/4/181) decouples rolling (pendulum)
from turning with a gear-rack yaw mechanism. **What none of these combine** is a
split tyre with a **non-rotating, levelled sensor axle**, a **twin-crown rubber
profile** for passive roll stiffness, and **one pendulum reacting both drives**.

How it works:
* Each tyre half is a spherical zone whose crown is offset outward by **d = 60 mm**:
  its surface is a sphere of radius R centred at (0, ±d, 0). Upright, the robot
  stands on **two patches 120 mm apart**. The centre seam sits 6 mm below the crowns,
  so it never touches flat ground.
* Each half has its own internal ring gear and its own drive motor. Both motors react
  on the same pendulum: the **sum** of their torques swings the pendulum (forward
  drive), and the **difference** cancels inside the pendulum and appears only as a
  **yaw couple at the ground**: τ_yaw = (F_L − F_R)·d.
* Turn in place: counter-rotate the halves. Each patch rolls on a circle of radius d,
  so the only resistance is pivot friction (~8.5 N·m). Each motor needs about 28 N·m
  at the tyre for that, well inside the 6374 + 30:1 drive's 100 N·m peak, and
  **independent of the pendulum limit** because the difference torque never reaches
  the pendulum.
* High-speed turns: the turn is commanded directly by the differential, and the bob
  leans the CoM inward to stop tip-over. Lateral acceleration is limited by
  a_tip ≈ g·(d + e_bob)/h_CoM.

## 4. Learning-based control: the plan

* **Asymmetric actor-critic** (Pinto et al., 2017): the actor sees only what the real
  robot measures (IMU, motor encoders, pendulum/bob encoders, LiDAR, depth). The critic
  additionally gets privileged simulator state (true velocity, friction, masses, slope,
  obstacle map). Only the actor is deployed.
* **Locomotion / teleoperation policy**: tracks (v, ω) joystick commands at 50 Hz,
  outputting the two drive torques and the bob setpoint. It is trained with domain
  randomisation (mass, CoM offset, friction, motor strength, latency, sensor noise,
  slopes, pushes).
* **Autonomous navigation policy**: at 10 Hz it outputs (v, ω) commands to the frozen
  locomotion policy. Inputs are a 2-D scan synthesised from the two pod LiDARs, **with
  the real forward blind wedge modelled**; a low-resolution depth image from the front
  stereo pair; encoder/IMU odometry; and the goal in the odometry frame. Its critic
  sees the true pose and a local occupancy grid.
* Prior art for this structure: RL spherical-pendulum tracking with zero-shot sim-to-real
  (arXiv:2309.14096), LiDAR-ray PPO navigation on differential-drive robots, and the
  Zhejiang group's MPC trajectory tracking on the RotunBot platform (arXiv:2303.18186,
  2205.14181). RL locomotion for a **pendulum-driven sphere** appears to be unpublished.

## Sources
* Split-hemisphere patent CN103387016B: https://eureka.patsnap.com/patent-CN103387016B
* *Mechanically Decoupled Rolling and Turning Design for Pendulum-Driven Unmanned Spherical Robots*, Actuators 15(4):181: https://www.mdpi.com/2076-0825/15/4/181
* GuardBot (pendulum drive, 22 in, 25 h endurance): https://physicstoday.aip.org/news/meet-guardbot
* Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning*: https://www.semanticscholar.org/paper/Asymmetric-Actor-Critic-for-Image-Based-Robot-Pinto-Andrychowicz/ae096b868323f74a68414ae8855e20669540e2ba
* *Tracking Control for a Spherical Pendulum via Curriculum RL*: https://arxiv.org/abs/2309.14096
* *Adaptive MPC multi-terrain trajectory tracking for mobile spherical robots*: https://arxiv.org/abs/2303.18186
* *Direction and trajectory tracking control for nonholonomic spherical robot (SMC + MPC)*: https://arxiv.org/abs/2205.14181
* *Deep RL with enhanced PPO for safe mobile robot navigation*: https://arxiv.org/abs/2405.16266
