# GYRA Mk2: learned control v2, built to work anywhere

v1 (doc 06) learned to drive on flat ground with pushes, and to navigate a random-obstacle
arena. v2 rebuilds both policies and their training stack with three goals:

* hold up on any surface and under any disturbance the robot is likely to meet;
* use production-style architecture, rewards, curriculum and evaluation;
* keep simulation honest: what the robot collides with is exactly what its sensors see and
  what the renders show.

| | v1 | v2 |
|---|---|---|
| Ground | flat plane (+ gravity-tilt slopes) | procedural heightfields: hills, rough + potholes, curbs/steps/ditches, ramps to 15°, blended mixed terrain |
| World build | compiled once, obstacles moved at run time (**broken**, see §0) | **recompiled every episode** (~5 ms); obstacles are convex mesh geoms, pedestrians are mocap mesh bodies |
| Disturbances | pushes ≤ 250 N, friction 0.5-1.2 | pushes ≤ 350 N + yaw kicks, steady wind + gusts, friction 0.25-1.2 with **mid-run surface change**, payload ≤ 10 kg anywhere, tyre wear, per-motor strength 0.75-1.1, torque noise, 0-40 ms latency, IMU bias / misalignment / glitches |
| Locomotion actor | MLP on 4 frames (80 ms) | **1 s history → temporal conv encoder → latent + concurrent state estimator** → MLP |
| Losses | PPO | PPO + estimator regression + left/right **symmetry** loss; return normalisation; KL-adaptive LR with early stop |
| Curriculum | global, step-based | **per-environment** level (terrain, disturbances, speed range) moved up on clean episodes and down on falls, plus 15 % replay of easier levels |
| Navigation | 1 world type, Euclidean progress | 5 world families, pedestrians, **geodesic** progress, global-planner subgoal on a **stale prior map** (or none), sensor faults |
| Evaluation | random seeds | fixed-seed **robustness matrices** (16 locomotion, 12 navigation scenarios) against classical and v1 baselines |

## 0. A simulation bug, found and fixed

While making the video I checked that every obstacle's collision geometry matched its
rendered geometry, and found that **in v1's navigation world, obstacles never
collided**. The arena compiled 22 obstacle slots once, then moved and resized them per
episode by writing `geom_pos`/`geom_size`. MuJoCo keeps the static world body's
bounding-volume hierarchy and each geom's AABB from compile time. The broad phase
therefore never paired the tyres with a moved obstacle, and contacts were silently
missed. Only the walls, which never move, registered.

Consequences and fix:

* v1's reported navigation collision rate (2 %) only counted walls. The v1 numbers in
  doc 06 are wrong and are restated below, measured in the corrected world.
* Updating the AABBs alone is not enough (8 117 of 15 792 probe states still
  disagreed): the world-body BVH is also stale.
* v2 compiles a fresh world every episode (5 ms). Every static obstacle and wall is a
  convex mesh geom at its real pose, and moving obstacles are mocap bodies. MuJoCo
  collides convex meshes as their convex hull. All obstacles are convex and every
  input vertex is kept (asserted), so the collision shape is the mesh itself.
* The renderer builds every obstacle, wall, pole, pedestrian and the terrain from the
  geometry read back from the compiled model. It then re-casts 4 000 MuJoCo probe rays
  in Blender. Test mission: **max disagreement 0.11 mm over 3 830 rays** (float32
  rounding).
* Terrain lookups (`sim/terrain.height_at`) reproduce MuJoCo's heightfield vertex grid
  and triangulation exactly: max |ray − lookup| = 0.005 mm.

## 1. Locomotion v2

### World and robot randomisation (per episode, scaled by the env's curriculum level k)
| | range at k = 1 |
|---|---|
| terrain | 20 % flat, else one of hills (p95 slope 12°), rough + potholes, curbs/steps to 6 cm and ditches, ramps to 15°, smoothly blended mix |
| incline | 0-14° uniform slope in 40 % of episodes |
| friction | μ 0.25-1.2; 30 % of episodes switch surface mid-run (e.g. asphalt → ice) |
| robot | link masses ±10 %, CoM ±6 mm, payload 0-10 kg anywhere on the spine, tyre radius −1.5/+1 % |
| actuators | per-motor strength 0.75-1.1, 5 % torque noise, bob worm-drive 45-75°/s, latency 0-2 steps |
| pushes | 1 every 3 s on average, 50-350 N for 60-300 ms, with yaw kicks to ±40 N·m |
| wind | 0-40 N steady + Ornstein-Uhlenbeck gusts (σ 10 N, τ 1 s) |
| IMU | gyro noise 0.02 rad/s + bias to ±0.03 rad/s, mounting error to ±1.5°, rare glitch frames |
| operator | steps or 0.5-1.5 s ramps every 1.5-5 s: stop, spin, straight, arcs (lateral accel ≤ min(4, 0.6 μ g), 20 % unconstrained); speed range 2 → 8.5 m/s with k |

### Policy
* **Actor:** 50 × 18 proprioceptive frames (1 s) → 3-layer temporal conv → 64-d latent
  → linear **estimator** (body velocity, local slope, friction, payload, wind, motor
  strength). [current frame, command, latent, estimate] → MLP 256-128-64 → residual
  torque/bob action on top of the classical controller, with **full authority**
  (±60 N·m sum, ±60 N·m differential, ±0.5 rad bob).
* **Critic:** estimator targets + privileged state (contacts, every randomised
  parameter, push/wind forces, true attitude and rates) + a 7 × 7 terrain height scan →
  MLP 512-256-128.
* **Deployable artifact:** one TorchScript module with the observation normaliser
  folded in (`rl/runs/loco2/actor.ts`).

### Reward (per step; every term is logged separately)
| term | weight | |
|---|---|---|
| speed tracking | +1.0 · exp(−e_v²/0.25) | |
| yaw-rate tracking | +0.75 · exp(−e_ω²/0.25) | |
| steady sensor platform | +0.25 · exp(−(ω_pitch² + ω_roll²)/0.25) | camera/LiDAR stability is the robot's selling point |
| spine attitude | −1.0 θ_pitch² − 0.2 max(\|roll\| − 0.15, 0)² | |
| action rate, smoothness | −0.02 \|Δa\|², −0.01 \|Δ²a\|² | |
| mechanical power | −4·10⁻⁴ Σ max(τω, 0) | |
| torque saturation | −0.3 · clipped torque / stall torque | |
| pendulum slosh | −0.01 ω_pend² | |
| bob near its stop | −0.5 max(\|bob\| − 0.6, 0) | |
| fall (pendulum loops, roll > 63°, spine > 34°) | −20, episode ends | |

### Results
(pending: training in progress)

## 2. Navigation v2
(pending)

## 3. Robustness matrices
(pending)
