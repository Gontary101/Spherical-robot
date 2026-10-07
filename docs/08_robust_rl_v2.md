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

### Reward (final v3.1; per step, every term logged separately)
Errors are measured against the **feasible reference** (sec. 2), not the raw operator command.

| term | weight | |
|---|---|---|
| speed tracking | +1.0 · exp(−e_v²/0.25) + 0.5 · exp(−e_v²/0.01) | coarse + fine kernels |
| yaw-rate tracking | +0.75 · exp(−e_ω²/0.25) + 0.5 · exp(−e_ω²/0.01) | |
| distance / heading hold | +0.5 · exp(−E_s²/0.0025) + 0.5 · exp(−E_ψ²/0.0025) | E = leaky (0.2/s) integrals of the along-track and heading errors: a persistent bias is never free |
| steady sensor platform | +0.25 · exp(−(ω_pitch² + ω_roll²)/0.25) | camera/LiDAR stability is the robot's selling point |
| spine attitude | −1.0 θ_pitch² − 0.5 θ_roll² | level in both axes |
| residual size | −0.02 \|a\|² | deviate from the classical loop only when it pays |
| action rate, smoothness | −0.02 \|Δa\|², −0.01 \|Δ²a\|² | |
| mechanical power | −4·10⁻⁴ Σ max(τω, 0) | |
| torque saturation | −0.3 · clipped torque / stall torque | |
| pendulum slosh | −0.01 ω_pend² | |
| bob near its stop | −0.5 max(\|bob\| − 0.6, 0) | |
| failure: **pod strike** (CAD pod geometry touches the ground), pendulum loop-over, spine > 34° | −20, episode ends | |

### Results
Final policy: **v3.1** (`rl/runs/loco31/best.pt`, `actor.ts`), fine-tuned from v3 (15M steps) for 9M steps.
Robustness on the CAD-exact Mk2.1 model, 16 scenarios × 24 episodes (`rl/eval_robust.py`):

| controller / hardware | falls | mean spine tilt | speed RMSE vs feasible ref |
|---|---|---|---|
| classical, Mk2 as built | 27.7 % | 0.99° | 0.48 m/s |
| v1 policy, Mk2 | 19.5 % | 0.89° | 0.46 m/s |
| v2 policy (15M), Mk2 | 16.8 % | 0.83° | 0.37 m/s |
| classical, Mk2.1 | 12.5 % | 0.26° | 0.46 m/s |
| **v3.1 policy, Mk2.1** | **1.8 %** (7 / 384) | **0.21°** | **0.33 m/s** |

The remaining v3.1 falls: 350 N lateral pushes 12.5 % (beyond the roll capacity of the hardware: even an ideal bob
lean cannot recover a 350 N × 0.3 s side push), 12° incline 12.5 %, rough ground 4 %; 13 of 16 scenarios have none.

Low-speed precision (`rl/precision_test.py`, flat, 4 seeds), which tight-space navigation depends on:

| | classical | v3 | **v3.1** |
|---|---|---|---|
| spin in place: heading error vs command | 73° | 31° | **0.8°** |
| spin in place: displacement | 1 cm | 21 cm | 10 cm |
| creep 0.3 m/s, 10 s: heading drift | 0.4° | 27° | 13° |
| hold still 6 s: heading / displacement | 0.3° / 2 cm | 12° / 43 cm | 9° / 31 cm |
| stop from 1 m/s: overrun | 84 cm | 63 cm | 57 cm |

How v3.1 came about: navigation exposed a yaw bias in v3 (−0.08 rad/s while "holding still"). Channel isolation
showed it came entirely from the differential-torque residual (switching it off gives 0.0005 rad/s), and it went
with the bob action pinned at ±1 in a self-reinforcing lean (the policy observes its own last action, so
leaned-left and leaned-right are both stable modes that still satisfy the symmetry loss). The σ = 0.5 tracking
kernels made such small persistent errors almost free. Integrated-error terms, roll penalty and residual cost fix
spinning and stopping; the classical controller's yaw integrator still holds better at a standstill, and the navigation
layer closes the heading loop at 10 Hz anyway.

## 2. Root-cause analysis of the remaining falls (why v2 plateaued)

At 15M steps v2 had the lowest tracking error and spine tilt, but its fall rate had
stalled (9.4 % mean over 16 scenarios, versus 10.2 % for v1 and 18 % for classical).
Rather than patch scenario by scenario, every fall was instrumented
(`rl/diagnose_falls.py`) and each suspected mechanism tested in isolation.

**What the falls are.** All of them, for every controller and in all seven failing
scenarios, are sideways roll-overs. Drive-torque saturation before a fall is ~0,
so the "pendulum torque limit" hypothesis was wrong.

**Mechanism 1: the pendulum spends roll stability.** On a 12° incline (to hold
position) or when braking from 6 m/s, the pendulum swings to 45-70°. The robot's roll
stiffness comes from its low CoM (CoM drop ∝ cos θ_pendulum) plus the twin-crown
stance, and the bob's lean axis tilts with the yoke. Large pendulum angles therefore
halve both the passive roll stiffness and the bob's authority. Traces show a growing
roll oscillation (−60° → +58° within 0.5 s) at exactly those moments. Longitudinal
and lateral capability are coupled: a **stability ellipse**, not independent limits.

**Mechanism 2: the only roll actuator is too slow.** The worm-driven bob (60°/s)
needs ~0.7 s to cross its range, which is comparable to the ~2 s rocking period. It
rate-saturates and arrives half a cycle late (roll +32° while the bob is still at
−15°), pumping energy into the oscillation. The feedback sign is correct; flipping
it falls at 16°.

**Mechanism 3: the simulation's failure definition was not physical.** "Fall" was
roll > 63°, and the tyre halves were full spheres. Per the CAD, each half is a sphere
cap truncated at the pod opening, and the LiDAR pod touches the ground at ~45-48° of
roll. The sim now carries the pods' collision geometry (stacked cylinders following
the CAD radial profile; strike at 44.5°), and a pod strike ends the episode.

**Mechanism 4: the task paid the policy to fight physics.** The reward demanded exact
tracking of any operator command, including speed/turn combinations outside the
coupled envelope. Tracking them anyway is precisely what produces mechanism 1.

**Two false leads, kept for the record.**
* "Lateral capability collapses above 4 m/s" was an artefact: fast straight-line
  test runs left the 60 m terrain patch and hit the geofence. Without it, Mk2.1 turns
  at > 4 m/s² at 6 m/s with ~0° roll. Steady turning is not roll-limited; transient
  braking-while-turning is (mechanism 1).
* A roll reaction wheel looked attractive (RT-G uses one). Measured, it adds ~1 point
  over the geometric fix and introduces gyroscopic roll-yaw-pitch coupling, so it was
  not adopted.

### Fixes, each at the layer of its cause
| layer | fix | evidence |
|---|---|---|
| hardware | **Mk2.1**: twin-crown offset 50 → 90 mm (width 700 → 780 mm), bob lean actuator 60 → 180°/s (back-drivable ball-screw in place of the worm), PI spine levelling | classical controller, same scenarios: falls 26.2 % → **10.9 %**, spine tilt 1.00° → **0.28°** (`rl/hw_study.py`) |
| simulation | pod collision geometry from CAD; failure = pod strike or pendulum loop | strike at 44.5° (CAD vertex analysis: 45-48°) |
| task | reward tracks a **feasible reference**: operator command shaped by the true slope (uniform + local terrain), friction, payload, motor strength and the stability ellipse. The policy still sees the raw command | — |
| curriculum | **ADR**: nine factors (terrain, slope, friction, pushes, wind, payload, actuator faults, sensor faults, speed), each with its own range, widened at ≥ 80 % and narrowed below 50 % success on pooled boundary tests | — |
| optimisation | entropy bonus removed, policy std capped at 0.35 (the bob action's std had drifted to its 0.5 cap) | — |

Hardware study (classical controller, fall %, 16 episodes per cell):

| scenario | Mk2 | lean 180°/s | crown 90 mm | wheel 8 N·m | wheel 15 N·m | **Mk2.1** | Mk2.1 + wheel |
|---|---|---|---|---|---|---|---|
| hills | 25 | 0 | 0 | 0 | 0 | **0** | 0 |
| rough + potholes | 19 | 12 | 12 | 6 | 6 | **6** | 0 |
| ramps to 15° | 19 | 12 | 6 | 12 | 6 | **0** | 6 |
| 12° incline | 88 | 81 | 56 | 88 | 88 | **56** | 56 |
| pushes to 350 N | 69 | 50 | 31 | 56 | 50 | **12** | 6 |
| wind 40 N + gusts | 44 | 38 | 19 | 31 | 31 | **12** | 12 |
| payload 10 kg off-centre | 38 | 44 | 12 | 31 | 25 | **12** | 6 |
| flat, 8 m/s | 38 | 56 | 44 | 50 | 38 | **50** | 50 |
| everything at once | 38 | 38 | 6 | 12 | 6 | **6** | 0 |
| **mean, all 16** | 26.2 | 23.4 | 12.9 | 20.7 | 17.6 | **10.9** | 9.8 |

The remaining classical failures (12° incline, 8 m/s) are braking/climbing-while-
turning beyond the stability ellipse: the classical controller tracks the raw command.
That is the job of the feasible-reference policy below.

## 3. Navigation v2
(pending)

## 4. Robustness matrices
(pending)
