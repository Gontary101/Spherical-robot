# GYRA Mk2: learned control (asymmetric PPO)

Two policies, trained in MuJoCo on the CAD-derived Mk2 model:

1. **Locomotion / teleoperation policy** (50 Hz): turns joystick commands (v, ω) into
   motor torques and a bob setpoint.
2. **Autonomous navigation policy** (10 Hz): turns pod LiDAR, stereo depth and
   odometry into (v, ω) commands for policy 1.

Both use the **asymmetric actor-critic** setup (Pinto et al., 2017). The actor only
gets signals the real robot measures, with noise and latency. The critic additionally
gets privileged simulator state. Only the actor is deployed.

Code: `rl/loco_env.py`, `rl/nav_env.py`, `rl/ppo.py`, `rl/vec_env.py`,
`rl/train_loco.py`, `rl/train_nav.py`, `rl/eval_loco.py`, `rl/eval_nav.py`.
Everything runs on CPU: 4 cores, ~2 k env-steps/s for locomotion, ~250 nav-steps/s
per core.

## 1. Locomotion policy

### Structure: residual RL over an estimator-fed classical loop
A first, fully end-to-end run plateaued: speed error ~0.6 m/s, falls ~35 % of
episodes, and exploration noise growing (log in `rl/runs/archive/loco_e2e/`). With
CPU-only compute, the faster route is **residual RL**. The classical cascade controller
(speed → pendulum angle → torque, yaw-rate → differential torque, lateral
acceleration → counter-lean) runs on *estimated* state (IMU + encoders + noise), and
the policy adds a bounded correction: ±25 N·m on the drive sum, ±25 N·m on the
differential, ±0.35 rad on the bob. The whole stack stays deployable, and learning
starts from a working controller.

### Observations
| | actor (deployable) | critic (privileged) |
|---|---|---|
| IMU | gyro (3), gravity direction (3) | true body velocity (3), yaw rate, roll, pitch |
| encoders | pendulum angle (sin, cos) + rate, both drive-motor rates, bob angle | pendulum absolute angle + rate, bob |
| controller | previous action (3), classical base action (3) | contacts L/R |
| command | v, ω | v, ω, tracking errors |
| world | — | friction (μ, torsional, rolling), mass scales (4), CoM offset (3), motor strength, gravity tilt (2), push active |
| history | 4 frames (18 each) → 74 inputs | current frame → 48 inputs |

Noise: gyro 0.02 rad/s, gravity 0.015, encoder rates 0.05 rad/s, plus 0-1 control
steps of action latency.

### Domain randomisation (per episode)
Friction 0.5-1.2 · torsional 0.008-0.025 m · rolling 0.002-0.008 m · body masses
±10 % · spine CoM ±5 mm · motor strength 0.8-1.1 · slopes up to 12° in a random
direction (curriculum) · random 50-250 N shoves (curriculum) · commands resampled
every 2-5 s (turn in place, straight, stop, mixed with |v·ω| ≤ 4 m/s²).

### Reward (per 20 ms step)
exp(−e_v²/0.5) + exp(−e_ω²/0.5) − 0.15|e_v| − 0.15|e_ω| − 0.05 ω_roll² −
0.5 max(|roll| − 0.35, 0)² − 2 θ_spine² − 0.05 Δa² − 0.02 a² − energy − pendulum
slosh; −10 on a fall (pendulum over the top, or roll > 63°).

### Curriculum
Commanded speed range 4 → 8.5 m/s, and difficulty (slope range, push rate/magnitude)
0.5 → 1.0. It advances when the smoothed tracking errors and fall rate pass the
thresholds.

## 2. Navigation policy

| | actor (deployable) | critic (privileged) |
|---|---|---|
| LiDAR | 72-beam 2-D scan × 2 frames, synthesised from both pod Mid-360s, **with the forward/rear blind wedge** (returns hidden by the tyre are dropped), 2 cm noise, 2 % dropouts | true unmasked noiseless 72-beam scan, nearest-obstacle distance |
| vision | 16 × 8 front stereo depth image (90° × 45°), noise ∝ range², < 0.35 m invalid | — |
| odometry | goal distance/bearing from encoder + gyro dead reckoning (gyro bias, wheel-radius error → drift) | true goal vector, odometry error |
| state | speed estimate, yaw rate, previous action, time | true v, ω |

Action: (v_cmd ∈ [−1, 3.5] m/s, ω_cmd ∈ ±2.5 rad/s) to the frozen locomotion policy.
Reward: 3 × progress − 0.01/step − proximity penalty − action-rate penalty, +10 on
reaching the goal (< 0.6 m), −10 on collision or fall. Worlds: a 14 × 14 m walled
arena with 8-22 random cylinders and boxes (0.25-0.9 m tall), start/goal 4-10 m apart.

Baseline: a classical **VFH-style gap follower** using the same masked scan and the
same drifting odometry, driving through the same low-level policy.

## 3. Results
*(filled in from `rl/runs/*/eval.json` when training completes)*
