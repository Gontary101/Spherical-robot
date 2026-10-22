# Plan: from "good" to "no collisions, precise following"

Starting point (doc 08):

| | now | target |
|---|---|---|
| locomotion falls (16 stress scenarios) | 1.8 % | ≤ 1 %, the rest documented as beyond the hardware |
| hold-still drift / heading | 31 cm / 9° | < 5 cm / < 2° |
| stop overrun from 1 m/s | 57 cm | within 10 cm of the physical braking bound |
| navigation success | 59 % | ≥ 85 % (the privileged oracle is at 88 %) |
| navigation collisions | 20 % | **< 1 % at-fault static**, < 2 % at-fault pedestrian |

## Why a better policy alone will not get there

A learned policy is a statistical object: it fails on the tail of its training distribution, and its collision
rate only falls slowly with more data. Real robots never rely on it for safety. They wrap planning in a safety layer
whose guarantee comes from a model of the robot's braking and a record of what has been seen: a command is executed
only if the robot can still stop before contact along the arc that command produces. Two things make that layer
possible here: a **braking/turning model of the low-level**, identified from simulation, and a **local occupancy
map with memory**, so obstacles that slip into the pods' forward blind wedge, a failed sensor sector or between
LiDAR beams are not forgotten. The learned policy is then trained with the safety layer in the loop, and penalised
when the layer has to intervene, so it learns to be efficient rather than to rely on being stopped.

The remaining success gap to the oracle has two causes: memory (the actor sees 0.2 s of scans) and perception. The
standard fix is privileged-teacher distillation: train a teacher on the true map, then a sensor-only student with
memory (local map CNN + GRU) imitating it (DAgger).

## Phases

| # | phase | deliverable | acceptance |
|---|---|---|---|
| 0 | **failure taxonomy** | `rl/diagnose_nav.py`: every failure classified (static / pedestrian / fall / timeout; speed at impact; obstacle seen or not and by which sensor; blind-wedge / dead-sector / thin-pole / perceived-but-not-avoided; at-fault vs struck while stopped; where timeouts stall); 32 episodes per scenario | ranked list of failure causes with counts |
| 1a | **low-level system identification** | braking distance d(v), turn response, latency of locomotion v3.1 across friction and slope | model with ≥ 95 % coverage margin |
| 1b | **egocentric occupancy map** | rolling 2-D grid (odometry frame) fusing masked LiDAR + depth, with free-space clearing and decay | blind-wedge obstacles retained; map vs truth error measured |
| 1c | **safety filter** | dynamic-window admissibility on the map: (v, ω) passes only if braking along its arc stops ≥ margin before the footprint touches an occupied cell; otherwise nearest admissible command | static at-fault collisions < 1 % with the current policy, VFH and oracle |
| 1d | **shielded training** | navigation fine-tuned with the filter in the loop + intervention penalty | success not below phase 1c, interventions ↓ |
| 2 | **teacher-student navigation** | privileged teacher (true geodesic + map), sensor student with local-map CNN + GRU, DAgger | success ≥ 85 % |
| 3 | **following precision** | low-level with structural integral action (residual cannot create steady-state yaw/position error) | hold < 5 cm / < 2°, stop overrun near bound, robustness ≤ 1 % falls |
| 4 | **pedestrians** | track/predict from the map, social margin in the filter; at-fault metric | at-fault pedestrian collisions < 2 % |
| 5 | **final evaluation** | 50 episodes per scenario, docs, new video | all targets reported, misses explained |

Every phase is measured with fixed seeds before and after, and nothing is kept that does not move its metric.

## Progress log — phase 1 (safety layer)

All numbers: current navigation policy (nav v2, 6 M, never trained with the filter), 16 episodes on each of mixed /
rooms with 1.0 m doors / forest of thin poles / depth camera dead, fixed seeds, Mk2.1 + locomotion v3.1, benchmark
disturbances (rough ground, 10 N wind + gusts, 3 kg off-centre payload, 150 N pushes).

| step | success | collision | timeout | what the traces showed |
|---|---|---|---|---|
| no filter | 66 % | **30 %** | 3 % | baseline |
| filter v1 (map + braking table) | 28 % | 9 % | 70 % | deadlocks |
| + spawn prior, free evidence persists, never-closer recovery, no chatter | 66 % | 9 % | 25 % | collisions all < 0.3 m/s, obstacle already mapped |
| + field-measured error envelope, 3-D LiDAR map | 53 % | 12.5 % | 34 % | every obstacle in the map ≥ 3 s before contact: creep at "stop", late braking at speed |
| + position hold at rest, straight + arc braking paths | 50 % | **4.7 %** | 45 % | 3 left, all < 0.45 m/s, reverse→forward overshoot |
| + anisotropic error, sub-cell map (doors) — privileged oracle for reference | 72 % | 1.6 % | 27 % | oracle approaches doors off-axis; filter correctly refuses |
| navigator fine-tuned with the filter, 1 M steps (envelope observed) | 61 % | **0 %** | 39 % | half the timeouts never leave the start: unknown wedge ahead |
| + near-field ground cameras, look-around primitive | **77 %** | 1.6 % | 22 % | the one collision during a 150 N push |

Root causes found and fixed (none of them was "tune a weight"):

1. **The pods cannot see 2.25 m straight ahead** past the Mk2.1 tyre. With the depth camera dead, the area ahead of
   the spawn point stayed unknown and every motion was inadmissible. The spawn disc is known free (start clearance
   > 0.75 m by construction), free evidence no longer decays (a robot standing still would otherwise forget the
   ground beside it, which the pods cannot re-observe), occupied evidence still fades (pedestrian ghosts).
2. **Recovery could not rank candidates** inside the guard band (every path's minimum is the present pose): now
   "never closer than now, toward the most room". The correction alternated ±0.75 rad/s each step, swinging the
   pendulum ~1 rad and translating the robot at 0.5 m/s while commanded to stop: switching and reversal cost.
3. **The response model was identified in calm conditions.** Under the benchmark's ground, wind and payload, the
   1 s path-prediction error is 12-23 cm p95 even at low speed. `sysid_low.py` now measures a p95 error table per
   horizon and speed in field conditions; the filter uses it over the reaction phase.
4. **The simulated LiDAR was a single plane at hub height** and missed low obstacles in every test world (the
   Mid-360s sweep 59° in elevation). The map scan is now cut from a horizontal ring plus 8 downward rings whose ground
   bands tile 0.4-6 m, so every obstacle taller than 10 cm is returned (ground segmentation assumed; 1 ms per scan).
5. **"Stop" meant "zero velocity"**: under disturbances the robot wobbled at 0.1-0.3 m/s and drifted up to 0.4 m in
   5 s into the obstacle it had stopped for. Stop now means "stay here": at rest, an outer loop on the along-track
   displacement (gain 2/s, best of 0.5-4) bounds the excursion to 0.13-0.17 m worst case.
6. **The backup stop was assumed to follow the commanded arc**; when the yaw rate dies during braking the robot stops
   on a straighter path. Both braking paths must now be clear.

7. **Doors were closed by the error model and by the map's quantisation.** The field error is almost all
   along-track (cross-track p95 2 cm at 0.3 s): the footprint is now inflated by the cross-track error and the path
   shifted by the along-track one. And 5 cm cells + 2 cm range noise grew every surface by 2.5-5 cm (a 1.0 m door
   was 0.90-0.95 m on the map): cells keep the mean of their hit points and distances are measured to it.
8. **A penalty the policy cannot predict teaches it to stand still.** The first shielded fine-tune (correction
   penalty 1.0, nothing about the filter observed) lost success 81 → 45 % while its reward rose. The navigator
   now observes the filter's safe-speed envelope (fastest admissible forward speed for 7 yaw rates), penalty 0.3.
9. **Nothing could see the floor just ahead of a robot at rest.** Pod LiDARs are blind 2.25 m ahead, the depth
   camera sees the floor beyond 1.6 m. The pods' ground cameras of the Mk1 design are now modelled (91 % of the
   0.3-1.5 m floor; the wedge ahead of the tyre stays blind, as the design doc says), and the filter has the
   design's answer for the rest: when only unknown space blocks the request, it turns in place to look.

## Progress log — phase 2 (navigator)

| navigator (all under the safety filter; 4 scenarios × 16) | success | collision | timeout |
|---|---|---|---|
| shield-trained RL navigator (scans + depth + filter envelope) | 76.6 % | 3.1 % | 20 % |
| + egocentric map branch, PPO fine-tune 0.9 M | flat in training | | |
| privileged teacher: sampling MPC on the true geodesic (before fixes) | 61 % | 1.6 % | 38 % |
| privileged teacher after the fixes below | 90.6 % | 1.6 % | 7.8 % |
| DAgger student of that teacher (map-aware NavActor, 16 rounds, 320 k labels) | 57.8 % | 6.2 % | 36 % |
| **deployable MPC** (onboard route field + local map + LIO), after the pose split | **90.6 %** | **0 %** | 9.4 % |

What the teacher exposed (each found with `rl/diagnose_stuck.py`, which classifies every stall by what blocks
the least progress command the teacher wants):

10. **Backing out was impossible.** 86 % of the teacher's stalls were reverses blocked by unknown space (the wedge
    exists behind the tyre too). The look-around primitive now serves reverse requests as well.
11. **Single never-observed cells in the wedge blocked everything.** The 0.1 m ground-camera grid left holes in the
    5 cm map; an unknown cell whose 8 neighbours are observed free is now not an obstacle (real obstacles leave
    occupied evidence and span ≥ 2 cells), and the samples are jittered.
12. **The teacher priced doors out** (clearance penalty from 0.35 m; a 1.0 m door leaves 0.107 m).
13. **The local map was built with a pose that drifts against the world**, smearing walls by the drift accumulated
    over the map's memory (1.0 m door rooms: 50 % → 75 % with drift removed). A LiDAR-inertial stack registers each
    scan to its local map, so the env now keeps a *local* pose (registered; map, filter, MPC clearance) and a
    *global* one (localised against the prior map, 3 cm / 0.3°; drifting LIO without a prior map; wheel odometry
    for both when LIO is lost).

Why imitation failed: the privileged teacher's advantage is information (true geodesic, true clearance) the
student cannot reconstruct, and averaging its discrete choices (left or right of an obstacle) makes the student
dither. The deployable MPC needs no imitation at all: it reaches the teacher's score with onboard information only.

## Full benchmark — deployable MPC + safety layer (13 scenarios × 16 episodes, pod contacts counted)

| scenario | success | collision | of which at fault | timeout |
|---|---|---|---|---|
| clutter | 88 % | 6 % | 6 % | 6 % |
| forest (thin poles) | 75 % | 12 % | 12 % | 12 % |
| rooms, 1.0 m doors | 81 % | 6 % | 6 % | 12 % |
| maze | 81 % | 0 % | 0 % | 19 % |
| mixed | 88 % | 6 % | 6 % | 6 % |
| maze, no map | 81 % | 0 % | 0 % | 19 % |
| mixed, 40 % stale map | 81 % | 6 % | 6 % | 12 % |
| 6 pedestrians to 1.4 m/s | 62 % | 25 % | 6 % (static) | 12 % |
| LiDAR 60° blind sector | 100 % | 0 % | 0 % | 0 % |
| depth camera dead | 100 % | 0 % | 0 % | 0 % |
| dust: spurious returns 3 % | 100 % | 0 % | 0 % | 0 % |
| LiDAR odometry lost (wheel + gyro) | 31 % | 6 % | 6 % | 62 % |
| everything at once | 94 % | 6 % | 0 % | 0 % |
| **all** | **81.7 %** | **5.8 %** | **3.8 %** | 12.5 % |

Before this plan (nav v2, no safety layer, tyre contacts only): 58.9 % success, 20.3 % collisions.

14. **Dust**: one spurious return marked a cell occupied. Isolated-return filter (real obstacles span ≥ 2
    adjacent dense rays): dust 62.5 → 100 %.
15. **People** are tracked from the dense LiDAR (5 cm centroid noise, smoothed velocity, 1 s coasting), kept out
    of the static map, and the filter checks every rollout point against each person's predicted position at that
    time (1.5 s horizon, +0.2 m/s deviation). Every pedestrian contact left is a person walking into a robot that is
    stopped or backing away (at-fault pedestrian collisions 0/208).

What is left: the static contacts are side grazes at the end of a stop (median 0.19 m/s, map knew the obstacle
1-4 cm away) — lateral drift that position hold does not correct, and hard braking while turning; LiDAR-odometry
loss (the map is built on drifting wheel odometry: needs scan-to-map registration); maze timeouts.

## Consolidated benchmark (after straight-braking recovery and the LIO-loss fallback)

Deployable MPC + safety layer, 13 scenarios × 16 episodes, pod contacts counted, benchmark disturbances (rough
ground, 10 N wind + gusts, 3 kg off-centre payload, **150 N shoves every ~6 s**):

| scenario | success | collision | at fault | timeout |
|---|---|---|---|---|
| clutter | 94 % | 0 % | 0 % | 6 % |
| forest (thin poles) | 88 % | 0 % | 0 % | 12 % |
| rooms, 1.0 m doors | 81 % | 0 % | 0 % | 19 % |
| maze | 69 % | 6 % | 6 % | 25 % |
| mixed | 88 % | 6 % | 6 % | 6 % |
| maze, no map | 75 % | 6 % | 6 % | 19 % |
| mixed, 40 % stale map | 75 % | 12 % | 12 % | 12 % |
| 6 pedestrians to 1.4 m/s | 56 % | 31 % | 6 % | 12 % |
| LiDAR 60° blind sector | 88 % | 12 % | 12 % | 0 % |
| depth camera dead | 100 % | 0 % | 0 % | 0 % |
| dust: spurious returns 3 % | 88 % | 12 % | 12 % | 0 % |
| LiDAR odometry lost | 81 % | 6 % | 6 % | 12 % |
| everything at once | 88 % | 0 % | 0 % | 12 % |
| **all (208)** | **82.2 %** | 7.2 % | 5.3 % | 10.6 % |

**Where the at-fault contacts come from.** 10 of the 11 at-fault static contacts happened within 1 s of a 150 N
shove (a shove is active or just over in ~20 % of random moments, so this is not chance): a shove adds up to ~1 m/s
sideways, which no margin compatible with a 1.0 m door absorbs. **Static collisions without an external shove:
1 in 208 episodes (0.5 %)** — the plan's < 1 % target. **At-fault pedestrian collisions: 0**; the 4 pedestrian contacts
are people walking into a robot that is stopped or backing away.

16. **Braking while turning** swung the pods into what is beside the path: recovery from > 0.3 m/s brakes straight.
17. **Losing LiDAR odometry** left the map and the route on wheel + gyro dead reckoning (31 % success): each dense
    scan is now registered against the local map (local error 1-4 cm, global 3-19 cm over 7-13 m): 81 %.

Remaining: timeouts in mazes (dead ends discovered late, the 1.5 s MPC horizon) and pedestrian-dense scenes; the
learned navigator (76.6 % on the 4-scenario set) is below the deployable MPC (87.5-90.6 % on the same set), so the
deployed stack uses the MPC as the navigator, the learned policy for locomotion.

## Final benchmark (stall recoveries, route tracking, field-identified state-dependent braking)

Deployable MPC + safety layer + locomotion v3.1 on Mk2.1, 13 scenarios × 16 episodes, pod contacts and 150 N
shoves counted:

| scenario | success | collision | at fault | timeout |
|---|---|---|---|---|
| clutter | 100 % | 0 % | 0 % | 0 % |
| forest (thin poles) | 100 % | 0 % | 0 % | 0 % |
| rooms, 1.0 m doors | 62 % | 0 % | 0 % | 38 % |
| maze | 88 % | 0 % | 0 % | 12 % |
| mixed | 94 % | 0 % | 0 % | 6 % |
| maze, no map | 88 % | 0 % | 0 % | 12 % |
| mixed, 40 % stale map | 81 % | 0 % | 0 % | 19 % |
| 6 pedestrians to 1.4 m/s | 88 % | 12 % | 0 % | 0 % |
| LiDAR 60° blind sector | 94 % | 0 % | 0 % | 6 % |
| depth camera dead | 100 % | 0 % | 0 % | 0 % |
| dust: spurious returns 3 % | 94 % | 0 % | 0 % | 6 % |
| LiDAR odometry lost | 88 % | 0 % | 0 % | 12 % |
| everything at once | 81 % | 6 % | 6 % | 12 % |
| **all (208)** | **88.9 %** | **1.4 %** | **0.5 %** | 9.6 % |

Start of this plan: 58.9 % success, 20.3 % collisions. The three contacts left: two pedestrians walking into the
robot (not at fault) and one 0.11 m/s side graze. The weak spot is 1.0 m doors for a 0.79 m robot (timeouts, never
contacts).

18. **The navigator reversed for minutes** (1.5 s horizon: reversing gains 0.75 m, turning round nothing): heading
    alignment term + small reverse cost.
19. **It reached doors off-centre**: it tracked the remaining route length, not the route line. It now penalises
    the cross-track distance from the planner's costmap-centred path.
20. **Stalls**: recoveries as production stacks have them — settle (reverse is non-minimum phase while the pendulum
    swings), back off along the seen-free trail, face the path, retry with stronger path tracking.
21. **Braking from a dynamic state was identified in calm conditions only.** In field conditions a stop from
    0.5 m/s takes up to 1.47 m with the pendulum still swung into a push (0.41 m from a neutral one). The filter
    now takes the braking distance by speed *and* pendulum angle, and extrapolates the measured acceleration
    through the dead time.
22. **The known-free disc had holes**: sampled at exactly the cell pitch, floating-point flooring skipped rows.

## Film

`media/gyra_mk21_film.mp4` (1080p, 24 fps, 6 min 58 s). Sections: title, assembly of the CAD
(79 parts in five groups), x-ray of the drive / levelling / lean / sensor subsystems, how it
moves, steers and stays level (the levelling plot is recorded mission data), the software
architecture, the safety filter on a real map snapshot, then the field mission
(`media/showcase.npz`: 17/17 waypoints, 238 m in 238.9 s, no collisions; the doorway section
plays at 2x) and the results card.

| step | command | output |
|---|---|---|
| record mission | `python rl/showcase.py` | `media/showcase.npz` |
| explainer scenes | `python render/explainer.py --scene move\|steer\|level\|arch\|filter --out F/explainer` | PNG frames |
| CAD shots + mission | `render/render_video.sh F` (Blender 4.2 Cycles, 12 fps rendered) | `F/robot`, `F/mission` |
| compose | `python render/compose_final.py --frames F --explainer F/explainer --out film.mp4 --results '[...]'` | 1080p film (12 to 24 fps by motion interpolation) |

The repository copy is a two-pass 1.7 Mbit/s encode to stay under GitHub's 100 MB file
limit. Fonts: Poppins (upright weights only), DejaVu Sans as fallback for Greek symbols.
High-quality (1080p or 4K, every frame, GPU) rendering: [10_render_film_tutorial.md](10_render_film_tutorial.md).


## Low-level stopping study

Question: the learned low level (loco31) stops in 0.75 m per m/s (median; p95 1.51, worst 2.77) and
creeps 0.26 m afterwards. Can a shorter stop make navigation safer? Answer so far: **no** - every
variant that stops shorter navigates worse, because this robot is non-minimum phase. The deployed
stack is unchanged (loco31, classical base, `GYRA_STOPCTRL=0`); everything below is opt-in.

Stops from field states (`rl/stop_ctrl.py --sweep-cascade [--policy-only]`, 120 stops, rough ground /
wind / payload) and on navigator-like command sequences (`--nav-bench`: the MPC's candidate speeds held
0.1-0.5 s, a third of them stops):

| low level | field stop, m per m/s (p50 / p95 / max) | creep p95 | post-stop surge p95 | yaw-rate error p50 |
|---|---|---|---|---|
| loco31 (deployed) | 0.75 / 1.51 / 2.77 | 0.26 m | 0.53 m/s | 0.13 rad/s |
| loco31 + external stop controller (`StopCascade`) | 0.45 / 0.77 / 0.91 | 0.04 m | 1.14 m/s | 0.26 rad/s |
| loco32: fine-tuned on the navigator command mix (`GYRA_CMD=nav`) | 0.74 / 1.30 / 1.69 | 0.14 m | 0.49 m/s | 0.12 rad/s |
| loco34: base with model feedforward (`GYRA_BASE=model`) + fine-tune | 0.47 / 0.87 / 1.42 | 0.40 m | 0.89 m/s | 0.19 rad/s |

Held-out navigation (`GYRA_SEED0=100`, as-built sensors with lane cameras, 208 episodes; each low
level with its own re-identified braking, reach and path-error tables):

| low level | success | collisions | at fault | timeouts |
|---|---|---|---|---|
| loco31 | 93.3 % | 1.9 % | 3 | 4.8 % |
| loco31 + stop controller (before bumpless hand-back and steering priority) | 78.8 % | 16.8 % | 31 | 3.8 % |
| loco32 | 89.9 % | 3.8 % | 1 (7 pedestrians walking into it) | 6.3 % |
| loco34 | 81.2 % | 15.4 % | 28 | 3.4 % |

What was found, in order:

1. **The classical base fights hard stops.** Its speed loop slews the reference at 2.5 m/s^2 but has
   no feedforward of the pendulum angle that deceleration needs (~64 deg); the PI asks for it only at
   ~1.3 m/s of error. Stopping from 2 m/s the policy's residual sat at its -60 N m limit while the
   base pushed back with +30..+54 N m. `ModelController` (`sim/gyra2_sim.py`) adds the identified
   model's feedforward (angle for the reference's acceleration, torque that holds the pendulum there,
   60 deg cap): the base alone stops in 0.26 / 1.04 / 2.52 m from 1 / 2 / 3 m/s (classical 0.49 /
   1.75 / 3.69), downhill 8 deg 0.58 / 1.82 / 3.19 (0.92 / 2.72 / 4.29).
2. **An external stop controller fights the policy.** Taking over the common drive torque during
   stops (`StopCascade`: pendulum-capped braking cascade, disturbance observer, equilibrium
   feedforward, bumpless hand-back, steering priority) stops best in isolation but halves turn
   tracking and doubles the surge; the navigator toggles stop / go every 0.1-0.5 s.
3. **Training never contained the navigator's commands** (10 % stops, half of them ramped, held
   1.5-5 s). `GYRA_CMD=nav` adds them; it trims the tail (loco32) but leaves the median.
4. **The reason shorter stops hurt: the sphere is non-minimum phase.** To brake, the motors swing the
   pendulum forward, and the reaction first drives the shell forward. Harder braking means a bigger
   lurch toward what the robot is stopping for; the filter's response model (dead time +
   acceleration limit) cannot represent it, and its 0.25 m/s command steps near walls then drive the
   loop into rocking (velocity traces of the new collisions: commands 0 / +-0.25, speed +-0.5-0.8).
   Shaping the reference (critically damped, `GYRA_WREF`) does not remove it: zero-residual surge p95
   0.74 / 1.08 / 1.15 m/s at w_ref 4 / 6 / 8 against 0.35 for the classical base.

What would make shorter stops pay off: a navigation-layer motion model with the pendulum state (the
reach table already indexes it), so the filter predicts the lurch and the MPC stops smoothly instead
of stepping; only then a faster low level. Tools: `rl/stop_ctrl.py` (`--identify`, `--sweep-cascade`,
`--nav-bench`, `GYRA_ZERO_RES=1` for the base alone), `rl/sysid_low.py --stops / --arcs-only` with
`GYRA_LOW` / `GYRA_BASE`, results in `rl/runs/nav2/bench/heldout_lane_*.json`.
