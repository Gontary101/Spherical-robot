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

The remaining timeouts are the navigator fighting a filter it has never seen: phase 1d (shielded fine-tuning with a
correction penalty, the executed command in the observation) is running.
