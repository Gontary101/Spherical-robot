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
