# Plan: a faster, calmer, more general navigation stack

Starting point (docs/09, "Low-level stopping study"): the navigation layer predicts motion with a
closed-loop response model that includes the pendulum (`rl/resp_model.py`, default). With the
deployed low level (loco31) every remaining collision is a pedestrian walking into the robot (0 at
fault in 416 episodes), success is 92-94 %, and **timeouts (5-6 %) are now the main failure**. A
low level that stops twice as short (loco34) still navigates worse (85 %, 4.3 % collisions): it rocks
and its turns chatter in tight rooms, while the safety filter corrects it every step.

## Rules for every phase

* **One pipeline per low level.** A candidate is always judged with its own response model, braking
  / reach tables and error envelope (`rl/lowlevel_pipeline.py`). Stale tables made two earlier
  comparisons wrong.
* **Paired, larger comparisons.** Held-out (`GYRA_SEED0=100`) and standard seeds, 208 episodes each,
  same seeds for both arms; report the success difference with a paired bootstrap 95 % interval.
* **Acceptance.** A change ships when (a) success is not lower beyond noise (interval includes or
  exceeds 0), (b) at-fault collisions do not rise, (c) nothing in the other benchmarks regresses.
* **Opt-in first, default after acceptance**, with an environment switch to restore the old
  behaviour. Commit every step.

## Phases

| # | phase | question it answers | acceptance |
|---|---|---|---|
| P0 | evaluation pipeline + paired statistics | is a candidate better, honestly measured? | reproduces the loco31 numbers |
| P1 | calm safety-filter corrections | is the rocking caused by the filter's 0 / +-0.25 steps? | loco34 at-fault collisions <= loco31's, no loco31 regression |
| P2 | doorways and stalls | can timeouts be halved without new collisions? | timeouts <= 3 %, collisions unchanged |
| P3 | low level trained on the real command stream | can the fast stopper also manoeuvre calmly? | navigator-like bench: turn error <= 0.13 rad/s, surge <= loco31's; nav >= loco31 |
| P4 | yielding to pedestrians | can the robot avoid people who walk into it? | pedestrian collisions halved, no at-fault rise |
| P5 | terrain-aware braking and response | less caution on good ground, same safety on bad | slopes / ice in navigation: 0 at-fault static collisions; flat: faster |
| P6 | online adaptation of the response model | does it stay right under wear, payload, terrain the training never saw? | prediction error under unseen conditions within the envelope; no regression |
| P7 | hardware identification procedure | can the build produce its own model and tables? | procedure + tool that turn logged commands / speed / pendulum into the files, verified on simulated logs |

### P0 - evaluation pipeline (`rl/lowlevel_pipeline.py`)

`python rl/lowlevel_pipeline.py --name loco34 --low rl/runs/loco34/final.pt --base model` runs, for
one low level: response-model identification and validation, braking / reach tables (`sysid_low
--stops`), error envelope of the response model's predictions (`--arcs-only`), held-out and standard
navigation benchmarks, and writes `rl/runs/nav2/bench/pipeline_<name>.json`. `--compare A B` prints the
paired comparison. Extra environment (e.g. a filter variant) passes through to the benchmarks and
is recorded in the report.

### P1 - calm corrections

The filter returns the admissible candidate nearest the request on a fixed grid, re-chosen every
0.1 s: near walls the outputs alternate 0 / +-0.25 m/s and, on this non-minimum-phase robot, every
switch swings the pendulum. With the response model the filter can price that: among admissible
candidates, add the predicted pendulum excursion and a sign-flip cost to the selection cost (the
admissibility test, i.e. the safety guarantee, is unchanged).

### P2 - doorways and stalls

Timeout forensics first (`rl/diagnose_stuck.py`): where the robot stands, what the filter blocks,
what the MPC wanted. Expected levers: the error envelope at crawl speed now that predictions are
better; a door-crossing mode (align on the door axis, then cross straight at a crawl).

### P3 - low level on the real command stream

Record the filtered command streams of the navigation benchmark (near walls, doors, pedestrians) and
replay them in locomotion training next to the operator commands; reward turn-rate tracking at low
speed and penalise speed rising in the direction of motion after a stop command. Fine-tune loco34,
then P0.

### P4 - yielding

The filter knows each tracked person's position and velocity. When a person's predicted path
crosses the robot's position within the horizon and the robot is slow, candidates that increase the
predicted minimum distance are preferred over standing still.

### P5 - terrain-aware braking and response

The locomotion policy's estimator already predicts slope, friction, payload and wind. Identify the
braking tables and the response model per condition class and select them from the estimate, falling
back to the worst case when the estimate is uncertain. Add slopes and low-friction patches to the
navigation scenarios to measure it.

### P6 - online adaptation

Recursive least squares with forgetting on the response model's parameters while driving; the error
envelope tracks live prediction residuals and may only shrink after a minimum amount of evidence.

### P7 - hardware procedure

A ten-minute identification run (navigator-like command sequences in an open space), logging
commands, LiDAR-inertial speed, yoke encoder and IMU; a tool that turns such a log into the response
model, braking tables and error envelope; verified on simulated logs with the same channels and
noise.

## Progress log

(filled in as phases complete)
