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

All numbers: deployed low level loco31 unless stated, held-out + standard seeds, 416 episodes, paired
against the previous default with a bootstrap 95 % interval (`rl/lowlevel_pipeline.py --compare`).

| step | change | success (paired) | collisions / at fault | timeouts | verdict |
|---|---|---|---|---|---|
| P0 | pipeline reproduces loco31 | 94.2 / 92.3 % (sets) | 9 / 1 | 19 | noise floor about +-2 % per set |
| P1 | calm filter corrections (`FILTER_CALM` 1 / 3) | -1.4 % (-3.1 .. +0.2) / -0.5 % | 10 / 1, 9 / 3 | | rejected |
| P2 | crawl speeds (`GYRA_CRAWL`) | +0.7 % (-1.2 .. +2.9) | 8 / 1 | 17 | kept |
| P2 | + door-crossing mode (`MPC_DOOR`) | +1.9 % (-0.2 .. +4.1) | 8 / 2 | 12 (2.9 % per set) | **default** |
| P4 | people-aware rollouts (`MPC_PED`) | -0.5 % (-1.7 .. +0.7) | 9 / 1 | | rejected |
| P4 | + speed limit near people | 0.0 % (-1.2 .. +1.2) | 7 / 1 | | rejected |
| P5 | slope-aware braking tables (`GYRA_TERRAIN`) | +1.0 % (-0.7 .. +2.6) | 3 / 0 | | off: worse on slopes |
| P3+P5 | loco35: model base + slope feedforward + replayed commands + lurch penalty | -4.3 % (-7.2 .. -1.4) | 23 / 21 | | rejected |
| P5 | slope feedforward in loco31's own base (`GYRA_KSLOPE`) | **+2.4 % (+0.2 .. +4.6)** | **1 / 0** | 9 | **default** |
| P5 | same, tables re-identified with it | +0.5 % (-1.4 .. +2.4) | 6 / 4 | 12 | tables too optimistic |
| P5 | same, tables merged over both identifications (`--merge`) | +1.7 % (-0.5 .. +3.8) | 3 / 0 | 10 | **default** |

Findings, in the order they changed the plan:

* **P1.** Pricing the filter's corrections by the predicted pendulum swing changes nothing: loco34's
  rocking is its own behaviour, not the filter's alternation.
* **P2.** Timeout forensics (`rl/diagnose_stuck.py --from <report>`): the filter held the robot in 11
  of 203 stall samples; the navigator mostly chose the best admissible progress command. The robot
  approached 1.0 m doors obliquely along the planner's grid route and wedged on the near jamb (a
  r = 0.393 m circle with the filter's margins fits only a centred, near-perpendicular crossing), and the
  stall recovery backed off and re-aligned to the same route. The door mode finds the passage axis,
  centres it, approaches on it and crosses at a crawl. (The "ghost" map cells the old diagnosis
  reported were an artefact of its 0.1 m truth grid: against exact geometry the map's hit points lie
  p50 0.0 / p90 1.5 cm from the surfaces.)
* **P4.** The remaining pedestrian collisions are people perceived only at contact (blind wedge,
  depth only): a sensing limit no planning cost changes.
* **P5.** Slopes were the largest weakness: on 8 / 12 deg 24 / 16 of 32 episodes succeeded, with 3 /
  12 at-fault static collisions, the robot rolling on at 0.2-0.45 m/s while commanded to stop. Cause:
  turning in place while stopped changes the slope's component along the heading, and the base's PI
  only followed it through its integrator (loco31 rolled 0.6-0.8 m in 5 s on 8 deg, up to 1.5 m on 12
  deg). The base now adds the standing pendulum angle for the measured slope along the heading (IMU +
  levelling angle; sin th = 2.46 sin slope) and the torque that holds it there, within the cascade's 80
  deg limit: drift turning in place 0.15-0.31 m (8 deg), 0.06-0.46 m (12 deg). It is zero on level
  ground, so the trained policy is unchanged there - and on the benchmark's rough ground, whose local
  slopes it also follows, collisions fell from 8 to 1. 12 deg remains poor (14 / 32): outside the
  envelope the low level was trained for (slopes to 10.5 deg) and the tables were identified at (8 deg).
  Slope-aware braking tables helped on level ground but not on slopes, where the failure was the low
  level, not the stopping distance; they stay off.
* **P3.** A low level fine-tuned on the recorded navigation command stream (10651 commands, switching
  every 0.2 s) with a lurch penalty, on the model base with slope feedforward (loco35), stops shorter
  (field p95 0.81 vs 1.51 m per m/s) but navigates worse: every low level on the aggressive base lurches
  more after a stop command (surge p95 0.84 vs 0.53 m/s), and navigation pays for it more than it gains.
* **P6.** One-step RLS adaptation of the response model made predictions worse everywhere (the model
  is fitted on multi-step error). An acceleration offset nudged by the 0.5 s speed error, beyond a dead
  band, leaves level ground unchanged and improves slopes (downhill 1 s p95 0.364 -> 0.319 m); kept as
  an opt-in tool (`resp_model.Adapter`), not wired into navigation: the slope feedforward measures the
  slope directly.
* **Tables are tail estimates.** Re-identifying loco31's tables with the slope feedforward gave less
  conservative extremes (0.38 vs 0.41-0.47 m from 0.5 m/s, 0.80 vs 0.85-1.15 m from 1.0 m/s) and lost
  1.9 % (-3.8 .. -0.2) against the same controller on the older tables. Every entry is the extreme of
  ~150 runs; the deployed tables are now the worst case over both identifications
  (`rl/lowlevel_pipeline.py --merge loco31 loco31k --into loco31m`).
* **P7.** `rl/hw_ident.py`: identification run, log channels, and the tool that turns a log into the
  response model, stopping distances and error envelope. Verified on a simulated 17-minute log with only
  the measured channels and their noise (LIO speed 0.03 m/s, gyro 0.02 rad/s, pendulum 0.003 rad):
  open-loop prediction from the log-identified model 0.10 / 0.18 / 0.27 m (p95 along-track at 0.5 / 1 /
  1.5 s) vs 0.10 / 0.20 / 0.29 m from the privileged one, lateral within 0.01 m; stopping distances
  from the log's 85 stops 0.39 / 0.81 / 1.17 / 2.42 / 2.60 m from 0.5 ... 2.5 m/s against the
  privileged field-condition table's 0.38 / 0.73 / 1.14 / 1.79 / 2.66 m (with its 5 cm): equal or larger
  except 6 cm short at 2.5 m/s, where the protocol has the fewest stops - on the robot, repeat the fast
  stops or take the larger of the log and a margin.

## Where it stands

Deployed: loco31 with the base's slope feedforward, the response model with the pendulum state,
crawl speeds and the door-crossing mode, tables merged over two identifications. Held-out 97.6 %,
standard 96.2 % success; 3 collisions in 416 episodes, none at fault (the start of this plan: 94.2 /
92.3 %, 9 collisions). Open: slopes beyond the low level's training envelope (12 deg: 14 / 32); the
pedestrian contacts, which need better coverage of the pods' blind wedge rather than planning; and a
low level that stops shorter without the lurch (every one tried so far traded one for the other).
