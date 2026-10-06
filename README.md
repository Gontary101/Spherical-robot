# GYRA: rubber-tyred spherical robots inspired by the RT-G (Mk1 → Mk2 + learned control)

![GYRA Mk1](media/gyra_hero.png)

This repo has three parts (the third is the Mk2 section below):

1. **A teardown of how the RT-G (Rotunbot/RotunBot) police robot actually works**, from
   the Zhejiang University paper behind it plus press and company sources:
   **[docs/01_RT-G_how_it_works.md](docs/01_RT-G_how_it_works.md)**
2. **GYRA Mk1**, a new design that keeps the RT-G's best idea (a wide rubber tyre
   rolling around a non-rotating sensor axle) and adds:
   * a **level sensor spine**, mechanically separate from the drive pendulum: the pods
     stay within 0.75° while the pendulum swings 46°;
   * a **scissored, counter-rotating CMG pair** that gives **pure roll torque**
     (stops lean-steer wobble) *or* **pure yaw torque** (turns on the spot at zero
     speed), with zero net momentum when idle;
   * an **equatorial internal ring-gear drive** (30:1) with **two anti-backlash
     pinions**;
   * a **self-locking worm lean drive** that holds a lean at zero power.

   Full spec: **[docs/02_GYRA_Mk1_design.md](docs/02_GYRA_Mk1_design.md)** ·
   simulation: **[docs/03_simulation.md](docs/03_simulation.md)** ·
   interactive 3-D viewer: **[docs/viewer/index.html](docs/viewer/index.html)**
   (open locally in a browser)

![annotated cutaway](media/gyra_cutaway_annotated.png)

## Mk2: split differential tyre + learned control (latest)

![GYRA Mk2](media/mk2_cutaway_annotated.png)

Mk2 drops the CMGs. The tyre becomes **two independently driven halves whose crowns
are offset 50 mm outward**, so the robot stands on two contact patches and steers with
the torque *difference* between the halves. The freed mass goes into an 11 kg tungsten
+ 936 Wh bob, and the pendulum can now swing a full 360°. On top of this sit two
**asymmetric-PPO policies** trained in MuJoCo on the CAD-derived model:
* a **locomotion/teleop policy**: a residual over the classical loop, fed by IMU and
  encoders;
* an **autonomous navigator**: pod LiDARs with the real blind wedge, stereo depth,
  drifting odometry.

| | RT-G | Mk1 | **Mk2 + learned control** |
|---|---|---|---|
| turn in place | can't | 9°/s | **176°/s** (~20×) |
| 6 m/s turn at 0.6 rad/s | — | — | **stable, 7° roll** (classical loop falls) |
| falls, random commands + randomised physics + shoves | — | — | **8 %** (classical: 67 %) |
| sensor-pod pitch while accelerating | ≈ 40° | 0.75° | **0.39°** |
| grade (sim) | 10° tested | 10° | **14°** |
| battery | 2.4 kWh @ 160 kg | 468 Wh @ 41 kg | **936 Wh @ 45 kg** |
| autonomous navigation, 100 random maps | claimed | — | **79 % success, 2 % collisions, 1.35 m/s** (VFH baseline: 72 %, 11 %, 0.86 m/s) |

What does *not* get 10× better, and why (gravity-limited grade and acceleration), is
covered in [docs/04](docs/04_mk2_research_and_bounds.md). Full numbers:
[docs/07_scorecard.md](docs/07_scorecard.md) · design: [docs/05](docs/05_GYRA_Mk2_design.md) ·
learning: [docs/06](docs/06_learned_control.md) · 3-D viewer: [docs/viewer/mk2.html](docs/viewer/mk2.html)

| Learned locomotion vs classical | Mk2 classical benchmark |
|---|---|
| ![rl](media/rl_loco_eval.png) | ![mk2](media/mk2_bench.png) |

---

## The short version of "how RT-G works"

* The **central rubber shell** (fibre-reinforced rubber with a tread, 27.6 kg) is the
  only part that rolls. **The side pods are bolted to the main shaft**, so cameras,
  LiDAR and GNSS see out directly; there's no transparent hull.
* A motor on the internal frame turns the shell. Its reaction torque **lifts a 73 kg
  pendulum** (battery inside), and gravity does the driving. Grade and acceleration are
  limited by m·g·L of the pendulum, not by the motor.
* **Steering = leaning.** A second motor swings the pendulum sideways. The spinning
  shell precesses toward the low side, like a rolling coin, so the turn radius grows
  with v².
* A **42 cm momentum wheel** with its axis along the direction of travel only damps
  roll wobble. That's what let the RT-G reach 10 m/s.
* Correction to the "axis stays stable" idea: the pods are held level only by gravity,
  and **they pitch with the pendulum** when the robot accelerates, brakes or climbs.
  GYRA fixes this.

## GYRA Mk1 at a glance (all derived from the CAD model)

| | | | |
|---|---|---|---|
| Diameter | **600 mm** | Mass | **41.4 kg** (CoM 59 mm below centre) |
| Top speed | **6.4 m/s** (23 km/h) | Drive | 2 × 6374 BLDC → belt → ring gear, **30:1** |
| Grade | **10°** demonstrated in sim (13.6° static) | Lean torque | **11.2 N·m** sustained, + **18.5 N·m** CMG burst |
| Turn radius | 1.3 m @ 1 m/s · 5.8 m @ 2 m/s (sim) | Turn in place | **yes** (CMG yaw mode) |
| Sensors | 2 × Livox Mid-360, 6 × GS cameras (0.56 m stereo baseline), dual-antenna RTK | Compute | Jetson Orin NX 16 GB |
| Battery | 468 Wh, ~4 h / 45 km @ 3 m/s | Water | floats at 41 % draft, chevron tread paddles |
| Parts | 89 | Interferences | **0 across 45 mechanism poses** (B-rep check) |

| Exploded | Section through the drive plane |
|---|---|
| ![exploded](media/gyra_exploded.png) | ![section](media/gyra_section_side.png) |
| **Leaning into a turn** | **X-ray** |
| ![lean](media/gyra_leaning_turn.png) | ![xray](media/gyra_xray.png) |

### Simulation highlights (MuJoCo, model generated from the CAD inertias)
| Lean-steer at 2 m/s: CMG off vs on | Turn in place at zero speed |
|---|---|
| ![turn](media/sim_turn.png) | ![spin](media/sim_spin.png) |

## Repository layout
```
cad/gyra2_cad.py         Mk2 parametric CAD (split tyre) -> cad/out2/
sim/gyra2_model.py       Mk2 MuJoCo model + motor torque-speed model
sim/gyra2_sim.py         Mk2 classical controller + benchmark
rl/loco_env.py           locomotion env (sensor-realistic actor obs, privileged critic, domain randomisation)
rl/nav_env.py            navigation env (pod-LiDAR blind wedge, stereo depth, drifting odometry)
rl/ppo.py, vec_env.py    asymmetric PPO + subprocess vector env
rl/train_*.py, eval_*.py training + evaluation; rl/runs/{loco,nav}/ deployed policies + eval.json
calc/scorecard.py        RT-G vs Mk1 vs Mk2 table from all result files
cad/params.py            single source of truth for dimensions
cad/gyra_cad.py          parametric CadQuery model → STEP / STL / mass properties / interference check
cad/export_glb.py        → cad/out/gyra_mk1.glb
cad/out/                 gyra_mk1.step (coloured assembly), stl/, glb, mass_properties.json, interference_report.json
calc/sizing.py           closed-form sizing from the CAD mass properties → calc/out/sizing.json
sim/gyra_model.py        MJCF generator (CAD inertias → MuJoCo)
sim/scenarios.py         closed-loop experiments → sim/out/summary.json, media/sim_*.png
render/render_gyra.py    Blender Cycles renders (hero, cutaway, x-ray, exploded, sections, poses)
render/annotate.py       visibility-checked call-out labels
render/build_viewer.py   self-contained three.js viewer → docs/viewer/index.html
docs/                    research report, design spec, simulation report
```

## Reproduce
```bash
pip install cadquery mujoco numpy scipy matplotlib trimesh pillow torch
python cad/gyra_cad.py --check        # CAD, exports, mass properties, 45-pose interference check
python cad/export_glb.py
python calc/sizing.py
python sim/scenarios.py
BLENDER=/path/to/blender render/render_all.sh
python render/annotate.py media/gyra_cutaway.png media/gyra_cutaway_annotated.png
python render/build_viewer.py            # Mk1 viewer;  `mk2` for the Mk2 viewer
# Mk2 + learned control
python cad/gyra2_cad.py --check && python cad/export_glb.py out2
python sim/gyra2_sim.py
python rl/train_loco.py --steps 4e6 && python rl/eval_loco.py
python rl/train_nav.py --loco rl/runs/loco/best.pt --steps 7e5 && python rl/eval_nav.py
python calc/scorecard.py
```
Tested with CadQuery 2.8 / OCCT 7.9, MuJoCo 3.15, Blender 4.2 LTS.

## Status
Mk1 is a **concept-level CAD and simulation design**, not yet built. Gears use
trapezoidal tooth approximations. Purchased parts are modelled as envelopes with their
real masses. The main open items are in §7 of the design doc: grade capability, seal
validation and flywheel spin-down/burst testing.
