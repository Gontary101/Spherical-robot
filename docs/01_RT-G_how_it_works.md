# How the RT-G actually works

This is a reverse-engineering report on the **RT-G** spherical patrol robot. It is built from
the only primary technical source available, the 2025 Zhejiang University paper on the
same machine, cross-checked against press coverage and the company's own statements.

---

## 1. Who makes it

| | |
|---|---|
| Product | **RT-G** "spherical amphibious patrol robot" (police variant shown in Wenzhou, Dec 2024) |
| Company | **Logon / Luoteng Technology** (逻腾科技, also styled *Rotunbot*), Qiantang district, Hangzhou |
| Founder | **Wang You** (王酉), associate professor, Institute of Cyber-Systems & Control, Zhejiang University |
| Research name | **RotunBot**. Paper: B. Zhang, F. Zhang, H. Chen, **Y. Wang**, J. Hao (*Luoteng Hangzhou Technology*), Z. Luo, G. Li, *"A High-Speed Capable Spherical Robot"*, arXiv:2511.01288 |
| History | Development started in 2017 (planetary-exploration inspiration). Mass production started in 2023 after a second funding round. Other sizes include 60 cm and 80 cm variants. |

Published numbers don't all agree. Press articles say **125 kg**. The paper and the Hangzhou
government article say **≈160 kg**. The other figures: shell radius **40 cm** (Ø 80 cm),
**35 km/h** claimed (10 m/s measured in the paper), 0→30 km/h in 2.5 s, "withstands 4 t
impacts", **10 h** endurance (≈120 km), −45…85 °C, amphibious.

## 2. The internal architecture (from the paper's CAD figure and Table I)

```
                 external component (sensor pod)          external component (sensor pod)
                 rigidly bolted to the MAIN SHAFT          rigidly bolted to the MAIN SHAFT
                        │                                            │
   ┌────────────────────┴────────────────────────────────────────────┴────────────┐
   │  ═══════════════════════  aluminium MAIN SHAFT (spans the sphere)  ════════  │
   │      │ (1) main drive: motor on the frame → gear → FLANGE bolted to SHELL    │
   │      │ (3) momentum-wheel motor → belt → MOMENTUM WHEEL Ø420 mm, 9.8 kg      │
   │      │     (wheel axis = direction of travel → it produces ROLL torque)       │
   │      │ (2) auxiliary drive: lifts the HEAVY PENDULUM sideways (about x)       │
   │      ▼                                                                       │
   │   HEAVY PENDULUM 73.4 kg (cast steel + lead, the battery is inside it)        │
   └──────────────────────────────────────────────────────────────────────────────┘
          SHELL 27.6 kg: polyester-fibre-reinforced RUBBER with an off-road tread
```

Mass breakdown (paper, Table I). Total ≈ 160 kg, shell radius 40 cm, pendulum arm ≈ 27 cm:

| component | material | kg |
|---|---|---|
| heavy pendulum (incl. battery) | cast steel + lead blocks | 73.4 |
| spherical shell (incl. flange plate) | polyester-fibre-coated rubber | 27.6 |
| momentum wheel | cast steel | 9.8 |
| external components (both sides) | nylon shield + sensors | 15.1 |
| main shaft + everything else | aluminium alloy | 34.3 |

Electronics: 13S, 50 Ah Li-ion pack (≈2.4 kWh). An STM32F407 talks to three servo drives
over CAN at 100 Hz. An onboard computer runs navigation and SLAM over UDP. Industry
coverage adds stereo cameras with a baseline over 50 cm, mounted in the side shells
(the "insect compound eye" camera ring), plus GNSS, an IMU and motor encoders. Larger
models use an Intel i7 with an RTX 2060.

### How it drives
The **main motor sits on the internal frame** and drives a gear meshed with a **flange
that is bolted to the shell**. The motor's reaction torque has nowhere to go except into
the frame. The frame rotates forward around the shaft and lifts the 73 kg pendulum off
vertical. The centre of mass moves ahead of the contact point and gravity rolls the ball.
**Drive torque is limited by gravity × pendulum offset, not by the motor**:

  τ_max = m_p · g · L_p · sin θ  ⇒  max grade ≈ asin(m_p L_p / (M R)) ≈ 18° for the RT-G.

### How it steers ("gyro steering")
The **auxiliary motor swings the pendulum sideways**, which moves the centre of mass
laterally and tips the shaft over. The shell is spinning, so it has angular momentum
**L** along the shaft. The sideways gravity torque τ, along the direction of travel,
makes **L** precess about the vertical: dL/dt = τ. The robot **yaws toward the side it
leans to**, like a rolling coin. The turn rate is Ω = τ / L, so the turn radius grows
with v². This is why the RT-G turns tightly at walking pace but needs very wide arcs at
35 km/h. The paper sums it up: *"RotunBot completes steering tasks by moving forward and
backward while in a tilted state."*

### What the "gyroscope" really does
The **momentum wheel** (axis along the direction of travel) is a **reaction wheel for the
roll axis**. Speeding it up or slowing it down puts a roll torque on the body without
moving the centre of mass. The paper splits the job between the two actuators:

* pendulum = **roll angle** setpoint (segmented PI + model feed-forward, eq. 3: θ_ref = θ_hope + sin⁻¹(m v² tan θ_hope / (m_p g r)))
* momentum wheel = **roll rate → 0** (segmented PD)

Without the wheel, the single-pendulum ball diverges in roll above about 2 m/s when
turning and decelerating. With it, the authors report 10 m/s straight-line runs with
less than 0.05 rad roll ripple. They also cleared a 17 cm step at speed, climbed 10°
slopes, and recovered from a 0.4 rad roll kick (hitting a 10 cm brick at 4 m/s) in
under 2 s.

## 3. Your hypothesis, checked

> *"the wheel turns with a motor while the axis stays stable, so you can put LiDAR and cameras on the sides, and use a rubber wheel instead of a transparent ball"*

**Mostly right.** The details:

1. ✅ **Only the central shell rotates.** The side pods bolt to the main shaft, so they
   never spin with the tyre. Sensors look out directly, with no transparent, scratched,
   dirty or refracting hull in the way. That is also why the shell can be an opaque,
   tough, **fibre-reinforced rubber tyre with a real tread**.
2. ⚠️ **"Stable" means gravity-stabilised, not gyro-stabilised.** The shaft and pods are
   held level only by the hanging pendulum, and they are rigid with the frame that
   pitches with it. When the RT-G accelerates, brakes or climbs, **the pods pitch with
   the pendulum** (tens of degrees in a hard launch). When it leans into a turn, they
   **roll**. The "compound-eye" software de-jitters the video using IMU attitude.
3. ✅ **It turns with gyroscopic physics**, but the gyroscope is the **spinning tyre
   itself**. The momentum wheel only damps roll wobble; it doesn't steer.
4. ⚠️ It is **non-holonomic, like a unicycle**. It can't roll sideways, and it turns
   only while rolling, or with fore-and-aft shuffling.

## 4. RT-G vs. "traditional" ball robots

| | Sphero / BB-8 / hamster-ball (internal drive unit) | Omni-wheel / ballbot-in-a-ball | Classic pendulum sphere (GroundBot, Rollo, BHQ) | **RT-G / RotunBot** |
|---|---|---|---|---|
| What rotates | whole shell, any direction | whole shell, any direction | whole shell (1-2 axes) | **central tyre band only (1 axis)** |
| Drive | small car or wheels pushing the shell inside | omni-wheels on the shell inside | pendulum CoM offset | **pendulum CoM offset (geared to flange)** |
| Sensors | inside, looking through the shell (or none) | inside | inside or in transparent side domes | **in opaque side pods on the axle, direct view** |
| Shell | transparent or smooth plastic | smooth | polycarbonate | **rubber tyre with tread, 27.6 kg** |
| Steering | differential drive inside | holonomic | lateral pendulum (wobbly) | **lateral pendulum + momentum-wheel roll damping** |
| Weakness | slips, low torque | slips, complex | wobble, slow | **heavy, grade-limited, pods still pitch** |

Its closest ancestor is **GroundBot** (Rotundus, Uppsala, 60 cm, glass camera domes on
the axle ends). RT-G's real contributions are scale, a **rubber traction tyre**, a
**high-speed controller** and the **momentum wheel that separates roll stabilisation
from steering**.

## 5. What I took from this for GYRA

* Keep: sensors on the non-rotating axle ends, a rubber tyre over most of the sphere,
  pendulum drive, lean-steering, sealed amphibious body.
* Fix: (a) the pods pitch with the pendulum, (b) a single reaction wheel saturates and
  only handles roll, (c) no turn-in-place, (d) the drive gear sits at small radius on a
  side flange, (e) the lateral pendulum must hold the lean against gravity with power.

See [`02_GYRA_Mk1_design.md`](02_GYRA_Mk1_design.md).

## Sources
* Zhang, Zhang, Chen, Wang, Hao, Luo, Li, *A High-Speed Capable Spherical Robot*, arXiv:2511.01288 (Zhejiang Univ. + Luoteng Hangzhou Technology): https://arxiv.org/abs/2511.01288
* Singhal, Modi, Gupta, Vachhani, *Wobble control of a pendulum actuated spherical robot*, arXiv:2301.06433: https://arxiv.org/abs/2301.06433
* Hangzhou government news, *Hangzhou's spherical robot grabs global attention*: https://www.ehangzhou.gov.cn/2025-02/21/c_292708.htm and https://www.ehangzhou.gov.cn/2025-02/27/c_292781.htm
* 机器人大讲堂, *"大黑球"火爆全网…浙大王酉团队全面起底*: https://leaderobot.com/news/4946
* 网易, *逻腾王酉：球形机器人布局千亿市场*: https://www.163.com/dy/article/IB5HUII10511C9QL.html
* New Atlas, *Chinese police trial amphibious crime-fighting robot sphere*: https://newatlas.com/robotics/chinese-police-amphibious-robot-ball/
* Interesting Engineering: https://interestingengineering.com/innovation/chinas-spherical-robot-cop-captures-criminals
* Yanko Design: https://www.yankodesign.com/2024/12/17/rolling-crime-fighting-machine-the-futuristic-robot-redesigning-law-enforcement/
* Robots Asia product page: https://www.robotsasia.com/Rotunbot-RT-G.htm
* Rotundus GroundBot: https://newatlas.com/rotundus-groundbot/20259/ and https://orionrobots.co.uk/wiki/rotundus.html
