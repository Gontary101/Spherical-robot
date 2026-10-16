# 11 Build: GYRA Mk2.1 as it will be made

The concept CAD (`cad/gyra2_cad.py`, docs 05) shows what the robot is; the **build CAD** (`cad/build/`) shows how
to make it. Every part that is machined, printed, cast, cut or bought is a solid with its material and process.
So is every screw, nut, washer, insert, PEM nut, dowel, circlip and E-clip, with its standard, length and
tightening torque. The holes they need are cut from the joints, not drawn by hand. The bought parts use their
datasheet dimensions, and the hardware has McMaster-Carr part numbers. The model is then checked by machine:

- static interference;
- thread engagement;
- floating parts;
- a continuous proof that nothing collides anywhere in the range of motion;
- an assembly-path check that every subassembly can go in along the path it really takes.

@@STATUS@@

## 1. Files and commands

| | |
|---|---|
| `cad/build/spec.py` | every dimension and every bought part (part number, datasheet source, verified flag) |
| `cad/build/lib.py` | geometry kernel, ISO fastener tables (4762, 7380, 10642, 4032/7040, 7089, 273 clearances, tap drills, counterbores, McMaster 94180A inserts), engagement rules, torques, `Joint` → holes + fasteners |
| `cad/build/tyre.py` `pod.py` `spine.py` `yoke.py` `bob.py` `cots.py` | the five bodies and the bought-part models |
| `cad/build/assemble.py` | builds everything, applies the joints, writes `out/parts.json`, `joints.json`, `connections.json`, `mass_properties.json` |
| `cad/build/checks.py` + `sweep.py` | static interference, joints, floating parts, continuous motion clearance → `out/checks.json` |
| `cad/build/assembly.py` | assembly-path check → `out/assembly_check.json` |
| `cad/build/calcs.py` | 36 hand calculations from the simulated loads → `out/calcs.md` |
| `cad/build/bom.py` | BOM with McMaster numbers → `out/bom.md`, `out/bom.csv` |
| `cad/build/guide.py` | per-step fastener / torque table (section 5) → `out/assembly_steps.md` |
| `cad/build/export.py` | assembly STEP, one STEP per drawing, STL (printed parts), DXF (plates), A3 PDF sheets with hole tables → `out/release/` |
| `rl/actuator_loads.py`, `rl/yoke_range.py`, `rl/build_mass_check.py` | actuator duty, used yoke angle, and the control re-checked on the build masses |

```bash
python cad/build/assemble.py                  # ~80 s
python -u cad/build/checks.py                 # static + joints + floating + continuous motion (4 processes)
python -u cad/build/assembly.py               # assembly paths
python cad/build/calcs.py && python cad/build/bom.py && python cad/build/guide.py
python cad/build/export.py                    # release package
python rl/build_mass_check.py --episodes 3    # deployed control on the build masses
```

## 2. What changed from the concept, and why

| | concept (Mk2.1) | build | why |
|---|---|---|---|
| Seam | wire-race slewing ring between the halves | **no seam bearing**: each half rides on its own pod (left: 8 DualVee W2 on a 90° V-track = axial + radial location; right: 8 INA LR201 crowned rollers on a cylindrical land = radial only) | a 600 mm slewing ring is heavy and costly, and two located bearings fight thermal growth between the CFRP axle and the GFRP tyre |
| Ring gear | separate ring bolted in | one-piece 7075, co-bonded in the drum mould | its OD (278–282) cannot pass the 276.5 mm seam opening afterwards |
| Battery | 13S4P 936 Wh on the bob | **12S2P Molicel P50B, 432 Wh**, two 6S2P modules on the yoke outside the arc plates | 13S exceeds the ODrive S1's 50.5 V limit; 936 Wh does not fit; 432 Wh gives about 23 km at the simulated 16 Wh/km |
| Compute | on the pendulum | Jetson on the spine (left bay) | the cameras and LiDARs are on the spine; the slip ring then carries only power, CAN and e-stop |
| Lean drive | | CubeMars AK70-10 + 40T m1.5 pinion in an internal rack R195 (6.5:1) | sim duty: p99 44 N·m at the bob, ≤ 3.5 rad/s |
| Level drive | | D6374 + 14T/96T m1.5 (6.86:1) | sim duty: 15 N·m (saturated), ≤ 10.4 rad/s |
| Drive | 24:1 | D6374 → HTD 5M P20/P32 → 24T pinion → 360T ring | same 24:1 |
| Motor encoders | (not modelled) | **AMT212B-V-OD absolute encoder on every D6374** | the S1s sit on heat spreaders away from the motors, so their on-board encoder cannot see the motor |
| Axle | 40 × 34 | **40 × 30 HM CFRP** (E ≥ 150 GPa) | gear-mesh stiffness: 0.19 mm sag into the mesh at 3 g, within the 0.35 mm backlash |
| Pendulum travel | "360° free" | **±102.5° yoke-to-spine, PU-bumpered stops** | see below |
| GNSS | antenna only | ANN-MB1 antenna + ZED-F9P receiver board (left bay, under the Jetson) | the ANN-MB1 is an active antenna, not a receiver |
| Cameras | | 6 × Waveshare OV9281 (A), square to the shell behind PC windows; 4 through a USB hub in the right bay | the Jetson carrier has 4 USB-A ports |
| Mass | 45.1 kg | @@MASS@@ | real hardware (fasteners, brackets, bearings, cases); the deployed control still works on it (section 8) |

**Why the pendulum no longer turns 360°.** The concept claimed the spine had nothing in the pendulum's swept
volume. With the real level drive this is false. A D6374 (Ø63 × 90) sits on the pendulum axis's side at 82.5 mm,
meshing with the 96T gear on the left hanger. At full lean (41.5°) the bob's inner wheels come within 81 mm of
the axle line, so with that lean they reach the level motor at 104.4° of yoke-to-spine angle. The arc-plate tips
reach its rear boss at 108.2°, whatever the lean. The deployed control uses up to **99.4°**, measured over all 16
robustness scenarios by `rl/yoke_range.py`; the worst is climbing a 12° incline.

Two PU-sleeved strikers on the left hanger meet the flanks of the level-bracket arm at ±102.5°. That leaves
3.1° over the control's maximum and 1.1° to the first collision, after up to 0.8° of bumper compression. The
continuous motion check proves the whole ±103.3° × ±41.5° range clear.

To win more travel later, the level drive can become an internal ring gear on the hanger. The motor would then sit
at 61.5 mm from the axle instead of 82.5 mm.

## 3. Bought parts

@@COTS@@

Fasteners, PEM nuts, inserts, dowels, keys, circlips: `cad/build/out/bom.md`. Each line has the standard, size,
quantity, torque and McMaster number, with the source of each number in `cad/build/mcmaster.json`.

## 4. Making the custom parts

The release package (`python cad/build/export.py`, then `cad/build/out/release/`) contains:

- `gyra_mk21_build.step`: the full assembly with every part named;
- `parts/<drawing>.step`: one per drawing, the geometry master;
- `dxf/<plate>.dxf`: plate profiles with all holes, for waterjet or laser;
- `stl/`: the printed parts;
- `drawings/GYRA_Mk21_drawing_set.pdf`: an A3 sheet per drawing. Each sheet has 3 views + iso, the material,
  process and finish, the mass, the overall size, and the hole table that the joints machined into the part
  (tapped / clearance / insert / PEM, position and axis).

| process | parts |
|---|---|
| waterjet + CNC (6061-T6 plate) | hangers (8), arc plates (6), carriage plates (6), outer plates (8), pod plates (8), lidar risers (5), GNSS plate (3, 5052) |
| CNC turned / milled | hub cartridges, hub spacer, level bracket, slip sleeve, bay clamps, pod hubs, rims, wheel studs (303), bob studs (303), cassette and track spacers, pitch strikers, encoder brackets |
| gears | ring gears 7075 m1.5 z360 (gear shaping or wire EDM, hard anodise); level gear 7075 z96; lean rack 7075 (wire EDM); pinion shafts 24T, lean pinion 40T, level pinion 14T (42CrMo4 QT, hob; drive pinions gas-nitrided) |
| tracks | AISI 1045, V edges induction-hardened HRC 53 to the DualVee track specification |
| sheet metal (5052-H32, laser + bend + PEM) | bay trays, mezzanine, battery cases, power box |
| MJF PA12 | pod shields, camera holders (heat-set inserts) |
| composites | drums: E-glass / epoxy 3 mm, vacuum infusion in a 2-piece female mould, ring gear placed in the mould; axle: roll-wrapped HM CFRP 40 × 30 |
| cast | treads: 70A PU cast in place in a closed mould |
| tungsten | 3 × W-Ni-Fe (ASTM B777 class 3) sectors, M8 tapped 16 deep in both end faces |
| PCB | LED ring PCB (24 × WS2812B-2020, 12→5 V buck) |

## 5. Assembly sequence

The order below is the one `assembly.py` verifies. Each move is simulated along its real path against everything
already installed. Torques and threadlocker for every fastener are in the table at the end of this section, which
is generated from the joints.

**T. Tyre halves (bench).**
1. Infuse each drum with its ring gear located in the mould on the crown-plane spigot (co-bonded). Add a DP490
   fillet at the gear's back edges.
2. Pass the rim ring in through the seam opening and bond it into the drum's outer edge with DP490 (30 mm band).
   Its OD clears the drum interior everywhere; verified along the whole path.
3. Cast the tread in place.

**B. Bob (bench).**
1. Tungsten blocks between the carriage plates: 12 × M8 FHCS class 10.9.
2. AK70-10 on the front carriage (8 × M3 FHCS), lean pinion on its output (6 × M3, counterbored).
3. V-wheel studs: the outer studs thread into the tungsten; the inner studs are eccentric, with washer + nyloc. Fit
   the wheels and DIN 6799 E-clips.

**Y. Yoke (bench).**
1. Left hanger: hub cartridge (spigot 66 h7, bearing 6908 + DIN 472), level gear, drive ODrive S1 on its heat
   spreader, pitch-stop strikers with PU sleeves.
2. Right hanger: hub cartridge, drive S1, slip-ring drive pin (pressed).
3. Arc plates: tracks on spacers, lean rack on spacers.
4. Frame: arc brackets join the arc plates to the hangers. Fit the power box (BMS, fuse, anti-spark, relay).
5. **2a. Run the bob onto the tracks** with the lean stops off. It rotates in about x from beyond the track end
   (verified from 120° down to 0). Set the inner eccentric studs for zero play. Fit the lean stops.
6. Drive cassettes, one per side:
   - pinion shaft with its 6001 bearings;
   - P32 pulley (key + set screw), belt, P20 on the D6374;
   - motor screws through the ±2 mm slots;
   - tension the belt (Gates 5M 15 mm: set with a sonic tension meter to the Design Flex value);
   - fit the encoder bracket (its foot slots move with the motor) and the AMT212;
   - mount the cassette on its 3 spacers.
7. Battery modules A and B on their spacers (PEM nuts in the case ends).

**S. Onto the axle.**
1. Hub spacer between the bearings. The yoke + bob unit slides onto the axle from +y.
2. Level drive from +y: bracket clamp on the axle with its land against the left inner race, pinion in mesh with
   the level gear, motor, encoder.
3. Slip ring from −y: sleeve clamp, rotor, housing engaging the yoke's drive pin.
4. Bays slide on from the axle ends:
   - left: Jetson on 25 mm standoffs, ZED-F9P on spacers underneath, GNSS plate and antenna on the clamp tabs;
   - right: level S1, mezzanine with the USB hub, DC-DC, IMU.

**5. Tyre halves.** Each slides on seam-first, held 15 mm off-centre toward its pinion so the ring gear clears the
drive cassette, then moves radially into mesh. The path is verified in 5 mm steps.

**6. Pods.** Each pod slides on from its axle end, then its hub clamp closes on the axle.
- Left: the V-wheels are turned out on their 3.5 mm eccentric studs during the slide, then set against the V-track.
- Right: the rollers clear the land by 0.3 mm and are preloaded on their 1 mm eccentrics.
- Then fit the LiDARs, cameras, light ring and shields.

**Checks before first power:**
- the yoke turns by hand to both pitch stops;
- the bob leans by hand to both lean stops;
- each tyre turns freely;
- backlash at both drive meshes is 0.15–0.45 mm (feeler or dial);
- belt tension.

@@STEPS@@

## 6. Wiring

| from | to | conductor | through |
|---|---|---|---|
| battery A (6S2P) + battery B (6S2P) in series | power box: XT90-S anti-spark → 40 A MIDI fuse → 12S BMS → 48 V bus | 10 AWG silicone | yoke |
| 48 V bus | drive S1 front, drive S1 rear | 12 AWG | yoke |
| 48 V bus + CAN | AK70-10 (XT30 + CAN JST) | 14 AWG + twisted pair | yoke → bob: service loop clipped along the arc plate (±42° travel) |
| 48 V bus, GND | slip ring 2 × 10 A rings each (paralleled) | 12 AWG | yoke → spine |
| e-stop loop (2 signal rings) | power-box relay coil (cuts the drive S1s and the AK70-10) | 22 AWG | spine ↔ yoke |
| CAN H/L + CAN GND (3 signal rings) | one bus: Jetson (J401 CAN) ↔ level S1 ↔ slip ring ↔ drive S1 F ↔ drive S1 R ↔ AK70-10 ↔ BMS; 120 Ω at the Jetson and at the AK70-10 | 22 AWG twisted pair | spine ↔ yoke |
| S1 (each) | D6374 phases (4 mm bullets); AMT212 RS485 + 5 V (S1 ISO port) | motor leads; 26 AWG shielded 6-wire | yoke / spine |
| spine 48 V | level S1; RECOM RPA60 48 → 12 V | 14 AWG | right bay |
| 12 V | Jetson J401 (12–19 V in), 2 × Mid-360 (9–27 V), LED rings (5 V buck on the LED PCB) | 16–20 AWG | axle slots at y ±170 / ±296 |
| Jetson USB-A 1 | USB hub (right bay) → cameras L0, L1, L2 + USB-GbE adapter (ASIX AX88179 class) → right Mid-360 | USB 2.0 | axle slots |
| Jetson USB-A 2, 3, 4 | cameras R0, R1, R2 | USB 2.0 (MJPEG) | axle slot −y |
| Jetson GbE | left Mid-360 (Livox M12 cable) | Cat5e | axle slot +y |
| Jetson UART1 (40-pin 8/10, 3.3 V) | IMU 3DM-CV7 UART (Samtec FTSH 10-pin pigtail, 3.3–5 V supply from 5 V) | 4-wire | right bay |
| Jetson I2C1 (40-pin 3/5) | ZED-F9P (Qwiic); ANN-MB1 on the SMA | Qwiic cable | left bay |

## 7. Commissioning

1. **ODrive S1 × 3 on CAN.** Node IDs: drive F = 1, drive R = 2, level = 3.
   - DC bus over-voltage trip 50.4 V; regen current limit set to what the BMS accepts.
2. **Encoders.** Set `encoder` to RS485 / AMT21 (event-driven mode for the -OD firmware). Then run motor
   calibration and encoder offset calibration per motor, with the belt off for the drive motors.
3. **Lean.** AK70-10 in MIT mode. Zero it at the mechanical centre (bob straight down). Software limits ±40°.
   The hard stops are at about ±42°.
4. **Level.** Zero it with the spine level on a flat floor. Software limit on the yoke-to-spine angle: ±100°. The
   hard stops are at ±102.5°.
5. **IMU.** Its x axis is along the robot's forward axis. Run the misalignment calibration on a level surface.
6. **GNSS.** Configure the ZED-F9P (u-center) for 115200 baud UBX-NAV-PVT at 10 Hz on UART2.

## 8. Verification

@@VERIFY@@

## 9. Open items: verify before ordering

@@OPEN@@
