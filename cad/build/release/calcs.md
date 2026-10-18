| group | check | value | allowable | ok | note |
|---|---|---|---|---|---|
| drive gear | pinion root bending, p99 | 16.9 MPa | 300.0 | yes | 42CrMo4 QT + nitrided, allowable bending ~300 MPa |
| drive gear | ring root bending, p99 | 11.8 MPa | 150.0 | yes | 7075-T6 fatigue ~150 MPa |
| drive gear | contact stress, p99 | 160 MPa | 250.0 | yes | hard-anodised 7075 vs steel; p99 kept under 250 MPa for wear life; peak is rare |
| drive gear | pinion root bending, peak (motor limit 82 N m) | 40.1 MPa | 300.0 | yes | 42CrMo4 QT + nitrided, allowable bending ~300 MPa |
| drive gear | ring root bending, peak (motor limit 82 N m) | 28.1 MPa | 150.0 | yes | 7075-T6 fatigue ~150 MPa |
| drive gear | contact stress, peak (motor limit 82 N m) | 247 MPa | 400.0 | yes | hard-anodised 7075 vs steel; p99 kept under 250 MPa for wear life; peak is rare |
| axle | bending stress, 1 g | 8.73 MPa | 400.0 | yes | HM CFRP, allowable well above (UD 0 deg strength > 1 GPa) |
| axle | mid-span sag, 1 g | 0.093 mm |  | yes |  |
| drive gear | pinion pushed into the ring by axle sag, 1 g | 0.062 mm | 0.35 | yes | design backlash: centre distance +0.20 mm -> 0.15-0.45 mm operating |
| axle | bending stress, bump 3 g | 26.2 MPa | 400.0 | yes | HM CFRP, allowable well above (UD 0 deg strength > 1 GPa) |
| axle | mid-span sag, bump 3 g | 0.28 mm |  | yes |  |
| drive gear | pinion pushed into the ring by axle sag, bump 3 g | 0.187 mm | 0.35 | yes | design backlash: centre distance +0.20 mm -> 0.15-0.45 mm operating |
| belt | tight-side tension at motor peak torque | 239 N |  |  | verify against Gates HTD 5M 15 mm allowable working tension (Design Flex) - 285-5M-15 at 20T; tooth-in-mesh 6.3 |
| bearings | 6001 hanger bearing A: radial load p99 | 235 N | 2370 | yes | L10 at p95 speed 1297 rpm: 174,255 h |
| bearings | 6001 outer-plate bearing B: radial load p99 | 21.7 N | 2370 | yes | L10 at p95 speed 1297 rpm: 221,836,200 h |
| bearings | 6908 hub bearing, 3 g bump (per bearing) | 363 N | 10000 | yes |  |
| support wheels | DualVee W2 radial per wheel, 1 g | 103 N | 2650 | yes |  |
| support wheels | DualVee W2 axial per wheel (crown moment), 1 g | 105 N | 625 | yes | bump case against 2 x working axial (short-duration); size 3 (W3, 1701 N axial) is the drop-in upgrade if field data demands |
| support wheels | LR201 roller radial per roller (floating side), 1 g | 103 N | 2850 | yes |  |
| support wheels | DualVee W2 radial per wheel, bump 3 g | 310 N | 2650 | yes |  |
| support wheels | DualVee W2 axial per wheel (crown moment), bump 3 g | 314 N | 1250 | yes | bump case against 2 x working axial (short-duration); size 3 (W3, 1701 N axial) is the drop-in upgrade if field data demands |
| support wheels | LR201 roller radial per roller (floating side), bump 3 g | 310 N | 2850 | yes |  |
| lean drive | AK70-10 output torque, rms | 2.8 N m | 8.3 | yes |  |
| lean drive | AK70-10 output torque, p99 | 7.12 N m | 24.8 | yes |  |
| lean drive | AK70-10 output torque, peak | 19.4 N m | 24.8 | yes |  |
| lean drive | pinion root bending at peak | 153 MPa | 300.0 | yes |  |
| lean drive | rack root bending at peak | 132 MPa | 150.0 | yes |  |
| lean drive | AK70-10 speed at max lean rate | 215 rpm | 310 | yes |  |
| level drive | D6374 torque at the level limit (15 N m) | 2.3 N m | 3.8 | yes |  |
| level drive | D6374 rms current | 18.8 A | 40.0 | yes | ODrive S1 40 A with heat spreader |
| drive | D6374 rms current (p rms tyre torque) | 9.76 A | 40.0 | yes |  |
| fixings | tungsten block (3.5 kg) on 4 x M8 10.9: shear per bolt at 5 g | 1.18 MPa | 200.0 | yes | bolts clamp: friction alone carries it at proper preload |
| fixings | battery module (1.3 kg) on 4 x M5: shear per bolt at 5 g | 1.12 MPa | 200.0 | yes | bolts clamp: friction alone carries it at proper preload |
| pitch stop | yoke-to-spine angle used by the deployed control (max, 16 scenarios) | 99.5 deg | 101.5 | yes | stop contact at 102.5 deg; first collision without the stop at 104.4 deg |
| pitch stop | inertia about the axle: pendulum / spine / reduced (what the bumper sees) | 0.106 kg m2 |  |  | pendulum 1.00, spine 0.119: on contact the spine is carried along |
| pitch stop | relative arrival rate the bumper absorbs within its 1.3 mm (0.8 deg) travel | 3.09 rad/s |  | yes | capacity 0.51 J; the control's fastest outward yoke rate above 90 deg is 2.23 rad/s |
| pitch stop | striker bending at full bumper travel (d12 C45 steel) | 115 MPa | 300.0 | yes | 780 N at 25 mm; arm flank bearing 16 MPa (6061) |
| pitch stop | level-bracket clamp slip torque vs stop torque at full bumper travel | 74.1 N m | 113.1 | yes | 2 x M5 split clamp on the 40 mm CFRP axle, friction 0.2 |
| pitch stop | fault: both drive motors at their torque limit from 60 deg (relative energy at the stop) | 9.71 J |  |  | far beyond any end stop: guarded by the drive S1s' CAN watchdog (idle on lost heartbeat), the +-100 deg software limit and the e-stop relay |
| battery | usable range at the simulated mix | 23 km |  | yes | 432 Wh, 85 % usable, 16 Wh/km |
| battery | pack current at p99 mechanical power (+60 W electronics) | 19.6 A | 120.0 | yes | 12S2P P50B: 2 x 60 A continuous; the two drive ODrives are current-limited to 60 A phase each |
| slip ring | spine power through 2 paralleled 10 A rings | 4.63 A | 20.0 | yes | Jetson 25 W + 2 LiDAR 13 W + cameras 6 W + level motor ~150 W peak |
