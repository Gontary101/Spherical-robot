"""
GYRA Mk2.1 build specification: every dimension the build CAD uses, and the bought parts with their datasheet data.

Frame (same as the concept CAD and the simulator): origin = sphere centre, +x forward, +y left (axle), +z up, mm.
The left tyre half spins about y, its crown sphere centred at (0, +D, 0); everything mirrors for the right half.

Changes from the concept model (cad/gyra2_cad.py) and why are listed in docs/11_build.md.
"""
import math

# ------------------------------------------------------------------ envelope (unchanged from the concept)
D = 90.0                       # crown offset of each half
R_OUT = 300.0                  # lug tips
R_TREAD = 296.0                # PU tread base
R_DRUM_OUT = 293.0             # GFRP drum, 3.0 mm wall
R_DRUM_IN = 290.0
U_EDGE = 235.0                 # drum spans local u = y - D up to the rim edge
SEAM = 2.5                     # each drum stops 2.5 mm short of y = 0 (5 mm seam, brush-sealed)


def drum_r(y, side=1):
    """Drum inner radius (about the y axis) at global y for the half on `side` (+1 left, -1 right)."""
    u = y - side * D
    return math.sqrt(max(R_DRUM_IN ** 2 - u ** 2, 0.0))


# ------------------------------------------------------------------ drive: ring gear + pinion
MODULE = 1.5
Z_RING = 360                   # internal, pitch r 270, made as 6 segments of 60 teeth
R_PITCH = MODULE * Z_RING / 2
R_RING_TIP = R_PITCH - MODULE                      # 268.5
R_RING_ROOT = R_PITCH + 1.25 * MODULE              # 271.875
R_SEG_OUT = 277.0                                  # segment back
GEAR_Y = (8.0, 24.0)           # |y| span of each half's ring gear (face 16)
Z_PINION = 24
R_PINION = MODULE * Z_PINION / 2                   # 18
PINION_FACE = (8.5, 23.5)      # |y|
DRIVE_ANGLE = 48.0             # pinion / motor centre angle from straight down, toward +x for the front (left-half) train
R_PIN_C = R_PITCH - R_PINION   # 252
R_MOT_C = 175.0
BELT_RATIO = 1.6               # HTD 5M 20T -> 32T

# ------------------------------------------------------------------ rim + support wheels
R_TRACK = 150.0                # V-track apex (left) / cylindrical land (right) radius
U_TRACK = 222.0                # local u of the wheel plane
U_POD_PLATE = (240.0, 248.0)   # pod plate (local u), 8 mm 6061
N_WHEELS = 8

# ------------------------------------------------------------------ spine
AXLE_OD, AXLE_ID = 40.0, 30.0  # roll-wrapped high-modulus CFRP tube, 5 mm wall (E_axial >= 150 GPa): gear-mesh stiffness, calcs.py
AXLE_END = 312.0               # |y| of each axle end (inside the pod hub)
BAY_Y = (176.0, 262.0)         # |y| span of the spine electronics bays (below the axle)

# ------------------------------------------------------------------ yoke
HANGER_Y = (27.0, 35.0)        # |y| of the two hanger plates (8 mm 6061)
HUB_BRG_Y = (25.0, 37.0)       # |y| of the 6908-2RS hub bearings (in cartridges bolted to the hangers)
ARC_X = (89.0, 95.0)           # |x| of the two arc plates (6 mm 6061)
ARC_R = (140.0, 238.0)         # arc plate radii (about the x axis)
ARC_HALF = 72.0                # arc plate half-angle (deg, from straight down); lean stops at +-68
TRACK_R = (152.0, 214.0)       # bob V-track ring segments (both edges are 90 deg tracks)
TRACK_X = 82.0                 # |x| of the track mid-plane (wheel plane); wheels end 1.4 mm short of the arc plate
TRACK_T = 6.35                 # DualVee size-2 track thickness
RACK_R = 195.0                 # lean rack pitch radius (internal teeth, m1.5), on the front arc plate
RACK_X = (69.0, 76.5)          # rack / pinion face
OUTER_PLATE_Y = (76.0, 84.0)   # |y| of each drive cassette's outer plate (motor on its outer face)
BELT_Y = 56.75                 # |y| of the belt centre plane
LEVEL_GEAR = dict(z=96, pinion=14, m=1.5, y=(35.0, 45.0))      # 6.86:1, centre distance 82.5
LEVEL_MOTOR_Y = 66.0           # level motor mount face (spine side), shaft toward -y

# ------------------------------------------------------------------ bob
BOB_X = (62.0, 68.0)           # |x| of the carriage plates (6 mm 6061)
BOB_HALF = 24.0                # carriage angular half-width (deg)
PITCH_STOP = 102.5             # yoke-to-spine pitch stop (deg, PU bumper contact; <= 0.8 deg compression). The deployed control
                               # uses up to 99.4 (rl/yoke_range.py); first collision (bob inner wheel vs level motor) at 104.4
BOB_LEAN = 40.0                # lean travel +- (deg), as simulated
W_R = (222.0, 246.0)           # tungsten block radii
LEAN_PINION_Z = 40             # 40T m1.5 (r 30) in the internal rack r 195: 6.5:1

# ------------------------------------------------------------------ battery (on the yoke)
CELL = dict(d=21.55, L=70.15, mass=0.071, pn="Molicel INR-21700-P50B", Ah=5.0, V=3.6)
PACK = "12S2P (2 modules of 6S2P in series), 43.2 V nominal, 50.4 V full, 10 Ah, 432 Wh"

# ------------------------------------------------------------------ bought parts (datasheet data; source per item)
# verified: True when the number was read from a manufacturer drawing / datasheet in this project
COTS = {
    "lidar": dict(pn="Livox Mid-360", body=(65.0, 65.0, 60.0), hole_rect=(48.0, 36.0), hole="M3 x 5 deep",
                  locating=dict(round=(0.0, -16.0, 3.0, 1.8), slot=(0.0, 23.0, 3.0, 1.8)), optical_h=47.0, mass=0.265,
                  power_W=6.5, connector="M12 A-code 12-pin", fov_v=(-7.0, 52.0), verified=True,
                  source="Livox Mid-360 User Manual v1.2 (2023.06), Appendix 'Dimensions' p.19"),
    "drive_motor": dict(pn="ODrive D6374-150KV (dual shaft)", od=63.0, base_d=48.8, base_t=4.0, can=(4.0, 67.0),
                        shaft=(10.0, 30.0), shaft2=(8.0, 16.0), holes=dict(pcd44=("M4", 4, 45.0), pcd30=("M4", 4, 45.0)),
                        hole_depth=6.0, mass=0.89, verified=True,
                        source="ODrive D6374 STEP (odriverobotics.com product page CAD link), measured in cad/build/vendor.py"),
    "vee_wheel": dict(pn="Bishop-Wisecarver DualVee W2X (size 2, steel, shielded)", od=30.73, vee=25.4, bore=9.53, width=11.13,
                      mass=0.039, working_radial_N=2650, working_axial_N=625, track_t=6.35, verified=True,
                      source="Bishop-Wisecarver DualVee catalogue 2025"),
    "track_roller": dict(pn="INA LR201-2RSR (crowned outer ring R500)", d=12.0, D=35.0, B=10.0, mass=0.047, Cr=6000, C0r=2850,
                         verified=True, source="Schaeffler catalogue data via distributor"),
    "hub_bearing": dict(pn="6908-2RS (NSK 6908DDU)", d=40.0, D=62.0, B=12.0, mass=0.112, Cr=15000, C0r=10000, verified=True,
                        source="NSK catalogue E1102m"),
    "shaft_bearing": dict(pn="6001-2RS (NSK 6001DDU)", d=12.0, D=28.0, B=8.0, mass=0.022, Cr=5600, C0r=2370, verified=True,
                          source="NSK catalogue E1102m"),
    "belt": dict(pn="Gates PowerGrip HTD 285-5M-15 (57 T)", alt="280 / 290 do not exist in the Gates list: 285 is the design length",
                 verified=True, source="Gates HTD 5M list (260, 270, 285, 295, 310 ...)"),
    "pulley_20": dict(pn="Gates P20-5M-15AL (MPB, rebore 10H7 + 3 mm keyway + M4 set screw)", pd=31.83, od=30.68, flange=34.93,
                      length=26.19, face=13.89, hub_d=20.65, mass=0.041, flange_gap_note="Gates data gives face 13.89 for a 15 mm belt: confirm the flange gap (model uses 16) on the delivered part", verified=True, source="Gates data (MROSupply)"),
    "pulley_32": dict(pn="Gates P32-5M-15AL (MPB, rebore 12H7 + DIN 6885 4x4 keyway + M5 set screw)", pd=50.93, od=49.78,
                      flange=53.98, length=27.79, face=13.89, hub_d=31.75, mass=0.113, flange_gap_note="Gates data gives face 13.89 for a 15 mm belt: confirm the flange gap (model uses 16) on the delivered part", verified=True, source="Gates data (MROSupply)"),
    "lean_actuator": dict(pn="CubeMars AK70-10 (KV100, 48 V)", od=89.0, length=50.25, mass=0.521, rated_Nm=8.3, peak_Nm=24.8,
                          rated_rpm_48V=310, output="6 x M3 x 7 on PCD 25, pilot d20", housing="8 x M3 x 10 on PCD 83",
                          verified=True, source="CubeMars brochure 2023 p.07 drawing"),
    "slip_ring": dict(pn="Moflon MT05012-0606 (6 x 10 A + 6 x 2 A signal)", bore=50.0, od=120.0, length=74.0, collar_d=75.0,
                      collar_len=8.6, tab=dict(w=19.5, t=4.0, slot_w=9.5, slot_r=73.0, r_end=78.0), rpm=600, verified=True,
                      source="Moflon MT050 drawing + catalogue page"),
    "clamp_collar": dict(pn="Ruland MCL-40-A (one-piece, 2024 Al)", bore=40.0, od=60.0, width=15.0, screw="M6x18", torque_Nm=16,
                         mass=0.0618, verified=True, source="Ruland datasheet"),
    "controller": dict(pn="ODrive S1 (ODS1) on heat spreader ACC-HS04", board=(66.0, 50.0), hs=(70.0, 70.0, 4.0),
                       hs_holes="4 x 4.2 on 60 x 60 (M4)", mass=0.035 + 0.06, current_cont_A=40, v_max=50.5, verified=True,
                       note="board hole positions only in ODrive's Onshape model; the heat-spreader pattern is what we mount to",
                       source="docs.odriverobotics.com S1 datasheet + ACC-HS04 listing"),
    "compute": dict(pn="NVIDIA Jetson Orin NX 16 GB on Seeed reComputer J401 carrier", board=(100.0, 80.0), height=35.0,
                    power_W=(10, 25), v_in=(9, 19), mass=0.25, verified=False,
                    note="carrier hole pattern from Seeed STEP before machining the bay plate", source="Seeed wiki"),
    "camera": dict(pn="Waveshare OV9281 1MP Mono USB Camera (A)", board=(25.0, 25.0), holes="4 x d2 on 21 x 21", depth=11.6,
                   lens_protrusion=5.5, fov=120, mass=0.027, verified=True, source="waveshare.com outline drawing"),
    "gnss_antenna": dict(pn="u-blox ANN-MB1-00 (L1/L5)", size=(82.0, 60.0, 22.5), flange_t=6.0, holes="2 x d4.5 on 68 (M4)",
                         mass=0.164, verified=True, source="u-blox ANN-MB1 datasheet R04 + distributors"),
    "encoder": dict(pn="CUI Devices (Same Sky) AMT212B-V-OD (AMT21 RS485 absolute, ODrive firmware)", width=28.58, r_top=15.49, h_top=15.33,
                    height=39.94, thick=10.30, thick_conn=11.70, holes="2 x d2.95 on PCD 19.05 (M2.5)", sleeve=8.0, mass=0.015, verified=True,
                    source="Same Sky AMT21 datasheet rev 1.11 (2026-06-23) p.7-8; ODrive docs hardware.html (S1 supports CUI-AMT212B-V-OD)",
                    note="the S1s are mounted on the hangers / bay, away from the motors, so their on-board encoder cannot see the motor;"
                         " RS485 to the S1's ISO/RS485 port, 5 V from the S1"),
    "gnss_receiver": dict(pn="SparkFun GPS-RTK2 Board - ZED-F9P (GPS-15136)", board=(43.5, 43.2), height=10.0, mass=0.015, verified=False,
                          note="under the GNSS plate, components up (<= 7.4 mm), in the gap above the axle; I2C (Qwiic) to the J401 40-pin"
                               " header (I2C1, pins 3/5), SMA to the ANN-MB1 cable; hole positions from SparkFun's board file",
                          source="sparkfun.com GPS-15136 (43.5 x 43.2 mm)"),
    "usb_hub": dict(pn="Waveshare 12694 USB HUB HAT (4 x USB 2.0, 65 x 30 mm, d3.0 holes)", board=(65.0, 30.0), height=16.0, mass=0.02,
                    verified=False, note="left-pod cameras L0-L2 + a USB-GbE adapter (ASIX AX88179 class) for the right Mid-360; cameras R0-R2"
                                         " go straight to the J401's other USB-A ports; hole pattern assumed 58 x 23"
                                         " (Pi Zero) - check the Waveshare drawing",
                    source="waveshare.com/usb-hub-hat.htm"),
    "imu": dict(pn="MicroStrain 3DM-CV7-AHRS", size=(38.0, 24.0, 8.6), mass=0.0083, verified=False,
                note="hole positions from the MicroStrain drawing before machining", source="MicroStrain product page"),
    "dcdc": dict(pn="RECOM RPA60-4812SFW (18-75 V in, 12 V 60 W, 2 x 1 in brick)", size=(50.8, 25.4, 10.2), mass=0.04,
                 verified=False, source="RECOM RPA60 datasheet"),
}
