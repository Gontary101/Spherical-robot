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
AXLE_OD, AXLE_ID = 40.0, 34.0  # roll-wrapped CFRP tube, 3 mm wall
AXLE_END = 312.0               # |y| of each axle end (inside the pod hub)
BAY_Y = (176.0, 262.0)         # |y| span of the spine electronics bays (below the axle)

# ------------------------------------------------------------------ yoke
HANGER_Y = (27.0, 35.0)        # |y| of the two hanger plates (8 mm 6061)
ARC_X = (86.0, 92.0)           # |x| of the two arc plates (6 mm 6061)
ARC_R = (140.0, 238.0)         # arc plate radii (about the x axis)
ARC_HALF = 64.0                # arc plate half-angle (deg, from straight down)
TRACK_R = (152.0, 214.0)       # bob V-track strip radii (edges are the wheel tracks)
TRACK_X = 75.0                 # |x| of the track strip mid-plane
TRACK_T = 6.0
OUTER_PLATE_Y = (70.0, 78.0)   # |y| of each drive cassette's outer plate

# ------------------------------------------------------------------ bob
BOB_X = (62.0, 68.0)           # |x| of the carriage plates (6 mm 6061)
BOB_HALF = 24.0                # carriage angular half-width (deg)
BOB_LEAN = 40.0                # lean travel +- (deg), as simulated
W_R = (210.0, 246.0)           # tungsten block radii
TUNGSTEN_KG = 11.0

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
}
