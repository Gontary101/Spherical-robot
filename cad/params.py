"""
GYRA Mk1 - single source of truth for geometry and mass parameters.

Frame convention (used by CAD, calculations and simulation):
    origin = geometric centre of the sphere
    +x     = forward (direction of travel)
    +y     = left, along the axle (the tyre spins about y)
    +z     = up
All lengths in millimetres, masses in kilograms, unless suffixed otherwise.
"""
from math import sqrt

# ----------------------------------------------------------------------------
# Envelope / tyre
# ----------------------------------------------------------------------------
R_OUT = 300.0            # lug-top (crown) radius -> 600 mm diameter robot
TREAD_LUG_H = 4.0        # lug height above the tread base
TREAD_BASE_T = 3.0       # continuous PU tread base (bonded to the GFRP drum)
SHELL_T = 2.5            # GFRP drum wall
R_TREAD_BASE = R_OUT - TREAD_LUG_H          # 296
R_SHELL_OUT = R_TREAD_BASE - TREAD_BASE_T   # 293
R_SHELL_IN = R_SHELL_OUT - SHELL_T          # 290
Y_EDGE = 235.0           # half-width of the rolling tyre (tyre spans |y| <= 235)
TREAD_GROOVE_W = 10.0    # centre circumferential groove (hides the equator bolts)

# Tread pattern (chevron off-road / paddle lugs)
N_LUG_PITCH = 36         # lug pitches around the circumference
LUG_ROWS = (             # (y_start, y_end, phase_fraction) per half of the tyre
    (8.0, 72.0, 0.0),
    (80.0, 150.0, 0.5),
    (158.0, 228.0, 0.0),
)
LUG_ARC_DEG = 5.0        # circumferential arc of one lug block
LUG_SKEW_DEG = 7.0       # chevron skew across the row

# Rim rings (aluminium, bonded into each tyre edge, carry the V-track)
RIM_TRACK_R = 150.0      # V-track inner edge radius (rollers ride on it)

# Equatorial internal ring gear (also the joint between the two drum halves)
GEAR_MODULE = 1.5
GEAR_Z_RING = 360
GEAR_R_PITCH = GEAR_MODULE * GEAR_Z_RING / 2.0   # 270
GEAR_FACE = 18.0                                 # |y| <= 9
GEAR_R_TIP = GEAR_R_PITCH - GEAR_MODULE          # 268.5 (internal gear)
GEAR_R_BODY = R_SHELL_IN                         # bonded/bolted to the drum

# ----------------------------------------------------------------------------
# Spine (non-rotating sensor axle) = axle + 2 sensor pods + CMG pair + compute
# ----------------------------------------------------------------------------
AXLE_OD, AXLE_ID = 40.0, 32.0
AXLE_HALF = 238.0
POD_PLATE_Y0, POD_PLATE_Y1 = 239.0, 245.0
POD_PLATE_R = 176.0
POD_ROLLER_N = 6
POD_ROLLER_R = 12.0      # V-roller radius
POD_ROLLER_PHASE_DEG = 0.0 

# CMG pair (scissored, counter-rotating). Gimbal axes parallel to y.
CMG_Y = 170.0            # |y| of each CMG centre
CMG_Z = 122.0            # above the axle
FLY_OD, FLY_ID, FLY_T = 100.0, 50.0, 30.0
CMG_HOUSING_R = 62.0     # spherical-ish guard radius
CMG_RPM = 10000.0
GIMBAL_ACT_D, GIMBAL_ACT_L = 80.0, 40.0   # CubeMars AK70-10 class actuator

# Spine electronics bays (below the axle, outside the pendulum's swept volume)
BAY_X = (-60.0, 60.0)
BAY_Y = (130.0, 222.0)
BAY_Z = (-88.0, -28.0)

# Leveling actuator (spine -> yoke), and clock-spring cable reel (other side)
LEVEL_MOTOR_D, LEVEL_MOTOR_L = 50.0, 38.0
LEVEL_MOTOR_POS = (62.0, 81.0, -52.0)      # centre (x, y, z)
CLOCKSPRING_D, CLOCKSPRING_Y = 110.0, (-64.0, -44.0)

# ----------------------------------------------------------------------------
# Pendulum = yoke (pitch body) + bob (rides a lateral arc on the yoke)
# ----------------------------------------------------------------------------
HUB_BEARING = (40.0, 62.0, 12.0)           # 6908-2RS on the axle
HANGER_Y = 53.0                            # hanger plates at |y| = 50..56
HANGER_T = 6.0
ARC_PLATE_X = 81.0                         # arc plates at |x| = 78..84
ARC_PLATE_T = 6.0
ARC_R_IN, ARC_R_OUT = 150.0, 240.0          # cradle arc radii (about the x axis)
ARC_HALF_DEG = 50.0
BOB_R_IN, BOB_R_OUT = 152.0, 238.0
BOB_HALF_DEG = 18.0                        # angular half-width of the bob block
BOB_X = 62.0                               # bob spans |x| <= 62
BOB_TRAVEL_DEG = 30.0                      # lateral travel +/-30 deg
PITCH_STOP_DEG = 75.0                      # yoke vs spine hard stop

# Drive: 2 x BLDC (6374 class) -> HTD belt 2:1 -> 24T pinion -> 360T ring (15:1)
DRIVE_PINION_Z = 24
DRIVE_PINION_RC = GEAR_R_PITCH - GEAR_MODULE * DRIVE_PINION_Z / 2.0    # 252
DRIVE_ANGLE_DEG = 42.0                     # pinions at +/-42 deg from bottom
DRIVE_MOTOR_RC = 175.0
DRIVE_MOTOR_D, DRIVE_MOTOR_L = 63.0, 74.0
BELT_RATIO = 2.0

# ----------------------------------------------------------------------------
# Purchased / cast items whose mass is fixed (everything else is computed from
# CAD volume x density in gyra_cad.py)
# ----------------------------------------------------------------------------
BOB_LEAD_KG = 8.4        # lead casting in the bob
BATTERY_KG = 1.9         # 13S2P 21700 pack incl. BMS


def shell_in_r_at(y: float) -> float:
    """Radius of the drum inner surface at axial position y."""
    return sqrt(max(R_SHELL_IN ** 2 - y ** 2, 0.0))


if __name__ == "__main__":
    print("rim radius at tyre edge:", round(sqrt(R_OUT**2 - Y_EDGE**2), 1), "mm")
    print("drive pinion centre radius:", DRIVE_PINION_RC, "mm")
