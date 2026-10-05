"""
GYRA Mk1 - first-order sizing, computed from the CAD mass properties.

    python cad/gyra_cad.py      # (re)generates cad/out/mass_properties.json
    python calc/sizing.py       # prints the sizing table, writes calc/out/sizing.json + charts

Every number here is a closed-form estimate; the MuJoCo model in sim/ checks the
dynamic claims (drive, lean-steer, CMG wobble damping, turn-in-place).
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "cad"))
import params as P  # noqa: E402

G = 9.81
OUT = os.path.join(ROOT, "calc", "out")
os.makedirs(OUT, exist_ok=True)

mp = json.load(open(os.path.join(ROOT, "cad", "out", "mass_properties.json")))
B = mp["bodies"]
R = P.R_OUT / 1000.0

M = mp["robot"]["mass"]
com_z = mp["robot"]["com_mm"][2] / 1000.0
m_tyre = B["tyre"]["mass"]
I_tyre = B["tyre"]["I_com_kgm2"][1][1]                  # about the spin (y) axis
m_yoke, m_bob = B["yoke"]["mass"], B["bob"]["mass"]
m_p = m_yoke + m_bob
mL_p = m_yoke * -B["yoke"]["com_mm"][2] / 1000 + m_bob * -B["bob"]["com_mm"][2] / 1000
L_p = mL_p / m_p
m_spine = M - m_tyre - m_p

# robot inertia about the vertical axis through the centre (for in-place yaw)
I_zz = mp["robot"]["I_com_kgm2"][2][2] + M * 0.0  # CoM is on the vertical through the centre

# --- straight-line drive ------------------------------------------------------
theta_max = math.radians(P.PITCH_STOP_DEG)
M_eff = M + I_tyre / R ** 2
tau_pend_max = m_p * G * L_p * math.sin(theta_max)        # max torque the pendulum can react
a_max = tau_pend_max / (R * M_eff)
slope_max = math.degrees(math.asin(min(1.0, mL_p * math.sin(theta_max) / (M * R))))
# the pendulum also has to swing up: worst case a full-throttle launch
ratio = (P.GEAR_Z_RING / P.DRIVE_PINION_Z) * P.BELT_RATIO   # 30:1
KV, V_BATT = 150.0, 13 * 3.7
rpm_motor = KV * V_BATT * 0.85
v_top = rpm_motor / ratio * 2 * math.pi / 60 * R
tau_motor_each = tau_pend_max / ratio / 2 / 0.9          # per motor, 90 % mesh+belt efficiency
# static step from the pendulum only (no momentum) and the RotunBot-scaled dynamic step
step_static = R * (1 - math.cos(math.radians(slope_max)))
step_dynamic = 0.17 / 0.40 * R                             # RotunBot cleared 17 cm at R = 40 cm

# --- lateral lean (steering) --------------------------------------------------
r_bob = -B["bob"]["com_mm"][2] / 1000
tau_lean = m_bob * G * r_bob * math.sin(math.radians(P.BOB_TRAVEL_DEG))

# --- CMG pair ---------------------------------------------------------------
I_fly = B["flyL"]["I_com_kgm2"][2][2]
w_fly = P.CMG_RPM * 2 * math.pi / 60
h = I_fly * w_fly
gimbal_rate = 4.0                                    # rad/s (AK70-10 class at ~10 N m)
tau_cmg_pair = 2 * h * gimbal_rate
E_fly = 0.5 * I_fly * w_fly ** 2
roll_impulse = 2 * h * math.sin(math.radians(60))    # scissor 0 -> +-60 deg
gimbal_torque_req = h * 2.0                          # body roll rate 2 rad/s -> gyroscopic load on the gimbal

# --- turning ----------------------------------------------------------------
# steady gyroscopic-precession steering of a rolling sphere:
#   tau_roll = Omega * (I_tyre + M R^2) * omega_spin  ->  rho = (I_tyre + M R^2) v^2 / (tau R)
J = I_tyre + M * R ** 2


def turn_radius(v, tau):
    return J * v ** 2 / (tau * R)


# --- turn in place with the CMG pair in "yaw mode" ----------------------------
mu, patch_a = 0.9, 0.035                              # rubber on asphalt, contact-patch radius
M_spin = 2 / 3 * mu * M * G * patch_a                 # pivot (spin) friction torque
dh_yaw = 2 * h * (math.cos(math.radians(30)) - math.cos(math.radians(150)))
t_sweep = math.radians(120) / gimbal_rate
tau_yaw = dh_yaw / t_sweep
w_end = max(0.0, (tau_yaw - M_spin) * t_sweep / I_zz)
alpha_f = M_spin / I_zz
yaw_per_sweep = 0.5 * w_end * t_sweep + w_end ** 2 / (2 * alpha_f)

# --- flotation & endurance ----------------------------------------------------
V_sphere = 4 / 3 * math.pi * R ** 3
V_disp = M / 1000.0


def cap_vol(hh):
    return math.pi * hh ** 2 * (3 * R - hh) / 3


lo, hi = 0.0, 2 * R
for _ in range(60):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if cap_vol(mid) < V_disp else (lo, mid)
draft = lo
E_batt = 13 * 3.6 * 10.0                              # 13S2P 21700 (5 Ah cells) -> Wh
crr, v_cruise = 0.03, 3.0
P_roll = crr * M * G * v_cruise / 0.82
P_hotel = 15 + 13 + 4 + 24 + 10                       # Orin NX, 2 LiDAR, cameras, CMG idle, misc
runtime_h = E_batt / (P_roll + P_hotel)

S = {
    "mass_total_kg": M, "com_below_centre_mm": -com_z * 1000,
    "mass_tyre_kg": m_tyre, "mass_pendulum_kg": m_p, "mass_spine_cmg_kg": m_spine,
    "pendulum_mL_kgm": mL_p, "pendulum_L_eff_mm": L_p * 1000,
    "I_tyre_spin_kgm2": I_tyre, "I_yaw_kgm2": I_zz,
    "drive_ratio": ratio, "v_top_ms": v_top, "v_top_kmh": v_top * 3.6,
    "tau_pendulum_max_Nm": tau_pend_max, "a_max_ms2": a_max, "slope_max_deg": slope_max,
    "motor_torque_each_Nm": tau_motor_each,
    "step_static_mm": step_static * 1000, "step_dynamic_est_mm": step_dynamic * 1000,
    "lean_torque_bob_Nm": tau_lean,
    "flywheel_I_kgm2": I_fly, "flywheel_h_Nms": h, "flywheel_energy_J": E_fly,
    "cmg_pair_roll_torque_Nm": tau_cmg_pair, "cmg_roll_impulse_Nms": roll_impulse,
    "gimbal_torque_required_Nm": gimbal_torque_req,
    "turn_radius_m": {f"{v:.0f} m/s": turn_radius(v, tau_lean) for v in (1, 2, 3, 4, 6)},
    "spin_friction_Nm": M_spin, "cmg_yaw_torque_Nm": tau_yaw, "yaw_per_sweep_deg": math.degrees(yaw_per_sweep),
    "draft_mm": draft * 1000, "draft_fraction": draft / (2 * R), "reserve_buoyancy_kg": (V_sphere - V_disp) * 1000,
    "battery_Wh": E_batt, "runtime_at_3ms_h": runtime_h, "range_at_3ms_km": runtime_h * 3.6 * v_cruise,
}

if __name__ == "__main__":
    json.dump(S, open(os.path.join(OUT, "sizing.json"), "w"), indent=1)
    w = max(len(k) for k in S)
    for k, v in S.items():
        if isinstance(v, dict):
            print(f"{k:{w}s}  " + ", ".join(f"{kk}: {vv:.1f}" for kk, vv in v.items()))
        else:
            print(f"{k:{w}s}  {v:10.3f}")

    # ---- chart: turning radius vs speed -------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
    v = np.linspace(0.3, 6.5, 200)
    # RotunBot estimate from its paper: 160 kg, R 0.4 m, 73.4 kg pendulum, L 0.27 m, +/-30 deg lateral swing
    J_rt = 27.6 * 0.4 ** 2 * 0.7 + 160 * 0.4 ** 2
    tau_rt = 73.4 * G * 0.27 * math.sin(math.radians(30))
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    series = [
        ("GYRA - bob only (sustained)", turn_radius(v, tau_lean), C1),
        ("GYRA - bob + CMG burst (< 0.4 s)", turn_radius(v, tau_lean + tau_cmg_pair), C2),
        ("RotunBot / RT-G (estimate)", J_rt * v ** 2 / (tau_rt * 0.4), C3),
    ]
    for name, y, c in series:
        ax.plot(v, y, color=c, lw=2, label=name)
        ax.annotate(name, (v[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=8.5, va="center")
    ax.set_ylim(0, 40)
    ax.set_xlim(0, 6.5)
    ax.set_xlabel("speed (m/s)", color=INK2)
    ax.set_ylabel("steady turn radius (m)", color=INK2)
    ax.set_title("Lean-steer turn radius grows with v^2 (gyroscopic precession)", color=INK, fontsize=11, loc="left")
    ax.grid(color=GRID, lw=0.8)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(colors=INK2)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)
    fig.subplots_adjust(right=0.70)
    fig.savefig(os.path.join(ROOT, "media", "chart_turn_radius.png"), facecolor=SURF)
    print("chart written")
