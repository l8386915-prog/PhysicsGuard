"""
PhysicsGuard - Physical Consistency and Closed-Loop Control Tests
==================================================================
Pytest test suite validating the foundational physical conservation laws,
closed-loop control behavior, and cybersecurity resilience of the EV
DC fast-charging simulator:
  a) Coolant Check:
     Logged Q_cool equals m_dot * cp * (T_out - T_in) within 1.0%.
  b) Whole-Run Energy Check:
     Heat generated = C_pack * Delta_T + Coolant Heat + Ambient Heat,
     computed via independent trapezoid integration on logged signals within 1.0%.
  c) Time-Step Convergence Check:
     Normal charge runs at dt = 1.0 s and dt = 0.25 s yield final true battery
     temperatures within 0.3 C.
  d) Twin Independence:
     Proves that the Digital Twin consumes strictly measured signals, never
     accesses ground-truth plant states, and maintains isolated parameters.
  e) Unprotected Cyberattack Check (A5):
     Sensor spoofing (freezing temp_cell_a at 25 C) combined with actuator command
     tampering (forcing pump to 0 L/min from t=60s) causes true battery temperature
     to end above 40 C and at least 5 C higher than normal.
  f) Thermal Cooling Impact:
     Pump OFF causes faster heating and significantly higher peak temperature.
  g) Monotonic State of Charge:
     SOC strictly increases monotonically during charging.
  h) Cell Safety Voltage Clamp:
     Cell terminal voltage never exceeds the safety cutoff threshold of 4.20 V.
  i) Digital Twin vs Plant Tracking:
     Twin tracks perturbed plant within 3.0 C maximum temperature error.
"""

import sys
import inspect
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import trapezoid

# Ensure package imports succeed regardless of pytest working directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
PKG_DIR = Path(__file__).resolve().parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from physicsguard import params
from physicsguard.plant import Plant
from physicsguard.twin import Twin
from physicsguard.controller import ChargerBMS
from physicsguard.simulate import run_session


def test_coolant_check():
    """
    Test B a: Coolant Heat Transfer Check.
    
    Validates that logged Q_cool equals m_dot * cp * (T_out - T_in) within 1.0%,
    computed directly from logged coolant temperatures and fluid flow rates.
    """
    plant = Plant(perturb=False, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=42)
    controller = ChargerBMS()
    df = run_session(plant, controller, seed=42, until_soc=params.SOC_TARGET)

    # Extract logged true signals
    q_cool_logged = df["true_q_cool"].values
    flow_lpm = df["true_pump_flow_lpm"].values
    t_in = df["true_coolant_temp_in"].values
    t_out = df["true_coolant_temp_out"].values

    # Compute coolant heat removal independently from logged variables
    flow_m3_s = (flow_lpm / 1000.0) / 60.0
    m_dot = params.COOLANT_DENSITY_KG_PER_M3 * flow_m3_s
    cp = params.COOLANT_SPECIFIC_HEAT_J_PER_KG_K
    q_cool_computed = m_dot * cp * (t_out - t_in)

    # Evaluate integral / cumulative heat discrepancy
    total_logged_j = np.sum(q_cool_logged)
    total_computed_j = np.sum(q_cool_computed)
    rel_error = abs(total_logged_j - total_computed_j) / total_logged_j

    # Also evaluate maximum pointwise error during active flow (after pump spin-up)
    active_mask = flow_lpm > 5.0
    pointwise_rel_err = np.max(
        np.abs(q_cool_logged[active_mask] - q_cool_computed[active_mask])
        / np.maximum(q_cool_logged[active_mask], 1.0)
    )

    print(f"\n[Coolant Check] Logged Q_cool sum: {total_logged_j:.1f} J, Computed sum: {total_computed_j:.1f} J")
    print(f"[Coolant Check] Cumulative Relative Error: {rel_error * 100:.6f}%")
    print(f"[Coolant Check] Peak Active Pointwise Relative Error: {pointwise_rel_err * 100:.6f}%")

    assert rel_error < 0.01, (
        f"Coolant heat mismatch: relative error {rel_error * 100:.4f}% exceeds 1% limit!"
    )
    assert pointwise_rel_err < 0.01


def test_whole_run_energy_check():
    """
    Test B b: Whole-Run Energy Balance Check (Trapezoid Integration).
    
    Computed independently from logged signals only:
      Heat Generated = Integral over time of N_cells * (I_cell^2 * R0(T) + Vp^2 / Rp)
    Compare against:
      Stored Heat   = C_pack * (T_end - T_start)
      Coolant Heat  = Integral over time of m_dot * cp * (T_out - T_in)
      Ambient Heat  = Integral over time of UA_amb * (T - T_amb)
      
    Assert relative error is under 1.0% and print it.
    """
    plant = Plant(perturb=False, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=42)
    controller = ChargerBMS()
    df = run_session(plant, controller, seed=42, until_soc=params.SOC_TARGET)

    t = df["time"].values
    i_cell = df["true_cell_current"].values
    t_batt = df["true_battery_temp"].values
    vp = df["true_vp"].values
    flow_lpm = df["true_pump_flow_lpm"].values
    t_in = df["true_coolant_temp_in"].values
    t_out = df["true_coolant_temp_out"].values
    t_amb = df["true_ambient_temp"].values

    # 1. Independent Heat Generation Rate:
    # R0(T) = R0_25 * exp(-0.02 * (T - 25))
    r0_t = params.CELL_R0_25C_OHM * np.exp(-params.CELL_R0_TEMP_COEFF * (t_batt - 25.0))
    q_gen_rate = params.PACK_TOTAL_CELLS * (
        (i_cell ** 2) * r0_t + (vp ** 2) / params.CELL_RP_OHM
    )
    heat_generated_j = float(trapezoid(q_gen_rate, t))

    # 2. Independent Energy Accounting:
    # Stored internal thermal energy:
    t_start = 25.0  # Initial pack temperature
    t_end = t_batt[-1]
    stored_energy_j = float(params.PACK_TOTAL_HEAT_CAPACITY_J_PER_K * (t_end - t_start))

    # Coolant heat removed:
    flow_m3_s = (flow_lpm / 1000.0) / 60.0
    m_dot = params.COOLANT_DENSITY_KG_PER_M3 * flow_m3_s
    q_cool_rate = m_dot * params.COOLANT_SPECIFIC_HEAT_J_PER_KG_K * (t_out - t_in)
    coolant_heat_j = float(trapezoid(q_cool_rate, t))

    # Ambient enclosure heat loss:
    q_amb_rate = params.CASING_UA_AMBIENT_W_PER_K * (t_batt - t_amb)
    ambient_heat_j = float(trapezoid(q_amb_rate, t))

    accounted_energy_j = stored_energy_j + coolant_heat_j + ambient_heat_j
    rel_error = abs(heat_generated_j - accounted_energy_j) / heat_generated_j

    print(f"\n[Whole-Run Energy Check] Heat Generated : {heat_generated_j:.1f} J")
    print(f"[Whole-Run Energy Check] Stored Energy  : {stored_energy_j:.1f} J")
    print(f"[Whole-Run Energy Check] Coolant Energy : {coolant_heat_j:.1f} J")
    print(f"[Whole-Run Energy Check] Ambient Energy : {ambient_heat_j:.1f} J")
    print(f"[Whole-Run Energy Check] Total Accounted: {accounted_energy_j:.1f} J")
    print(f"[Whole-Run Energy Check] Relative Error : {rel_error * 100:.6f}%")

    assert rel_error < 0.01, (
        f"Whole-run energy balance error {rel_error * 100:.4f}% exceeds 1% limit!"
    )


def test_time_step_check():
    """
    Test B c: Time-Step Invariance Check.
    
    Simulate normal charge at dt = 1.0 s and dt = 0.25 s;
    final true battery temperature must match within 0.3 C.
    """
    plant_1 = Plant(perturb=False, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=42)
    ctrl_1 = ChargerBMS()
    df_1 = run_session(plant_1, ctrl_1, seed=42, dt=1.0)
    t_final_1 = df_1["true_battery_temp"].iloc[-1]

    plant_025 = Plant(perturb=False, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=42)
    ctrl_025 = ChargerBMS()
    df_025 = run_session(plant_025, ctrl_025, seed=42, dt=0.25)
    t_final_025 = df_025["true_battery_temp"].iloc[-1]

    temp_diff = abs(t_final_1 - t_final_025)
    print(f"\n[Time-Step Check] dt=1.0s Final Temp : {t_final_1:.3f} C")
    print(f"[Time-Step Check] dt=0.25s Final Temp: {t_final_025:.3f} C")
    print(f"[Time-Step Check] Discrepancy        : {temp_diff:.4f} C (Limit: 0.3 C)")

    assert temp_diff <= 0.3, (
        f"Time-step discretization difference {temp_diff:.4f} C exceeds 0.3 C threshold!"
    )


def test_twin_independence():
    """
    Test B d: Twin Independence & Observer Isolation.
    
    Proves that the Digital Twin only receives measured current, measured pump flow,
    measured ambient, and measured coolant inlet, and NEVER receives plant true
    values, plant objects, or perturbed plant parameters.
    """
    # 1. Method signature verification
    sig = inspect.signature(Twin.step_estimator)
    allowed_params = ["self", "i_pack_meas", "pump_flow_meas", "T_amb_meas", "T_cool_in_meas", "dt"]
    for param_name in sig.parameters.keys():
        assert param_name in allowed_params, f"Unexpected parameter {param_name} in Twin.step_estimator!"

    # 2. Structural independence from Plant instance
    twin = Twin(initial_soc=0.20, initial_temp_c=25.0)
    plant = Plant(perturb=True, seed=123, age_factor=1.4)

    # Assert twin retains nominal constants completely different from perturbed plant
    assert twin.r0 != plant.r0_25
    assert twin.c_pack != plant.c_pack
    assert not hasattr(twin, "plant")

    # 3. Feed synthetic measured dictionary to step_estimator (no Plant passed)
    synthetic_meas = {
        "pack_current": 128.0,
        "pump_flow_lpm": 10.0,
        "ambient_temp": 25.0,
        "coolant_temp_in": 25.0
    }
    pred = twin.step_estimator(
        i_pack_meas=synthetic_meas["pack_current"],
        pump_flow_meas=synthetic_meas["pump_flow_lpm"],
        T_amb_meas=synthetic_meas["ambient_temp"],
        T_cool_in_meas=synthetic_meas["coolant_temp_in"],
        dt=1.0
    )
    assert "temp_cell_a" in pred
    assert "pack_voltage" in pred

    # 4. Modifying internal hidden plant variables has ZERO effect on Twin
    t_twin_before = twin.T_batt
    plant.T_batt = 9999.0
    plant.vp = 50.0
    twin.step_estimator(
        i_pack_meas=synthetic_meas["pack_current"],
        pump_flow_meas=synthetic_meas["pump_flow_lpm"],
        T_amb_meas=synthetic_meas["ambient_temp"],
        T_cool_in_meas=synthetic_meas["coolant_temp_in"],
        dt=1.0
    )
    # The twin advances by normal expected thermal rise, unaffected by plant corruption
    delta_t = twin.T_batt - t_twin_before
    assert 0.0 < delta_t < 0.1, "Twin was corrupted by external plant state!"


def test_unprotected_cyberattack():
    """
    Test A5: Unprotected Cyberattack Scenario.
    
    Sensor tamper freezes temp_cell_a at 25 C AND command tamper forces pump
    flow to 0 L/min starting from t = 60 s.
    
    Criteria:
      1. True battery temperature must end above 40 C.
      2. True battery temperature must end at least 5 C higher than normal run.
      3. Print peak temperature.
    """
    # 1. Baseline normal charge
    plant_normal = Plant(perturb=False, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=42)
    ctrl_normal = ChargerBMS()
    df_normal = run_session(plant_normal, ctrl_normal, seed=42)
    normal_peak_c = df_normal["true_battery_temp"].max()
    normal_end_c = df_normal["true_battery_temp"].iloc[-1]

    # 2. Cyberattack run
    def sensor_freeze_tamper(t: float, meas: dict) -> dict:
        meas["temp_cell_a"] = 25.0  # Blind the BMS to temperature rise
        return meas

    def pump_kill_tamper(t: float, cmds: tuple) -> tuple:
        curr, pump = cmds
        if t >= 60.0:
            pump = 0.0  # Cut off coolant loop circulation
        return (curr, pump)

    plant_attack = Plant(perturb=False, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=42)
    ctrl_attack = ChargerBMS()
    df_attack = run_session(
        plant_attack,
        ctrl_attack,
        seed=42,
        sensor_tamper=sensor_freeze_tamper,
        command_tamper=pump_kill_tamper
    )

    attack_end_c = df_attack["true_battery_temp"].iloc[-1]
    attack_peak_c = df_attack["true_battery_temp"].max()
    delta_over_normal = attack_end_c - normal_end_c

    print(f"\n[Unprotected Attack] Normal Run Peak Temp : {normal_peak_c:.2f} C (End: {normal_end_c:.2f} C)")
    print(f"[Unprotected Attack] Attack Run Peak Temp : {attack_peak_c:.2f} C (End: {attack_end_c:.2f} C)")
    print(f"[Unprotected Attack] Excess Heating Delta : {delta_over_normal:.2f} C")

    assert attack_end_c > 40.0, (
        f"Attack end temperature {attack_end_c:.2f} C did not exceed 40.0 C requirement!"
    )
    assert delta_over_normal >= 5.0, (
        f"Attack temperature rise ({delta_over_normal:.2f} C) is not at least 5 C higher than normal!"
    )


def test_pump_off_heats_faster():
    """
    Test 5b: With the pump off, the pack heats faster and to a higher temperature.
    """
    plant_pump_on = Plant(perturb=False, initial_soc=0.20, initial_temp_c=25.0)
    plant_pump_off = Plant(perturb=False, initial_soc=0.20, initial_temp_c=25.0)

    for _ in range(1000):
        plant_pump_on.step(dt=1.0, current_cmd_a=128.0, pump_cmd_lpm=params.PUMP_NOMINAL_FLOW_LPM)
        plant_pump_off.step(dt=1.0, current_cmd_a=128.0, pump_cmd_lpm=0.0)

    temp_rise_pump_on = plant_pump_on.T_batt - plant_pump_on.initial_T_batt
    temp_rise_pump_off = plant_pump_off.T_batt - plant_pump_off.initial_T_batt

    print(f"\n[Cooling Test] Temp Rise with Pump ON : {temp_rise_pump_on:.2f} C")
    print(f"[Cooling Test] Temp Rise with Pump OFF: {temp_rise_pump_off:.2f} C")

    assert temp_rise_pump_off > temp_rise_pump_on
    assert temp_rise_pump_off - temp_rise_pump_on > 3.0


def test_soc_increases_monotonically():
    """
    Test 5c: SOC increases monotonically during charge.
    """
    plant = Plant(perturb=False, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=42)
    ctrl = ChargerBMS()
    df = run_session(plant, ctrl, seed=42)

    soc_diffs = np.diff(df["true_soc"].values)
    min_diff = np.min(soc_diffs)

    print(f"\n[SOC Test] Initial: {df['true_soc'].iloc[0]:.4f}, Final: {df['true_soc'].iloc[-1]:.4f}")
    print(f"[SOC Test] Minimum Step Delta: {min_diff:.8f}")

    assert min_diff >= 0.0
    assert df["true_soc"].iloc[-1] >= params.SOC_TARGET


def test_cell_voltage_never_exceeds_max():
    """
    Test 5d: Cell terminal voltage never exceeds 4.20 V.
    """
    plant = Plant(perturb=False, initial_soc=0.75, initial_temp_c=25.0, sensor_seed=42)
    ctrl = ChargerBMS()
    df = run_session(plant, ctrl, seed=42)

    max_v_cell = df["true_cell_voltage"].max()
    print(f"\n[Voltage Test] Max Cell Voltage Observed: {max_v_cell:.4f} V (Limit: 4.20 V)")

    assert max_v_cell <= params.CELL_V_MAX + 1e-5


def test_twin_vs_plant_tracking():
    """
    Test 5e: Digital Twin vs Perturbed Plant Tracking.
    
    Maximum temperature error between perturbed plant and nominal twin stays < 3.0 C.
    """
    plant = Plant(perturb=True, seed=42, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=42)
    ctrl = ChargerBMS()
    df = run_session(plant, ctrl, seed=42)

    twin = Twin(initial_soc=0.20, initial_temp_c=25.0)
    max_temp_error = 0.0

    for _, row in df.iterrows():
        twin.step_estimator(
            i_pack_meas=row["pack_current"],
            pump_flow_meas=row["pump_flow_lpm"],
            T_amb_meas=row["ambient_temp"],
            T_cool_in_meas=row["coolant_temp_in"],
            dt=1.0
        )
        err = abs(row["true_battery_temp"] - twin.T_batt)
        if err > max_temp_error:
            max_temp_error = err

    print(f"\n[Twin Tracking Test] Max Temperature Error: {max_temp_error:.2f} C (Limit: 3.0 C)")
    assert max_temp_error < 3.0
