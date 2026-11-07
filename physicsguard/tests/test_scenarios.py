"""
PhysicsGuard - Phase 3 Scenario Suite & Effect Check Tests
==========================================================
Pytest suite validating:
  1. Every scenario runs and two runs with the same seed yield identical CSVs.
  2. event_active is False before start_s and True after.
  3. Effect checks:
     - freeze -> measured signal constant after start.
     - drift -> (measured - true) grows at set rate within 10%.
     - spike -> offset equals magnitude during the window.
     - replay -> measured equals earlier recorded values.
     - abusive_command -> applied current 192 A after start.
     - pump_off -> applied pump 0 and peak true temp > normal + 3 C.
     - pump_degradation -> true flow near 6 L/min while applied pump command stays 10.
     - aged_cell -> late-charge pack voltage at least 2 V above normal.
     - dead_sensor_nan -> NaN appears in logged measurement AND run completes without NaN in applied current.
  4. demo_hot_attack -> peak true temp > 50 C and print crossing times for 45 C and 50 C.
  5. normal run -> after first 10 s and before last 5 s, applied current never drops below 100 A.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import pytest

# Ensure local package imports succeed
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
PKG_DIR = Path(__file__).resolve().parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from physicsguard import params
from physicsguard.plant import Plant
from physicsguard.controller import ChargerBMS
from physicsguard.simulate import run_session
from physicsguard.scenarios import build_scenario, Scenario
from physicsguard.run_scenarios import SCENARIO_NAMES


def _run_single_scenario(name: str, seed: int = 42) -> Tuple[pd.DataFrame, Scenario]:
    sc = build_scenario(name)
    plant_kwargs = dict(sc.plant_kwargs)
    init_soc = plant_kwargs.pop("initial_soc", params.SOC_INITIAL)
    init_temp = plant_kwargs.pop(
        "initial_temp_c",
        sc.inlet_c if sc.inlet_c is not None else params.T_AMBIENT_NOMINAL_C
    )

    plant = Plant(
        perturb=False,
        initial_soc=init_soc,
        initial_temp_c=init_temp,
        sensor_seed=seed,
        **plant_kwargs
    )
    controller = ChargerBMS()
    df = run_session(
        plant=plant,
        controller=controller,
        seed=seed,
        scenario=sc
    )
    return df, sc


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_scenario_runs_and_deterministic(scenario_name: str):
    """Every scenario runs and two runs with the same seed give identical CSVs."""
    df1, sc1 = _run_single_scenario(scenario_name, seed=42)
    df2, sc2 = _run_single_scenario(scenario_name, seed=42)

    assert len(df1) > 0, f"Scenario {scenario_name} returned empty DataFrame!"
    pd.testing.assert_frame_equal(df1, df2)


def test_scenario_labels_event_active():
    """event_active is False before start_s and True after."""
    for name in SCENARIO_NAMES:
        if name in ("normal", "aged_cell", "hot_day", "demo_hot_attack"):
            continue  # normal has no start_s; aged_cell/hot_day/demo_hot_attack start at t = 0
        df, sc = _run_single_scenario(name, seed=42)
        start_s = sc.start_s
        assert start_s is not None

        # Before start_s: all rows must have event_active == False
        before_mask = df["time_s"] < start_s
        if before_mask.any():
            assert not df.loc[before_mask, "event_active"].any(), (
                f"Scenario {name} has event_active=True before start_s={start_s}!"
            )

        # After start_s: all rows must have event_active == True
        after_mask = df["time_s"] > start_s
        if after_mask.any():
            assert df.loc[after_mask, "event_active"].all(), (
                f"Scenario {name} has event_active=False after start_s={start_s}!"
            )


def test_freeze_effect():
    """freeze -> measured signal constant after start."""
    df, sc = _run_single_scenario("freeze_temp_a", seed=42)
    start_s = sc.start_s

    post_df = df[df["time_s"] > start_s]
    vals = post_df["temp_cell_a"].values
    assert len(vals) > 50
    # All values must be identical to the frozen value
    assert np.all(vals == vals[0]), "Frozen signal values are not strictly constant!"


def test_drift_effect():
    """drift -> (measured - true) grows at the set rate within 10%."""
    df, sc = _run_single_scenario("drift_temp_a", seed=42)
    start_s = sc.start_s

    post_df = df[df["time_s"] > start_s]
    t = post_df["time_s"].values
    meas = post_df["temp_cell_a"].values
    true_val = post_df["true_battery_temp"].values
    diff = meas - true_val

    # Linear fit of (measured - true) vs time in minutes
    t_min = (t - start_s) / 60.0
    slope, _ = np.polyfit(t_min, diff, 1)

    # Default temperature drift rate is -1.0 C/min
    expected_rate = -1.0
    rel_error = abs(slope - expected_rate) / abs(expected_rate)
    print(f"\n[Drift Test] Fitted rate: {slope:.4f} C/min, Expected: {expected_rate:.4f} C/min (Error: {rel_error*100:.2f}%)")
    assert rel_error <= 0.10, f"Drift rate {slope:.4f} differs from {expected_rate} by more than 10%!"


def test_spike_effect():
    """spike -> offset equals magnitude during the window."""
    df, sc = _run_single_scenario("spike_temp_a", seed=42)
    start_s = sc.start_s
    duration_s = 60.0
    magnitude = -15.0

    in_window = df[(df["time_s"] >= start_s) & (df["time_s"] < start_s + duration_s)]
    before_window = df[(df["time_s"] >= start_s - 10) & (df["time_s"] < start_s)]

    # Before window, meas - true has mean ~ 0 (just noise)
    base_diff = (before_window["temp_cell_a"] - before_window["true_battery_temp"]).mean()
    spike_diff = (in_window["temp_cell_a"] - in_window["true_battery_temp"]).mean()

    effective_offset = spike_diff - base_diff
    print(f"\n[Spike Test] Measured offset: {effective_offset:.3f} C (Expected: {magnitude:.3f} C)")
    assert abs(effective_offset - magnitude) < 0.20, "Spike magnitude does not match expected offset!"


def test_replay_effect():
    """replay -> measured equals the earlier recorded values."""
    df, sc = _run_single_scenario("replay_temp_a", seed=42)
    start_s = sc.start_s
    window_s = 120.0

    # At t = start_s + 1, measured must equal recorded value from start_s - window_s + 1
    t_sample = start_s + 10.0
    t_earlier = start_s - window_s + 10.0

    meas_replay = df.loc[df["time_s"] == t_sample, "temp_cell_a"].iloc[0]
    meas_original = df.loc[df["time_s"] == t_earlier, "temp_cell_a"].iloc[0]

    assert meas_replay == meas_original, (
        f"Replayed value {meas_replay} does not match original recorded value {meas_original}!"
    )


def test_abusive_command_effect():
    """abusive_command -> applied current 192 A after start."""
    df, sc = _run_single_scenario("abusive_command", seed=42)
    start_s = sc.start_s

    post_df = df[df["time_s"] > start_s]
    applied = post_df["applied_current_a"].values
    assert np.all(applied == 192.0), "Applied current was not exactly 192.0 A after start_s!"


def test_pump_off_effect():
    """pump_off -> applied pump 0 and peak true temp > normal + 3 C."""
    df_norm, _ = _run_single_scenario("normal", seed=42)
    df_pump_off, sc = _run_single_scenario("pump_off", seed=42)
    start_s = sc.start_s

    post_df = df_pump_off[df_pump_off["time_s"] > start_s]
    assert np.all(post_df["applied_pump_lpm"].values == 0.0), "Pump command was not 0 after start_s!"

    peak_normal = df_norm["true_battery_temp"].max()
    peak_pump_off = df_pump_off["true_battery_temp"].max()
    temp_delta = peak_pump_off - peak_normal

    print(f"\n[Pump Off Test] Normal Peak: {peak_normal:.2f} C, Pump Off Peak: {peak_pump_off:.2f} C (Delta: {temp_delta:.2f} C)")
    assert temp_delta > 3.0, f"Pump off temperature increase {temp_delta:.2f} C was not > 3.0 C over normal!"


def test_pump_degradation_effect():
    """pump_degradation -> true flow near 6 L/min while the applied pump command stays 10."""
    df, sc = _run_single_scenario("pump_degradation", seed=42)
    start_s = sc.start_s

    post_df = df[df["time_s"] > start_s + 15.0]  # Allow time for pump hydrodynamic settling
    charging_post_df = post_df[post_df["applied_current_a"] > 0]

    # Command stays 10 L/min
    assert np.all(charging_post_df["applied_pump_lpm"].values == 10.0), (
        "Applied pump command did not stay at 10.0 L/min during charging!"
    )

    # True flow settled near 6.0 L/min (fraction 0.6 * 10)
    mean_flow = float(charging_post_df["true_pump_flow_lpm"].mean())
    print(f"\n[Pump Degradation Test] Settled true pump flow: {mean_flow:.3f} L/min")
    assert abs(mean_flow - 6.0) < 0.1, f"True flow {mean_flow:.3f} L/min not near 6.0 L/min!"


def test_aged_cell_effect():
    """aged_cell -> late-charge pack voltage at least 2 V above normal."""
    df_norm, _ = _run_single_scenario("normal", seed=42)
    df_aged, _ = _run_single_scenario("aged_cell", seed=42)

    # Compare late charge at t = 1000 s (both actively charging at 128 A)
    v_norm = df_norm.loc[df_norm["time_s"] == 1000.0, "true_pack_voltage"].iloc[0]
    v_aged = df_aged.loc[df_aged["time_s"] == 1000.0, "true_pack_voltage"].iloc[0]
    delta_v = v_aged - v_norm

    print(f"\n[Aged Cell Test] Normal Pack Voltage: {v_norm:.2f} V, Aged: {v_aged:.2f} V (Delta: {delta_v:.2f} V)")
    assert delta_v >= 2.0, f"Aged pack voltage rise {delta_v:.2f} V was not at least 2.0 V above normal!"


def test_dead_sensor_nan_effect():
    """dead_sensor_nan -> NaN appears in logged measurement AND run completes without NaN in applied current."""
    df, sc = _run_single_scenario("dead_sensor_nan_temp_a", seed=42)
    start_s = sc.start_s

    post_df = df[df["time_s"] > start_s]
    assert post_df["temp_cell_a"].isna().any(), "NaN did not appear in logged temp_cell_a measurement!"
    assert not df["applied_current_a"].isna().any(), "NaN appeared in applied current!"
    assert len(df) > 1000, "Run did not complete full session!"


def test_demo_hot_attack_effect():
    """demo_hot_attack: print peak true temperature and first time it exceeds 45 C and 50 C. Assert peak > 50 C."""
    df, _ = _run_single_scenario("demo_hot_attack", seed=42)
    peak_temp = float(df["true_battery_temp"].max())

    above_45 = df[df["true_battery_temp"] > 45.0]
    above_50 = df[df["true_battery_temp"] > 50.0]

    t_cross_45 = float(above_45["time_s"].iloc[0]) if not above_45.empty else None
    t_cross_50 = float(above_50["time_s"].iloc[0]) if not above_50.empty else None

    print(f"\n========================================================")
    print(f"demo_hot_attack Thermal Assessment:")
    print(f"  Peak True Battery Temperature: {peak_temp:.2f} °C")
    print(f"  First time exceeding 45 °C    : {t_cross_45:.1f} s")
    print(f"  First time exceeding 50 °C    : {t_cross_50:.1f} s")
    print(f"========================================================")

    assert peak_temp > 50.0, f"Peak temperature {peak_temp:.2f} C was not > 50.0 C!"
    assert t_cross_45 is not None
    assert t_cross_50 is not None


def test_normal_run_current():
    """normal run: after first 10 s and before last 5 s, applied current never drops below 100 A."""
    df, _ = _run_single_scenario("normal", seed=42)
    t_end = df["time_s"].iloc[-1]

    # Filter window: 10 s < t < (t_end - 5 s)
    window_df = df[(df["time_s"] > 10.0) & (df["time_s"] < t_end - 5.0)]
    min_current = float(window_df["applied_current_a"].min())

    print(f"\n[Normal Run Current Test] Minimum applied current in [10s, {t_end-5:.0f}s]: {min_current:.2f} A")
    assert min_current >= 100.0, f"Applied current dropped to {min_current:.2f} A (< 100 A limit) during charging window!"
