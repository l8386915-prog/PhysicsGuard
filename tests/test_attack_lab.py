"""
tests/test_attack_lab.py - Test Suite for Phase 6.5 Attack Lab
==============================================================
Validates:
  1. Freeze holds signal value constant.
  2. Drift slope matches commanded rate within 10%.
  3. set_value current override reaches actuator in unprotected run.
  4. Stacked attacks apply both effects simultaneously.
  5. Invalid inputs are clamped to physical boundaries.
  6. Smoke run of demo_hot_attack via the builder gives peak > 50 °C unprotected
     and <= 45 °C protected.
"""

import math
import pytest
import numpy as np
import pandas as pd

from physicsguard.attacks_custom import (
    build_custom_scenario,
    validate_and_clamp_spec,
    load_preset,
    get_preset_options,
    CLAMP_BOUNDS,
)
from sim_engine import run_custom_attack_simulation, compute_param_hash


def test_freeze_holds_value_constant():
    """Verify that freeze attack locks the measured sensor reading at start_s."""
    sc = build_custom_scenario([
        {"target": "temp_cell_a", "type": "freeze", "start_s": 100.0, "duration_s": None}
    ])
    assert sc.sensor_tamper is not None

    # Step through before and after start_s
    meas_99 = {"temp_cell_a": 28.5, "temp_cell_b": 28.6}
    res_99 = sc.sensor_tamper(99.0, meas_99.copy())
    assert res_99["temp_cell_a"] == 28.5

    # At start_s
    meas_100 = {"temp_cell_a": 29.2, "temp_cell_b": 29.3}
    res_100 = sc.sensor_tamper(100.0, meas_100.copy())
    frozen_val = res_100["temp_cell_a"]
    assert frozen_val == 29.2

    # After start_s with fluctuating inputs
    for t, noisy_val in [(101.0, 31.4), (150.0, 35.8), (200.0, 42.1)]:
        res_t = sc.sensor_tamper(t, {"temp_cell_a": noisy_val, "temp_cell_b": noisy_val})
        assert res_t["temp_cell_a"] == frozen_val, f"Value changed at t={t}: expected {frozen_val}, got {res_t['temp_cell_a']}"


def test_drift_slope_within_10_percent():
    """Verify that drift rate per minute matches the commanded slope within 10%."""
    commanded_rate_per_min = -6.0  # -0.1 °C/sec
    sc = build_custom_scenario([
        {
            "target": "temp_cell_a",
            "type": "drift",
            "magnitude": commanded_rate_per_min,
            "start_s": 100.0,
            "duration_s": None
        }
    ])
    assert sc.sensor_tamper is not None

    base_temp = 30.0
    t_start = 100.0
    t_end = 220.0  # 120 seconds duration = 2 minutes

    res_start = sc.sensor_tamper(t_start, {"temp_cell_a": base_temp})
    res_end = sc.sensor_tamper(t_end, {"temp_cell_a": base_temp})

    val_start = res_start["temp_cell_a"]
    val_end = res_end["temp_cell_a"]

    measured_slope_per_sec = (val_end - val_start) / (t_end - t_start)
    expected_slope_per_sec = commanded_rate_per_min / 60.0

    relative_error = abs(measured_slope_per_sec - expected_slope_per_sec) / abs(expected_slope_per_sec)
    assert relative_error <= 0.10, f"Drift slope error {relative_error*100:.2f}% exceeds 10% tolerance"


def test_set_value_current_override_reaches_actuator_unprotected():
    """Verify set_value override on cmd_current reaches applied current in unprotected run."""
    target_current = 192.0
    start_time = 300.0
    res = run_custom_attack_simulation(
        attack_specs=[
            {"target": "cmd_current", "type": "set_value", "magnitude": target_current, "start_s": start_time}
        ],
        ambient_c=25.0,
        inlet_c=25.0,
        seed=42
    )
    assert "error" not in res, res.get("error")
    df_u = res["df_unpr"]

    # During attack period (time_s > start_time while actively charging), applied current equals target_current
    attack_rows = df_u[(df_u["time_s"] > start_time) & (df_u["applied_current_a"] > 0)]
    assert len(attack_rows) > 10
    applied_currents = attack_rows["applied_current_a"].values
    assert np.allclose(applied_currents, target_current, atol=1e-3), (
        f"Unprotected applied current {applied_currents[:5]} does not match override {target_current}"
    )


def test_stacked_attacks_apply_both():
    """Verify up to stacked attacks apply both sensor tamper and command tamper."""
    sc = build_custom_scenario([
        {"target": "temp_cell_a", "type": "freeze", "start_s": 200.0},
        {"target": "cmd_pump", "type": "set_value", "magnitude": 0.0, "start_s": 200.0}
    ])
    assert sc.sensor_tamper is not None
    assert sc.command_tamper is not None

    # Before attack
    s_pre = sc.sensor_tamper(199.0, {"temp_cell_a": 25.5})
    c_pre = sc.command_tamper(199.0, (128.0, 10.0))
    assert s_pre["temp_cell_a"] == 25.5
    assert c_pre[1] == 10.0

    # During attack
    s_post = sc.sensor_tamper(250.0, {"temp_cell_a": 38.0})
    c_post = sc.command_tamper(250.0, (128.0, 10.0))
    assert s_post["temp_cell_a"] == 25.5  # Frozen at pre-attack value
    assert c_post[1] == 0.0               # Commanded pump killed to 0


def test_invalid_input_is_clamped():
    """Verify invalid and out-of-range inputs are safely clamped without crashing."""
    # Negative start_s clamped to 0
    s1 = validate_and_clamp_spec({"start_s": -100.0})
    assert s1["start_s"] == 0.0

    # Start > 3000 clamped to 3000
    s2 = validate_and_clamp_spec({"start_s": 4500.0})
    assert s2["start_s"] == 3000.0

    # Current > 250 clamped to 250
    s3 = validate_and_clamp_spec({"target": "cmd_current", "type": "set_value", "magnitude": 999.0})
    assert s3["magnitude"] == 250.0

    # Negative current clamped to 0
    s4 = validate_and_clamp_spec({"target": "cmd_current", "type": "set_value", "magnitude": -50.0})
    assert s4["magnitude"] == 0.0

    # Pump > 20 clamped to 20
    s5 = validate_and_clamp_spec({"target": "cmd_pump", "type": "set_value", "magnitude": 50.0})
    assert s5["magnitude"] == 20.0

    # Temperature > 120 clamped to 120, < -20 clamped to -20
    s6 = validate_and_clamp_spec({"target": "temp_cell_a", "type": "set_value", "magnitude": 150.0})
    assert s6["magnitude"] == 120.0

    s7 = validate_and_clamp_spec({"target": "temp_cell_a", "type": "set_value", "magnitude": -80.0})
    assert s7["magnitude"] == -20.0

    # Non-numeric / NaN input safe fallback
    s8 = validate_and_clamp_spec({"target": "invalid_xyz", "type": "bad_type", "start_s": float("nan"), "magnitude": None})
    assert s8["target"] == "temp_cell_a"
    assert s8["type"] == "freeze"
    assert s8["start_s"] == 300.0


def test_demo_hot_attack_smoke_run_via_builder():
    """Smoke run: demo_hot_attack via builder gives peak > 50 °C unprotected and <= 45 °C protected."""
    specs, env = load_preset("demo_hot_attack")
    assert len(specs) == 2

    res = run_custom_attack_simulation(
        attack_specs=specs,
        ambient_c=env.get("ambient_c", 35.0),
        inlet_c=env.get("inlet_c", 35.0),
        seed=42
    )
    assert "error" not in res, res.get("error")
    kpis = res["kpis"]

    assert kpis["peak_t_unpr"] > 50.0, f"Unprotected peak {kpis['peak_t_unpr']} not > 50.0 °C"
    assert kpis["peak_t_prot"] <= 45.0, f"Protected peak {kpis['peak_t_prot']} not <= 45.0 °C"
    assert kpis["detected"] is True
    assert kpis["detected_label"] == "YES"
    assert kpis["detection_latency_s"] is not None


def test_honesty_rule_no_physical_effect():
    """Verify honesty rule: attack on coolant_temp_out reports NO PHYSICAL EFFECT IN THIS MODEL."""
    res = run_custom_attack_simulation(
        attack_specs=[
            {"target": "coolant_temp_out", "type": "drift", "magnitude": -2.0, "start_s": 300.0}
        ],
        ambient_c=25.0,
        inlet_c=25.0,
        seed=42
    )
    assert "error" not in res, res.get("error")
    kpis = res["kpis"]
    assert kpis["has_physical_effect"] is False
    assert kpis["physical_effect_label"] == "NO PHYSICAL EFFECT IN THIS MODEL"


def test_timeout_handling():
    """Verify graceful friendly message on wall-clock timeout without crash."""
    res = run_custom_attack_simulation(
        attack_specs=[
            {"target": "cmd_current", "type": "set_value", "magnitude": 192.0, "start_s": 300.0}
        ],
        ambient_c=25.0,
        inlet_c=25.0,
        seed=999,
        timeout_wall_clock_s=0.000001
    )
    assert "error" in res
    assert "timed out" in res["error"].lower()


def test_app_compiles_cleanly():
    """Verify that app.py has no syntax errors and compiles cleanly."""
    import py_compile
    py_compile.compile("app.py", doraise=True)

