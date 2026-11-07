"""
tests/test_app_smoke.py - Smoke Tests for Streamlit Application Pipeline
========================================================================
Validates that the dashboard engine, simulation runs, fallback demo caching,
plain-alarm baseline, and moment explanation run cleanly and headlessly.
"""

import pytest
import math
import numpy as np
import pandas as pd
from pathlib import Path

from sim_engine import (
    ALL_SCENARIOS, MAIN_DEMO_SCENARIOS, DEMO_CACHE_DIR,
    run_full_simulation, load_from_cache, explain_moment,
    read_live_esp32_sensor, PlainAlarmChargerBMS, InstrumentedPhysicsGuard
)


def test_app_imports_and_constants():
    """Verify core scenarios and demo cache directory are defined."""
    assert len(ALL_SCENARIOS) == 17
    assert len(MAIN_DEMO_SCENARIOS) == 4
    assert DEMO_CACHE_DIR.exists()


def test_load_from_demo_cache():
    """Verify that cached demo data loads correctly for the main demo scenario."""
    res = load_from_cache("demo_hot_attack")
    assert res is not None, "Failed to load demo_hot_attack from cache"
    assert "df_unpr" in res and len(res["df_unpr"]) > 0
    assert "df_prot" in res and len(res["df_prot"]) > 0
    assert "df_plain" in res
    assert "kpis" in res

    k = res["kpis"]
    assert k["scenario"] == "demo_hot_attack"
    assert k["peak_t_unpr"] > 50.0  # Unprotected overheats
    assert k["peak_t_prot"] <= 45.0  # Protected holds below 45 C
    assert k["overall_badge"] == "ATTACK"


def test_headless_simulation_run():
    """Execute one live simulation scenario headlessly and check output integrity."""
    res = run_full_simulation("pump_off", ambient_c=35.0, inlet_c=35.0, seed=42)
    assert res is not None

    df_u = res["df_unpr"]
    df_p = res["df_prot"]
    df_plain = res["df_plain"]
    kpis = res["kpis"]

    # Trajectory validation
    assert len(df_u) > 100
    assert len(df_p) > 100
    assert "true_battery_temp" in df_u.columns
    assert "true_battery_temp" in df_p.columns

    # Residual columns present in protected df
    for r_col in ["R1a", "R1b", "R2", "R3", "Rv", "Rflow", "Rcur"]:
        assert r_col in df_p.columns, f"Missing residual column {r_col}"

    # KPIs check
    assert kpis["peak_t_unpr"] > 45.0  # Pump off overheats in hot ambient
    assert kpis["peak_t_prot"] <= 45.0  # Protected stays safe
    assert kpis["overall_badge"] in ["ATTACK", "FAULT", "NORMAL"]
    assert kpis["final_mode"] in ["NOMINAL", "DEGRADED", "LIMP_HOME", "SHUTDOWN"]


def test_plain_alarm_baseline_behavior():
    """Verify plain-alarm baseline stops on >45 C, and handles NaNs safely."""
    ctrl_both = PlainAlarmChargerBMS(use_both=True)
    ctrl_a = PlainAlarmChargerBMS(use_both=False)

    # 1. Normal temperature -> charging continues
    cmd_curr, cmd_pump = ctrl_both.compute_commands({"temp_cell_a": 35.0, "temp_cell_b": 35.2, "soc_estimate": 0.5})
    assert cmd_curr > 0.0
    assert not ctrl_both.alarm_tripped

    # 2. Overheating in B only (A is spoofed/frozen)
    cmd_curr_a, _ = ctrl_a.compute_commands({"temp_cell_a": 25.0, "temp_cell_b": 46.5, "soc_estimate": 0.5})
    cmd_curr_both, _ = ctrl_both.compute_commands({"temp_cell_a": 25.0, "temp_cell_b": 46.5, "soc_estimate": 0.5})

    assert not ctrl_a.alarm_tripped  # A-only baseline blinded!
    assert ctrl_both.alarm_tripped   # Main baseline trips
    assert cmd_curr_both == 0.0

    # 3. NaN sensor safety -> never crash
    ctrl_nan = PlainAlarmChargerBMS()
    cmd_nan, _ = ctrl_nan.compute_commands({"temp_cell_a": float("nan"), "temp_cell_b": float("nan"), "soc_estimate": 0.5})
    assert isinstance(cmd_nan, float)


def test_explain_moment_generation():
    """Verify that explain_moment generates valid physical explanations with numbers."""
    row = pd.Series({
        "temp_cell_a": 25.0,
        "temp_cell_b": 35.5,
        "trusted_temp": 34.2,
        "pump_flow_lpm": 10.0,
        "applied_current_a": 128.0,
        "gate_mode": "DEGRADED",
        "diagnosis": "SENSOR_FROZEN",
        "intervened": True
    })

    text = explain_moment(312.0, row)
    assert "t=312 s" in text
    assert "25.0 °C" in text
    assert "DEGRADED" in text
    assert "SENSOR_FROZEN" in text


def test_live_sensor_fallback():
    """Verify live serial sensor falls back gracefully and non-blockingly."""
    val = read_live_esp32_sensor(port="COM_NON_EXISTENT_9999")
    assert val is None
