"""
PhysicsGuard - Phase 4 Guard Tests
====================================
Pytest suite validating the PhysicsGuard observer:

1. Guard never uses true_* values (source code test).
2. Guard never raises on NaN input.
3. Each of the key attack/fault scenarios is detected within 120 s of start_s.
4. Normal, aged_cell, hot_day and pump_degradation never give is_attack = True.
"""

import ast
import sys
import inspect
import math
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
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
from physicsguard.guard import PhysicsGuard, Verdict


# ---------------------------------------------------------------------------
# Helper: run a scenario with guard and return all verdicts
# ---------------------------------------------------------------------------

def _run_with_guard(
    sc_name: str,
    seed: int = 42,
    max_steps: int = 7200
) -> Tuple[List[Verdict], Scenario]:
    """Run a named scenario with a fresh PhysicsGuard and return verdicts list."""
    sc = build_scenario(sc_name)
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
    guard = PhysicsGuard()

    ambient_c = float(sc.ambient_c) if sc.ambient_c is not None else params.T_AMBIENT_NOMINAL_C
    inlet_c = float(sc.inlet_c) if sc.inlet_c is not None else params.T_COOLANT_INLET_NOMINAL_C

    df = run_session(
        plant,
        controller,
        seed=seed,
        scenario=sc,
        guard=guard,
        ambient_c=ambient_c,
        inlet_c=inlet_c,
        max_duration_s=float(max_steps),
    )
    verdicts = [
        Verdict(
            t=row["time_s"],
            diagnosis=row.get("diagnosis", "NORMAL"),
            alarm_level=row.get("alarm_level", "none"),
            is_attack=row.get("is_attack"),
            blamed_signals=row.get("blamed_signals", []),
            reasons=row.get("reasons", []),
        )
        for _, row in df.iterrows()
    ]
    return verdicts, sc


def _first_alarm_after(
    verdicts: List[Verdict],
    start_s: float,
    accepted_diags: List[str]
) -> Optional[float]:
    """Return the first alarm time (t) after start_s matching the expected diagnosis set."""
    for v in verdicts:
        if v.t >= start_s and v.alarm_level in ("warning", "critical"):
            if v.diagnosis in accepted_diags:
                return v.t
    return None


# ---------------------------------------------------------------------------
# Test 1: Guard and Mitigation never read true_* values (source-code test)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod_name", ["guard.py", "mitigation.py"])
def test_guard_never_reads_true_values(mod_name: str):
    """
    Parse source code as an AST and verify no attribute access
    or string literals starting with true_* appear anywhere.
    """
    target_path = PKG_DIR / mod_name
    assert target_path.exists(), f"{mod_name} not found at {target_path}"
    source = target_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_prefixes = ("true_",)

    class TrueAttrVisitor(ast.NodeVisitor):
        def __init__(self):
            self.violations: List[str] = []

        def visit_Attribute(self, node):
            if isinstance(node.attr, str):
                for prefix in forbidden_prefixes:
                    if node.attr.startswith(prefix):
                        self.violations.append(
                            f"Line {node.lineno}: .{node.attr}"
                        )
            self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                for prefix in forbidden_prefixes:
                    if node.value.startswith(prefix):
                        self.violations.append(
                            f"Line {node.lineno}: string '{node.value}'"
                        )
            self.generic_visit(node)

    visitor = TrueAttrVisitor()
    visitor.visit(tree)

    if visitor.violations:
        pytest.fail(
            f"{mod_name} references true_* values:\n" +
            "\n".join(visitor.violations)
        )
    print(f"\n[Source AST Test] {mod_name}: no true_* references found.")


# ---------------------------------------------------------------------------
# Test 2: Guard never raises on NaN input
# ---------------------------------------------------------------------------

def test_guard_no_raise_on_nan_input():
    """
    Feed a guard with a stream of measurements containing NaN values in every
    sensor channel and ensure no exception is raised.
    """
    guard = PhysicsGuard()
    nan_meas = {
        "pack_voltage": float("nan"),
        "pack_current": float("nan"),
        "soc_estimate": float("nan"),
        "temp_cell_a": float("nan"),
        "temp_cell_b": float("nan"),
        "coolant_temp_in": float("nan"),
        "coolant_temp_out": float("nan"),
        "pump_flow_lpm": float("nan"),
        "ambient_temp": float("nan"),
        "charger_power_cmd_kw": float("nan"),
    }

    try:
        for t in range(1, 21):
            v = guard.update(float(t), nan_meas, (128.0, 10.0))
            assert isinstance(v, Verdict), "Expected a Verdict object"
            assert v.alarm_level in ("none", "warning", "critical")
    except Exception as exc:
        pytest.fail(f"Guard raised on NaN input: {exc!r}")

    print(f"\n[NaN Test] Guard handled 20 steps of all-NaN input without raising.")


# ---------------------------------------------------------------------------
# Test 3: Detection within allowed time limit of start_s
# ---------------------------------------------------------------------------

DETECTION_SCENARIOS = {
    "freeze_temp_a":               (["SENSOR_FROZEN"], 120.0),
    "spike_temp_a":                (["SENSOR_SPOOF"], 120.0),
    "drift_temp_a":                (["SENSOR_SPOOF", "COMMAND_RESPONSE_MISMATCH"], 120.0),
    "replay_temp_a":               (["SENSOR_SPOOF", "SENSOR_FROZEN"], 120.0),
    "dead_sensor_nan_temp_a":      (["SENSOR_HW_FAULT"], 120.0),
    "pump_degradation":            (["ACTUATOR_FAULT"], 120.0),
    "abusive_command":             (["UNSAFE_COMMAND"], 120.0),
    "pump_off":                    (["UNSAFE_COMMAND"], 120.0),
    "coordinated":                 (["UNSAFE_COMMAND", "COMMAND_RESPONSE_MISMATCH", "SENSOR_SPOOF"], 120.0),
    "demo_hot_attack":             (["SENSOR_FROZEN", "SENSOR_SPOOF", "UNSAFE_COMMAND"], 120.0),
    "drift_current":              (["COMMAND_RESPONSE_MISMATCH", "SENSOR_SPOOF"], 240.0),
    "drift_voltage":              (["SENSOR_SPOOF", "COMMAND_RESPONSE_MISMATCH"], 240.0),
}


@pytest.mark.parametrize("sc_name,config", list(DETECTION_SCENARIOS.items()))
def test_detection_within_120s(sc_name: str, config: Tuple[List[str], float]):
    """
    Each specified attack/fault scenario must be detected within its allowed
    time limit of start_s (120 s or 240 s).
    """
    accepted_diags, limit_s = config
    verdicts, sc = _run_with_guard(sc_name, seed=42)
    start_s = sc.start_s if sc.start_s is not None else 0.0

    first_t = _first_alarm_after(verdicts, start_s, accepted_diags)
    lag = (first_t - start_s) if first_t is not None else None

    print(
        f"\n[Detection Test] {sc_name}: start_s={start_s:.0f}s, "
        f"first_detection={first_t}, lag={lag}s (limit: {limit_s:.0f}s)"
    )

    assert first_t is not None, (
        f"MISSED: {sc_name} was not detected after start_s={start_s:.0f}s "
        f"(accepted diagnoses: {accepted_diags})"
    )
    assert lag <= limit_s, (
        f"SLOW: {sc_name} detected {lag:.0f}s after start_s (limit: {limit_s:.0f}s)"
    )


# ---------------------------------------------------------------------------
# Test 4: Normal, aged_cell, hot_day, pump_degradation never is_attack = True
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sc_name", [
    "normal", "aged_cell", "hot_day", "pump_degradation"
])
def test_no_false_attack_flag(sc_name: str):
    """
    Normal, aged_cell, hot_day and pump_degradation must never produce
    is_attack = True verdicts.
    """
    verdicts, sc = _run_with_guard(sc_name, seed=42)
    attack_verdicts = [v for v in verdicts if v.is_attack is True]

    if attack_verdicts:
        examples = [(v.t, v.diagnosis, v.reasons) for v in attack_verdicts[:3]]
        pytest.fail(
            f"Scenario '{sc_name}' produced is_attack=True verdicts. "
            f"Examples: {examples}"
        )

    print(f"\n[No-Attack Test] {sc_name}: {len(verdicts)} steps, 0 is_attack=True verdicts. PASS")
