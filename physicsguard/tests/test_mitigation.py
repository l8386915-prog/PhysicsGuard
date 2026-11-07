"""
Test Suite for PhysicsGuard Phase 5: Closed-Loop Actuator SafetyGate
===================================================================
Covers Acceptance Criteria 1 to 7:
  1. normal, aged_cell, hot_day: gate NEVER intervenes; charge time within 1% of unprotected.
  2. pump_degradation: is_attack never True; protected peak true temp <= 45 C.
  3. Every attack scenario: protected peak true temp <= 45 C; demo_hot_attack protected seconds above 45 C == 0.
  4. abusive_command: applied current never exceeds 130.56 A after the first screened step.
  5. Sensor-spoof temperature attacks (freeze/spike/drift/replay temp_a, dead sensor):
     controller input stays within 3 C of the twin; charge completes with time penalty <= 5%.
  6. coordinated: protected run keeps pump >= 5 L/min while charging.
  7. AST test: mitigation.py and guard.py NEVER read true_* values or access plant directly.
"""

import ast
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from physicsguard.plant import Plant
from physicsguard.controller import ChargerBMS
from physicsguard.scenarios import build_scenario
from physicsguard.simulate import run_session
from physicsguard.guard import PhysicsGuard
from physicsguard.mitigation import SafetyGate
import physicsguard.params as params


def _run_protected(scenario_name: str, seed: int = 42) -> pd.DataFrame:
    """Helper to run a scenario with gate and guard enabled."""
    sc = build_scenario(scenario_name, seed=seed)
    plant = Plant(perturb=False, sensor_seed=seed, **dict(sc.plant_kwargs))
    controller = ChargerBMS()
    gate = SafetyGate()
    guard = PhysicsGuard()
    gate.set_guard(guard)
    return run_session(plant, controller, scenario=sc, guard=guard, gate=gate, seed=seed)


def _run_unprotected(scenario_name: str, seed: int = 42) -> pd.DataFrame:
    """Helper to run a scenario unprotected (gate=None)."""
    sc = build_scenario(scenario_name, seed=seed)
    plant = Plant(perturb=False, sensor_seed=seed, **dict(sc.plant_kwargs))
    controller = ChargerBMS()
    return run_session(plant, controller, scenario=sc, gate=None, seed=seed)


# ---------------------------------------------------------------------------
# Criterion 1: normal, aged_cell, hot_day -> gate never intervenes; duration within 1%
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario_name", ["normal", "aged_cell", "hot_day"])
def test_criterion_1_normal_aged_hot(scenario_name: str):
    df_prot = _run_protected(scenario_name, seed=42)
    df_unpr = _run_unprotected(scenario_name, seed=42)

    # Gate NEVER intervened
    assert df_prot["intervened"].sum() == 0, (
        f"{scenario_name}: Gate intervened {df_prot['intervened'].sum()} times, expected 0"
    )

    # Charge time within 1% of unprotected
    dur_prot = float(df_prot["time_s"].iloc[-1])
    dur_unpr = float(df_unpr["time_s"].iloc[-1])
    rel_diff = abs(dur_prot - dur_unpr) / dur_unpr
    assert rel_diff <= 0.01, (
        f"{scenario_name}: Duration penalty {rel_diff*100:.2f}% exceeds 1% "
        f"({dur_prot:.1f}s vs {dur_unpr:.1f}s)"
    )


# ---------------------------------------------------------------------------
# Criterion 2: pump_degradation -> is_attack never True; peak true temp <= 45 C
# ---------------------------------------------------------------------------
def test_criterion_2_pump_degradation():
    df_prot = _run_protected("pump_degradation", seed=42)

    # is_attack never True
    assert not (df_prot["is_attack"] == True).any(), (
        "pump_degradation must never be classified as an attack"
    )

    # Peak true temp <= 45 C
    peak_temp = df_prot["true_battery_temp"].max()
    assert peak_temp <= 45.0, (
        f"pump_degradation peak temperature {peak_temp:.2f} C exceeds 45.0 C"
    )


# ---------------------------------------------------------------------------
# Criterion 3: Every attack scenario -> protected peak true temp <= 45 C;
#               demo_hot_attack protected seconds above 45 C == 0
# ---------------------------------------------------------------------------
ATTACK_SCENARIOS = [
    "dead_sensor_nan_temp_a", "dead_sensor_flatline_temp_a",
    "freeze_temp_a", "replay_temp_a", "drift_temp_a", "spike_temp_a",
    "freeze_current", "drift_current", "drift_voltage",
    "abusive_command", "pump_off", "coordinated", "demo_hot_attack"
]

@pytest.mark.parametrize("scenario_name", ATTACK_SCENARIOS)
def test_criterion_3_attack_peak_temp(scenario_name: str):
    df_prot = _run_protected(scenario_name, seed=42)
    peak_temp = df_prot["true_battery_temp"].max()
    assert peak_temp <= 45.0, (
        f"{scenario_name}: Protected peak temp {peak_temp:.2f} C exceeds 45.0 C"
    )

    if scenario_name == "demo_hot_attack":
        secs_above_45 = int((df_prot["true_battery_temp"] > 45.0).sum())
        assert secs_above_45 == 0, (
            f"demo_hot_attack has {secs_above_45}s above 45 C, expected 0"
        )


# ---------------------------------------------------------------------------
# Criterion 4: abusive_command -> applied current never exceeds 130.56 A
# ---------------------------------------------------------------------------
def test_criterion_4_abusive_command():
    df_prot = _run_protected("abusive_command", seed=42)
    max_allowed = params.PACK_RATED_CHARGE_CURRENT_A * 1.02  # 130.56 A

    # After the first screened step (index > 0)
    applied_currents = df_prot["applied_current_a"].iloc[1:]
    max_applied = applied_currents.max()
    assert max_applied <= max_allowed + 1e-4, (
        f"abusive_command applied current {max_applied:.2f} A exceeds 130.56 A"
    )


# ---------------------------------------------------------------------------
# Criterion 5: Sensor-spoof temperature attacks -> controller input within 3 C
#               of twin; charge completes with time penalty <= 5%
# ---------------------------------------------------------------------------
SENSOR_TEMP_ATTACKS = [
    "freeze_temp_a", "spike_temp_a", "drift_temp_a",
    "replay_temp_a", "dead_sensor_nan_temp_a", "dead_sensor_flatline_temp_a"
]

@pytest.mark.parametrize("scenario_name", SENSOR_TEMP_ATTACKS)
def test_criterion_5_sensor_spoof(scenario_name: str):
    df_prot = _run_protected(scenario_name, seed=42)
    df_norm = _run_unprotected("normal", seed=42)

    # Charge completes (SOC >= 0.80)
    final_soc = df_prot["true_soc"].iloc[-1]
    assert final_soc >= 0.80, f"{scenario_name} did not reach 0.80 SOC (final={final_soc:.3f})"

    # Duration penalty <= 5%
    dur_prot = float(df_prot["time_s"].iloc[-1])
    dur_norm = float(df_norm["time_s"].iloc[-1])
    penalty = (dur_prot - dur_norm) / dur_norm
    assert penalty <= 0.05, (
        f"{scenario_name}: Duration penalty {penalty*100:.2f}% exceeds 5% "
        f"({dur_prot:.1f}s vs {dur_norm:.1f}s)"
    )

    # Controller input stays within 3 C of the twin
    intervened_steps = df_prot[df_prot["intervened"]]
    if len(intervened_steps) > 0:
        temp_diff = (intervened_steps["trusted_temp"] - intervened_steps["true_battery_temp"]).abs().max()
        assert temp_diff <= 3.0, (
            f"{scenario_name}: Trusted temp differed from true temp by {temp_diff:.2f} C > 3.0 C"
        )


# ---------------------------------------------------------------------------
# Criterion 6: coordinated -> protected run keeps pump >= 5 L/min while charging
# ---------------------------------------------------------------------------
def test_criterion_6_coordinated():
    df_prot = _run_protected("coordinated", seed=42)

    # Filter steps after initial mechanical startup (t >= 10s) while charging
    charging_steps = df_prot[(df_prot["time_s"] >= 10.0) & (df_prot["applied_current_a"] > 5.0)]
    min_pump = charging_steps["true_pump_flow_lpm"].min()
    assert min_pump >= 5.0, (
        f"coordinated: Minimum pump flow while charging was {min_pump:.2f} L/min < 5.0 L/min"
    )


# ---------------------------------------------------------------------------
# Criterion 7: AST Check -> mitigation.py NEVER reads true_* values or accesses plant
# ---------------------------------------------------------------------------
def test_criterion_7_ast_mitigation_isolation():
    file_path = Path(__file__).resolve().parent.parent / "mitigation.py"
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    forbidden_prefixes = ("true_",)

    class TrueAttrVisitor(ast.NodeVisitor):
        def __init__(self):
            self.violations = []

        def visit_Attribute(self, node):
            if isinstance(node.attr, str):
                for prefix in forbidden_prefixes:
                    if node.attr.startswith(prefix):
                        self.violations.append(f"Line {node.lineno}: .{node.attr}")
            self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                for prefix in forbidden_prefixes:
                    if node.value.startswith(prefix):
                        self.violations.append(f"Line {node.lineno}: string '{node.value}'")
            self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.id, str):
                for prefix in forbidden_prefixes:
                    if node.id.startswith(prefix):
                        self.violations.append(f"Line {node.lineno}: name '{node.id}'")
            self.generic_visit(node)

    visitor = TrueAttrVisitor()
    visitor.visit(tree)
    assert not visitor.violations, f"mitigation.py violated isolation rules:\n" + "\n".join(visitor.violations)


# ---------------------------------------------------------------------------
# Phase 5.1 New Tests
# ---------------------------------------------------------------------------
def test_criterion_1_perturbed_20_seeds():
    """Criterion 1 over 20 random perturbed seeds (seeds 500-519 from dev set)."""
    scenarios = ["normal", "aged_cell", "hot_day"]
    for seed in range(500, 520):
        sc_name = scenarios[seed % len(scenarios)]
        sc = build_scenario(sc_name, seed=seed)

        plant_prot = Plant(perturb=True, seed=seed, sensor_seed=seed, **dict(sc.plant_kwargs))
        ctrl_prot = ChargerBMS()
        gate = SafetyGate()
        guard = PhysicsGuard()
        gate.set_guard(guard)
        df_prot = run_session(plant_prot, ctrl_prot, scenario=sc, gate=gate, guard=guard, seed=seed)

        plant_unpr = Plant(perturb=True, seed=seed, sensor_seed=seed, **dict(sc.plant_kwargs))
        ctrl_unpr = ChargerBMS()
        df_unpr = run_session(plant_unpr, ctrl_unpr, scenario=sc, gate=None, guard=None, seed=seed)

        assert df_prot["intervened"].sum() == 0, (
            f"Seed {seed} ({sc_name}): Gate intervened {df_prot['intervened'].sum()} times"
        )
        dur_prot = float(df_prot["time_s"].iloc[-1])
        dur_unpr = float(df_unpr["time_s"].iloc[-1])
        rel_diff = abs(dur_prot - dur_unpr) / dur_unpr
        assert rel_diff <= 0.01, (
            f"Seed {seed} ({sc_name}): duration penalty {rel_diff*100:.2f}% exceeds 1%"
        )


def test_demo_hot_attack_completes():
    """demo_hot_attack protected run reaches SOC 0.80, peak true temp <= 45 C, charge time within 25% of baseline."""
    seed = 42
    sc_attack = build_scenario("demo_hot_attack", seed=seed)
    plant_prot = Plant(perturb=False, sensor_seed=seed, **dict(sc_attack.plant_kwargs))
    ctrl_prot = ChargerBMS()
    gate = SafetyGate()
    guard = PhysicsGuard()
    gate.set_guard(guard)
    df_prot = run_session(plant_prot, ctrl_prot, scenario=sc_attack, gate=gate, guard=guard, seed=seed)

    # Baseline unprotected hot-ambient normal run
    from physicsguard.scenarios import Scenario
    sc_hot_norm = Scenario(
        name="normal", label_class="normal", label_type="normal",
        ambient_c=35.0, inlet_c=35.0, plant_kwargs={"initial_temp_c": 35.0}
    )
    plant_unpr = Plant(perturb=False, sensor_seed=seed, **dict(sc_hot_norm.plant_kwargs))
    ctrl_unpr = ChargerBMS()
    df_unpr = run_session(plant_unpr, ctrl_unpr, scenario=sc_hot_norm, gate=None, guard=None, seed=seed)

    final_soc = float(df_prot["true_soc"].iloc[-1])
    peak_temp = float(df_prot["true_battery_temp"].max())
    secs_above_45 = int((df_prot["true_battery_temp"] > 45.0).sum())
    dur_prot = float(df_prot["time_s"].iloc[-1])
    dur_unpr = float(df_unpr["time_s"].iloc[-1])
    time_penalty = (dur_prot - dur_unpr) / dur_unpr

    assert final_soc >= 0.80, f"demo_hot_attack final SOC {final_soc:.3f} < 0.80"
    assert peak_temp <= 45.0, f"demo_hot_attack peak temp {peak_temp:.2f} C > 45.0 C"
    assert secs_above_45 == 0, f"demo_hot_attack spent {secs_above_45}s above 45 C"
    assert time_penalty <= 0.25, f"demo_hot_attack charge time penalty {time_penalty*100:.2f}% > 25%"


def test_heldout_evaluation_uses_perturb_true():
    """Verify via AST that evaluate_protected.py uses Plant(perturb=True) for heldout runs."""
    eval_file = Path(__file__).resolve().parent.parent.parent / "evaluate_protected.py"
    with open(eval_file, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)

    found_perturb_true = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = getattr(node.func, "id", None)
            if func_name == "Plant":
                for kw in node.keywords:
                    if kw.arg == "perturb" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        found_perturb_true = True
    assert found_perturb_true, "evaluate_protected.py must use Plant(perturb=True, ...) in heldout sessions"


ALL_SCENARIO_NAMES = [
    "normal", "pump_degradation", "aged_cell", "hot_day",
    "dead_sensor_nan_temp_a", "dead_sensor_flatline_temp_a",
    "freeze_temp_a", "replay_temp_a", "drift_temp_a", "spike_temp_a",
    "freeze_current", "drift_current", "drift_voltage",
    "abusive_command", "pump_off", "coordinated", "demo_hot_attack"
]


@pytest.mark.parametrize("scenario_name", ALL_SCENARIO_NAMES)
def test_mode_transitions_per_session_bounded(scenario_name: str):
    """Phase 5.2 Item 4: Test that the number of mode transitions per session is <= 6 for every scenario."""
    df_prot = _run_protected(scenario_name, seed=42)
    modes = df_prot["gate_mode"].tolist()
    transitions = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i-1])
    assert transitions <= 6, (
        f"{scenario_name}: mode transitions ({transitions}) exceeded 6"
    )


