"""
PhysicsGuard - Quick Evaluation Report
=======================================
Runs all 17 scenarios at seed 42 with the PhysicsGuard detector attached.
For each scenario reports:
  - Detection time after start_s (first warning/critical alarm matching expected family)
  - Diagnosis class
  - is_attack value
  - False alarms before start_s

Also runs 20 extra normal runs (seeds 200-219) to measure false alarm rate.

Saves:
  - outputs/guard_report.csv
  - Timeline PNGs for: demo_hot_attack, coordinated, pump_off, drift_temp_a
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Path setup
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
PARENT_DIR = ROOT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from physicsguard import params
    from physicsguard.plant import Plant
    from physicsguard.controller import ChargerBMS
    from physicsguard.simulate import run_session
    from physicsguard.scenarios import build_scenario, Scenario
    from physicsguard.run_scenarios import SCENARIO_NAMES
    from physicsguard.guard import PhysicsGuard, Verdict
except ImportError:
    import params
    from plant import Plant
    from controller import ChargerBMS
    from simulate import run_session
    from scenarios import build_scenario, Scenario
    from run_scenarios import SCENARIO_NAMES
    from guard import PhysicsGuard, Verdict


# ---------------------------------------------------------------------------
# Expected diagnosis family per scenario name
# ---------------------------------------------------------------------------
EXPECTED_FAMILY: Dict[str, Optional[str]] = {
    "freeze_temp_a":              "SENSOR_FROZEN",
    "dead_sensor_flatline_temp_a": "SENSOR_FROZEN",
    "spike_temp_a":               "SENSOR_SPOOF",
    "drift_temp_a":               "SENSOR_SPOOF",
    "replay_temp_a":              "SENSOR_SPOOF",
    "dead_sensor_nan_temp_a":     "SENSOR_HW_FAULT",
    "pump_degradation":           "ACTUATOR_FAULT",
    "abusive_command":            "UNSAFE_COMMAND",
    "pump_off":                   "UNSAFE_COMMAND",
    "coordinated":                "UNSAFE_COMMAND",          # or COMMAND_RESPONSE_MISMATCH
    "drift_current":              "COMMAND_RESPONSE_MISMATCH",
    "drift_voltage":              "SENSOR_SPOOF",            # or COMMAND_RESPONSE_MISMATCH
    "demo_hot_attack":            "SENSOR_FROZEN",           # or SENSOR_SPOOF first
    # Must remain NORMAL:
    "normal":                     None,
    "aged_cell":                  None,
    "hot_day":                    None,
    # freeze_current changes nothing physical: "no observable effect"
    "freeze_current":             None,
}

# Scenarios for which detection is expected (missing = honest miss)
DETECTION_EXPECTED = {
    name for name, exp in EXPECTED_FAMILY.items()
    if exp is not None and name not in ("freeze_current",)
}

# Scenarios that must stay NORMAL
MUST_BE_NORMAL = {"normal", "aged_cell", "hot_day"}

# For COORDINATED, accept either UNSAFE_COMMAND or COMMAND_RESPONSE_MISMATCH
MULTI_ACCEPT: Dict[str, List[str]] = {
    "coordinated":    ["UNSAFE_COMMAND", "COMMAND_RESPONSE_MISMATCH"],
    "drift_voltage":  ["SENSOR_SPOOF", "COMMAND_RESPONSE_MISMATCH"],
    "demo_hot_attack": ["SENSOR_FROZEN", "SENSOR_SPOOF"],
}

# Scenarios to generate timeline PNGs for
TIMELINE_SCENARIOS = {"demo_hot_attack", "coordinated", "pump_off", "drift_temp_a"}

# Colour map for diagnosis bands
DIAG_COLORS = {
    "NORMAL":                   "#2ecc71",
    "SENSOR_HW_FAULT":          "#e74c3c",
    "SENSOR_FROZEN":            "#3498db",
    "SENSOR_SPOOF":             "#e67e22",
    "ACTUATOR_FAULT":           "#9b59b6",
    "COMMAND_RESPONSE_MISMATCH": "#f39c12",
    "UNSAFE_COMMAND":           "#c0392b",
}


# ---------------------------------------------------------------------------
# Core session runner with guard
# ---------------------------------------------------------------------------

def run_session_with_guard(
    plant: Plant,
    controller: ChargerBMS,
    scenario: Scenario,
    guard: Optional[PhysicsGuard],
    seed: int = 42,
    ambient_c: float = 25.0,
    inlet_c: float = 25.0,
    max_steps: int = 7200,
) -> Tuple[pd.DataFrame, List[Verdict]]:
    """
    Run a closed-loop session with an optional PhysicsGuard.

    Delegates to simulate.run_session(..., guard=guard).
    """
    df = run_session(
        plant=plant,
        controller=controller,
        seed=seed,
        scenario=scenario,
        guard=guard,
        ambient_c=ambient_c,
        inlet_c=inlet_c,
        max_duration_s=float(max_steps),
    )
    if guard is not None:
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
    else:
        verdicts = [Verdict(t=row["time_s"]) for _, row in df.iterrows()]

    return df, verdicts


def _build_scenario_plant(sc_name: str, seed: int = 42) -> Tuple[Plant, ChargerBMS, Scenario]:
    """Build plant and controller for a named scenario."""
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
    return plant, ChargerBMS(), sc


def _run_one(sc_name: str, seed: int = 42, with_guard: bool = True):
    """Run a single scenario with guard and return (df, verdicts, sc)."""
    plant, ctrl, sc = _build_scenario_plant(sc_name, seed)
    guard = PhysicsGuard() if with_guard else None
    df, verdicts = run_session_with_guard(plant, ctrl, sc, guard, seed=seed)
    return df, verdicts, sc


# ---------------------------------------------------------------------------
# Detection analysis
# ---------------------------------------------------------------------------

def _check_detection(
    sc_name: str,
    df: pd.DataFrame,
    verdicts: List[Verdict],
    sc: Scenario,
) -> Dict[str, Any]:
    """
    Analyse verdicts for a scenario and return a detection result dict.
    """
    start_s = sc.start_s if sc.start_s is not None else 0.0
    expected = EXPECTED_FAMILY.get(sc_name)
    acceptable = MULTI_ACCEPT.get(sc_name, [expected] if expected else [])

    # False alarms before start_s (for attack/fault scenarios)
    # For normal scenarios, any alarm is a false alarm
    fa_before = 0
    first_detection_t: Optional[float] = None
    first_diagnosis = "NORMAL"
    first_is_attack = None

    for v in verdicts:
        t = v.t
        is_alarming = v.alarm_level in ("warning", "critical")
        matches = v.diagnosis in acceptable

        if sc_name in MUST_BE_NORMAL:
            # Any alarm is a false alarm
            if is_alarming:
                fa_before += 1
        elif sc_name == "freeze_current":
            # No observable effect expected
            pass
        else:
            # Before start: count false alarms
            if t < start_s:
                if is_alarming:
                    fa_before += 1
            else:
                # After start: look for first matching detection
                if first_detection_t is None and is_alarming and matches:
                    first_detection_t = t
                    first_diagnosis = v.diagnosis
                    first_is_attack = v.is_attack

    # Time from start_s to first detection
    detection_lag: Optional[float] = None
    if first_detection_t is not None:
        detection_lag = first_detection_t - start_s

    # Build result dict
    if sc_name == "freeze_current":
        status = "no_observable_effect"
    elif sc_name in MUST_BE_NORMAL:
        status = "NORMAL" if fa_before == 0 else f"FALSE_ALARM({fa_before})"
    elif expected is None:
        status = "NORMAL"
    elif first_detection_t is not None:
        status = "DETECTED"
    else:
        status = "MISSED"

    return {
        "scenario": sc_name,
        "expected_family": expected or "NORMAL",
        "status": status,
        "detection_time_after_start_s": detection_lag,
        "first_diagnosis": first_diagnosis,
        "is_attack": first_is_attack,
        "false_alarms_before_start": fa_before,
        "start_s": start_s,
    }


# ---------------------------------------------------------------------------
# Timeline PNG generation
# ---------------------------------------------------------------------------

def _save_timeline_png(
    sc_name: str,
    df: pd.DataFrame,
    sc: Scenario,
    save_path: Path,
) -> None:
    """
    Save a two-panel timeline PNG:
    Top: true battery temperature, sensor A, sensor B, coolant outlet.
    Bottom: diagnosis as coloured bands.
    Vertical line at start_s.
    """
    t = df["time_s"].values
    true_temp = df["true_battery_temp"].values
    temp_a = df["temp_cell_a"].values
    temp_b = df["temp_cell_b"].values
    coolant_out = df["coolant_temp_out"].values
    diags = df["diagnosis"].values

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Top panel
    ax_top.plot(t, true_temp, "k-", lw=2.0, label="True Battery Temp")
    ax_top.plot(t, temp_a, color="#1f77b4", lw=1.2, alpha=0.85, label="Sensor A (temp_cell_a)")
    ax_top.plot(t, temp_b, color="#ff7f0e", lw=1.2, alpha=0.85, label="Sensor B (temp_cell_b)")
    ax_top.plot(t, coolant_out, color="#2ca02c", lw=1.2, alpha=0.7, label="Coolant Outlet")
    ax_top.set_ylabel("Temperature [°C]", fontsize=11)
    ax_top.set_title(f"PhysicsGuard Timeline: {sc_name}", fontsize=13, fontweight="bold")
    ax_top.grid(True, linestyle=":", alpha=0.5)
    ax_top.legend(loc="upper left", fontsize=9)

    # Add start line
    start_s = sc.start_s if sc.start_s is not None else 0.0
    if start_s > 0:
        ax_top.axvline(x=start_s, color="red", lw=2.0, ls="--", alpha=0.8, label=f"start_s={start_s:.0f}s")

    # Bottom panel: diagnosis bands
    unique_diags = list(DIAG_COLORS.keys())
    diag_to_num = {d: i for i, d in enumerate(unique_diags)}
    diag_nums = np.array([diag_to_num.get(d, 0) for d in diags], dtype=float)

    # Colour each time step as a filled rectangle
    for i in range(len(t) - 1):
        d = diags[i]
        color = DIAG_COLORS.get(d, "#cccccc")
        ax_bot.axvspan(t[i], t[i + 1], facecolor=color, alpha=0.9)

    # Add start line
    if start_s > 0:
        ax_bot.axvline(x=start_s, color="red", lw=2.0, ls="--", alpha=0.8)

    # Legend patches
    patches = [
        mpatches.Patch(facecolor=c, label=d)
        for d, c in DIAG_COLORS.items()
    ]
    ax_bot.legend(handles=patches, loc="upper right", fontsize=7, ncol=2)
    ax_bot.set_yticks([])
    ax_bot.set_ylabel("Diagnosis", fontsize=10)
    ax_bot.set_xlabel("Time [s]", fontsize=11)
    ax_bot.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved timeline -> {save_path}")


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(seed: int = 42) -> pd.DataFrame:
    """
    Run the full evaluation: 17 scenarios + 20 extra normal runs.

    Returns
    -------
    pd.DataFrame : guard_report
    """
    outputs_dir = ROOT_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PhysicsGuard - Quick Evaluation (Observe-Only)")
    print("=" * 80)

    report_rows: List[Dict[str, Any]] = []

    for sc_name in SCENARIO_NAMES:
        print(f"\n  Running: {sc_name}")
        df, verdicts, sc = _run_one(sc_name, seed=seed, with_guard=True)
        result = _check_detection(sc_name, df, verdicts, sc)
        report_rows.append(result)

        det_lag = result["detection_time_after_start_s"]
        status = result["status"]
        diag = result["first_diagnosis"]
        fa = result["false_alarms_before_start"]
        is_atk = result["is_attack"]

        print(
            f"    Status: {status:<30} "
            f"Detection lag: {f'{det_lag:.0f}s' if det_lag is not None else 'N/A':>8}  "
            f"Diag: {diag:<30} "
            f"is_attack: {str(is_atk):<8} "
            f"FA_before: {fa}"
        )

        # Timeline PNG for selected scenarios
        if sc_name in TIMELINE_SCENARIOS:
            png_path = outputs_dir / f"timeline_{sc_name}.png"
            _save_timeline_png(sc_name, df, sc, png_path)

    # ------------------------------------------------------------------
    # Extra normal runs: seeds 200-219
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Extra Normal Runs (seeds 200-219) - False Alarm Analysis")
    print("=" * 80)

    total_session_seconds = 0
    total_false_alarms = 0
    import random as _random

    for extra_seed in range(200, 220):
        rng_extra = _random.Random(extra_seed)
        perturb_frac = rng_extra.uniform(-0.10, 0.10)
        plant = Plant(
            perturb=True,
            seed=extra_seed,
            initial_soc=params.SOC_INITIAL,
            initial_temp_c=params.T_AMBIENT_NOMINAL_C,
            sensor_seed=extra_seed
        )
        ctrl = ChargerBMS()
        sc_normal = build_scenario("normal")
        guard = PhysicsGuard()
        df_extra, verdicts_extra = run_session_with_guard(
            plant, ctrl, sc_normal, guard, seed=extra_seed
        )
        n_steps = len(df_extra)
        n_fa = sum(
            1 for v in verdicts_extra if v.alarm_level in ("warning", "critical")
        )
        total_session_seconds += n_steps
        total_false_alarms += n_fa
        print(f"  seed {extra_seed}: {n_steps:4d} s, {n_fa:3d} alarms")

    fa_per_hour = (total_false_alarms / total_session_seconds * 3600) if total_session_seconds > 0 else 0.0
    print(f"\nTotal session time: {total_session_seconds} s ({total_session_seconds/3600:.2f} h)")
    print(f"Total false alarms: {total_false_alarms}")
    print(f"False alarm rate  : {fa_per_hour:.2f} alarms/hour")

    # Add extra normal summary to report
    report_rows.append({
        "scenario": "extra_normal_20runs",
        "expected_family": "NORMAL",
        "status": f"FA_rate={fa_per_hour:.2f}/hr",
        "detection_time_after_start_s": None,
        "first_diagnosis": "NORMAL",
        "is_attack": None,
        "false_alarms_before_start": total_false_alarms,
        "start_s": None,
    })

    # ------------------------------------------------------------------
    # Save guard report CSV
    # ------------------------------------------------------------------
    df_report = pd.DataFrame(report_rows)
    csv_path = outputs_dir / "guard_report.csv"
    df_report.to_csv(csv_path, index=False)
    print(f"\nSaved guard report -> {csv_path}")

    # ------------------------------------------------------------------
    # Print final summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Guard Report Summary")
    print("=" * 80)
    print(df_report.to_string(index=False))
    print("=" * 80)
    print(f"\nFalse alarm rate on 20 extra normal runs: {fa_per_hour:.2f} alarms/hour")

    return df_report


if __name__ == "__main__":
    evaluate()
