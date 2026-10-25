"""
PhysicsGuard - Closed-Loop Evaluation Pipeline
==============================================
Evaluates PhysicsGuard + SafetyGate closed-loop mitigation system against:
  - Phase 5 Criterion 1: Unprotected vs Protected on 17 Scenarios.
  - Phase 5.1 Task 3: 200 Held-Out Sessions (fresh seeds 700 to 899) with perturb=True,
    randomized aging/ambient/inlet/pump derating, Wilson 95% CIs, and
    charge-time penalty vs matching unprotected normal baseline.
  - Phase 5.1 Task 4 & Addition D: Hot day attacks (35 C ambient / 35 C inlet),
    seed 42 + perturbed seeds 500-504 (random start_s in [200, 600]).
  - Phase 5.1 Addition F: Side-by-side comparison plots with matching x-axes
    and one-line metric summaries.

Isolation Rules:
  - SafetyGate and PhysicsGuard NEVER access true_* variables or scenario labels.
  - Deterministic evaluation reproducible with specified seeds.
"""

import sys
import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Path configuration
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physicsguard import params
from physicsguard.plant import Plant
from physicsguard.controller import ChargerBMS
from physicsguard.scenarios import build_scenario, Scenario
from physicsguard.simulate import run_session
from physicsguard.guard import PhysicsGuard, Verdict
from physicsguard.mitigation import SafetyGate
from physicsguard.attacks import (
    freeze, spike, drift, replay, abusive_command, pump_off, coordinated, demo_hot_attack
)
from physicsguard.faults import (
    pump_degradation, aged_cell, hot_day, dead_sensor
)

OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a binomial proportion k/n.
    For 95% confidence, z = 1.95996 (~1.96).
    """
    if n <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = k / n
    denominator = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1.0 - p) / n + (z**2) / (4.0 * (n**2)))
    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)
    return float(ci_lower), float(ci_upper)


def clean_df(df: pd.DataFrame, latched: bool = False) -> pd.DataFrame:
    """
    Drop t=0 pre-step sample and last 2 stop-latch zero samples.
    """
    d = df[df["time_s"] > 0].copy()
    if latched and len(d) > 2:
        d = d.iloc[:-2].copy()
    return d


# ===========================================================================
# C1: Unprotected vs Protected Baseline Evaluation (17 Scenarios)
# ===========================================================================

EVAL_SCENARIOS = [
    "normal",
    "pump_degradation",
    "aged_cell",
    "hot_day",
    "dead_sensor_nan_temp_a",
    "dead_sensor_flatline_temp_a",
    "freeze_temp_a",
    "replay_temp_a",
    "drift_temp_a",
    "spike_temp_a",
    "freeze_current",
    "drift_current",
    "drift_voltage",
    "abusive_command",
    "pump_off",
    "coordinated",
    "demo_hot_attack",
]


def run_c1_evaluation(seed: int = 42) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("C1: Unprotected vs Protected Evaluation (17 Scenarios, Seed 42)")
    print("=" * 78)

    rows = []

    for name in EVAL_SCENARIOS:
        sc = build_scenario(name, seed=seed)

        # 1. Unprotected run
        plant_unpr = Plant(perturb=False, sensor_seed=seed, **dict(sc.plant_kwargs))
        ctrl_unpr = ChargerBMS()
        df_unpr = run_session(plant_unpr, ctrl_unpr, scenario=sc, gate=None, guard=None, seed=seed)
        df_unpr_clean = clean_df(df_unpr, getattr(ctrl_unpr, "latched_stop", False))

        # 2. Protected run
        plant_prot = Plant(perturb=False, sensor_seed=seed, **dict(sc.plant_kwargs))
        ctrl_prot = ChargerBMS()
        gate = SafetyGate()
        guard = PhysicsGuard()
        gate.set_guard(guard)
        df_prot = run_session(plant_prot, ctrl_prot, scenario=sc, gate=gate, guard=guard, seed=seed)
        df_prot_clean = clean_df(df_prot, getattr(ctrl_prot, "latched_stop", False))

        # Metrics extraction (using cleaned records without t=0 or stop-latch zeros)
        peak_t_unpr = float(df_unpr_clean["true_battery_temp"].max())
        peak_t_prot = float(df_prot_clean["true_battery_temp"].max())

        s45_unpr = int((df_unpr_clean["true_battery_temp"] > 45.0).sum())
        s45_prot = int((df_prot_clean["true_battery_temp"] > 45.0).sum())

        s50_unpr = int((df_unpr_clean["true_battery_temp"] > 50.0).sum())
        s50_prot = int((df_prot_clean["true_battery_temp"] > 50.0).sum())

        soc_unpr = float(df_unpr_clean["true_soc"].iloc[-1])
        soc_prot = float(df_prot_clean["true_soc"].iloc[-1])

        dur_unpr = float(df_unpr_clean["time_s"].iloc[-1])
        dur_prot = float(df_prot_clean["time_s"].iloc[-1])

        # 3. Matching normal baseline for this exact scenario's ambient/inlet
        amb = sc.ambient_c if sc.ambient_c is not None else params.T_AMBIENT_NOMINAL_C
        inlet = sc.inlet_c if sc.inlet_c is not None else params.T_COOLANT_INLET_NOMINAL_C
        init_t = sc.plant_kwargs.get("initial_temp_c", amb)
        p_base = Plant(perturb=False, sensor_seed=seed, initial_soc=params.SOC_INITIAL, initial_temp_c=init_t)
        c_base = ChargerBMS()
        sc_base = Scenario(name="normal", label_class="normal", label_type="normal",
                           ambient_c=amb, inlet_c=inlet, plant_kwargs={"initial_temp_c": init_t})
        df_base = run_session(p_base, c_base, scenario=sc_base, gate=None, guard=None, seed=seed, ambient_c=amb, inlet_c=inlet)
        df_base_clean = clean_df(df_base, getattr(c_base, "latched_stop", False))
        dur_norm_base = float(df_base_clean["time_s"].iloc[-1])
        penalty_vs_norm = (dur_prot - dur_norm_base) / dur_norm_base if dur_norm_base > 0 else 0.0

        interv_mask = df_prot_clean["intervened"].values
        interv_times = df_prot_clean.loc[interv_mask, "time_s"].values
        t_first_interv = float(interv_times[0]) if len(interv_times) > 0 else None
        num_interv = int(interv_mask.sum())

        final_mode = str(df_prot_clean["gate_mode"].iloc[-1])

        row = {
            "scenario": name,
            "peak_true_temp_unprotected": round(peak_t_unpr, 2),
            "peak_true_temp_protected": round(peak_t_prot, 2),
            "seconds_above_45C_unprotected": s45_unpr,
            "seconds_above_45C_protected": s45_prot,
            "seconds_above_50C_unprotected": s50_unpr,
            "seconds_above_50C_protected": s50_prot,
            "final_soc_unprotected": round(soc_unpr, 3),
            "final_soc_protected": round(soc_prot, 3),
            "charge_time_unprotected": round(dur_unpr, 1),
            "charge_time_protected": round(dur_prot, 1),
            "charge_time_normal_baseline": round(dur_norm_base, 1),
            "charge_time_penalty_vs_baseline_pct": round(penalty_vs_norm * 100, 2),
            "time_to_first_intervention": round(t_first_interv, 1) if t_first_interv is not None else "N/A",
            "num_intervened_seconds": num_interv,
            "final_mode": final_mode,
        }
        rows.append(row)

        print(
            f"  {name:27s} | Unpr Peak: {peak_t_unpr:5.1f}C (>45:{s45_unpr:3d}s) | "
            f"Prot Peak: {peak_t_prot:5.1f}C (>45:{s45_prot:3d}s) | "
            f"Chg: {dur_prot:5.0f}s (Norm: {dur_norm_base:5.0f}s, {penalty_vs_norm*100:+5.1f}%) | Mode: {final_mode}"
        )

    df_res = pd.DataFrame(rows)
    csv_path = OUTPUTS_DIR / "protected_vs_unprotected.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved C1 results -> {csv_path}")
    return df_res


# ===========================================================================
# C2: Held-Out Batch Evaluation (Seeds 900 to 1199: 300 Sessions)
# ===========================================================================

def run_c2_heldout() -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("Phase 5.2 Held-Out Evaluation on Fresh Seeds 900 to 1199 (300 Sessions, perturb=True)")
    print("=" * 78)

    rng = np.random.default_rng(2026)

    # 100 non-attacks across seeds 900-999 (25 per type)
    non_attack_types = ["normal", "aged_cell", "hot_day", "pump_degradation"]
    # 200 attacks across seeds 1000-1199 (>= 15 per type, 12 types)
    attack_types = [
        "freeze_temp_a", "spike_temp_a", "drift_temp_a", "replay_temp_a",
        "dead_sensor_flatline_temp_a", "dead_sensor_nan_temp_a",
        "abusive_command", "pump_off", "coordinated", "drift_current",
        "drift_voltage", "demo_hot_attack"
    ]

    session_records = []
    latencies_by_type: Dict[str, List[float]] = {atk: [] for atk in attack_types}
    detected_by_type: Dict[str, int] = {atk: 0 for atk in attack_types}
    total_by_type: Dict[str, int] = {atk: 0 for atk in attack_types}

    non_attack_interventions = 0
    non_attack_false_alarms = 0
    total_non_attacks = 0

    unprotected_below_45_count = 0
    protected_below_45_count = 0
    unprotected_below_50_count = 0
    protected_below_50_count = 0

    # Charge penalty grouped by class
    charge_penalties_by_class: Dict[str, List[float]] = {}

    for seed in range(900, 1200):
        is_attack_session = (seed >= 1000)

        if not is_attack_session:
            cat = non_attack_types[(seed - 900) % len(non_attack_types)]
            session_class = cat
            if cat == "normal":
                amb = float(rng.uniform(20.0, 30.0))
                sc = Scenario(
                    name="normal", label_class="normal", label_type="normal",
                    ambient_c=amb, inlet_c=amb, plant_kwargs={"initial_temp_c": amb}
                )
            elif cat == "aged_cell":
                af = float(rng.uniform(1.10, 1.60))
                amb = float(rng.uniform(20.0, 30.0))
                sc = aged_cell(age_factor=af)
                sc.ambient_c = amb
                sc.inlet_c = amb
                sc.plant_kwargs["initial_temp_c"] = amb
            elif cat == "hot_day":
                amb = float(rng.uniform(30.0, 40.0))
                inlet = float(rng.uniform(30.0, 38.0))
                sc = hot_day(ambient_c=amb, inlet_c=inlet)
                sc.plant_kwargs["initial_temp_c"] = amb
            else:  # pump_degradation
                flow_derate = float(rng.uniform(0.40, 0.80))
                sc = pump_degradation(fraction=flow_derate)
                amb = params.T_AMBIENT_NOMINAL_C
                inlet = params.T_COOLANT_INLET_NOMINAL_C
                sc.ambient_c = amb
                sc.inlet_c = inlet
                sc.plant_kwargs["initial_temp_c"] = amb
            attack_type = None
            start_s = None
        else:
            atk_name = attack_types[(seed - 1000) % len(attack_types)]
            session_class = atk_name
            start_s = float(round(rng.uniform(200.0, 800.0)))
            attack_type = atk_name

            if atk_name == "freeze_temp_a":
                sc = freeze("temp_cell_a", start_s=start_s)
            elif atk_name == "spike_temp_a":
                sc = spike("temp_cell_a", start_s=start_s)
            elif atk_name == "drift_temp_a":
                sc = drift("temp_cell_a", start_s=start_s)
            elif atk_name == "replay_temp_a":
                sc = replay("temp_cell_a", start_s=start_s)
            elif atk_name == "dead_sensor_flatline_temp_a":
                sc = dead_sensor("temp_cell_a", "flatline", start_s=start_s)
            elif atk_name == "dead_sensor_nan_temp_a":
                sc = dead_sensor("temp_cell_a", "nan", start_s=start_s)
            elif atk_name == "abusive_command":
                sc = abusive_command(start_s=start_s)
            elif atk_name == "pump_off":
                sc = pump_off(start_s=start_s)
            elif atk_name == "coordinated":
                sc = coordinated(start_s=start_s)
            elif atk_name == "drift_current":
                sc = drift("pack_current", start_s=start_s)
            elif atk_name == "drift_voltage":
                sc = drift("pack_voltage", start_s=start_s)
            else:  # demo_hot_attack
                sc = demo_hot_attack()
                sc.start_s = start_s

            amb = sc.ambient_c if sc.ambient_c is not None else params.T_AMBIENT_NOMINAL_C
            inlet = sc.inlet_c if sc.inlet_c is not None else params.T_COOLANT_INLET_NOMINAL_C
            sc.ambient_c = amb
            sc.inlet_c = inlet
            if "initial_temp_c" not in sc.plant_kwargs:
                sc.plant_kwargs["initial_temp_c"] = amb

        # Phase 5.2 Item 1: Matching Unprotected Normal Baseline for this EXACT perturbed session
        base_amb = sc.ambient_c if sc.ambient_c is not None else params.T_AMBIENT_NOMINAL_C
        base_inlet = sc.inlet_c if sc.inlet_c is not None else params.T_COOLANT_INLET_NOMINAL_C
        base_init_t = sc.plant_kwargs.get("initial_temp_c", base_amb)

        p_base = Plant(perturb=True, seed=seed, sensor_seed=seed,
                       initial_soc=params.SOC_INITIAL, initial_temp_c=base_init_t)
        c_base = ChargerBMS()
        sc_base = Scenario(name="normal", label_class="normal", label_type="normal",
                           ambient_c=base_amb, inlet_c=base_inlet,
                           plant_kwargs={"initial_temp_c": base_init_t})
        df_base = run_session(p_base, c_base, scenario=sc_base, gate=None, guard=None, seed=seed,
                              ambient_c=base_amb, inlet_c=base_inlet)
        df_base_clean = clean_df(df_base, getattr(c_base, "latched_stop", False))
        normal_baseline_dur = float(df_base_clean["time_s"].iloc[-1])

        # Run Unprotected (Plant with perturb=True for all sessions)
        p_unpr = Plant(perturb=True, seed=seed, sensor_seed=seed, **dict(sc.plant_kwargs))
        c_unpr = ChargerBMS()
        df_unpr = run_session(p_unpr, c_unpr, scenario=sc, gate=None, guard=None, seed=seed,
                              ambient_c=base_amb, inlet_c=base_inlet)
        df_unpr_clean = clean_df(df_unpr, getattr(c_unpr, "latched_stop", False))

        # Run Protected (Plant with perturb=True for all sessions)
        p_prot = Plant(perturb=True, seed=seed, sensor_seed=seed, **dict(sc.plant_kwargs))
        c_prot = ChargerBMS()
        gate = SafetyGate()
        guard = PhysicsGuard()
        gate.set_guard(guard)
        df_prot = run_session(p_prot, c_prot, scenario=sc, gate=gate, guard=guard, seed=seed,
                              ambient_c=base_amb, inlet_c=base_inlet)
        df_prot_clean = clean_df(df_prot, getattr(c_prot, "latched_stop", False))

        peak_t_unpr = float(df_unpr_clean["true_battery_temp"].max())
        peak_t_prot = float(df_prot_clean["true_battery_temp"].max())
        dur_unpr = float(df_unpr_clean["time_s"].iloc[-1])
        dur_prot = float(df_prot_clean["time_s"].iloc[-1])

        if peak_t_unpr <= 45.0:
            unprotected_below_45_count += 1
        if peak_t_prot <= 45.0:
            protected_below_45_count += 1

        if peak_t_unpr <= 50.0:
            unprotected_below_50_count += 1
        if peak_t_prot <= 50.0:
            protected_below_50_count += 1

        # Charge-time penalty vs the matching unprotected NORMAL baseline
        if normal_baseline_dur > 0:
            penalty = (dur_prot - normal_baseline_dur) / normal_baseline_dur
            charge_penalties_by_class.setdefault(session_class, []).append(penalty)

        num_interv = int(df_prot_clean["intervened"].sum())
        intervened_bool = (num_interv > 0)

        # Check false alarms (is_attack == True on non-attack session)
        has_false_alarm = False
        if not is_attack_session:
            total_non_attacks += 1
            if intervened_bool:
                non_attack_interventions += 1
            # Check false alarms from guard
            alarms = df_prot_clean[df_prot_clean["is_attack"] == True]
            if len(alarms) > 0:
                non_attack_false_alarms += 1
                has_false_alarm = True
        else:
            total_by_type[attack_type] += 1
            # Check detection latency after start_s
            after_start = df_prot_clean[
                (df_prot_clean["time_s"] >= start_s) &
                (df_prot_clean["alarm_level"].isin(["warning", "critical"]))
            ]
            if len(after_start) > 0:
                det_t = float(after_start["time_s"].iloc[0])
                latency = max(0.0, det_t - start_s)
                latencies_by_type[attack_type].append(latency)
                detected_by_type[attack_type] += 1

        session_records.append({
            "seed": seed,
            "category": "attack" if is_attack_session else "non-attack",
            "type": session_class,
            "start_s": start_s,
            "peak_temp_unpr": peak_t_unpr,
            "peak_temp_prot": peak_t_prot,
            "dur_unpr": dur_unpr,
            "dur_prot": dur_prot,
            "norm_baseline_dur": normal_baseline_dur,
            "intervened_seconds": num_interv,
            "final_mode": df_prot_clean["gate_mode"].iloc[-1]
        })

    # Summary Statistics & Reporting
    summary_rows = []

    print("\n" + "=" * 95)
    print("Phase 5.2 Held-Out Evaluation (Seeds 900-1199: 300 Sessions, perturb=True)")
    print("=" * 95)

    print("\n--- Attack Detection Rates and Latency (Protected vs Unprotected) ---")
    print(f"{'Attack Type':<26} {'Protected Det':>16} {'95% Wilson CI':>18} {'Med Lat [s]':>12} {'p95 Lat [s]':>12} {'Unprotected Det':>16}")
    print("-" * 105)

    for atk in attack_types:
        n_tot = total_by_type[atk]
        n_det = detected_by_type[atk]
        det_rate = (n_det / n_tot) if n_tot > 0 else 1.0
        ci_lo, ci_hi = wilson_ci(n_det, n_tot)
        lats = latencies_by_type[atk]
        med_lat = float(np.median(lats)) if lats else 0.0
        p95_lat = float(np.percentile(lats, 95)) if lats else 0.0

        u_det = 0
        u_ci_lo, u_ci_hi = wilson_ci(0, n_tot)

        summary_rows.append({
            "metric_type": "attack_detection",
            "class_or_type": atk,
            "protected_count": f"{n_det} of {n_tot}",
            "protected_rate": round(det_rate, 4),
            "protected_ci_95": f"[{ci_lo:.3f}, {ci_hi:.3f}]",
            "unprotected_count": f"0 of {n_tot}",
            "unprotected_rate": 0.0,
            "unprotected_ci_95": f"[{u_ci_lo:.3f}, {u_ci_hi:.3f}]",
            "median_lat_s": round(med_lat, 1),
            "p95_lat_s": round(p95_lat, 1),
            "sample_count": n_tot
        })
        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]"
        print(f"{atk:<26} {n_det:>2d} of {n_tot:<2d} ({det_rate*100:5.1f}%) {ci_str:>18} {med_lat:>11.1f}s {p95_lat:>11.1f}s   0 of {n_tot:<2d} ( 0.0%)")

    # Non-attack false interventions and alarms
    false_interv_rate = non_attack_interventions / float(total_non_attacks)
    fi_ci_lo, fi_ci_hi = wilson_ci(non_attack_interventions, total_non_attacks)

    false_alarm_rate = non_attack_false_alarms / float(total_non_attacks)
    fa_ci_lo, fa_ci_hi = wilson_ci(non_attack_false_alarms, total_non_attacks)

    print("\n--- Non-Attack False Interventions and False Alarms (100 Sessions) ---")
    print(f"False Interventions : {non_attack_interventions} of {total_non_attacks} ({false_interv_rate*100:.2f}%, 95% Wilson CI: [{fi_ci_lo:.3f}, {fi_ci_hi:.3f}]) | Unprotected: 0 of {total_non_attacks}")
    print(f"False Alarms (Guard): {non_attack_false_alarms} of {total_non_attacks} ({false_alarm_rate*100:.2f}%, 95% Wilson CI: [{fa_ci_lo:.3f}, {fa_ci_hi:.3f}]) | Unprotected: 0 of {total_non_attacks}")

    summary_rows.append({
        "metric_type": "non_attack_metric",
        "class_or_type": "false_intervention_rate",
        "protected_count": f"{non_attack_interventions} of {total_non_attacks}",
        "protected_rate": round(false_interv_rate, 4),
        "protected_ci_95": f"[{fi_ci_lo:.3f}, {fi_ci_hi:.3f}]",
        "unprotected_count": f"0 of {total_non_attacks}",
        "unprotected_rate": 0.0,
        "unprotected_ci_95": f"[{fi_ci_lo:.3f}, {fi_ci_hi:.3f}]",
        "median_lat_s": None,
        "p95_lat_s": None,
        "sample_count": total_non_attacks
    })
    summary_rows.append({
        "metric_type": "non_attack_metric",
        "class_or_type": "false_alarm_rate",
        "protected_count": f"{non_attack_false_alarms} of {total_non_attacks}",
        "protected_rate": round(false_alarm_rate, 4),
        "protected_ci_95": f"[{fa_ci_lo:.3f}, {fa_ci_hi:.3f}]",
        "unprotected_count": f"0 of {total_non_attacks}",
        "unprotected_rate": 0.0,
        "unprotected_ci_95": f"[{fa_ci_lo:.3f}, {fa_ci_hi:.3f}]",
        "median_lat_s": None,
        "p95_lat_s": None,
        "sample_count": total_non_attacks
    })

    # Thermal safety rates
    total_sessions = 300
    unpr_45_rate = unprotected_below_45_count / float(total_sessions)
    u45_ci_lo, u45_ci_hi = wilson_ci(unprotected_below_45_count, total_sessions)
    prot_45_rate = protected_below_45_count / float(total_sessions)
    p45_ci_lo, p45_ci_hi = wilson_ci(protected_below_45_count, total_sessions)

    unpr_50_rate = unprotected_below_50_count / float(total_sessions)
    u50_ci_lo, u50_ci_hi = wilson_ci(unprotected_below_50_count, total_sessions)
    prot_50_rate = protected_below_50_count / float(total_sessions)
    p50_ci_lo, p50_ci_hi = wilson_ci(protected_below_50_count, total_sessions)

    print("\n--- Thermal Safety Rates Across All 300 Sessions (Protected vs Unprotected) ---")
    print(f"Peak True Temp <= 45 C: Protected {protected_below_45_count} of {total_sessions} ({prot_45_rate*100:.1f}%, 95% Wilson CI: [{p45_ci_lo:.3f}, {p45_ci_hi:.3f}]) | Unprotected {unprotected_below_45_count} of {total_sessions} ({unpr_45_rate*100:.1f}%, 95% Wilson CI: [{u45_ci_lo:.3f}, {u45_ci_hi:.3f}])")
    print(f"Peak True Temp <= 50 C: Protected {protected_below_50_count} of {total_sessions} ({prot_50_rate*100:.1f}%, 95% Wilson CI: [{p50_ci_lo:.3f}, {p50_ci_hi:.3f}]) | Unprotected {unprotected_below_50_count} of {total_sessions} ({unpr_50_rate*100:.1f}%, 95% Wilson CI: [{u50_ci_lo:.3f}, {u50_ci_hi:.3f}])")

    summary_rows.append({
        "metric_type": "thermal_safety",
        "class_or_type": "peak_temp_below_45C",
        "protected_count": f"{protected_below_45_count} of {total_sessions}",
        "protected_rate": round(prot_45_rate, 4),
        "protected_ci_95": f"[{p45_ci_lo:.3f}, {p45_ci_hi:.3f}]",
        "unprotected_count": f"{unprotected_below_45_count} of {total_sessions}",
        "unprotected_rate": round(unpr_45_rate, 4),
        "unprotected_ci_95": f"[{u45_ci_lo:.3f}, {u45_ci_hi:.3f}]",
        "median_lat_s": None,
        "p95_lat_s": None,
        "sample_count": total_sessions
    })
    summary_rows.append({
        "metric_type": "thermal_safety",
        "class_or_type": "peak_temp_below_50C",
        "protected_count": f"{protected_below_50_count} of {total_sessions}",
        "protected_rate": round(prot_50_rate, 4),
        "protected_ci_95": f"[{p50_ci_lo:.3f}, {p50_ci_hi:.3f}]",
        "unprotected_count": f"{unprotected_below_50_count} of {total_sessions}",
        "unprotected_rate": round(unpr_50_rate, 4),
        "unprotected_ci_95": f"[{u50_ci_lo:.3f}, {u50_ci_hi:.3f}]",
        "median_lat_s": None,
        "p95_lat_s": None,
        "sample_count": total_sessions
    })

    # Charge-time penalty per class
    print("\n--- Charge-Time Penalty vs Matching Unprotected Normal Baseline per Class ---")
    print(f"{'Class / Type':<26} {'Count':>6} {'Median Penalty':>16} {'p95 Penalty':>14}")
    print("-" * 65)

    for cls_name, penalties in sorted(charge_penalties_by_class.items()):
        med_p = float(np.median(penalties))
        p95_p = float(np.percentile(penalties, 95))
        summary_rows.append({
            "metric_type": "charge_time_penalty",
            "class_or_type": cls_name,
            "protected_count": f"{len(penalties)}",
            "protected_rate": round(med_p, 4),
            "protected_ci_95": None,
            "unprotected_count": "baseline",
            "unprotected_rate": 0.0,
            "unprotected_ci_95": None,
            "median_lat_s": round(med_p * 100.0, 2),
            "p95_lat_s": round(p95_p * 100.0, 2),
            "sample_count": len(penalties)
        })
        print(f"{cls_name:<26} {len(penalties):>6d} {med_p*100:>15.2f}% {p95_p*100:>13.2f}%")

    df_summary = pd.DataFrame(summary_rows)
    csv_path_v2 = OUTPUTS_DIR / "heldout_report_v2.csv"
    df_summary.to_csv(csv_path_v2, index=False)
    csv_path = OUTPUTS_DIR / "heldout_report.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved Phase 5.2 Held-Out Report -> {csv_path_v2}")
    return df_summary


# ===========================================================================
# Task 4 & Addition D: Stronger, Fair Hot Day Attack Evaluation
# ===========================================================================

HOT_ATTACK_SCENARIOS = ["pump_off", "abusive_command", "coordinated", "demo_hot_attack"]


def run_task4_hot_attacks() -> None:
    print("\n" + "=" * 78)
    print("Task 4 & Addition D: Hot Day Attacks Evaluation (35 C Ambient / 35 C Inlet)")
    print("=" * 78)

    # 1. outputs/hot_day_attacks.csv on seed 42 (Unprotected vs Protected)
    seed_42_rows = []
    for sc_name in HOT_ATTACK_SCENARIOS:
        # Build base scenario
        if sc_name == "demo_hot_attack":
            sc_unpr = demo_hot_attack()
            sc_prot = demo_hot_attack()
        elif sc_name == "pump_off":
            sc_unpr = pump_off(start_s=300.0)
            sc_prot = pump_off(start_s=300.0)
        elif sc_name == "abusive_command":
            sc_unpr = abusive_command(start_s=300.0)
            sc_prot = abusive_command(start_s=300.0)
        else:  # coordinated
            sc_unpr = coordinated(start_s=300.0)
            sc_prot = coordinated(start_s=300.0)

        for s_obj in [sc_unpr, sc_prot]:
            s_obj.ambient_c = 35.0
            s_obj.inlet_c = 35.0
            s_obj.plant_kwargs["initial_temp_c"] = 35.0

        # Unprotected
        p_u = Plant(perturb=False, sensor_seed=42, **dict(sc_unpr.plant_kwargs))
        c_u = ChargerBMS()
        df_u = run_session(p_u, c_u, scenario=sc_unpr, gate=None, guard=None, seed=42)
        df_u_c = clean_df(df_u, getattr(c_u, "latched_stop", False))

        # Protected
        p_p = Plant(perturb=False, sensor_seed=42, **dict(sc_prot.plant_kwargs))
        c_p = ChargerBMS()
        gate = SafetyGate()
        guard = PhysicsGuard()
        gate.set_guard(guard)
        df_p = run_session(p_p, c_p, scenario=sc_prot, gate=gate, guard=guard, seed=42)
        df_p_c = clean_df(df_p, getattr(c_p, "latched_stop", False))

        for mode_lbl, df_sub in [("unprotected", df_u_c), ("protected", df_p_c)]:
            pk_t = float(df_sub["true_battery_temp"].max())
            s45 = int((df_sub["true_battery_temp"] > 45.0).sum())
            s50 = int((df_sub["true_battery_temp"] > 50.0).sum())
            fin_soc = float(df_sub["true_soc"].iloc[-1])
            chg_time = float(df_sub["time_s"].iloc[-1])
            completed = bool(fin_soc >= 0.80)

            seed_42_rows.append({
                "scenario": sc_name,
                "mode": mode_lbl,
                "peak_temp_c": round(pk_t, 2),
                "seconds_above_45C": s45,
                "seconds_above_50C": s50,
                "charge_completed": completed,
                "final_soc": round(fin_soc, 3),
                "charge_time_s": round(chg_time, 1)
            })

    df_hot_42 = pd.DataFrame(seed_42_rows)
    csv_hot_42 = OUTPUTS_DIR / "hot_day_attacks.csv"
    df_hot_42.to_csv(csv_hot_42, index=False)
    print(f"Saved Task 4 hot day attacks -> {csv_hot_42}")

    # 2. Addition D: Hot perturbed evaluation on seeds 500-504 as well as seed 42
    # with perturb=True and random start_s in [200, 600]
    eval_seeds = [42, 500, 501, 502, 503, 504]
    rng_hot = np.random.default_rng(5555)

    report_rows = []

    for seed in eval_seeds:
        for sc_name in HOT_ATTACK_SCENARIOS:
            start_s = 60.0 if (sc_name == "demo_hot_attack" and seed == 42) else float(round(rng_hot.uniform(200.0, 600.0)))
            if sc_name == "demo_hot_attack":
                sc = demo_hot_attack()
                sc.start_s = start_s
            elif sc_name == "pump_off":
                sc = pump_off(start_s=start_s)
            elif sc_name == "abusive_command":
                sc = abusive_command(start_s=start_s)
            else:  # coordinated
                sc = coordinated(start_s=start_s)

            sc.ambient_c = 35.0
            sc.inlet_c = 35.0
            sc.plant_kwargs["initial_temp_c"] = 35.0

            # Unprotected run
            p_u = Plant(perturb=True, seed=seed, sensor_seed=seed, **dict(sc.plant_kwargs))
            c_u = ChargerBMS()
            df_u = run_session(p_u, c_u, scenario=sc, gate=None, guard=None, seed=seed)
            df_u_c = clean_df(df_u, getattr(c_u, "latched_stop", False))

            # Protected run
            p_p = Plant(perturb=True, seed=seed, sensor_seed=seed, **dict(sc.plant_kwargs))
            c_p = ChargerBMS()
            gate = SafetyGate()
            guard = PhysicsGuard()
            gate.set_guard(guard)
            df_p = run_session(p_p, c_p, scenario=sc, gate=gate, guard=guard, seed=seed)
            df_p_c = clean_df(df_p, getattr(c_p, "latched_stop", False))

            for mode_lbl, df_sub in [("unprotected", df_u_c), ("protected", df_p_c)]:
                pk_t = float(df_sub["true_battery_temp"].max())
                s45 = int((df_sub["true_battery_temp"] > 45.0).sum())
                s50 = int((df_sub["true_battery_temp"] > 50.0).sum())
                fin_soc = float(df_sub["true_soc"].iloc[-1])
                chg_time = float(df_sub["time_s"].iloc[-1])
                completed = bool(fin_soc >= 0.80)

                report_rows.append({
                    "seed": seed,
                    "scenario": sc_name,
                    "mode": mode_lbl,
                    "start_s": start_s,
                    "peak_temp_c": round(pk_t, 2),
                    "seconds_above_45C": s45,
                    "seconds_above_50C": s50,
                    "charge_completed": completed,
                    "final_soc": round(fin_soc, 3),
                    "charge_time_s": round(chg_time, 1)
                })

    df_hot_report = pd.DataFrame(report_rows)
    csv_hot_report = OUTPUTS_DIR / "hot_protection_report.csv"
    df_hot_report.to_csv(csv_hot_report, index=False)
    print(f"Saved Addition D hot protection report -> {csv_hot_report}")

    # Summary: median and worst case per scenario and mode
    summary_list = []
    for sc_name in HOT_ATTACK_SCENARIOS:
        for mode_lbl in ["unprotected", "protected"]:
            sub = df_hot_report[(df_hot_report["scenario"] == sc_name) & (df_hot_report["mode"] == mode_lbl)]
            summary_list.append({
                "scenario": sc_name,
                "mode": mode_lbl,
                "sample_count": len(sub),
                "peak_temp_median_c": round(float(sub["peak_temp_c"].median()), 2),
                "peak_temp_worst_c": round(float(sub["peak_temp_c"].max()), 2),
                "seconds_above_45C_median": int(sub["seconds_above_45C"].median()),
                "seconds_above_45C_worst": int(sub["seconds_above_45C"].max()),
                "seconds_above_50C_median": int(sub["seconds_above_50C"].median()),
                "seconds_above_50C_worst": int(sub["seconds_above_50C"].max()),
                "final_soc_median": round(float(sub["final_soc"].median()), 3),
                "final_soc_worst": round(float(sub["final_soc"].min()), 3),
                "charge_time_s_median": round(float(sub["charge_time_s"].median()), 1),
                "charge_time_s_worst": round(float(sub["charge_time_s"].max()), 1),
            })

    df_hot_summary = pd.DataFrame(summary_list)
    csv_hot_summary = OUTPUTS_DIR / "hot_protection_summary.csv"
    df_hot_summary.to_csv(csv_hot_summary, index=False)
    print(f"Saved Addition D hot protection summary -> {csv_hot_summary}")


# ===========================================================================
# C3 & Addition F: Side-by-Side Comparison Plots
# ===========================================================================

PLOT_SCENARIOS = [
    "demo_hot_attack",
    "pump_off",
    "coordinated",
    "abusive_command",
    "drift_temp_a",
    "pump_degradation"
]

MODE_COLORS = {
    "NOMINAL": "#2ca02c",    # green
    "DEGRADED": "#ff7f0e",   # orange
    "LIMP_HOME": "#1f77b4",  # blue
    "SHUTDOWN": "#d62728",   # red
}


def generate_c3_plots(seed: int = 42) -> None:
    print("\n" + "=" * 78)
    print(f"C3 / Addition F: Generating Side-by-Side Plots (seed {seed})")
    print("=" * 78)

    for sc_name in PLOT_SCENARIOS:
        sc = build_scenario(sc_name, seed=seed)

        # 1. Unprotected
        p_unpr = Plant(perturb=False, sensor_seed=seed, **dict(sc.plant_kwargs))
        c_unpr = ChargerBMS()
        df_unpr = run_session(p_unpr, c_unpr, scenario=sc, gate=None, guard=None, seed=seed)

        # 2. Protected
        p_prot = Plant(perturb=False, sensor_seed=seed, **dict(sc.plant_kwargs))
        c_prot = ChargerBMS()
        gate = SafetyGate()
        guard = PhysicsGuard()
        gate.set_guard(guard)
        df_prot = run_session(p_prot, c_prot, scenario=sc, gate=gate, guard=guard, seed=seed)

        # Clean samples: no t=0 pre-step, and drop stop-latch zeros if latched
        df_unpr = clean_df(df_unpr, getattr(c_unpr, "latched_stop", False))
        df_prot = clean_df(df_prot, getattr(c_prot, "latched_stop", False))

        peak_u = float(df_unpr["true_battery_temp"].max())
        peak_p = float(df_prot["true_battery_temp"].max())
        time_u = float(df_unpr["time_s"].iloc[-1])
        time_p = float(df_prot["time_s"].iloc[-1])
        soc_u = float(df_unpr["true_soc"].iloc[-1])
        soc_p = float(df_prot["true_soc"].iloc[-1])

        # Matching normal-condition baseline
        amb = sc.ambient_c if sc.ambient_c is not None else params.T_AMBIENT_NOMINAL_C
        inlet = sc.inlet_c if sc.inlet_c is not None else params.T_COOLANT_INLET_NOMINAL_C
        init_t = sc.plant_kwargs.get("initial_temp_c", amb)
        p_norm = Plant(perturb=False, sensor_seed=seed, initial_soc=params.SOC_INITIAL, initial_temp_c=init_t)
        c_norm = ChargerBMS()
        sc_norm = Scenario(name="normal", label_class="normal", label_type="normal",
                           ambient_c=amb, inlet_c=inlet, plant_kwargs={"initial_temp_c": init_t})
        df_norm = run_session(p_norm, c_norm, scenario=sc_norm, gate=None, guard=None, seed=seed, ambient_c=amb, inlet_c=inlet)
        df_norm_clean = clean_df(df_norm, getattr(c_norm, "latched_stop", False))
        time_norm = float(df_norm_clean["time_s"].iloc[-1])

        t_unpr = df_unpr["time_s"].values
        t_prot = df_prot["time_s"].values
        max_t = max(t_unpr[-1], t_prot[-1])

        fig, axes = plt.subplots(4, 2, figsize=(15, 12), gridspec_kw={"height_ratios": [3, 2.5, 2.5, 1]})
        fig.suptitle(f"PhysicsGuard Phase 5.2: {sc_name} (Unprotected vs Protected) | Normal Baseline Charge Time: {time_norm:.0f}s", fontsize=13, fontweight="bold")

        # ------------------------------------------------------------------
        # Row 1: True Temperature with single-line summary under title
        # ------------------------------------------------------------------
        # Left (Unprotected)
        ax = axes[0, 0]
        ax.plot(t_unpr, df_unpr["true_battery_temp"], color="#d62728", lw=2, label="True Battery Temp")
        ax.axhline(45.0, color="#ff7f0e", ls="--", lw=1.5, label="Warning (45°C)")
        ax.axhline(50.0, color="#7f0000", ls=":", lw=1.5, label="Trip (50°C)")
        ax.set_ylabel("Temp [°C]")
        ax.set_title(
            f"UNPROTECTED (No Gate)\nPeak Temp: {peak_u:.1f}°C | Charge Time: {time_u:.0f}s | Final SOC: {soc_u:.3f}",
            fontsize=10
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", framealpha=0.8)
        ax.set_xlim(0, max_t)

        # Right (Protected)
        ax = axes[0, 1]
        ax.plot(t_prot, df_prot["true_battery_temp"], color="#2ca02c", lw=2, label="True Battery Temp")
        ax.axhline(45.0, color="#ff7f0e", ls="--", lw=1.5, label="Warning (45°C)")
        ax.axhline(50.0, color="#7f0000", ls=":", lw=1.5, label="Trip (50°C)")
        ax.set_ylabel("Temp [°C]")
        ax.set_title(
            f"PROTECTED (PhysicsGuard + SafetyGate)\nPeak Temp: {peak_p:.1f}°C | Charge Time: {time_p:.0f}s (Baseline: {time_norm:.0f}s) | Final SOC: {soc_p:.3f}",
            fontsize=10
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", framealpha=0.8)
        ax.set_xlim(0, max_t)

        # ------------------------------------------------------------------
        # Row 2: Sensors & Trusted Temp
        # ------------------------------------------------------------------
        # Left (Unprotected)
        ax = axes[1, 0]
        ax.plot(t_unpr, df_unpr["temp_cell_a"], label="Sensor Temp A", alpha=0.8)
        ax.plot(t_unpr, df_unpr["temp_cell_b"], label="Sensor Temp B", alpha=0.8)
        ax.plot(t_unpr, df_unpr["ambient_temp"], label="Ambient", alpha=0.5, ls=":")
        ax.set_ylabel("Sensors [°C]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", framealpha=0.8)
        ax.set_xlim(0, max_t)

        # Right (Protected)
        ax = axes[1, 1]
        ax.plot(t_prot, df_prot["temp_cell_a"], label="Raw Temp A", alpha=0.5)
        ax.plot(t_prot, df_prot["temp_cell_b"], label="Raw Temp B", alpha=0.5)
        ax.plot(t_prot, df_prot["trusted_temp"], color="#1f77b4", lw=2, label="Gate Trusted Temp")
        ax.set_ylabel("Sensors [°C]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", framealpha=0.8)
        ax.set_xlim(0, max_t)

        # ------------------------------------------------------------------
        # Row 3: Applied Current & Pump Flow
        # ------------------------------------------------------------------
        # Left (Unprotected)
        ax_curr = axes[2, 0]
        ax_pump = ax_curr.twinx()
        l1 = ax_curr.plot(t_unpr, df_unpr["applied_current_a"], color="#1f77b4", label="Current [A]")
        l2 = ax_pump.plot(t_unpr, df_unpr["applied_pump_lpm"], color="#2ca02c", ls="--", label="Pump [L/min]")
        ax_curr.set_ylabel("Current [A]", color="#1f77b4")
        ax_pump.set_ylabel("Pump [L/min]", color="#2ca02c")
        ax_curr.grid(True, alpha=0.3)
        ax_curr.legend(l1 + l2, ["Current [A]", "Pump [L/min]"], loc="upper right", framealpha=0.8)
        ax_curr.set_xlim(0, max_t)

        # Right (Protected)
        ax_curr = axes[2, 1]
        ax_pump = ax_curr.twinx()
        l1 = ax_curr.plot(t_prot, df_prot["applied_current_a"], color="#1f77b4", label="Current [A]")
        l2 = ax_pump.plot(t_prot, df_prot["applied_pump_lpm"], color="#2ca02c", ls="--", label="Pump [L/min]")
        ax_curr.set_ylabel("Current [A]", color="#1f77b4")
        ax_pump.set_ylabel("Pump [L/min]", color="#2ca02c")
        ax_curr.grid(True, alpha=0.3)
        ax_curr.legend(l1 + l2, ["Current [A]", "Pump [L/min]"], loc="upper right", framealpha=0.8)
        ax_curr.set_xlim(0, max_t)

        # ------------------------------------------------------------------
        # Row 4: Mode Band
        # ------------------------------------------------------------------
        # Left (Unprotected - always N/A)
        ax = axes[3, 0]
        ax.fill_between(t_unpr, 0, 1, color="#2ca02c", alpha=0.6, label="NOMINAL (Unprotected)")
        ax.set_yticks([])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Mode")
        ax.legend(loc="center", framealpha=0.8)
        ax.set_xlim(0, max_t)

        # Right (Protected - colored gate_mode band)
        ax = axes[3, 1]
        modes = df_prot["gate_mode"].values
        curr_mode = modes[0]
        start_t = t_prot[0]
        for idx in range(1, len(modes)):
            if modes[idx] != curr_mode:
                ax.axvspan(start_t, t_prot[idx], color=MODE_COLORS.get(curr_mode, "#2ca02c"), alpha=0.6)
                start_t = t_prot[idx]
                curr_mode = modes[idx]
        ax.axvspan(start_t, t_prot[-1], color=MODE_COLORS.get(curr_mode, "#2ca02c"), alpha=0.6)

        patches = [mpatches.Patch(color=c, label=m) for m, c in MODE_COLORS.items()]
        ax.set_yticks([])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Mode")
        ax.legend(handles=patches, loc="upper right", ncol=4, fontsize=8, framealpha=0.8)
        ax.set_xlim(0, max_t)

        plt.tight_layout()
        save_path = OUTPUTS_DIR / f"compare_{sc_name}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved plot -> {save_path}")

    print("\nAll comparison plots generated successfully with matching x-axes.")


# ===========================================================================
# Main Execution
# ===========================================================================

if __name__ == "__main__":
    # C1: Standard 17 scenarios evaluation
    df_c1 = run_c1_evaluation(seed=42)

    # C2 / Task 3: 200 held-out perturbed sessions
    df_c2 = run_c2_heldout()

    # Task 4 & Addition D: Hot day attacks evaluation
    run_task4_hot_attacks()

    # C3 / Addition F: Plotting
    generate_c3_plots(seed=42)

    print("\n" + "=" * 78)
    print("PHASE 5.2 EVALUATION PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 78)
