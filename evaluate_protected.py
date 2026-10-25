"""
PhysicsGuard Phase 5 - Closed-Loop Evaluation Script
===================================================
Implements Part C:
  C1. Evaluates all 17 scenarios Unprotected vs Protected (seed 42).
      Saves outputs/protected_vs_unprotected.csv.
  C2. Held-out random batch on seeds 300-399 (100 sessions):
      40 non-attacks (normal, aged, hot, pump_degradation) + 60 attacks with random start_s in [200, 800]s.
      Reports false-intervention rate, detection latencies, peak temperature safety, charge time penalty.
      Saves outputs/heldout_report.csv.
  C3. Side-by-side comparison plots (unprotected left, protected right) for:
      demo_hot_attack, pump_off, coordinated, abusive_command, drift_temp_a, pump_degradation.
      Saves outputs/compare_<name>.png. (Excludes t=0 sample).
"""

import sys
import os
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

ALL_17_SCENARIOS = [
    "normal", "pump_degradation", "aged_cell", "hot_day",
    "dead_sensor_nan_temp_a", "dead_sensor_flatline_temp_a",
    "freeze_temp_a", "replay_temp_a", "drift_temp_a", "spike_temp_a",
    "freeze_current", "drift_current", "drift_voltage",
    "abusive_command", "pump_off", "coordinated", "demo_hot_attack"
]

OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# C1: All 17 Scenarios Unprotected vs Protected
# ===========================================================================

def run_c1_evaluation(seed: int = 42) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print(f"C1: Running all 17 scenarios Unprotected vs Protected (seed {seed})")
    print("=" * 78)

    rows = []

    for name in ALL_17_SCENARIOS:
        sc = build_scenario(name, seed=seed)

        # 1. Unprotected run
        plant_unpr = Plant(perturb=False, sensor_seed=seed, **dict(sc.plant_kwargs))
        ctrl_unpr = ChargerBMS()
        df_unpr = run_session(plant_unpr, ctrl_unpr, scenario=sc, gate=None, guard=None, seed=seed)

        # 2. Protected run
        plant_prot = Plant(perturb=False, sensor_seed=seed, **dict(sc.plant_kwargs))
        ctrl_prot = ChargerBMS()
        gate = SafetyGate()
        guard = PhysicsGuard()
        gate.set_guard(guard)
        df_prot = run_session(plant_prot, ctrl_prot, scenario=sc, gate=gate, guard=guard, seed=seed)

        # Metrics extraction
        peak_t_unpr = float(df_unpr["true_battery_temp"].max())
        peak_t_prot = float(df_prot["true_battery_temp"].max())

        s45_unpr = int((df_unpr["true_battery_temp"] > 45.0).sum())
        s45_prot = int((df_prot["true_battery_temp"] > 45.0).sum())

        s50_unpr = int((df_unpr["true_battery_temp"] > 50.0).sum())
        s50_prot = int((df_prot["true_battery_temp"] > 50.0).sum())

        soc_unpr = float(df_unpr["true_soc"].iloc[-1])
        soc_prot = float(df_prot["true_soc"].iloc[-1])

        dur_unpr = float(df_unpr["time_s"].iloc[-1])
        dur_prot = float(df_prot["time_s"].iloc[-1])

        interv_mask = df_prot["intervened"].values
        interv_times = df_prot.loc[interv_mask, "time_s"].values
        t_first_interv = float(interv_times[0]) if len(interv_times) > 0 else None
        num_interv = int(interv_mask.sum())

        final_mode = str(df_prot["gate_mode"].iloc[-1])

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
            "time_to_first_intervention": round(t_first_interv, 1) if t_first_interv is not None else "N/A",
            "num_intervened_seconds": num_interv,
            "final_mode": final_mode,
        }
        rows.append(row)

        print(
            f"  {name:27s} | Unpr Peak: {peak_t_unpr:5.1f}C (>45:{s45_unpr:3d}s) | "
            f"Prot Peak: {peak_t_prot:5.1f}C (>45:{s45_prot:3d}s) | "
            f"Interv: {num_interv:4d}s | Mode: {final_mode}"
        )

    df_res = pd.DataFrame(rows)
    csv_path = OUTPUTS_DIR / "protected_vs_unprotected.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved C1 results -> {csv_path}")
    return df_res


# ===========================================================================
# C2: Held-Out Batch Evaluation (Seeds 300 to 399)
# ===========================================================================

def run_c2_heldout() -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("C2: Held-Out Evaluation on Seeds 300 to 399 (100 sessions)")
    print("=" * 78)

    rng = np.random.default_rng(2026)

    # 40 non-attacks across seeds 300-339
    non_attack_types = ["normal", "aged_cell", "hot_day", "pump_degradation"]
    # 60 attacks across seeds 340-399
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
    total_non_attacks = 0
    protected_below_45_count = 0
    charge_penalties: List[float] = []

    for seed in range(300, 400):
        is_attack_session = (seed >= 340)

        if not is_attack_session:
            # Non-attack category (seeds 300-339)
            cat = non_attack_types[(seed - 300) % len(non_attack_types)]
            if cat == "normal":
                init_temp = float(rng.uniform(20.0, 30.0))
                sc = Scenario(
                    name="normal", label_class="normal", label_type="normal",
                    ambient_c=init_temp, inlet_c=init_temp, plant_kwargs={"initial_temp_c": init_temp}
                )
            elif cat == "aged_cell":
                af = float(rng.uniform(1.10, 1.30))
                sc = aged_cell(age_factor=af)
            elif cat == "hot_day":
                amb = float(rng.uniform(32.0, 40.0))
                sc = hot_day(ambient_c=amb, inlet_c=amb)
            else:  # pump_degradation
                flow_derate = float(rng.uniform(0.20, 0.50))
                sc = pump_degradation(fraction=flow_derate)
            attack_type = None
            start_s = None
        else:
            # Attack category (seeds 340-399)
            atk_name = attack_types[(seed - 340) % len(attack_types)]
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
                start_s = 60.0

        # Run Unprotected
        p_unpr = Plant(perturb=False, sensor_seed=seed, **dict(sc.plant_kwargs))
        c_unpr = ChargerBMS()
        df_unpr = run_session(p_unpr, c_unpr, scenario=sc, gate=None, guard=None, seed=seed)

        # Run Protected
        p_prot = Plant(perturb=False, sensor_seed=seed, **dict(sc.plant_kwargs))
        c_prot = ChargerBMS()
        gate = SafetyGate()
        guard = PhysicsGuard()
        gate.set_guard(guard)
        df_prot = run_session(p_prot, c_prot, scenario=sc, gate=gate, guard=guard, seed=seed)

        peak_t_unpr = float(df_unpr["true_battery_temp"].max())
        peak_t_prot = float(df_prot["true_battery_temp"].max())
        dur_unpr = float(df_unpr["time_s"].iloc[-1])
        dur_prot = float(df_prot["time_s"].iloc[-1])
        num_interv = int(df_prot["intervened"].sum())
        intervened_bool = (num_interv > 0)

        if peak_t_prot <= 45.0:
            protected_below_45_count += 1

        if dur_unpr > 0:
            penalty = (dur_prot - dur_unpr) / dur_unpr
            charge_penalties.append(penalty)

        if not is_attack_session:
            total_non_attacks += 1
            if cat in ("normal", "aged_cell", "hot_day") and intervened_bool:
                non_attack_interventions += 1
        else:
            total_by_type[attack_type] += 1
            # Check detection latency after start_s
            after_start = df_prot[(df_prot["time_s"] >= start_s) & (df_prot["alarm_level"].isin(["warning", "critical"]))]
            if len(after_start) > 0:
                det_t = float(after_start["time_s"].iloc[0])
                latency = max(0.0, det_t - start_s)
                latencies_by_type[attack_type].append(latency)
                detected_by_type[attack_type] += 1

        session_records.append({
            "seed": seed,
            "category": "attack" if is_attack_session else "non-attack",
            "type": attack_type if is_attack_session else cat,
            "start_s": start_s,
            "peak_temp_unpr": peak_t_unpr,
            "peak_temp_prot": peak_t_prot,
            "dur_unpr": dur_unpr,
            "dur_prot": dur_prot,
            "intervened_seconds": num_interv,
            "final_mode": df_prot["gate_mode"].iloc[-1]
        })

    # Summary Statistics
    false_interv_rate = (non_attack_interventions / 30.0) if total_non_attacks >= 30 else 0.0
    safe_temp_fraction = protected_below_45_count / 100.0
    mean_charge_penalty = float(np.mean(charge_penalties))

    summary_rows = []
    print("\nHeld-Out Batch Performance Summary:")
    print("-" * 65)
    print(f"False-Intervention Rate on (normal/aged/hot): {false_interv_rate*100:.1f}%")
    print(f"Protected Sessions Peak Temp <= 45 C:        {safe_temp_fraction*100:.1f}% (target 100%)")
    print(f"Mean Charge Time Penalty across Batch:        {mean_charge_penalty*100:.2f}%")
    print("-" * 65)
    print(f"{'Attack Type':<26} {'Det Rate':>9} {'Med Lat [s]':>12} {'95th Lat [s]':>13}")
    print("-" * 65)

    for atk in attack_types:
        n_tot = total_by_type[atk]
        n_det = detected_by_type[atk]
        det_rate = (n_det / n_tot) if n_tot > 0 else 1.0
        lats = latencies_by_type[atk]
        med_lat = float(np.median(lats)) if lats else 0.0
        p95_lat = float(np.percentile(lats, 95)) if lats else 0.0

        summary_rows.append({
            "metric_type": "attack_detection",
            "name": atk,
            "detection_rate": round(det_rate, 3),
            "median_latency_s": round(med_lat, 1),
            "p95_latency_s": round(p95_lat, 1),
            "sample_count": n_tot
        })
        print(f"{atk:<26} {det_rate*100:>8.1f}% {med_lat:>11.1f}s {p95_lat:>12.1f}s")

    summary_rows.append({
        "metric_type": "aggregate",
        "name": "false_intervention_rate",
        "detection_rate": round(false_interv_rate, 4),
        "median_latency_s": 0.0,
        "p95_latency_s": 0.0,
        "sample_count": 30
    })
    summary_rows.append({
        "metric_type": "aggregate",
        "name": "fraction_peak_temp_below_45C",
        "detection_rate": round(safe_temp_fraction, 4),
        "median_latency_s": 0.0,
        "p95_latency_s": 0.0,
        "sample_count": 100
    })
    summary_rows.append({
        "metric_type": "aggregate",
        "name": "mean_charge_time_penalty",
        "detection_rate": round(mean_charge_penalty, 4),
        "median_latency_s": 0.0,
        "p95_latency_s": 0.0,
        "sample_count": 100
    })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = OUTPUTS_DIR / "heldout_report.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved C2 held-out report -> {csv_path}")
    return df_summary


# ===========================================================================
# C3: Side-by-Side Comparison Plots
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
    print(f"C3: Generating Side-by-Side Plots (seed {seed})")
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

        # Filter out t=0 sample
        df_unpr = df_unpr[df_unpr["time_s"] > 0].copy()
        df_prot = df_prot[df_prot["time_s"] > 0].copy()

        fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex="col", gridspec_kw={"height_ratios": [3, 2.5, 2.5, 1]})
        fig.suptitle(f"PhysicsGuard Phase 5: {sc_name} (Unprotected vs Protected)", fontsize=14, fontweight="bold")

        t_unpr = df_unpr["time_s"].values
        t_prot = df_prot["time_s"].values

        # ------------------------------------------------------------------
        # Row 1: True Temperature
        # ------------------------------------------------------------------
        # Left (Unprotected)
        ax = axes[0, 0]
        ax.plot(t_unpr, df_unpr["true_battery_temp"], color="#d62728", lw=2, label="True Battery Temp")
        ax.axhline(45.0, color="#ff7f0e", ls="--", lw=1.5, label="Warning (45°C)")
        ax.axhline(50.0, color="#7f0000", ls=":", lw=1.5, label="Trip (50°C)")
        ax.set_ylabel("Temp [°C]")
        ax.set_title("UNPROTECTED (No Gate)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", framealpha=0.8)

        # Right (Protected)
        ax = axes[0, 1]
        ax.plot(t_prot, df_prot["true_battery_temp"], color="#2ca02c", lw=2, label="True Battery Temp")
        ax.axhline(45.0, color="#ff7f0e", ls="--", lw=1.5, label="Warning (45°C)")
        ax.axhline(50.0, color="#7f0000", ls=":", lw=1.5, label="Trip (50°C)")
        ax.set_ylabel("Temp [°C]")
        ax.set_title("PROTECTED (PhysicsGuard + SafetyGate)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", framealpha=0.8)

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

        # Right (Protected)
        ax = axes[1, 1]
        ax.plot(t_prot, df_prot["temp_cell_a"], label="Raw Temp A", alpha=0.5)
        ax.plot(t_prot, df_prot["temp_cell_b"], label="Raw Temp B", alpha=0.5)
        ax.plot(t_prot, df_prot["trusted_temp"], color="#1f77b4", lw=2, label="Gate Trusted Temp")
        ax.set_ylabel("Sensors [°C]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", framealpha=0.8)

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

        # Right (Protected)
        ax_curr = axes[2, 1]
        ax_pump = ax_curr.twinx()
        l1 = ax_curr.plot(t_prot, df_prot["applied_current_a"], color="#1f77b4", label="Current [A]")
        l2 = ax_pump.plot(t_prot, df_prot["applied_pump_lpm"], color="#2ca02c", ls="--", label="Pump [L/min]")
        ax_curr.set_ylabel("Current [A]", color="#1f77b4")
        ax_pump.set_ylabel("Pump [L/min]", color="#2ca02c")
        ax_curr.grid(True, alpha=0.3)
        ax_curr.legend(l1 + l2, ["Current [A]", "Pump [L/min]"], loc="upper right", framealpha=0.8)

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

        # Right (Protected - colored gate_mode band)
        ax = axes[3, 1]
        modes = df_prot["gate_mode"].values
        # Create colored spans for mode transitions
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

        plt.tight_layout()
        save_path = OUTPUTS_DIR / f"compare_{sc_name}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved plot -> {save_path}")

    print("\nAll comparison plots generated successfully.")


# ===========================================================================
# Main Execution
# ===========================================================================

if __name__ == "__main__":
    # C1
    df_c1 = run_c1_evaluation(seed=42)

    # C2
    df_c2 = run_c2_heldout()

    # C3
    generate_c3_plots(seed=42)

    print("\n" + "=" * 78)
    print("EVALUATION COMPLETED SUCCESSFULLY.")
    print("=" * 78)
