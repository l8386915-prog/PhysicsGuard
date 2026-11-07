"""
PhysicsGuard - Scenario Suite Runner
====================================
Runs all physical fault, cyberattack, and baseline normal scenarios.

Generates and saves:
  1. outputs/scenarios/<name>.csv: Complete telemetry log for each scenario.
  2. outputs/scenario_summary.csv: Overview metrics for all scenarios.
  3. outputs/scenarios/<name>.png: Two-panel plot for each scenario.
  4. outputs/scenario_overview.png: Subplot comparison of true battery temperatures by class.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local package imports succeed
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

try:
    from physicsguard import params
    from physicsguard.plant import Plant
    from physicsguard.controller import ChargerBMS
    from physicsguard.simulate import run_session
    from physicsguard.scenarios import build_scenario, Scenario
except ImportError:
    import params
    from plant import Plant
    from controller import ChargerBMS
    from simulate import run_session
    from scenarios import build_scenario, Scenario


SCENARIO_NAMES = [
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


def plot_scenario(sc_name: str, df: pd.DataFrame, sc: Scenario, save_path: Path) -> None:
    """
    Save two-panel plot for a scenario:
      Top: true battery temp, sensor A, sensor B, coolant outlet.
      Bottom: applied current and applied pump.
      Mark start_s with a vertical line.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    t = df["time"].values

    # Top panel: Temperatures
    ax_top.plot(t, df["true_battery_temp"].values, label="True Battery Temp", color="black", lw=2.0)
    ax_top.plot(t, df["temp_cell_a"].values, label="Sensor A (temp_cell_a)", color="#1f77b4", lw=1.2, alpha=0.85)
    ax_top.plot(t, df["temp_cell_b"].values, label="Sensor B (temp_cell_b)", color="#ff7f0e", lw=1.2, alpha=0.85)
    ax_top.plot(t, df["coolant_temp_out"].values, label="Coolant Outlet", color="#2ca02c", lw=1.2, alpha=0.85)

    ax_top.set_ylabel("Temperature [°C]")
    ax_top.set_title(f"Scenario: {sc_name} (Class: {sc.label_class}, Type: {sc.label_type})", fontweight="bold")
    ax_top.grid(True, linestyle=":", alpha=0.6)

    # Bottom panel: Actuation Commands
    ax_bot.plot(t, df["applied_current_a"].values, label="Applied Current [A]", color="#d62728", lw=1.8)
    ax_bot.set_ylabel("Current [A]", color="#d62728")
    ax_bot.tick_params(axis="y", labelcolor="#d62728")
    ax_bot.grid(True, linestyle=":", alpha=0.6)

    # Twin axis for pump flow command
    ax_pump = ax_bot.twinx()
    ax_pump.plot(t, df["applied_pump_lpm"].values, label="Applied Pump [L/min]", color="#17becf", lw=1.8, linestyle="--")
    ax_pump.set_ylabel("Pump Flow [L/min]", color="#17becf")
    ax_pump.tick_params(axis="y", labelcolor="#17becf")

    # Mark start_s with a vertical line
    if sc.start_s is not None and sc.start_s > 0:
        ax_top.axvline(x=sc.start_s, color="red", linestyle="--", lw=1.5, alpha=0.8, label=f"start_s ({sc.start_s:.0f}s)")
        ax_bot.axvline(x=sc.start_s, color="red", linestyle="--", lw=1.5, alpha=0.8)

    ax_top.legend(loc="upper left", framealpha=0.9, fontsize=9)
    # Combine bottom legends
    lines_curr, labels_curr = ax_bot.get_legend_handles_labels()
    lines_pump, labels_pump = ax_pump.get_legend_handles_labels()
    ax_bot.legend(lines_curr + lines_pump, labels_curr + labels_pump, loc="upper right", framealpha=0.9, fontsize=9)

    ax_bot.set_xlabel("Elapsed Time [s]")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_overview(scenario_dfs: Dict[str, Tuple[pd.DataFrame, Scenario]], save_path: Path) -> None:
    """
    Save scenario overview plot: true battery temperature of all scenarios,
    one subplot per class ("normal", "fault", "attack").
    """
    classes = ["normal", "fault", "attack"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax, cls_name in zip(axes, classes):
        ax.set_title(f"Class: {cls_name.upper()}", fontweight="bold", fontsize=12)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("True Battery Temp [°C]" if cls_name == "normal" else "")
        ax.grid(True, linestyle=":", alpha=0.6)

        for name, (df, sc) in scenario_dfs.items():
            if sc.label_class == cls_name:
                ax.plot(df["time"].values, df["true_battery_temp"].values, lw=1.8, label=name)

        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    fig.suptitle("PhysicsGuard - Scenario Suite Battery Temperature Profiles", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_all_scenarios(
    save_dir: Path | None = None,
    seed: int = 42,
    default_start_s: float = 300.0
) -> pd.DataFrame:
    """
    Execute all scenarios and generate all summary and visual artifacts.
    """
    if save_dir is None:
        save_dir = ROOT_DIR / "outputs"
    scenarios_dir = save_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    scenario_dfs: Dict[str, Tuple[pd.DataFrame, Scenario]] = {}

    print("=" * 80)
    print("PhysicsGuard - Executing Complete Scenario Evaluation Suite")
    print("=" * 80)

    for sc_name in SCENARIO_NAMES:
        # Build scenario (demo_hot_attack uses its own times)
        if sc_name == "demo_hot_attack":
            sc = build_scenario(sc_name)
        elif sc_name in ("aged_cell", "hot_day", "normal"):
            sc = build_scenario(sc_name)
        else:
            sc = build_scenario(sc_name, start_s=default_start_s)

        # Plant setup
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

        # Save CSV
        csv_path = scenarios_dir / f"{sc_name}.csv"
        df.to_csv(csv_path, index=False)

        # Save PNG plot
        png_path = scenarios_dir / f"{sc_name}.png"
        plot_scenario(sc_name, df, sc, png_path)

        # Compute summary statistics
        true_temps = df["true_battery_temp"].values
        peak_temp = float(np.max(true_temps))
        end_temp = float(true_temps[-1])
        secs_above_45 = int(np.sum(true_temps > 45.0))
        max_applied_curr = float(np.max(df["applied_current_a"].values))

        summary_rows.append({
            "name": sc.name,
            "class": sc.label_class,
            "type": sc.label_type,
            "peak_true_temp": round(peak_temp, 2),
            "seconds_true_temp_above_45C": secs_above_45,
            "end_true_temp": round(end_temp, 2),
            "max_applied_current": round(max_applied_curr, 2),
        })

        scenario_dfs[sc_name] = (df, sc)
        print(f"  [OK] {sc_name:<28} | Peak: {peak_temp:5.2f} C | >45C: {secs_above_45:4d}s | Max Curr: {max_applied_curr:5.1f} A")

    # Save summary CSV
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = save_dir / "scenario_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)

    # Save overview plot
    overview_png_path = save_dir / "scenario_overview.png"
    plot_overview(scenario_dfs, overview_png_path)

    print("\n" + "=" * 80)
    print("PhysicsGuard - Scenario Summary Table")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    print(f"Artifacts saved:")
    print(f"  - Scenarios CSVs & PNGs -> {scenarios_dir}")
    print(f"  - Scenario Summary CSV  -> {summary_csv_path}")
    print(f"  - Overview PNG          -> {overview_png_path}\n")

    return df_summary


if __name__ == "__main__":
    run_all_scenarios()
