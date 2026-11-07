"""
PhysicsGuard - Normal Fast-Charging Session Runner
==================================================
Runs a complete closed-loop baseline DC fast-charging session from SOC 20% to 80%
under normal nominal conditions using ChargerBMS and run_session:
  - Ambient temperature: 25.0 C
  - Coolant inlet temperature: 25.0 C
  - Closed-loop pump command: 10.0 L/min during charge, 3.0 L/min idle
  - Rated charging current: 4.0 A / cell (128.0 A pack) with thermal derating

Outputs Generated (saved to outputs/):
  1. signals.csv: CSV file containing time, 10 measured sensor channels,
     and ground truth states.
  2. plot1_current_voltage.png: Pack current and terminal voltage vs time.
  3. plot2_soc.png: State of Charge (SOC) progression vs time.
  4. plot3_temperatures.png: Temperature sensor channels (A, B, coolant in/out, true temp).
  5. plot4_power.png: Charger power command delivered to the pack over time.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local package imports succeed whether run from workspace root or package directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

try:
    from physicsguard import params
    from physicsguard.plant import Plant
    from physicsguard.twin import Twin
    from physicsguard.controller import ChargerBMS
    from physicsguard.simulate import run_session
except ImportError:
    import params
    from plant import Plant
    from twin import Twin
    from controller import ChargerBMS
    from simulate import run_session


def run_normal_charge(save_dir: Path | None = None, seed: int = 42) -> pd.DataFrame:
    """
    Simulate one normal closed-loop fast-charging session until 80% SOC is reached.
    
    Parameters
    ----------
    save_dir : Path, optional
        Directory where CSV and PNG artifacts will be saved.
        Defaults to project outputs/ directory.
    seed : int, default 42
        Random seed for repeatable sensor noise generation.
        
    Returns
    -------
    pd.DataFrame
        Logged simulation signals table.
    """
    if save_dir is None:
        save_dir = ROOT_DIR / "outputs"

    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PhysicsGuard - Starting Normal Closed-Loop EV Fast-Charging Simulation")
    print("=" * 70)
    print(f"Simulation Seed     : {seed}")
    print(f"Pack Architecture   : {params.PACK_N_SERIES}s{params.PACK_N_PARALLEL}p ({params.PACK_TOTAL_CELLS} cells)")
    print(f"Initial SOC         : {params.SOC_INITIAL * 100:.1f}%")
    print(f"Target SOC          : {params.SOC_TARGET * 100:.1f}%")
    print(f"Nominal Pack Current: {params.PACK_RATED_CHARGE_CURRENT_A:.1f} A (4.0 A/cell)")
    print(f"Coolant Pump Command: {params.PUMP_NOMINAL_FLOW_LPM:.1f} L/min (charging) / 3.0 L/min (idle)")
    print(f"Coolant Inlet Temp  : {params.T_COOLANT_INLET_NOMINAL_C:.1f} C")
    print(f"Ambient Temperature : {params.T_AMBIENT_NOMINAL_C:.1f} C")
    print("-" * 70)

    # Initialize physical plant (unperturbed nominal baseline)
    plant = Plant(
        perturb=False,
        initial_soc=params.SOC_INITIAL,
        initial_temp_c=params.T_AMBIENT_NOMINAL_C,
        sensor_seed=seed
    )

    # Initialize closed-loop BMS controller
    controller = ChargerBMS(
        nominal_current_a=params.PACK_RATED_CHARGE_CURRENT_A,
        soc_cutoff=params.SOC_TARGET,
        temp_derate_start_c=40.0,
        temp_derate_stop_c=50.0,
        pump_charging_lpm=params.PUMP_NOMINAL_FLOW_LPM,
        pump_idle_lpm=3.0
    )

    # Run closed-loop session via simulate.run_session
    df = run_session(
        plant=plant,
        controller=controller,
        seed=seed,
        until_soc=params.SOC_TARGET,
        ambient_c=params.T_AMBIENT_NOMINAL_C,
        inlet_c=params.T_COOLANT_INLET_NOMINAL_C,
        dt=params.SIMULATION_DT_S
    )

    # Calculate summary metrics
    duration_s = df["time"].iloc[-1]
    duration_min = duration_s / 60.0
    final_soc = df["soc_estimate"].iloc[-1] * 100.0
    peak_temp_c = df["true_battery_temp"].max()
    final_temp_c = df["true_battery_temp"].iloc[-1]
    avg_delta_t_cool = (df["coolant_temp_out"] - df["coolant_temp_in"]).tail(300).mean()

    print(f"Charge Completed in : {duration_s:.0f} seconds ({duration_min:.2f} minutes)")
    print(f"Final SOC           : {final_soc:.2f}%")
    print(f"Peak Battery Temp   : {peak_temp_c:.2f} C (trip limit is {params.CELL_TEMP_TRIP_C} C)")
    print(f"End Battery Temp    : {final_temp_c:.2f} C")
    print(f"Coolant Temp Rise   : {avg_delta_t_cool:.2f} K (outlet - inlet)")
    print("-" * 70)

    # --------------------------------------------------------------------------
    # Save CSV File into outputs/
    # --------------------------------------------------------------------------
    csv_path = save_dir / "signals.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset -> {csv_path}")

    # --------------------------------------------------------------------------
    # Generate 4 Engineering Plots (Clean, High-Contrast Style)
    # --------------------------------------------------------------------------
    time_min = df["time"] / 60.0

    # PLOT 1: Pack Current & Pack Terminal Voltage
    fig1, ax1_curr = plt.subplots(figsize=(8, 5))
    color_curr = "#1f77b4"
    color_volt = "#d62728"

    ax1_curr.set_xlabel("Time (minutes)", fontsize=11, fontweight="bold")
    ax1_curr.set_ylabel("Pack Current (A)", color=color_curr, fontsize=11, fontweight="bold")
    ax1_curr.plot(time_min, df["pack_current"], color=color_curr, label="Pack Current (A)", linewidth=1.5)
    ax1_curr.tick_params(axis="y", labelcolor=color_curr)
    ax1_curr.grid(True, linestyle="--", alpha=0.5)

    ax1_volt = ax1_curr.twinx()
    ax1_volt.set_ylabel("Pack Voltage (V)", color=color_volt, fontsize=11, fontweight="bold")
    ax1_volt.plot(time_min, df["pack_voltage"], color=color_volt, label="Pack Voltage (V)", linewidth=1.5)
    ax1_volt.tick_params(axis="y", labelcolor=color_volt)

    plt.title("PhysicsGuard: Fast-Charge Current & Terminal Voltage", fontsize=12, fontweight="bold", pad=12)
    fig1.tight_layout()
    plot1_path = save_dir / "plot1_current_voltage.png"
    fig1.savefig(plot1_path, dpi=300)
    plt.close(fig1)
    print(f"Saved plot 1 -> {plot1_path}")

    # PLOT 2: State of Charge (SOC)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(time_min, df["soc_estimate"] * 100.0, color="#2ca02c", linewidth=2.0, label="SOC Estimate (%)")
    ax2.axhline(params.SOC_TARGET * 100.0, color="#d62728", linestyle="--", label=f"Target ({params.SOC_TARGET*100:.0f}%)")
    ax2.set_xlabel("Time (minutes)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("State of Charge (%)", fontsize=11, fontweight="bold")
    ax2.set_title("PhysicsGuard: Battery Pack State of Charge (SOC)", fontsize=12, fontweight="bold", pad=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")
    fig2.tight_layout()
    plot2_path = save_dir / "plot2_soc.png"
    fig2.savefig(plot2_path, dpi=300)
    plt.close(fig2)
    print(f"Saved plot 2 -> {plot2_path}")

    # PLOT 3: Temperatures (Sensor A, Sensor B, Coolant In, Coolant Out, True Temp)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(time_min, df["temp_cell_a"], label="Temp Sensor A (Pack)", color="#e377c2", alpha=0.7, linewidth=1.2)
    ax3.plot(time_min, df["temp_cell_b"], label="Temp Sensor B (Pack Offset)", color="#8c564b", alpha=0.7, linewidth=1.2)
    ax3.plot(time_min, df["coolant_temp_in"], label="Coolant Inlet (25 C)", color="#17becf", linestyle=":", linewidth=1.5)
    ax3.plot(time_min, df["coolant_temp_out"], label="Coolant Outlet", color="#ff7f0e", linewidth=1.5)
    ax3.plot(time_min, df["true_battery_temp"], label="True Battery Temp", color="#d62728", linewidth=2.0)
    ax3.axhline(params.CELL_TEMP_WARN_C, color="#ffbb78", linestyle="--", label=f"Warn Limit ({params.CELL_TEMP_WARN_C} C)")
    ax3.axhline(params.CELL_TEMP_TRIP_C, color="#9467bd", linestyle="--", label=f"Trip Limit ({params.CELL_TEMP_TRIP_C} C)")
    ax3.set_xlabel("Time (minutes)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Temperature (deg C)", fontsize=11, fontweight="bold")
    ax3.set_title("PhysicsGuard: Thermal Dynamics & Liquid Cooling Response", fontsize=12, fontweight="bold", pad=12)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", fontsize=9)
    fig3.tight_layout()
    plot3_path = save_dir / "plot3_temperatures.png"
    fig3.savefig(plot3_path, dpi=300)
    plt.close(fig3)
    print(f"Saved plot 3 -> {plot3_path}")

    # PLOT 4: Charger Command Power (kW)
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(time_min, df["charger_power_cmd_kw"], color="#9467bd", linewidth=2.0, label="Charger Power (kW)")
    ax4.set_xlabel("Time (minutes)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Power Command (kW)", fontsize=11, fontweight="bold")
    ax4.set_title("PhysicsGuard: Charger Power Command Delivered", fontsize=12, fontweight="bold", pad=12)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right")
    fig4.tight_layout()
    plot4_path = save_dir / "plot4_power.png"
    fig4.savefig(plot4_path, dpi=300)
    plt.close(fig4)
    print(f"Saved plot 4 -> {plot4_path}")

    print("=" * 70)
    print("Normal fast-charge simulation completed successfully.")
    print("=" * 70)

    return df


if __name__ == "__main__":
    run_normal_charge()
