"""
PhysicsGuard - Plant/Twin Mismatch Study
========================================
Quantifies state estimation error between the physical plant and the digital twin
across nominal, perturbed, aged, extreme environmental, and degraded actuator scenarios:
  (a) Nominal plant
  (b) Random +/-10% parameters (seeds 1 to 5)
  (c) Aged cell (R0 x1.3)
  (d) Hot day (ambient 35 C, coolant inlet 35 C)
  (e1) Pump degraded to 6 L/min (twin told 10 L/min)
  (e2) Pump degraded to 6 L/min (twin sees measured flow)

Generates outputs/results_twin_error.csv and prints the mismatch summary table.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

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
    from physicsguard.twin import Twin
    from physicsguard.controller import ChargerBMS
    from physicsguard.simulate import run_session
except ImportError:
    import params
    from plant import Plant
    from twin import Twin
    from controller import ChargerBMS
    from simulate import run_session


def evaluate_scenario(
    name: str,
    plant: Plant,
    controller: ChargerBMS,
    seed: int = 42,
    ambient_c: float = 25.0,
    inlet_c: float = 25.0,
    sensor_tamper=None,
    command_tamper=None,
    twin_flow_override: float | None = None
) -> Dict[str, Any]:
    """
    Simulate a scenario and compute max twin-vs-plant temperature and voltage errors.
    """
    initial_temp = plant.T_batt
    initial_soc = plant.soc
    twin = Twin(initial_soc=initial_soc, initial_temp_c=initial_temp)

    # Run session
    df = run_session(
        plant=plant,
        controller=controller,
        seed=seed,
        sensor_tamper=sensor_tamper,
        command_tamper=command_tamper,
        until_soc=params.SOC_TARGET,
        ambient_c=ambient_c,
        inlet_c=inlet_c,
        dt=1.0
    )

    # Step twin across the recorded session
    temp_errors = []
    voltage_errors = []

    for _, row in df.iterrows():
        flow_meas = twin_flow_override if twin_flow_override is not None else row["pump_flow_lpm"]
        
        twin.step_estimator(
            i_pack_meas=row["pack_current"],
            pump_flow_meas=flow_meas,
            T_amb_meas=row["ambient_temp"],
            T_cool_in_meas=row["coolant_temp_in"],
            dt=1.0
        )

        t_err = abs(row["true_battery_temp"] - twin.T_batt)
        v_err = abs(row["true_pack_voltage"] - twin.v_pack)
        temp_errors.append(t_err)
        voltage_errors.append(v_err)

    max_t_err = max(temp_errors)
    max_v_err = max(voltage_errors)
    peak_plant_temp = df["true_battery_temp"].max()
    peak_twin_temp = twin.T_batt

    return {
        "Scenario": name,
        "Max_Temp_Error_C": round(max_t_err, 3),
        "Max_Pack_Voltage_Error_V": round(max_v_err, 3),
        "Peak_Plant_Temp_C": round(peak_plant_temp, 2),
        "End_Twin_Temp_C": round(peak_twin_temp, 2),
    }


def run_mismatch_study(save_dir: Path | None = None) -> pd.DataFrame:
    """
    Execute all required mismatch scenarios and save results_twin_error.csv.
    """
    if save_dir is None:
        save_dir = Path(__file__).resolve().parent.parent / "outputs"
    save_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    # (a) Nominal plant
    plant_a = Plant(perturb=False, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=42)
    ctrl_a = ChargerBMS()
    results.append(evaluate_scenario("(a) Nominal plant", plant_a, ctrl_a, seed=42))

    # (b) Random +/-10% parameters, seeds 1 to 5
    for s in range(1, 6):
        plant_b = Plant(perturb=True, seed=s, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=s)
        ctrl_b = ChargerBMS()
        results.append(evaluate_scenario(f"(b) Perturbed +/-10% (seed {s})", plant_b, ctrl_b, seed=s))

    # (c) Aged cell with R0 x1.3
    plant_c = Plant(perturb=False, age_factor=1.3, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=42)
    ctrl_c = ChargerBMS()
    results.append(evaluate_scenario("(c) Aged cell (R0 x1.3)", plant_c, ctrl_c, seed=42))

    # (d) Hot day, ambient 35 C and coolant inlet 35 C
    plant_d = Plant(perturb=False, initial_soc=0.20, initial_temp_c=35.0, sensor_seed=42)
    ctrl_d = ChargerBMS()
    results.append(
        evaluate_scenario(
            "(d) Hot day (amb 35C, inlet 35C)",
            plant_d,
            ctrl_d,
            seed=42,
            ambient_c=35.0,
            inlet_c=35.0
        )
    )

    # (e1) Pump degraded to 6 L/min (twin told 10 L/min)
    plant_e1 = Plant(perturb=False, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=42)
    ctrl_e1 = ChargerBMS()
    degrade_pump = lambda t, cmds: (cmds[0], min(cmds[1], 6.0))
    results.append(
        evaluate_scenario(
            "(e1) Pump degraded 6 L/min (twin told 10 L/min)",
            plant_e1,
            ctrl_e1,
            seed=42,
            command_tamper=degrade_pump,
            twin_flow_override=10.0
        )
    )

    # (e2) Pump degraded to 6 L/min (twin using measured flow)
    plant_e2 = Plant(perturb=False, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=42)
    ctrl_e2 = ChargerBMS()
    results.append(
        evaluate_scenario(
            "(e2) Pump degraded 6 L/min (twin using measured flow)",
            plant_e2,
            ctrl_e2,
            seed=42,
            command_tamper=degrade_pump,
            twin_flow_override=None
        )
    )

    df_results = pd.DataFrame(results)
    csv_path = save_dir / "results_twin_error.csv"
    df_results.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("PhysicsGuard - Digital Twin vs Plant Mismatch Summary Table")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)
    print(f"Saved results -> {csv_path}\n")

    return df_results


if __name__ == "__main__":
    run_mismatch_study()
