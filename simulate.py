"""
PhysicsGuard - Closed-Loop Fast-Charging Session Simulator
==========================================================
Coordinates closed-loop simulation runs between the physical plant,
the Charger/BMS controller, and optional cybersecurity attack tamper functions.

Architecture:
  Plant.measure()
        │
        ▼
  [sensor_tamper(t, meas)]   <--- Cyberattack spoofing (e.g. freeze temperature sensor)
        │
        ▼
   ChargerBMS controller     <--- Computes (current_cmd_a, pump_cmd_lpm)
        │
        ▼
  [command_tamper(t, cmds)]  <--- Cyberattack command override (e.g. disable coolant pump)
        │
        ▼
   Plant.step()
"""

from typing import Callable, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

from . import params
from .plant import Plant
from .controller import ChargerBMS


def run_session(
    plant: Plant,
    controller: ChargerBMS,
    seed: Optional[int] = None,
    sensor_tamper: Optional[Callable[[float, Dict[str, float]], Dict[str, float]]] = None,
    command_tamper: Optional[Callable[[float, Tuple[float, float]], Tuple[float, float]]] = None,
    until_soc: float = 0.80,
    ambient_c: float = 25.0,
    inlet_c: float = 25.0,
    dt: float = 1.0,
    max_duration_s: float = 7200.0
) -> pd.DataFrame:
    """
    Simulate a closed-loop fast-charging session until target SOC is achieved.

    Parameters
    ----------
    plant : Plant
        Physical plant model of the EV battery pack and cooling system.
    controller : ChargerBMS
        Closed-loop Charger / BMS controller.
    seed : int, optional
        Random seed for reproducible sensor noise.
    sensor_tamper : callable, optional
        Function (t, measured_dict) -> dict applied BETWEEN plant.measure() and controller.
    command_tamper : callable, optional
        Function (t, (current_cmd, pump_cmd)) -> (current_cmd, pump_cmd) applied BETWEEN
        controller and plant.
    until_soc : float, default 0.80
        Target State of Charge to terminate charging session.
    ambient_c : float, default 25.0 C
        Ambient air temperature.
    inlet_c : float, default 25.0 C
        Coolant loop inlet temperature.
    dt : float, default 1.0 s
        Integration time step in seconds.
    max_duration_s : float, default 7200.0 s
        Maximum allowed simulation duration (timeout safety).

    Returns
    -------
    pd.DataFrame
        DataFrame with measured signals, true signals, requested commands,
        and applied commands.
    """
    if seed is not None:
        plant.sensor_rng = np.random.default_rng(seed)

    records = []
    # Initial measurement sampled prior to first control step
    meas = plant.measure()

    max_steps = int(max_duration_s / dt)
    step_count = 0

    while plant.soc < until_soc and step_count < max_steps:
        t = plant.time_s

        # 1. Sensor tamper (intercepts measurement before reaching controller)
        if sensor_tamper is not None:
            meas_for_ctrl = sensor_tamper(t, meas.copy())
        else:
            meas_for_ctrl = meas.copy()

        # 2. Controller determines commanded current and pump flow
        req_current_a, req_pump_lpm = controller.compute_commands(meas_for_ctrl)

        # 3. Command tamper (intercepts actuation commands before reaching physical plant)
        if command_tamper is not None:
            appl_current_a, appl_pump_lpm = command_tamper(
                t, (req_current_a, req_pump_lpm)
            )
        else:
            appl_current_a, appl_pump_lpm = req_current_a, req_pump_lpm

        # 4. Advance physical plant by time step dt
        plant.step(
            dt=dt,
            current_cmd_a=appl_current_a,
            pump_cmd_lpm=appl_pump_lpm,
            T_cool_in_c=inlet_c,
            T_amb_c=ambient_c
        )

        # 5. Sample updated state and post-step instrumentation
        meas = plant.measure()
        true_state = plant.get_true_state()

        row = {
            "time": true_state["time_s"],
            "time_s": true_state["time_s"],
            # Measured signals (10 channels)
            "pack_voltage": meas["pack_voltage"],
            "pack_current": meas["pack_current"],
            "soc_estimate": meas["soc_estimate"],
            "temp_cell_a": meas["temp_cell_a"],
            "temp_cell_b": meas["temp_cell_b"],
            "coolant_temp_in": meas["coolant_temp_in"],
            "coolant_temp_out": meas["coolant_temp_out"],
            "pump_flow_lpm": meas["pump_flow_lpm"],
            "ambient_temp": meas["ambient_temp"],
            "charger_power_cmd_kw": meas["charger_power_cmd_kw"],
            # True plant state signals
            "true_battery_temp": true_state["true_battery_temp"],
            "true_soc": true_state["true_soc"],
            "true_pack_voltage": true_state["true_pack_voltage"],
            "true_pack_current": true_state["true_pack_current"],
            "true_cell_voltage": true_state["true_cell_voltage"],
            "true_cell_current": true_state["true_cell_current"],
            "true_vp": true_state["true_vp"],
            "true_q_heat": true_state["true_q_heat"],
            "true_q_cool": true_state["true_q_cool"],
            "true_q_amb": true_state["true_q_amb"],
            "true_pump_flow_lpm": true_state["true_pump_flow_lpm"],
            "true_coolant_temp_in": true_state["true_coolant_temp_in"],
            "true_coolant_temp_out": true_state["true_coolant_temp_out"],
            "true_ambient_temp": true_state["true_ambient_temp"],
            "true_charger_power_kw": true_state["true_charger_power_kw"],
            # Requested commands (from controller)
            "requested_current_a": float(req_current_a),
            "requested_pump_lpm": float(req_pump_lpm),
            # Applied commands (after tamper, delivered to plant)
            "applied_current_a": float(appl_current_a),
            "applied_pump_lpm": float(appl_pump_lpm),
            # Direct aliases
            "current_cmd_a": float(appl_current_a),
            "pump_cmd_lpm": float(appl_pump_lpm),
        }
        records.append(row)
        step_count += 1

    return pd.DataFrame(records)
