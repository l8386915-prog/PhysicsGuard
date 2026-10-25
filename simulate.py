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
    plant_hook: Optional[Callable[[float, Plant], None]] = None,
    scenario: Optional[Any] = None,
    guard: Optional[Any] = None,
    gate: Optional[Any] = None,
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
        Function (t, measured_dict) -> dict applied to sensor measurements.
    command_tamper : callable, optional
        Function (t, (current_cmd, pump_cmd)) -> (current_cmd, pump_cmd) applied
        between controller and plant.
    plant_hook : callable, optional
        Function (t, plant) called every step before plant.step to alter plant
        parameters dynamically over time.
    scenario : Scenario, optional
        Scenario descriptor specifying faults, cyberattacks, and metadata.
    guard : PhysicsGuard, optional
        Anomaly detector observer.
    gate : SafetyGate, optional
        Actuator protection gate (virtual sensor and command screening).
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
        applied commands, gate telemetry, and scenario metadata.
    """
    if seed is not None:
        plant.sensor_rng = np.random.default_rng(seed)

    # Ingest scenario parameters if provided
    if scenario is not None:
        if sensor_tamper is None and scenario.sensor_tamper is not None:
            sensor_tamper = scenario.sensor_tamper
        if command_tamper is None and scenario.command_tamper is not None:
            command_tamper = scenario.command_tamper
        if plant_hook is None and scenario.plant_hook is not None:
            plant_hook = scenario.plant_hook
        if scenario.ambient_c is not None and ambient_c == 25.0:
            ambient_c = float(scenario.ambient_c)
        if scenario.inlet_c is not None and inlet_c == 25.0:
            inlet_c = float(scenario.inlet_c)
        if getattr(scenario, "plant_kwargs", None):
            for k, v in scenario.plant_kwargs.items():
                if k == "initial_temp_c":
                    plant.T_batt = float(v)
                    plant.initial_T_batt = float(v)
                elif k == "age_factor":
                    plant.age_factor = float(v)
                elif hasattr(plant, k):
                    setattr(plant, k, v)

    # Reset controller state if supported
    if hasattr(controller, "reset"):
        controller.reset()

    # Attach guard to gate if not already set
    if gate is not None and guard is not None:
        if hasattr(gate, "set_guard") and getattr(gate, "guard", None) is None:
            gate.set_guard(guard)

    records = []
    max_steps = int(max_duration_s / dt)
    step_count = 0
    latched_steps = 0

    # Initial measurement sampled prior to first step
    meas_raw = plant.measure()
    if sensor_tamper is not None:
        meas = sensor_tamper(plant.time_s, meas_raw.copy())
    else:
        meas = meas_raw.copy()

    while step_count < max_steps:
        t = plant.time_s

        # Stage 1: Virtual sensor filtering for controller (if gate enabled)
        if gate is not None:
            meas_for_ctrl = gate.sanitize(t, meas)
        else:
            meas_for_ctrl = meas

        # 1. Controller determines commanded current and pump flow
        req_current_a, req_pump_lpm = controller.compute_commands(meas_for_ctrl)

        # 2. Command tamper intercepts actuation commands before actuator gate
        if command_tamper is not None:
            tampered_current_a, tampered_pump_lpm = command_tamper(
                t, (req_current_a, req_pump_lpm)
            )
        else:
            tampered_current_a, tampered_pump_lpm = req_current_a, req_pump_lpm

        # Stage 2: Actuator gate screening before delivery to plant
        if gate is not None:
            appl_current_a, appl_pump_lpm = gate.screen(
                t, (tampered_current_a, tampered_pump_lpm), meas
            )
        else:
            appl_current_a, appl_pump_lpm = tampered_current_a, tampered_pump_lpm

        # 3. Plant hook called every step before plant.step
        if plant_hook is not None:
            plant_hook(t, plant)

        # 4. Advance physical plant by time step dt
        plant.step(
            dt=dt,
            current_cmd_a=appl_current_a,
            pump_cmd_lpm=appl_pump_lpm,
            T_cool_in_c=inlet_c,
            T_amb_c=ambient_c
        )

        # 5. Sample updated state and post-step instrumentation
        true_state = plant.get_true_state()
        t_post = true_state["time_s"]
        meas_raw = plant.measure()

        if sensor_tamper is not None:
            meas = sensor_tamper(t_post, meas_raw.copy())
        else:
            meas = meas_raw.copy()

        row = {
            "time": t_post,
            "time_s": t_post,
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

        # Gate telemetry columns
        if gate is not None:
            row["gate_mode"] = str(getattr(gate, "mode", "NOMINAL"))
            row["gate_action"] = str(getattr(gate, "last_action", "none"))
            row["trusted_temp"] = float(getattr(gate, "last_trusted_temp", meas["temp_cell_a"]))
            row["applied_vs_requested_current"] = float(appl_current_a - req_current_a)
            row["intervened"] = bool(getattr(gate, "intervened", False))
        else:
            row["gate_mode"] = "NOMINAL"
            row["gate_action"] = "none"
            trusted_val = guard.trusted_temperature() if guard is not None and hasattr(guard, "trusted_temperature") else float(meas.get("temp_cell_a", 25.0))
            row["trusted_temp"] = float(trusted_val)
            row["applied_vs_requested_current"] = float(appl_current_a - req_current_a)
            row["intervened"] = False

        # Scenario columns if scenario is provided
        if scenario is not None:
            row["scenario"] = scenario.name
            row["label_class"] = scenario.label_class
            row["label_type"] = scenario.label_type
            if scenario.start_s is not None and scenario.label_class != "normal":
                row["event_active"] = bool(t_post >= scenario.start_s)
            else:
                row["event_active"] = False

        # Guard verdict logging if guard provided
        if guard is not None:
            verdict = guard.update(
                t_post,
                dict(meas),
                (float(appl_current_a), float(appl_pump_lpm)),
                requested_cmds=(float(tampered_current_a), float(tampered_pump_lpm))
            )
            row["diagnosis"] = verdict.diagnosis
            row["alarm_level"] = verdict.alarm_level
            row["is_attack"] = verdict.is_attack
            row["blamed_signals"] = verdict.blamed_signals
            row["reasons"] = verdict.reasons

        records.append(row)
        step_count += 1

        # Check termination condition
        if getattr(controller, "latched_stop", False):
            latched_steps += 1
            if latched_steps >= 2:
                break
        elif until_soc < getattr(controller, "soc_cutoff", 0.80) and plant.soc >= until_soc:
            break
        elif plant.soc >= until_soc and (command_tamper is None or appl_current_a == 0.0):
            break
        elif plant.soc >= 0.85:
            break

    return pd.DataFrame(records)

