"""
PhysicsGuard - Physical & Hardware Fault Injectors
===================================================
Injectors modelling physical component degradation and sensor hardware failures.
All fault functions return a Scenario object with label_class="fault".
"""

from typing import Dict, Any, Optional
from .scenarios import Scenario


def pump_degradation(start_s: float = 300.0, fraction: float = 0.6) -> Scenario:
    """
    Simulates mechanical degradation or clogging of the coolant pump.
    
    plant_hook limits the actual pump capacity to fraction * 10 L/min from start_s.
    The applied pump COMMAND stays 10 L/min (the fault is below the command interface).
    """
    max_flow = float(fraction * 10.0)

    def hook(t: float, plant: Any) -> None:
        if t >= start_s:
            plant.max_pump_flow_lpm = max_flow
            if plant.pump_flow_lpm > max_flow:
                plant.pump_flow_lpm = max_flow

    return Scenario(
        name="pump_degradation",
        label_class="fault",
        label_type="pump_degradation",
        start_s=start_s,
        plant_hook=hook
    )


def aged_cell(age_factor: float = 1.4) -> Scenario:
    """
    Simulates cell electrochemical ageing from t = 0.
    
    Increases cell internal ohmic resistance by age_factor multiplier.
    """
    def hook(t: float, plant: Any) -> None:
        plant.age_factor = float(age_factor)

    return Scenario(
        name="aged_cell",
        label_class="fault",
        label_type="aged_cell",
        start_s=0.0,
        plant_hook=hook,
        plant_kwargs={"age_factor": float(age_factor)}
    )


def hot_day(ambient_c: float = 38.0, inlet_c: float = 35.0) -> Scenario:
    """
    Simulates extreme hot environment operating conditions from t = 0.
    """
    return Scenario(
        name="hot_day",
        label_class="fault",
        label_type="hot_day",
        start_s=0.0,
        ambient_c=float(ambient_c),
        inlet_c=float(inlet_c),
        plant_kwargs={"initial_temp_c": float(inlet_c)}
    )


def dead_sensor(signal: str, mode: str, start_s: float = 300.0) -> Scenario:
    """
    Simulates sensor hardware transducer failure starting at start_s.
    
    Modes:
      - "nan": sensor outputs NaN
      - "zero": sensor drops to 0.0
      - "flatline": sensor is frozen at its value and is noise-free
    """
    mode_lower = mode.lower()
    if mode_lower not in ("nan", "zero", "flatline"):
        raise ValueError(f"Invalid mode '{mode}'. Choose 'nan', 'zero', or 'flatline'.")

    frozen_val: Optional[float] = None

    def tamper(t: float, meas: Dict[str, float]) -> Dict[str, float]:
        nonlocal frozen_val
        if t < start_s:
            frozen_val = float(meas.get(signal, 0.0))
        else:
            if mode_lower == "nan":
                meas[signal] = float("nan")
            elif mode_lower == "zero":
                meas[signal] = 0.0
            elif mode_lower == "flatline":
                if frozen_val is None:
                    frozen_val = float(meas.get(signal, 0.0))
                meas[signal] = float(frozen_val)
        return meas

    return Scenario(
        name=f"dead_sensor_{mode_lower}_{signal}",
        label_class="fault",
        label_type="dead_sensor",
        start_s=start_s,
        sensor_tamper=tamper
    )
