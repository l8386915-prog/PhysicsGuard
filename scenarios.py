"""
PhysicsGuard - Scenario Definitions and Registry
================================================
Defines the Scenario dataclass and registry for generating fault and attack scenarios.

Every Scenario encapsulates:
  - name: unique scenario identifier string
  - label_class: ground-truth classification ("normal", "fault", or "attack")
  - label_type: specific event type name
  - start_s: activation timestamp in seconds
  - sensor_tamper: function (t, meas) -> meas
  - command_tamper: function (t, cmds) -> cmds
  - plant_hook: function (t, plant) -> None
  - ambient_c: environment ambient override
  - inlet_c: coolant loop inlet override
  - plant_kwargs: kwargs for Plant initialization
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, Tuple
import pandas as pd


@dataclass
class Scenario:
    """
    Specification for a physical fault, cyberattack, or normal baseline session.
    """
    name: str
    label_class: str  # "normal" | "fault" | "attack"
    label_type: str
    start_s: Optional[float] = None
    sensor_tamper: Optional[Callable[[float, Dict[str, float]], Dict[str, float]]] = None
    command_tamper: Optional[Callable[[float, Tuple[float, float]], Tuple[float, float]]] = None
    plant_hook: Optional[Callable[[float, Any], None]] = None
    ambient_c: Optional[float] = None
    inlet_c: Optional[float] = None
    plant_kwargs: Dict[str, Any] = field(default_factory=dict)


def build_scenario(name: str, **kwargs) -> Scenario:
    """
    Registry factory function to construct predefined scenarios by name.

    Supported names:
      - normal
      - pump_degradation
      - aged_cell
      - hot_day
      - dead_sensor_nan_temp_a
      - dead_sensor_flatline_temp_a
      - freeze_temp_a
      - replay_temp_a
      - drift_temp_a
      - spike_temp_a
      - freeze_current
      - drift_current
      - drift_voltage
      - abusive_command
      - pump_off
      - coordinated
      - demo_hot_attack
    """
    from . import faults
    from . import attacks

    if name == "normal":
        return Scenario(
            name="normal",
            label_class="normal",
            label_type="normal",
            start_s=None
        )

    # Physical Faults
    elif name == "pump_degradation":
        start_s = kwargs.get("start_s", 300.0)
        fraction = kwargs.get("fraction", 0.6)
        sc = faults.pump_degradation(start_s=start_s, fraction=fraction)
        sc.name = "pump_degradation"
        return sc

    elif name == "aged_cell":
        age_factor = kwargs.get("age_factor", 1.4)
        sc = faults.aged_cell(age_factor=age_factor)
        sc.name = "aged_cell"
        return sc

    elif name == "hot_day":
        ambient_c = kwargs.get("ambient_c", 38.0)
        inlet_c = kwargs.get("inlet_c", 35.0)
        sc = faults.hot_day(ambient_c=ambient_c, inlet_c=inlet_c)
        sc.name = "hot_day"
        return sc

    elif name == "dead_sensor_nan_temp_a":
        start_s = kwargs.get("start_s", 300.0)
        sc = faults.dead_sensor(signal="temp_cell_a", mode="nan", start_s=start_s)
        sc.name = "dead_sensor_nan_temp_a"
        return sc

    elif name == "dead_sensor_flatline_temp_a":
        start_s = kwargs.get("start_s", 300.0)
        sc = faults.dead_sensor(signal="temp_cell_a", mode="flatline", start_s=start_s)
        sc.name = "dead_sensor_flatline_temp_a"
        return sc

    # Cyberattacks
    elif name == "freeze_temp_a":
        start_s = kwargs.get("start_s", 300.0)
        sc = attacks.freeze(signal="temp_cell_a", start_s=start_s)
        sc.name = "freeze_temp_a"
        return sc

    elif name == "replay_temp_a":
        start_s = kwargs.get("start_s", 300.0)
        window_s = kwargs.get("window_s", 120.0)
        sc = attacks.replay(signal="temp_cell_a", start_s=start_s, window_s=window_s)
        sc.name = "replay_temp_a"
        return sc

    elif name == "drift_temp_a":
        start_s = kwargs.get("start_s", 300.0)
        rate_per_min = kwargs.get("rate_per_min", -1.0)
        sc = attacks.drift(signal="temp_cell_a", start_s=start_s, rate_per_min=rate_per_min)
        sc.name = "drift_temp_a"
        return sc

    elif name == "spike_temp_a":
        start_s = kwargs.get("start_s", 300.0)
        magnitude = kwargs.get("magnitude", -15.0)
        duration_s = kwargs.get("duration_s", 60.0)
        sc = attacks.spike(signal="temp_cell_a", start_s=start_s, magnitude=magnitude, duration_s=duration_s)
        sc.name = "spike_temp_a"
        return sc

    elif name == "freeze_current":
        start_s = kwargs.get("start_s", 300.0)
        sc = attacks.freeze(signal="pack_current", start_s=start_s)
        sc.name = "freeze_current"
        return sc

    elif name == "drift_current":
        start_s = kwargs.get("start_s", 300.0)
        rate_per_min = kwargs.get("rate_per_min", -2.0)
        sc = attacks.drift(signal="pack_current", start_s=start_s, rate_per_min=rate_per_min)
        sc.name = "drift_current"
        return sc

    elif name == "drift_voltage":
        start_s = kwargs.get("start_s", 300.0)
        rate_per_min = kwargs.get("rate_per_min", -2.0)
        sc = attacks.drift(signal="pack_voltage", start_s=start_s, rate_per_min=rate_per_min)
        sc.name = "drift_voltage"
        return sc

    elif name == "abusive_command":
        start_s = kwargs.get("start_s", 300.0)
        sc = attacks.abusive_command(start_s=start_s)
        sc.name = "abusive_command"
        return sc

    elif name == "pump_off":
        start_s = kwargs.get("start_s", 300.0)
        sc = attacks.pump_off(start_s=start_s)
        sc.name = "pump_off"
        return sc

    elif name == "coordinated":
        start_s = kwargs.get("start_s", 300.0)
        ref_df = kwargs.get("reference_df", None)
        sc = attacks.coordinated(start_s=start_s, reference_df=ref_df)
        sc.name = "coordinated"
        return sc

    elif name == "demo_hot_attack":
        sc = attacks.demo_hot_attack()
        sc.name = "demo_hot_attack"
        return sc

    else:
        raise ValueError(f"Unknown scenario name: '{name}'")
