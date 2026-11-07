"""
PhysicsGuard - Cyberattack Injectors
====================================
Injectors modelling cyberattacks targeting sensor telemetries or actuator commands.
All attack functions return a Scenario object with label_class="attack".
Every sensor attack works on ANY of the 10 measured signals through a signal argument.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from .scenarios import Scenario


def freeze(signal: str, start_s: float = 300.0) -> Scenario:
    """
    Sensor attack: holds the last good measured value starting from start_s.
    """
    frozen_val: Optional[float] = None

    def tamper(t: float, meas: Dict[str, float]) -> Dict[str, float]:
        nonlocal frozen_val
        if t < start_s:
            frozen_val = float(meas.get(signal, 0.0))
        else:
            if frozen_val is None:
                frozen_val = float(meas.get(signal, 0.0))
            meas[signal] = float(frozen_val)
        return meas

    return Scenario(
        name=f"freeze_{signal}",
        label_class="attack",
        label_type="freeze",
        start_s=start_s,
        sensor_tamper=tamper
    )


def replay(signal: str, start_s: float = 300.0, window_s: float = 120.0) -> Scenario:
    """
    Sensor attack: loops the signal's own recorded values from [start_s - window_s, start_s].
    """
    history: List[Tuple[float, float]] = []
    window_vals: Optional[List[float]] = None

    def tamper(t: float, meas: Dict[str, float]) -> Dict[str, float]:
        nonlocal window_vals
        val = float(meas.get(signal, 0.0))
        if t < start_s:
            history.append((t, val))
        else:
            if window_vals is None:
                # Filter recorded values in window [start_s - window_s, start_s]
                window_vals = [
                    v for (t_rec, v) in history
                    if (start_s - window_s) <= t_rec <= start_s
                ]
                if not window_vals:
                    window_vals = [val]
            idx = int(round(t - start_s)) % len(window_vals)
            meas[signal] = window_vals[idx]
        return meas

    return Scenario(
        name=f"replay_{signal}",
        label_class="attack",
        label_type="replay",
        start_s=start_s,
        sensor_tamper=tamper
    )


def drift(signal: str, start_s: float = 300.0, rate_per_min: Optional[float] = None) -> Scenario:
    """
    Sensor attack: adds a growing ramp offset.
    Defaults:
      - Temperatures: -1.0 C/min
      - Pack voltage: -2.0 V/min
      - Current: -2.0 A/min
    """
    if rate_per_min is None:
        sig_lower = signal.lower()
        if "temp" in sig_lower:
            rate_per_min = -1.0
        elif "voltage" in sig_lower:
            rate_per_min = -2.0
        elif "current" in sig_lower:
            rate_per_min = -2.0
        else:
            rate_per_min = -1.0

    rate_per_sec = float(rate_per_min) / 60.0

    def tamper(t: float, meas: Dict[str, float]) -> Dict[str, float]:
        if t >= start_s:
            offset = rate_per_sec * (t - start_s)
            meas[signal] = float(meas.get(signal, 0.0) + offset)
        return meas

    return Scenario(
        name=f"drift_{signal}",
        label_class="attack",
        label_type="drift",
        start_s=start_s,
        sensor_tamper=tamper
    )


def spike(
    signal: str,
    start_s: float = 300.0,
    magnitude: Optional[float] = None,
    duration_s: float = 60.0
) -> Scenario:
    """
    Sensor attack: injects a constant step offset during [start_s, start_s + duration_s].
    Default magnitude for temperatures is -15.0 C.
    """
    if magnitude is None:
        if "temp" in signal.lower():
            magnitude = -15.0
        else:
            magnitude = 10.0

    step_val = float(magnitude)

    def tamper(t: float, meas: Dict[str, float]) -> Dict[str, float]:
        if start_s <= t < (start_s + duration_s):
            meas[signal] = float(meas.get(signal, 0.0) + step_val)
        return meas

    return Scenario(
        name=f"spike_{signal}",
        label_class="attack",
        label_type="spike",
        start_s=start_s,
        sensor_tamper=tamper
    )


def abusive_command(start_s: float = 300.0) -> Scenario:
    """
    Actuator attack: commands 192.0 A (1.5x nominal 128 A) from start_s, ignoring derating.
    """
    def tamper(t: float, cmds: Tuple[float, float]) -> Tuple[float, float]:
        curr, pump = cmds
        if t >= start_s:
            curr = 192.0
        return curr, pump

    return Scenario(
        name="abusive_command",
        label_class="attack",
        label_type="abusive_command",
        start_s=start_s,
        command_tamper=tamper
    )


def pump_off(start_s: float = 300.0) -> Scenario:
    """
    Actuator attack: overrides commanded pump flow to 0.0 L/min from start_s.
    """
    def tamper(t: float, cmds: Tuple[float, float]) -> Tuple[float, float]:
        curr, pump = cmds
        if t >= start_s:
            pump = 0.0
        return curr, pump

    return Scenario(
        name="pump_off",
        label_class="attack",
        label_type="pump_off",
        start_s=start_s,
        command_tamper=tamper
    )


def coordinated(
    start_s: float = 300.0,
    reference_df: Optional[pd.DataFrame] = None
) -> Scenario:
    """
    Stealth coordinated attack:
    Sensors temp_cell_a, temp_cell_b, coolant_temp_out, pump_flow_lpm replay
    a healthy reference run (same seed), while the applied pump command is forced to 0.
    """
    ref_data: Optional[pd.DataFrame] = reference_df

    def _get_reference() -> pd.DataFrame:
        nonlocal ref_data
        if ref_data is None:
            from .plant import Plant
            from .controller import ChargerBMS
            from .simulate import run_session
            p = Plant(perturb=False, initial_soc=0.20, initial_temp_c=25.0, sensor_seed=42)
            c = ChargerBMS()
            ref_data = run_session(p, c, seed=42)
        return ref_data

    def sensor_tamper(t: float, meas: Dict[str, float]) -> Dict[str, float]:
        if t >= start_s:
            df_ref = _get_reference()
            idx = int(round(t))
            if 0 <= idx < len(df_ref):
                ref_row = df_ref.iloc[idx]
            else:
                ref_row = df_ref.iloc[-1]
            meas["temp_cell_a"] = float(ref_row["temp_cell_a"])
            meas["temp_cell_b"] = float(ref_row["temp_cell_b"])
            meas["coolant_temp_out"] = float(ref_row["coolant_temp_out"])
            meas["pump_flow_lpm"] = float(ref_row["pump_flow_lpm"])
        return meas

    def command_tamper(t: float, cmds: Tuple[float, float]) -> Tuple[float, float]:
        curr, pump = cmds
        if t >= start_s:
            pump = 0.0
        return curr, pump

    return Scenario(
        name="coordinated",
        label_class="attack",
        label_type="coordinated",
        start_s=start_s,
        sensor_tamper=sensor_tamper,
        command_tamper=command_tamper
    )


def demo_hot_attack() -> Scenario:
    """
    Severe composite cyberattack demo:
      - Ambient 35 C, Inlet 35 C
      - temp_cell_a frozen at 25 C from t = 0
      - pump_off (pump commanded to 0) from t = 60 s
    """
    def sensor_tamper(t: float, meas: Dict[str, float]) -> Dict[str, float]:
        meas["temp_cell_a"] = 25.0
        return meas

    def command_tamper(t: float, cmds: Tuple[float, float]) -> Tuple[float, float]:
        curr, pump = cmds
        if t >= 60.0:
            pump = 0.0
        return curr, pump

    return Scenario(
        name="demo_hot_attack",
        label_class="attack",
        label_type="demo_hot_attack",
        start_s=0.0,
        sensor_tamper=sensor_tamper,
        command_tamper=command_tamper,
        ambient_c=35.0,
        inlet_c=35.0,
        plant_kwargs={"initial_temp_c": 35.0}
    )
