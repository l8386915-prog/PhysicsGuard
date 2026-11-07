"""
physicsguard/attacks_custom.py - Generic, Composable Cyberattack Injectors
==========================================================================
Enables dynamic construction and stacking of sensor and actuator cyberattacks
from arbitrary user-defined specifications.

Phase 6.5 Attack Lab injector module.
"""

import math
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import pandas as pd

from .scenarios import Scenario


# ---------------------------------------------------------------------------
# Supported Targets and Types
# ---------------------------------------------------------------------------
SENSOR_TARGETS = [
    "temp_cell_a",
    "temp_cell_b",
    "coolant_temp_out",
    "pump_flow_lpm",
    "pack_current",
    "pack_voltage",
    "soc_estimate",
]

COMMAND_TARGETS = [
    "cmd_current",
    "cmd_pump",
]

ALL_TARGETS = SENSOR_TARGETS + COMMAND_TARGETS

ATTACK_TYPES = [
    "freeze",
    "drift",
    "spike",
    "set_value",
    "dead_nan",
    "dead_zero",
    "scale",
    "replay",
]

# Physical clamping bounds
CLAMP_BOUNDS = {
    "start_s": (0.0, 3000.0),
    "cmd_current": (0.0, 250.0),
    "pack_current": (0.0, 250.0),
    "cmd_pump": (0.0, 20.0),
    "pump_flow_lpm": (0.0, 20.0),
    "temp_cell_a": (-20.0, 120.0),
    "temp_cell_b": (-20.0, 120.0),
    "coolant_temp_out": (-20.0, 120.0),
    "duration_s": (0.0, 7200.0),
}


def clamp_value(val: Any, min_val: float, max_val: float, default: float = 0.0) -> float:
    """Safely clamp a numeric value to [min_val, max_val], handling None/NaN/invalid."""
    if val is None:
        return default
    try:
        f = float(val)
        if not math.isfinite(f):
            return default
        return float(min(max(f, min_val), max_val))
    except (ValueError, TypeError):
        return default


def validate_and_clamp_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clamp an attack specification dict.
    
    Fields:
      - target: sensor or command channel
      - type: attack injector type
      - start_s: clamped to [0, 3000] s
      - duration_s: clamped to >= 0 s (None/inf means until session end)
      - magnitude: clamped according to physical target bounds
    """
    spec = dict(spec)
    target = str(spec.get("target", "temp_cell_a")).strip().lower()
    if target not in ALL_TARGETS:
        target = "temp_cell_a"

    atk_type = str(spec.get("type", "freeze")).strip().lower()
    if atk_type not in ATTACK_TYPES:
        atk_type = "freeze"

    start_s = clamp_value(spec.get("start_s", 300.0), CLAMP_BOUNDS["start_s"][0], CLAMP_BOUNDS["start_s"][1], 300.0)

    dur_raw = spec.get("duration_s")
    if dur_raw is not None and str(dur_raw).strip() != "" and str(dur_raw).lower() != "none":
        try:
            dur_f = float(dur_raw)
            if math.isfinite(dur_f) and dur_f > 0:
                duration_s = max(0.0, dur_f)
            else:
                duration_s = None
        except (ValueError, TypeError):
            duration_s = None
    else:
        duration_s = None

    mag_raw = spec.get("magnitude", 0.0)
    mag_f = clamp_value(mag_raw, -1000.0, 1000.0, 0.0)

    # Specific clamping for set_value & target limits
    if atk_type == "set_value":
        if target in ("cmd_current", "pack_current"):
            mag_f = clamp_value(mag_f, CLAMP_BOUNDS["cmd_current"][0], CLAMP_BOUNDS["cmd_current"][1], 128.0)
        elif target in ("cmd_pump", "pump_flow_lpm"):
            mag_f = clamp_value(mag_f, CLAMP_BOUNDS["cmd_pump"][0], CLAMP_BOUNDS["cmd_pump"][1], 0.0)
        elif target in ("temp_cell_a", "temp_cell_b", "coolant_temp_out"):
            mag_f = clamp_value(mag_f, CLAMP_BOUNDS["temp_cell_a"][0], CLAMP_BOUNDS["temp_cell_a"][1], 25.0)

    return {
        "target": target,
        "type": atk_type,
        "start_s": start_s,
        "duration_s": duration_s,
        "magnitude": mag_f,
    }


# ---------------------------------------------------------------------------
# Composable Attack Injector Factory
# ---------------------------------------------------------------------------
def build_custom_scenario(
    attack_specs: List[Dict[str, Any]],
    name: str = "custom_attack",
    ambient_c: Optional[float] = None,
    inlet_c: Optional[float] = None,
    plant_kwargs: Optional[Dict[str, Any]] = None,
) -> Scenario:
    """
    Construct a Scenario object from up to 4 stacked generic attack specs.
    
    Each spec is validated and clamped. Sensor attacks are routed to
    sensor_tamper, and cmd_* attacks are routed to command_tamper.
    """
    clamped_specs = [validate_and_clamp_spec(s) for s in attack_specs if isinstance(s, dict)]
    # Support up to 4 stacked attacks
    clamped_specs = clamped_specs[:4]

    sensor_specs = [s for s in clamped_specs if s["target"] in SENSOR_TARGETS]
    cmd_specs = [s for s in clamped_specs if s["target"] in COMMAND_TARGETS]

    # Overall start timestamp
    if clamped_specs:
        min_start_s = min(s["start_s"] for s in clamped_specs)
    else:
        min_start_s = 300.0

    # Sensor tamper state closures
    frozen_sensor_vals: Dict[int, Optional[float]] = {i: None for i in range(len(sensor_specs))}
    sensor_histories: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(sensor_specs))}
    replay_windows: Dict[int, Optional[List[float]]] = {i: None for i in range(len(sensor_specs))}

    def sensor_tamper(t: float, meas: Dict[str, float]) -> Dict[str, float]:
        meas = dict(meas)
        for i, atk in enumerate(sensor_specs):
            target = atk["target"]
            start_s = atk["start_s"]
            dur = atk["duration_s"]
            atk_type = atk["type"]
            mag = atk["magnitude"]

            is_active = (t >= start_s) and (dur is None or t < (start_s + dur))
            val_now = float(meas.get(target, 0.0))

            if not is_active:
                if t <= start_s:
                    frozen_sensor_vals[i] = val_now
                    sensor_histories[i].append((t, val_now))
                continue

            if atk_type == "freeze":
                if t <= start_s or frozen_sensor_vals[i] is None:
                    frozen_sensor_vals[i] = val_now
                meas[target] = float(frozen_sensor_vals[i])

            elif atk_type == "drift":
                rate_per_sec = mag / 60.0
                offset = rate_per_sec * (t - start_s)
                meas[target] = float(val_now + offset)

            elif atk_type == "spike":
                meas[target] = float(val_now + mag)

            elif atk_type == "set_value":
                meas[target] = float(mag)

            elif atk_type == "dead_nan":
                meas[target] = float("nan")

            elif atk_type == "dead_zero":
                meas[target] = 0.0

            elif atk_type == "scale":
                meas[target] = float(val_now * mag)

            elif atk_type == "replay":
                w_s = mag if mag > 0 else 120.0
                if replay_windows[i] is None:
                    hist = sensor_histories[i]
                    w_vals = [v for (t_rec, v) in hist if (start_s - w_s) <= t_rec <= start_s]
                    replay_windows[i] = w_vals if w_vals else [val_now]
                w = replay_windows[i]
                meas[target] = w[int(round(t - start_s)) % len(w)]

        return meas

    # Command tamper state closures
    frozen_cmd_vals: Dict[int, Optional[float]] = {i: None for i in range(len(cmd_specs))}

    def command_tamper(t: float, cmds: Tuple[float, float]) -> Tuple[float, float]:
        curr, pump = cmds
        curr = float(curr)
        pump = float(pump)

        for i, atk in enumerate(cmd_specs):
            target = atk["target"]
            start_s = atk["start_s"]
            dur = atk["duration_s"]
            atk_type = atk["type"]
            mag = atk["magnitude"]

            is_active = (t >= start_s) and (dur is None or t < (start_s + dur))

            if target == "cmd_current":
                if not is_active:
                    if t <= start_s:
                        frozen_cmd_vals[i] = curr
                    continue

                if atk_type == "set_value":
                    curr = float(mag)
                elif atk_type == "freeze":
                    if t <= start_s or frozen_cmd_vals[i] is None:
                        frozen_cmd_vals[i] = curr
                    curr = float(frozen_cmd_vals[i])
                elif atk_type == "drift":
                    curr = float(curr + (mag / 60.0) * (t - start_s))
                elif atk_type == "spike":
                    curr = float(curr + mag)
                elif atk_type == "dead_zero":
                    curr = 0.0
                elif atk_type == "dead_nan":
                    curr = float("nan")
                elif atk_type == "scale":
                    curr = float(curr * mag)

            elif target == "cmd_pump":
                if not is_active:
                    if t <= start_s:
                        frozen_cmd_vals[i] = pump
                    continue

                if atk_type == "set_value":
                    pump = float(mag)
                elif atk_type == "freeze":
                    if t <= start_s or frozen_cmd_vals[i] is None:
                        frozen_cmd_vals[i] = pump
                    pump = float(frozen_cmd_vals[i])
                elif atk_type == "drift":
                    pump = float(pump + (mag / 60.0) * (t - start_s))
                elif atk_type == "spike":
                    pump = float(pump + mag)
                elif atk_type == "dead_zero":
                    pump = 0.0
                elif atk_type == "dead_nan":
                    pump = float("nan")
                elif atk_type == "scale":
                    pump = float(pump * mag)

        return curr, pump

    label_types = [f"{s['target']}_{s['type']}" for s in clamped_specs]
    comb_label_type = "+".join(label_types) if label_types else "custom_attack"

    p_kwargs = dict(plant_kwargs) if plant_kwargs is not None else {}

    return Scenario(
        name=name,
        label_class="attack",
        label_type=comb_label_type,
        start_s=min_start_s,
        sensor_tamper=sensor_tamper if sensor_specs else None,
        command_tamper=command_tamper if cmd_specs else None,
        ambient_c=ambient_c,
        inlet_c=inlet_c,
        plant_kwargs=p_kwargs
    )


# ---------------------------------------------------------------------------
# Presets Registry for 1-Click Demos
# ---------------------------------------------------------------------------
PRESET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "pump_off": {
        "description": "Overrides commanded coolant pump flow to 0.0 L/min at t=300s.",
        "specs": [
            {"target": "cmd_pump", "type": "set_value", "magnitude": 0.0, "start_s": 300.0, "duration_s": None}
        ],
        "env": {"ambient_c": 35.0, "inlet_c": 35.0}
    },
    "abusive_command": {
        "description": "Forces 192.0 A (1.5x nominal) charging current at t=300s, ignoring derating.",
        "specs": [
            {"target": "cmd_current", "type": "set_value", "magnitude": 192.0, "start_s": 300.0, "duration_s": None}
        ],
        "env": {"ambient_c": 35.0, "inlet_c": 35.0}
    },
    "freeze_temp_a": {
        "description": "Freezes temperature sensor A at its t=300s value indefinitely.",
        "specs": [
            {"target": "temp_cell_a", "type": "freeze", "magnitude": 0.0, "start_s": 300.0, "duration_s": None}
        ],
        "env": {"ambient_c": 25.0, "inlet_c": 25.0}
    },
    "drift_temp_a": {
        "description": "Applies a -1.0 °C/min slow negative drift to temperature sensor A at t=300s.",
        "specs": [
            {"target": "temp_cell_a", "type": "drift", "magnitude": -1.0, "start_s": 300.0, "duration_s": None}
        ],
        "env": {"ambient_c": 25.0, "inlet_c": 25.0}
    },
    "coordinated": {
        "description": "Stealth stacked attack: freezes temperature sensor A and shuts off pump at t=300s.",
        "specs": [
            {"target": "temp_cell_a", "type": "freeze", "magnitude": 0.0, "start_s": 300.0, "duration_s": None},
            {"target": "cmd_pump", "type": "set_value", "magnitude": 0.0, "start_s": 300.0, "duration_s": None}
        ],
        "env": {"ambient_c": 35.0, "inlet_c": 35.0}
    },
    "demo_hot_attack": {
        "description": "Severe composite attack in hot ambient (35°C): temp_cell_a spoofed to 25°C at t=0s, pump killed at t=60s.",
        "specs": [
            {"target": "temp_cell_a", "type": "set_value", "magnitude": 25.0, "start_s": 0.0, "duration_s": None},
            {"target": "cmd_pump", "type": "set_value", "magnitude": 0.0, "start_s": 60.0, "duration_s": None}
        ],
        "env": {"ambient_c": 35.0, "inlet_c": 35.0, "initial_temp_c": 35.0}
    },
}


def get_preset_options() -> List[str]:
    """Return available preset scenario names."""
    return list(PRESET_REGISTRY.keys())


def load_preset(name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (specs, env_overrides) for a given preset."""
    if name in PRESET_REGISTRY:
        preset = PRESET_REGISTRY[name]
        return [dict(s) for s in preset["specs"]], dict(preset.get("env", {}))
    return [], {}
