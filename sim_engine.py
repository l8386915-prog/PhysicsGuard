"""
sim_engine.py - Core Simulation Engine & Data Pipeline for PhysicsGuard Dashboard
================================================================================
Provides unified simulation execution, plain-alarm baselines, residual logging,
caching, and moment-by-moment physical explanation synthesis.
"""

import os
import json
import math
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
import numpy as np
import pandas as pd

from physicsguard import params
from physicsguard.plant import Plant
from physicsguard.controller import ChargerBMS
from physicsguard.scenarios import build_scenario, Scenario
from physicsguard.simulate import run_session
from physicsguard.guard import PhysicsGuard, Verdict
from physicsguard.mitigation import SafetyGate
from physicsguard.attacks_custom import (
    build_custom_scenario,
    validate_and_clamp_spec,
    load_preset,
    get_preset_options,
    PRESET_REGISTRY,
    ALL_TARGETS,
    ATTACK_TYPES,
)

# ---------------------------------------------------------------------------
# Constants & Scenarios
# ---------------------------------------------------------------------------
MAIN_DEMO_SCENARIOS = ["pump_off", "abusive_command", "coordinated", "demo_hot_attack"]

ALL_SCENARIOS = [
    # Main demo set (35 C default)
    "pump_off",
    "abusive_command",
    "coordinated",
    "demo_hot_attack",
    # Additional physical faults & baseline
    "normal",
    "aged_cell",
    "hot_day",
    "pump_degradation",
    # Sensor cyberattacks
    "freeze_temp_a",
    "drift_temp_a",
    "spike_temp_a",
    "replay_temp_a",
    "dead_sensor_nan_temp_a",
    "dead_sensor_flatline_temp_a",
    "drift_current",
    "drift_voltage",
    "freeze_current",
]

DEMO_CACHE_DIR = Path(__file__).resolve().parent / "demo_cache"


# ---------------------------------------------------------------------------
# Helper Classes
# ---------------------------------------------------------------------------
def _is_valid_num(val: Any) -> bool:
    if val is None:
        return False
    try:
        f = float(val)
        return math.isfinite(f)
    except (ValueError, TypeError):
        return False


class PlainAlarmChargerBMS(ChargerBMS):
    """
    Plain threshold-based alarm controller.
    Trips and stops charging (0 A) if measured temperature exceeds 45.0 C.
    - use_both=True: MAIN baseline checks max(temp_cell_a, temp_cell_b) >= 45 C.
    - use_both=False: Weaker A-only baseline checks temp_cell_a >= 45 C.
    Operates strictly on measured sensor values without digital twin or gate.
    """
    def __init__(self, use_both: bool = True, trip_threshold_c: float = 45.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_both = use_both
        self.trip_threshold_c = float(trip_threshold_c)
        self.alarm_tripped: bool = False
        self.trip_time_s: Optional[float] = None

    def reset(self) -> None:
        super().reset()
        self.alarm_tripped = False
        self.trip_time_s = None

    def compute_commands(self, measured: Dict[str, Any]) -> Tuple[float, float]:
        if self.alarm_tripped:
            return 0.0, self.pump_idle_lpm

        ta = measured.get("temp_cell_a") if isinstance(measured, dict) else None
        tb = measured.get("temp_cell_b") if isinstance(measured, dict) else None

        t_candidates = []
        if _is_valid_num(ta):
            t_candidates.append(float(ta))
        if self.use_both and _is_valid_num(tb):
            t_candidates.append(float(tb))

        if t_candidates and max(t_candidates) >= self.trip_threshold_c:
            self.alarm_tripped = True
            # Let caller inspect trip time via current simulation state if needed
            return 0.0, self.pump_idle_lpm

        return super().compute_commands(measured)


class InstrumentedPhysicsGuard(PhysicsGuard):
    """
    PhysicsGuard observer that retains complete verdict history per second,
    including individual physics residuals, diagnosis, blamed channels, and reasons.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verdicts_history: List[Verdict] = []

    def update(self, *args, **kwargs) -> Verdict:
        v = super().update(*args, **kwargs)
        self.verdicts_history.append(v)
        return v


def clean_simulation_df(df: pd.DataFrame, latched: bool = False) -> pd.DataFrame:
    """Drop t=0 pre-step record and trailing stop-latch records."""
    d = df[df["time_s"] > 0].copy()
    if latched and len(d) > 2:
        d = d.iloc[:-2].copy()
    return d


# ---------------------------------------------------------------------------
# Simulation Execution Function
# ---------------------------------------------------------------------------
def run_full_simulation(
    scenario_name: str,
    ambient_c: Optional[float] = None,
    inlet_c: Optional[float] = None,
    seed: int = 42,
    start_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Execute a full comparative charging session for a scenario.
    Runs:
      1. Unprotected (no gate, no guard)
      2. Protected (SafetyGate + PhysicsGuard with digital twin)
      3. Plain-Alarm Baseline (max(A, B) >= 45 C -> trip)
      4. Plain-Alarm A-only Baseline (A >= 45 C -> trip)
      5. Matching Normal Baseline (for charge-time penalty denominator)
    """
    # 1. Build scenario
    build_kwargs = {}
    if start_s is not None and scenario_name != "normal" and scenario_name != "demo_hot_attack":
        build_kwargs["start_s"] = float(start_s)

    sc = build_scenario(scenario_name, **build_kwargs)

    # Ambient / inlet defaults
    if scenario_name in MAIN_DEMO_SCENARIOS:
        default_amb = 35.0
        default_inlet = 35.0
    else:
        default_amb = sc.ambient_c if sc.ambient_c is not None else params.T_AMBIENT_NOMINAL_C
        default_inlet = sc.inlet_c if sc.inlet_c is not None else params.T_COOLANT_INLET_NOMINAL_C

    eff_amb = float(ambient_c) if ambient_c is not None else default_amb
    eff_inlet = float(inlet_c) if inlet_c is not None else default_inlet

    sc.ambient_c = eff_amb
    sc.inlet_c = eff_inlet

    # Initial plant temperature
    init_temp = sc.plant_kwargs.get("initial_temp_c", eff_amb)
    plant_kwargs = dict(sc.plant_kwargs)
    plant_kwargs["initial_temp_c"] = float(init_temp)

    # -----------------------------------------------------------------------
    # 2. Unprotected Run
    # -----------------------------------------------------------------------
    p_unpr = Plant(perturb=False, sensor_seed=seed, **plant_kwargs)
    c_unpr = ChargerBMS()
    raw_df_u = run_session(
        p_unpr, c_unpr, scenario=sc, gate=None, guard=None, seed=seed,
        ambient_c=eff_amb, inlet_c=eff_inlet
    )
    df_u = clean_simulation_df(raw_df_u, getattr(c_unpr, "latched_stop", False))

    # -----------------------------------------------------------------------
    # 3. Protected Run
    # -----------------------------------------------------------------------
    p_prot = Plant(perturb=False, sensor_seed=seed, **plant_kwargs)
    c_prot = ChargerBMS()
    gate = SafetyGate()
    guard = InstrumentedPhysicsGuard()
    gate.set_guard(guard)

    raw_df_p = run_session(
        p_prot, c_prot, scenario=sc, gate=gate, guard=guard, seed=seed,
        ambient_c=eff_amb, inlet_c=eff_inlet
    )
    df_p = clean_simulation_df(raw_df_p, getattr(c_prot, "latched_stop", False))

    # Attach residuals and verdict details
    n_prot = len(df_p)
    history = guard.verdicts_history[:n_prot]

    def _get_res(idx: int, key: str) -> Optional[float]:
        if idx < len(history) and history[idx].residuals is not None:
            return history[idx].residuals.get(key)
        return None

    df_p["R1a"] = [_get_res(i, "R1a") for i in range(n_prot)]
    df_p["R1b"] = [_get_res(i, "R1b") for i in range(n_prot)]
    df_p["R2"] = [_get_res(i, "R2") for i in range(n_prot)]
    df_p["R3"] = [_get_res(i, "R3") for i in range(n_prot)]
    df_p["Rv"] = [_get_res(i, "Rv") for i in range(n_prot)]
    df_p["Rflow"] = [_get_res(i, "Rflow") for i in range(n_prot)]
    df_p["Rcur"] = [_get_res(i, "Rcur") for i in range(n_prot)]

    # -----------------------------------------------------------------------
    # 4. Plain-Alarm Baseline (Main: max(A, B) >= 45 C)
    # -----------------------------------------------------------------------
    p_plain = Plant(perturb=False, sensor_seed=seed, **plant_kwargs)
    c_plain = PlainAlarmChargerBMS(use_both=True)
    raw_df_plain = run_session(
        p_plain, c_plain, scenario=sc, gate=None, guard=None, seed=seed,
        ambient_c=eff_amb, inlet_c=eff_inlet
    )
    df_plain = clean_simulation_df(raw_df_plain, getattr(c_plain, "latched_stop", False))

    # -----------------------------------------------------------------------
    # 5. Plain-Alarm A-only Baseline (temp_cell_a >= 45 C)
    # -----------------------------------------------------------------------
    p_plain_a = Plant(perturb=False, sensor_seed=seed, **plant_kwargs)
    c_plain_a = PlainAlarmChargerBMS(use_both=False)
    raw_df_plain_a = run_session(
        p_plain_a, c_plain_a, scenario=sc, gate=None, guard=None, seed=seed,
        ambient_c=eff_amb, inlet_c=eff_inlet
    )
    df_plain_a = clean_simulation_df(raw_df_plain_a, getattr(c_plain_a, "latched_stop", False))

    # -----------------------------------------------------------------------
    # 6. Matching Normal Baseline Run
    # -----------------------------------------------------------------------
    p_norm = Plant(perturb=False, sensor_seed=seed, initial_soc=params.SOC_INITIAL, initial_temp_c=init_temp)
    c_norm = ChargerBMS()
    sc_norm = Scenario(
        name="normal", label_class="normal", label_type="normal",
        ambient_c=eff_amb, inlet_c=eff_inlet, plant_kwargs={"initial_temp_c": init_temp}
    )
    raw_df_norm = run_session(
        p_norm, c_norm, scenario=sc_norm, gate=None, guard=None, seed=seed,
        ambient_c=eff_amb, inlet_c=eff_inlet
    )
    df_norm = clean_simulation_df(raw_df_norm, getattr(c_norm, "latched_stop", False))
    dur_norm_base = float(df_norm["time_s"].iloc[-1]) if len(df_norm) > 0 else 1594.0

    # -----------------------------------------------------------------------
    # 7. Extract High-Level KPIs
    # -----------------------------------------------------------------------
    peak_t_u = float(df_u["true_battery_temp"].max()) if len(df_u) > 0 else 0.0
    peak_t_p = float(df_p["true_battery_temp"].max()) if len(df_p) > 0 else 0.0
    peak_t_plain = float(df_plain["true_battery_temp"].max()) if len(df_plain) > 0 else 0.0
    peak_t_plain_a = float(df_plain_a["true_battery_temp"].max()) if len(df_plain_a) > 0 else 0.0

    s45_u = int((df_u["true_battery_temp"] > 45.0).sum()) if len(df_u) > 0 else 0
    s45_p = int((df_p["true_battery_temp"] > 45.0).sum()) if len(df_p) > 0 else 0
    s45_plain = int((df_plain["true_battery_temp"] > 45.0).sum()) if len(df_plain) > 0 else 0
    s45_plain_a = int((df_plain_a["true_battery_temp"] > 45.0).sum()) if len(df_plain_a) > 0 else 0

    s50_u = int((df_u["true_battery_temp"] > 50.0).sum()) if len(df_u) > 0 else 0
    s50_p = int((df_p["true_battery_temp"] > 50.0).sum()) if len(df_p) > 0 else 0
    s50_plain = int((df_plain["true_battery_temp"] > 50.0).sum()) if len(df_plain) > 0 else 0
    s50_plain_a = int((df_plain_a["true_battery_temp"] > 50.0).sum()) if len(df_plain_a) > 0 else 0

    dur_u = float(df_u["time_s"].iloc[-1]) if len(df_u) > 0 else 0.0
    dur_p = float(df_p["time_s"].iloc[-1]) if len(df_p) > 0 else 0.0
    dur_plain = float(df_plain["time_s"].iloc[-1]) if len(df_plain) > 0 else 0.0
    dur_plain_a = float(df_plain_a["time_s"].iloc[-1]) if len(df_plain_a) > 0 else 0.0

    soc_u = float(df_u["true_soc"].iloc[-1]) if len(df_u) > 0 else 0.0
    soc_p = float(df_p["true_soc"].iloc[-1]) if len(df_p) > 0 else 0.0
    soc_plain = float(df_plain["true_soc"].iloc[-1]) if len(df_plain) > 0 else 0.0
    soc_plain_a = float(df_plain_a["true_soc"].iloc[-1]) if len(df_plain_a) > 0 else 0.0

    penalty_pct = round(((dur_p - dur_norm_base) / dur_norm_base) * 100.0, 2) if dur_norm_base > 0 else 0.0

    # Detection & Badge Info from Guard
    # Note: Badge is determined from guard.is_attack, NOT scenario label
    guard_is_attack_vals = [v.is_attack for v in history if v.is_attack is not None]
    if any(v is True for v in guard_is_attack_vals):
        overall_badge = "ATTACK"
    elif any(v is False for v in guard_is_attack_vals):
        overall_badge = "FAULT"
    else:
        overall_badge = "NORMAL"

    final_mode = str(df_p["gate_mode"].iloc[-1]) if "gate_mode" in df_p.columns and len(df_p) > 0 else "NOMINAL"

    kpis = {
        "scenario": scenario_name,
        "ambient_c": eff_amb,
        "inlet_c": eff_inlet,
        "seed": seed,
        "overall_badge": overall_badge,
        "final_mode": final_mode,
        "dur_norm_base": dur_norm_base,
        # Unprotected
        "peak_t_unpr": round(peak_t_u, 2),
        "s45_unpr": s45_u,
        "s50_unpr": s50_u,
        "dur_unpr": round(dur_u, 1),
        "soc_unpr": round(soc_u, 3),
        # Protected
        "peak_t_prot": round(peak_t_p, 2),
        "s45_prot": s45_p,
        "s50_prot": s50_p,
        "dur_prot": round(dur_p, 1),
        "soc_prot": round(soc_p, 3),
        "penalty_pct": penalty_pct,
        # Plain Alarm
        "plain_tripped": bool(c_plain.alarm_tripped),
        "peak_t_plain": round(peak_t_plain, 2),
        "s45_plain": s45_plain,
        "dur_plain": round(dur_plain, 1),
        "soc_plain": round(soc_plain, 3),
        # Plain Alarm A-only
        "plain_a_tripped": bool(c_plain_a.alarm_tripped),
        "peak_t_plain_a": round(peak_t_plain_a, 2),
        "s45_plain_a": s45_plain_a,
        "dur_plain_a": round(dur_plain_a, 1),
        "soc_plain_a": round(soc_plain_a, 3),
    }

    return {
        "df_unpr": df_u,
        "df_prot": df_p,
        "df_plain": df_plain,
        "df_plain_a": df_plain_a,
        "kpis": kpis,
        "gate_events": gate.event_log,
        "verdicts": history,
    }


# ---------------------------------------------------------------------------
# Cache Persistence
# ---------------------------------------------------------------------------
def save_to_cache(results: Dict[str, Any], scenario_name: str, cache_dir: Path = DEMO_CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results["df_unpr"].to_csv(cache_dir / f"{scenario_name}_unprotected.csv", index=False)
    results["df_prot"].to_csv(cache_dir / f"{scenario_name}_protected.csv", index=False)
    results["df_plain"].to_csv(cache_dir / f"{scenario_name}_plain_alarm.csv", index=False)
    results["df_plain_a"].to_csv(cache_dir / f"{scenario_name}_plain_alarm_a.csv", index=False)

    with open(cache_dir / f"{scenario_name}_meta.json", "w") as f:
        json.dump(results["kpis"], f, indent=2)


def load_from_cache(scenario_name: str, cache_dir: Path = DEMO_CACHE_DIR) -> Optional[Dict[str, Any]]:
    unpr_file = cache_dir / f"{scenario_name}_unprotected.csv"
    prot_file = cache_dir / f"{scenario_name}_protected.csv"
    plain_file = cache_dir / f"{scenario_name}_plain_alarm.csv"
    plain_a_file = cache_dir / f"{scenario_name}_plain_alarm_a.csv"
    meta_file = cache_dir / f"{scenario_name}_meta.json"

    if not (unpr_file.exists() and prot_file.exists() and meta_file.exists()):
        return None

    try:
        df_u = pd.read_csv(unpr_file)
        df_p = pd.read_csv(prot_file)
        df_plain = pd.read_csv(plain_file) if plain_file.exists() else df_u
        df_plain_a = pd.read_csv(plain_a_file) if plain_a_file.exists() else df_u

        with open(meta_file, "r") as f:
            kpis = json.load(f)

        return {
            "df_unpr": df_u,
            "df_prot": df_p,
            "df_plain": df_plain,
            "df_plain_a": df_plain_a,
            "kpis": kpis,
            "gate_events": [],
            "verdicts": [],
        }
    except Exception as e:
        print(f"Error reading cache for {scenario_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Explain This Moment Text Generator
# ---------------------------------------------------------------------------
def explain_moment(
    t_sec: float,
    row_prot: pd.Series,
    verdict_at_t: Optional[Verdict] = None,
    recent_gate_event: Optional[Dict[str, Any]] = None
) -> str:
    """
    Constructs a plain-English physical explanation sentence at playback time t.
    Example:
      't=312 s: temperature sensor A reads 25.0 °C but the physics model expects 34.1 °C
       (flow 10.0 L/min, current 128.0 A). Verdict: SENSOR_FROZEN. Gate switched to sensor B.'
    """
    t_val = int(round(t_sec))
    ta_raw = row_prot.get("temp_cell_a")
    tb_raw = row_prot.get("temp_cell_b")
    ta_str = f"{float(ta_raw):.1f} °C" if _is_valid_num(ta_raw) else "sensor dead (NaN)"
    tb_str = f"{float(tb_raw):.1f} °C" if _is_valid_num(tb_raw) else "sensor dead (NaN)"

    trusted_t = float(row_prot.get("trusted_temp", 25.0))
    flow = float(row_prot.get("pump_cmd_lpm", row_prot.get("pump_flow_lpm", 10.0)))
    current = float(row_prot.get("applied_current_a", row_prot.get("pack_current", 128.0)))
    mode = str(row_prot.get("gate_mode", "NOMINAL"))
    diagnosis = str(row_prot.get("diagnosis", "NORMAL"))
    intervened = bool(row_prot.get("intervened", False))

    reasons = []
    if verdict_at_t is not None and getattr(verdict_at_t, "reasons", None):
        reasons = verdict_at_t.reasons
    elif "reasons" in row_prot and isinstance(row_prot["reasons"], list):
        reasons = row_prot["reasons"]

    # Reason synthesis
    if diagnosis == "NORMAL" and not intervened:
        return (
            f"t={t_val} s: Thermal physics in equilibrium. Sensors read A={ta_str}, B={tb_str}; "
            f"digital twin temperature is {trusted_t:.1f} °C. Gate is in {mode} mode, delivering "
            f"{current:.1f} A with {flow:.1f} L/min cooling flow."
        )

    action_text = ""
    if recent_gate_event:
        action_text = f" Gate action: {recent_gate_event.get('reason', recent_gate_event.get('action', 'intervened'))}."
    elif intervened:
        action_text = f" Gate mode: {mode} (limiting current to {current:.1f} A, pump flow {flow:.1f} L/min)."

    primary_reason = reasons[0] if reasons else f"Observed physical mismatch in diagnosis category '{diagnosis}'."

    # Format specific physical details
    if "freeze" in diagnosis.lower() or "frozen" in diagnosis.lower():
        return (
            f"t={t_val} s: Sensor freeze detected. Temperature sensor A reads {ta_str} while the digital twin "
            f"predicts {trusted_t:.1f} °C under active charging ({current:.1f} A, {flow:.1f} L/min). "
            f"Verdict: {diagnosis}.{action_text}"
        )
    elif "nan" in ta_str.lower():
        return (
            f"t={t_val} s: Sensor A hardware failure: reading is NaN. "
            f"Verdict: SENSOR_HW_FAULT. Gate switched to trusted sensor B ({tb_str}) in {mode} mode."
        )
    elif "abusive" in diagnosis.lower() or "unsafe" in diagnosis.lower():
        return (
            f"t={t_val} s: Requested charge command exceeds safe operating limits. "
            f"Verdict: {diagnosis}. Gate clamped applied current to {current:.1f} A to prevent battery overheating."
        )
    elif "actuator" in diagnosis.lower() or flow < 3.0:
        return (
            f"t={t_val} s: Cooling pump flow ({flow:.1f} L/min) severely degraded or zeroed while charging. "
            f"Verdict: {diagnosis}. Gate escalated to {mode} mode, derating current to {current:.1f} A."
        )
    else:
        return (
            f"t={t_val} s: {primary_reason}. Sensors: A={ta_str}, B={tb_str}, trusted model={trusted_t:.1f} °C. "
            f"Verdict: {diagnosis}.{action_text}"
        )


def read_live_esp32_sensor(port: Optional[str] = None, baudrate: int = 115200) -> Optional[float]:
    """
    Attempts to read one temperature value from an ESP32/Arduino over serial.
    Never blocks the UI: sets timeout=0.05s.
    Falls back silently to None if no device is connected.
    """
    try:
        import serial
        import serial.tools.list_ports
        available_ports = [p.device for p in serial.tools.list_ports.comports()]
        if not available_ports and port is None:
            return None
        target_port = port if port is not None else available_ports[0]
        with serial.Serial(target_port, baudrate=baudrate, timeout=0.05) as ser:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                for token in line.replace(",", " ").split():
                    try:
                        val = float(token)
                        if 0.0 <= val <= 100.0:
                            return val
                    except ValueError:
                        pass
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Phase 6.5 Attack Lab Simulation Pipeline
# ---------------------------------------------------------------------------
_CUSTOM_SIM_CACHE: Dict[str, Dict[str, Any]] = {}


def compute_param_hash(
    specs: List[Dict[str, Any]],
    ambient_c: float,
    inlet_c: float,
    seed: int,
    plant_variation: bool,
) -> str:
    """Compute a deterministic SHA-256 hash from attack parameters for caching."""
    clamped = [validate_and_clamp_spec(s) for s in specs if isinstance(s, dict)]
    canonical_dict = {
        "specs": clamped,
        "ambient_c": round(float(ambient_c), 2),
        "inlet_c": round(float(inlet_c), 2),
        "seed": int(seed),
        "plant_variation": bool(plant_variation),
    }
    encoded = json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def run_custom_attack_simulation(
    attack_specs: List[Dict[str, Any]],
    ambient_c: float = 25.0,
    inlet_c: float = 25.0,
    seed: int = 42,
    plant_variation: bool = False,
    timeout_wall_clock_s: float = 30.0,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 6.5 Attack Lab comparative session:
      1. Unprotected (compromised sensors/commands, no guard/gate)
      2. Protected (PhysicsGuard + SafetyGate + Digital Twin)
      3. Plain-Alarm Baseline (trip on max(A, B) >= 45.0 C)
      4. Matching Normal Baseline (same seed and conditions, unattacked)

    Features:
      - Validates and clamps inputs.
      - Cached by parameter hash.
      - Max 30 s wall-clock timeout safety.
      - Strict honesty rules for detection and physical effect.
    """
    t_wall_start = time.time()

    def _check_timeout():
        if (time.time() - t_wall_start) > timeout_wall_clock_s:
            raise TimeoutError(f"Simulation timed out after exceeding {timeout_wall_clock_s:.0f} s wall-clock time limit.")

    # 1. Validate specs & parameter hash
    clamped_specs = [validate_and_clamp_spec(s) for s in attack_specs if isinstance(s, dict)]
    p_hash = compute_param_hash(clamped_specs, ambient_c, inlet_c, seed, plant_variation)

    if p_hash in _CUSTOM_SIM_CACHE:
        if progress_callback:
            progress_callback(1.0, "Loaded from parameter cache.")
        return _CUSTOM_SIM_CACHE[p_hash]

    eff_amb = float(ambient_c)
    eff_inlet = float(inlet_c)
    init_temp = eff_amb
    plant_kwargs = {"initial_temp_c": float(init_temp)}

    # Build custom scenario
    sc = build_custom_scenario(
        clamped_specs,
        name="custom_attack",
        ambient_c=eff_amb,
        inlet_c=eff_inlet,
        plant_kwargs=plant_kwargs
    )

    try:
        # -------------------------------------------------------------------
        # Run 1: Unprotected Run
        # -------------------------------------------------------------------
        if progress_callback:
            progress_callback(0.15, "Simulating car under attack (Unprotected)...")
        _check_timeout()

        p_unpr = Plant(perturb=plant_variation, seed=seed, sensor_seed=seed, **plant_kwargs)
        c_unpr = ChargerBMS()
        raw_df_u = run_session(
            p_unpr, c_unpr, scenario=sc, gate=None, guard=None, seed=seed,
            ambient_c=eff_amb, inlet_c=eff_inlet, max_duration_s=7200.0
        )
        df_u = clean_simulation_df(raw_df_u, getattr(c_unpr, "latched_stop", False))
        _check_timeout()

        # -------------------------------------------------------------------
        # Run 2: Protected Run (PhysicsGuard + SafetyGate)
        # -------------------------------------------------------------------
        if progress_callback:
            progress_callback(0.40, "Simulating with PhysicsGuard & SafetyGate...")
        _check_timeout()

        p_prot = Plant(perturb=plant_variation, seed=seed, sensor_seed=seed, **plant_kwargs)
        c_prot = ChargerBMS()
        gate = SafetyGate()
        guard = InstrumentedPhysicsGuard()
        gate.set_guard(guard)

        raw_df_p = run_session(
            p_prot, c_prot, scenario=sc, gate=gate, guard=guard, seed=seed,
            ambient_c=eff_amb, inlet_c=eff_inlet, max_duration_s=7200.0
        )
        df_p = clean_simulation_df(raw_df_p, getattr(c_prot, "latched_stop", False))
        _check_timeout()

        # Attach residuals and verdict details
        n_prot = len(df_p)
        history = guard.verdicts_history[:n_prot]

        def _get_res(idx: int, key: str) -> Optional[float]:
            if idx < len(history) and history[idx].residuals is not None:
                return history[idx].residuals.get(key)
            return None

        df_p["R1a"] = [_get_res(i, "R1a") for i in range(n_prot)]
        df_p["R1b"] = [_get_res(i, "R1b") for i in range(n_prot)]
        df_p["R2"] = [_get_res(i, "R2") for i in range(n_prot)]
        df_p["R3"] = [_get_res(i, "R3") for i in range(n_prot)]
        df_p["Rv"] = [_get_res(i, "Rv") for i in range(n_prot)]
        df_p["Rflow"] = [_get_res(i, "Rflow") for i in range(n_prot)]
        df_p["Rcur"] = [_get_res(i, "Rcur") for i in range(n_prot)]

        # -------------------------------------------------------------------
        # Run 3: Plain-Alarm Baseline (max(A, B) >= 45°C)
        # -------------------------------------------------------------------
        if progress_callback:
            progress_callback(0.70, "Simulating standard 45°C thermal alarm baseline...")
        _check_timeout()

        p_plain = Plant(perturb=plant_variation, seed=seed, sensor_seed=seed, **plant_kwargs)
        c_plain = PlainAlarmChargerBMS(use_both=True)
        raw_df_plain = run_session(
            p_plain, c_plain, scenario=sc, gate=None, guard=None, seed=seed,
            ambient_c=eff_amb, inlet_c=eff_inlet, max_duration_s=7200.0
        )
        df_plain = clean_simulation_df(raw_df_plain, getattr(c_plain, "latched_stop", False))
        _check_timeout()

        # -------------------------------------------------------------------
        # Run 4: Matching Normal Baseline Run (For honest charge time & physics delta)
        # -------------------------------------------------------------------
        if progress_callback:
            progress_callback(0.90, "Evaluating matching normal baseline...")
        _check_timeout()

        p_norm = Plant(perturb=plant_variation, seed=seed, sensor_seed=seed, initial_soc=params.SOC_INITIAL, initial_temp_c=init_temp)
        c_norm = ChargerBMS()
        sc_norm = Scenario(
            name="normal", label_class="normal", label_type="normal",
            ambient_c=eff_amb, inlet_c=eff_inlet, plant_kwargs={"initial_temp_c": init_temp}
        )
        raw_df_norm = run_session(
            p_norm, c_norm, scenario=sc_norm, gate=None, guard=None, seed=seed,
            ambient_c=eff_amb, inlet_c=eff_inlet, max_duration_s=7200.0
        )
        df_norm = clean_simulation_df(raw_df_norm, getattr(c_norm, "latched_stop", False))
        dur_norm_base = float(df_norm["time_s"].iloc[-1]) if len(df_norm) > 0 else 1594.0
        peak_t_norm = float(df_norm["true_battery_temp"].max()) if len(df_norm) > 0 else 0.0
        soc_norm = float(df_norm["true_soc"].iloc[-1]) if len(df_norm) > 0 else 0.0

        # -------------------------------------------------------------------
        # Extract KPIs & Honesty Metrics
        # -------------------------------------------------------------------
        peak_t_u = float(df_u["true_battery_temp"].max()) if len(df_u) > 0 else 0.0
        peak_t_p = float(df_p["true_battery_temp"].max()) if len(df_p) > 0 else 0.0
        peak_t_plain = float(df_plain["true_battery_temp"].max()) if len(df_plain) > 0 else 0.0

        s45_u = int((df_u["true_battery_temp"] > 45.0).sum()) if len(df_u) > 0 else 0
        s45_p = int((df_p["true_battery_temp"] > 45.0).sum()) if len(df_p) > 0 else 0
        s45_plain = int((df_plain["true_battery_temp"] > 45.0).sum()) if len(df_plain) > 0 else 0

        s50_u = int((df_u["true_battery_temp"] > 50.0).sum()) if len(df_u) > 0 else 0
        s50_p = int((df_p["true_battery_temp"] > 50.0).sum()) if len(df_p) > 0 else 0
        s50_plain = int((df_plain["true_battery_temp"] > 50.0).sum()) if len(df_plain) > 0 else 0

        dur_u = float(df_u["time_s"].iloc[-1]) if len(df_u) > 0 else 0.0
        dur_p = float(df_p["time_s"].iloc[-1]) if len(df_p) > 0 else 0.0
        dur_plain = float(df_plain["time_s"].iloc[-1]) if len(df_plain) > 0 else 0.0

        soc_u = float(df_u["true_soc"].iloc[-1]) if len(df_u) > 0 else 0.0
        soc_p = float(df_p["true_soc"].iloc[-1]) if len(df_p) > 0 else 0.0
        soc_plain = float(df_plain["true_soc"].iloc[-1]) if len(df_plain) > 0 else 0.0

        charge_completed_prot = bool(soc_p >= 0.795)
        charge_completed_unpr = bool(soc_u >= 0.795)

        penalty_pct = round(((dur_p - dur_norm_base) / dur_norm_base) * 100.0, 2) if dur_norm_base > 0 else 0.0

        # Physical Effect Check (Honesty Rule):
        # Unprotected vs matching normal run
        diff_peak_t = abs(peak_t_u - peak_t_norm)
        diff_dur = abs(dur_u - dur_norm_base)
        diff_soc = abs(soc_u - soc_norm)
        has_physical_effect = (diff_peak_t > 0.1) or (diff_dur > 2.0) or (diff_soc > 0.005)

        # Detection & Latency Check (Honesty Rule):
        atk_start = float(sc.start_s) if sc.start_s is not None else 0.0
        first_alarm_time = None
        first_alarm_verdict = None

        for v in history:
            if (v.alarm_level and str(v.alarm_level).lower() != "none") or (v.diagnosis and v.diagnosis != "NORMAL") or (v.is_attack is not None):
                first_alarm_time = float(v.t)
                first_alarm_verdict = v
                break

        if first_alarm_time is not None:
            detected = True
            detected_label = "YES"
            detection_latency_s = max(0.0, round(first_alarm_time - atk_start, 1))
        else:
            detected = False
            detected_label = "NOT DETECTED"
            detection_latency_s = None

        # Diagnosis badge & blamed signals
        guard_is_attack_vals = [v.is_attack for v in history if v.is_attack is not None]
        if any(v is True for v in guard_is_attack_vals):
            overall_badge = "ATTACK"
        elif any(v is False for v in guard_is_attack_vals):
            overall_badge = "FAULT"
        else:
            overall_badge = "NORMAL"

        blamed_signals = sorted(list(set(
            sig for v in history if v.blamed_signals for sig in v.blamed_signals
        )))
        diagnoses_seen = sorted(list(set(
            v.diagnosis for v in history if v.diagnosis and v.diagnosis != "NORMAL"
        )))
        final_mode = str(df_p["gate_mode"].iloc[-1]) if "gate_mode" in df_p.columns and len(df_p) > 0 else "NOMINAL"

        # Card status color determination
        if peak_t_p > 45.0 or (not charge_completed_prot and final_mode == "SHUTDOWN"):
            card_status = "red"
        elif peak_t_p <= 45.0 and (final_mode in ("DEGRADED", "LIMP_HOME") or penalty_pct > 15.0 or not detected):
            card_status = "amber"
        else:
            card_status = "green"

        # Synthesize plain-English timeline
        timeline_events: List[str] = []
        timeline_events.append(
            f"**t = 0 s**: Fast-charging session initiated at {eff_amb:.1f} °C ambient, {eff_inlet:.1f} °C inlet (seed {seed})."
        )
        for s in clamped_specs:
            dur_str = f" for {s['duration_s']:.0f} s" if s.get("duration_s") else " until session end"
            timeline_events.append(
                f"**t = {s['start_s']:.0f} s**: Cyberattack activated on **{s['target']}** (type: `{s['type']}`, magnitude: {s['magnitude']:.1f}{dur_str})."
            )

        if detected and first_alarm_verdict is not None:
            reasons_str = "; ".join(first_alarm_verdict.reasons[:2]) if first_alarm_verdict.reasons else "Physical anomaly registered."
            timeline_events.append(
                f"**t = {first_alarm_time:.0f} s**: PhysicsGuard detected anomaly (+{detection_latency_s:.1f} s latency). Verdict: **{first_alarm_verdict.diagnosis}** on blamed channels {first_alarm_verdict.blamed_signals}. Details: {reasons_str}"
            )
        elif not detected:
            timeline_events.append(
                "**No alarm raised**: PhysicsGuard observed residuals within nominal noise thresholds across all channels."
            )

        for ev in gate.event_log:
            ev_t = ev.get("t", 0.0)
            reason = ev.get("reason", ev.get("action", "intervened"))
            mode_to = ev.get("mode_to", "")
            mode_str = f" to {mode_to}" if mode_to else ""
            timeline_events.append(
                f"**t = {ev_t:.0f} s**: SafetyGate intervened{mode_str}: {reason}."
            )

        if not has_physical_effect:
            timeline_events.append(
                "**Note**: Attack produced NO physical deviation in the battery pack (BMS controller does not close loop on this sensor channel)."
            )
        else:
            timeline_events.append(
                f"**Session End**: Unprotected pack peaked at **{peak_t_u:.1f} °C** ({s50_u} s > 50 °C). PhysicsGuard held true pack temperature to **{peak_t_p:.1f} °C**."
            )

        kpis = {
            "param_hash": p_hash,
            "attack_specs": clamped_specs,
            "ambient_c": eff_amb,
            "inlet_c": eff_inlet,
            "seed": seed,
            "plant_variation": plant_variation,
            # Honesty check outputs
            "detected": detected,
            "detected_label": detected_label,
            "detection_latency_s": detection_latency_s,
            "has_physical_effect": has_physical_effect,
            "physical_effect_label": "NO PHYSICAL EFFECT IN THIS MODEL" if not has_physical_effect else "PHYSICAL EFFECT CONFIRMED",
            "overall_badge": overall_badge,
            "blamed_signals": blamed_signals,
            "diagnoses_seen": diagnoses_seen,
            "card_status": card_status,
            # Unprotected KPIs
            "peak_t_unpr": round(peak_t_u, 2),
            "s45_unpr": s45_u,
            "s50_unpr": s50_u,
            "dur_unpr": round(dur_u, 1),
            "soc_unpr": round(soc_u, 3),
            "completed_unpr": charge_completed_unpr,
            # Protected KPIs
            "peak_t_prot": round(peak_t_p, 2),
            "s45_prot": s45_p,
            "s50_prot": s50_p,
            "dur_prot": round(dur_p, 1),
            "soc_prot": round(soc_p, 3),
            "completed_prot": charge_completed_prot,
            "dur_norm_base": dur_norm_base,
            "penalty_pct": penalty_pct,
            "final_mode": final_mode,
            # Plain Alarm KPIs
            "plain_tripped": bool(c_plain.alarm_tripped),
            "peak_t_plain": round(peak_t_plain, 2),
            "s45_plain": s45_plain,
            "s50_plain": s50_plain,
            "dur_plain": round(dur_plain, 1),
            "soc_plain": round(soc_plain, 3),
        }

        result = {
            "df_unpr": df_u,
            "df_prot": df_p,
            "df_plain": df_plain,
            "df_norm": df_norm,
            "kpis": kpis,
            "gate_events": gate.event_log,
            "verdicts": history,
            "timeline": timeline_events,
            "param_hash": p_hash,
        }

        # Cache by parameter hash
        _CUSTOM_SIM_CACHE[p_hash] = result

        if progress_callback:
            progress_callback(1.0, "Simulation complete.")

        return result

    except TimeoutError as te:
        return {"error": str(te), "param_hash": p_hash}
    except Exception as e:
        return {"error": f"Simulation encountered an unexpected condition: {e}", "param_hash": p_hash}


