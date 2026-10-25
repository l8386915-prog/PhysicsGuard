"""
sim_engine.py - Core Simulation Engine & Data Pipeline for PhysicsGuard Dashboard
================================================================================
Provides unified simulation execution, plain-alarm baselines, residual logging,
caching, and moment-by-moment physical explanation synthesis.
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

from physicsguard import params
from physicsguard.plant import Plant
from physicsguard.controller import ChargerBMS
from physicsguard.scenarios import build_scenario, Scenario
from physicsguard.simulate import run_session
from physicsguard.guard import PhysicsGuard, Verdict
from physicsguard.mitigation import SafetyGate

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

