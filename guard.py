"""
PhysicsGuard - Physics-Based Anomaly Detector (Observe-Only Mode)
=================================================================
This module implements the PhysicsGuard class which performs physics-based
anomaly detection on a real-time stream of sensor measurements and delivered
commands.

IMPORTANT ISOLATION RULES:
  - The guard NEVER reads true_* values.
  - The guard NEVER reads scenario labels.
  - The guard NEVER accesses the plant object.
  - The guard maintains its own private Digital Twin.
  - The guard uses only the DELIVERED commands and MEASURED sensors.

Verdict dataclass is returned every second with:
  - t             : current simulation time [s]
  - diagnosis     : diagnosis class string
  - alarm_level   : "none" | "warning" | "critical"
  - is_attack     : True | False | None
  - blamed_signals: list of signal names flagged
  - reasons       : list of plain-English explanation strings with numbers
  - residuals     : dict of residual values computed this step

Diagnosis classes:
  NORMAL, SENSOR_HW_FAULT, SENSOR_FROZEN, SENSOR_SPOOF,
  ACTUATOR_FAULT, COMMAND_RESPONSE_MISMATCH, UNSAFE_COMMAND
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import copy
import numpy as np

from . import params
from .twin import Twin


# ---------------------------------------------------------------------------
# Physical validity ranges for each sensor channel
# ---------------------------------------------------------------------------
SIGNAL_RANGES: Dict[str, Tuple[float, float]] = {
    "temp_cell_a":          (-20.0, 90.0),
    "temp_cell_b":          (-20.0, 90.0),
    "coolant_temp_in":      (-20.0, 90.0),
    "coolant_temp_out":     (-20.0, 90.0),
    "ambient_temp":         (-20.0, 90.0),
    "pack_voltage":         (200.0, 450.0),
    "pack_current":         (-10.0, 300.0),
    "pump_flow_lpm":        (-1.0,  30.0),
    "soc_estimate":         (0.0,   1.0),
    "charger_power_cmd_kw": (-1.0,  200.0),
}

# Signals considered "noisy" enough to apply the flatline test.
# Constant-by-design signals (like coolant_temp_in from a stable chiller)
# are excluded to avoid false alarms.
NOISY_SIGNALS = {
    "temp_cell_a", "temp_cell_b", "pack_voltage",
    "pack_current", "pump_flow_lpm",
}

FLATLINE_WINDOW = 10          # consecutive identical samples to flag
PERSISTENCE_SECS = 5          # seconds a soft check must hold before alerting
CALIB_WINDOW_S = 60.0         # seconds for learning the R3 offset
ACTUATOR_GRACE_S = 15.0       # ignore Rflow check for this many seconds after cmd change
VOLTAGE_IGNORE_S = 120.0      # seconds before C5 voltage CUSUM is active
FLOW_THRESHOLD_FOR_C4 = 3.0   # min measured flow to run coolant energy balance check
LOOKAHEAD_S = 180             # seconds to simulate ahead for C7c
MAX_TEMP_LOOKAHEAD_C = 45.0   # flag if look-ahead predicts pack temp above this

# Large-jump threshold for instant C2 alarm (bypass persistence)
C2_JUMP_INSTANT_THRESHOLD_C = 5.0

# Default thresholds (overridden by calibrate.py via thresholds.json at runtime)
_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "R1_threshold_c":     3.0,
    "R1_diff_threshold_c": 2.0,
    "R2_threshold_kj":    1.5,
    "R3_threshold_c":     2.5,
    "Rv_threshold_v":     8.0,
    "Rflow_threshold":    3.5,
    "Rcur_threshold":     10.0,
    "cusum_k_R3":         0.5,
    "cusum_h_R3":         15.0,
    "cusum_k_Rv":         1.0,
    "cusum_h_Rv":         20.0,
    "cusum_k_Rcur":       0.5,
    "cusum_h_Rcur":       5.0,
}


def _load_thresholds() -> Dict[str, float]:
    """Load calibrated thresholds from outputs/thresholds.json if available."""
    import json
    from pathlib import Path
    json_path = Path(__file__).resolve().parent.parent / "outputs" / "thresholds.json"
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            out = dict(_DEFAULT_THRESHOLDS)
            out.update(data)
            return out
        except Exception:
            pass
    return dict(_DEFAULT_THRESHOLDS)


@dataclass
class Verdict:
    """
    Per-step output from PhysicsGuard.update().

    Fields
    ------
    t               : float  - simulation time [s]
    diagnosis       : str    - one of the diagnosis class strings
    alarm_level     : str    - "none" | "warning" | "critical"
    is_attack       : bool or None - True=attack, False=fault/hw, None=undetermined
    blamed_signals  : list[str]   - sensor or channel names implicated
    reasons         : list[str]   - human-readable explanation strings
    residuals       : dict        - computed residual values this step
    """
    t: float
    diagnosis: str = "NORMAL"
    alarm_level: str = "none"
    is_attack: Optional[bool] = None
    blamed_signals: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    residuals: Dict[str, float] = field(default_factory=dict)


class _CUSUM:
    """
    Two-sided CUSUM detector.
    Tracks cumulative sums S+ and S- of (residual - k) and (-residual - k).
    Alarm fires when either exceeds threshold h.
    """
    def __init__(self, k: float, h: float):
        self.k = k
        self.h = h
        self.s_pos = 0.0
        self.s_neg = 0.0

    def update(self, residual: float) -> bool:
        """Update CUSUM and return True if alarm threshold exceeded."""
        self.s_pos = max(0.0, self.s_pos + residual - self.k)
        self.s_neg = max(0.0, self.s_neg - residual - self.k)
        return (self.s_pos >= self.h) or (self.s_neg >= self.h)

    def reset(self):
        self.s_pos = 0.0
        self.s_neg = 0.0


def run_lookahead(
    twin_state: Twin,
    i_pack: float,
    pump_lpm: float,
    T_amb: float,
    T_in: float,
    steps: int = LOOKAHEAD_S
) -> float:
    """
    Simulate a copy of the twin forward in time holding the delivered commands.
    Returns predicted battery temperature in Celsius. Pure physical prediction helper.
    """
    T_batt = float(twin_state.T_batt)
    vp = float(twin_state.vp)
    flow = float(twin_state.pump_flow_lpm)
    target_flow = float(pump_lpm)
    tau_pump = float(twin_state.tau_pump)

    i_cell = float(i_pack) / twin_state.n_parallel
    i_cell_sq_r0 = (i_cell ** 2) * twin_state.r0
    n_cells = twin_state.n_cells
    inv_rp_cp = 1.0 / (twin_state.rp * twin_state.cp)
    inv_cp = 1.0 / twin_state.cp
    inv_rp = 1.0 / twin_state.rp
    c_pack_inv = 1.0 / twin_state.c_pack
    ua_amb = twin_state.ua_ambient
    ua_cold = twin_state.ua_cold_plate
    rho_cp = twin_state.rho_coolant * twin_state.cp_coolant / 60000.0

    for _ in range(steps):
        if flow != target_flow:
            flow = max(0.0, min(50.0, flow + (target_flow - flow) / tau_pump))
        vp += (-vp * inv_rp_cp + i_cell * inv_cp)
        q_heat = n_cells * (i_cell_sq_r0 + (vp ** 2) * inv_rp)
        if flow < 1.0:
            q_cool = 0.0
        else:
            c_dot = rho_cp * flow
            ntu = ua_cold / c_dot
            q_cool = (1.0 - math.exp(-ntu)) * c_dot * (T_batt - T_in)
        q_amb = ua_amb * (T_batt - T_amb)
        T_batt += (q_heat - q_cool - q_amb) * c_pack_inv

    return float(T_batt)


class PhysicsGuard:
    """
    Physics-Based Anomaly Detector for EV DC Fast-Charging sessions.

    Maintains an internal Digital Twin driven by DELIVERED commands and
    MEASURED environmental inputs. Seven check families produce Verdict
    objects indicating detected anomalies.

    The guard NEVER accesses the plant object or true_* signals.
    """

    def __init__(self):
        # Load calibrated thresholds (fall back to defaults if not available)
        self._thr = _load_thresholds()

        # Internal Twin (observer mode - driven by delivered commands)
        self._twin = Twin(
            initial_soc=params.SOC_INITIAL,
            initial_temp_c=params.T_AMBIENT_NOMINAL_C
        )

        # ------------------------------------------------------------------
        # Last valid values (used as fallback when a sensor is invalid)
        # ------------------------------------------------------------------
        self._last_valid: Dict[str, float] = {
            "pack_current":   0.0,
            "pump_flow_lpm":  0.0,
            "ambient_temp":   params.T_AMBIENT_NOMINAL_C,
            "coolant_temp_in": params.T_COOLANT_INLET_NOMINAL_C,
            "temp_cell_a":    params.T_AMBIENT_NOMINAL_C,
            "temp_cell_b":    params.T_AMBIENT_NOMINAL_C,
            "pack_voltage":   params.PACK_N_SERIES * 3.6,
            "coolant_temp_out": params.T_COOLANT_INLET_NOMINAL_C,
        }

        # ------------------------------------------------------------------
        # Flatline detection: sliding window of recent values per signal
        # ------------------------------------------------------------------
        self._history: Dict[str, List[float]] = {
            sig: [] for sig in NOISY_SIGNALS
        }

        # ------------------------------------------------------------------
        # C2 Redundancy: learn R3 offset (temp_cell_a - temp_cell_b) in
        # the first CALIB_WINDOW_S seconds, then monitor with CUSUM
        # ------------------------------------------------------------------
        self._r3_baseline_vals: List[float] = []
        self._r3_offset: Optional[float] = None       # learned fixed offset
        self._cusum_r3 = _CUSUM(
            k=self._thr.get("cusum_k_R3", 0.3),
            h=self._thr.get("cusum_h_R3", 3.0)
        )

        # ------------------------------------------------------------------
        # C5 Electrical voltage CUSUM
        # ------------------------------------------------------------------
        self._cusum_rv = _CUSUM(
            k=self._thr.get("cusum_k_Rv", 0.5),
            h=self._thr.get("cusum_h_Rv", 5.0)
        )

        # ------------------------------------------------------------------
        # C6 Electrical current CUSUM
        # ------------------------------------------------------------------
        self._cusum_rcur = _CUSUM(
            k=self._thr.get("cusum_k_Rcur", 0.5),
            h=self._thr.get("cusum_h_Rcur", 5.0)
        )

        # ------------------------------------------------------------------
        # Persistence counters: each soft check counts consecutive seconds
        # before raising an alarm
        # ------------------------------------------------------------------
        # Keys: check identifier string -> consecutive seconds flagged
        self._persist: Dict[str, int] = {}

        # ------------------------------------------------------------------
        # Actuator echo & model tracking
        # ------------------------------------------------------------------
        self._last_pump_cmd: float = 0.0
        self._last_pump_cmd_change_t: float = -999.0
        self._last_curr_cmd: float = 0.0
        self._cmd_pump_flow: float = 0.0  # first-order lag model of delivered pump cmd

        # ------------------------------------------------------------------
        # Recent state for public trust & diagnostics
        # ------------------------------------------------------------------
        self._last_valid_flags: Dict[str, bool] = {}
        self._last_active_findings: List[Dict] = []
        self._last_resolved: Dict[str, float] = {}
        self._last_verdict: Optional[Verdict] = None

        # ------------------------------------------------------------------
        # Internal time tracking
        # ------------------------------------------------------------------
        self._t_prev: float = -1.0
        self._first_step: bool = True  # True until first update() call completes

    # ======================================================================
    # PUBLIC API
    # ======================================================================

    def update(
        self,
        t: float,
        measured: Dict[str, Any],
        delivered_cmds: Tuple[float, float],
        requested_cmds: Optional[Tuple[float, float]] = None
    ) -> Verdict:
        """
        Advance the guard by one second and return a Verdict.

        Parameters
        ----------
        t : float
            Current simulation time in seconds.
        measured : dict
            Sensor measurements as delivered to the controller
            (AFTER any sensor tamper injection).
        delivered_cmds : (float, float)
            (current_cmd_a, pump_cmd_lpm) as delivered to the plant
            (AFTER any command tamper injection).
        requested_cmds : (float, float), optional
            (req_current_a, req_pump_lpm) as requested by controller
            (BEFORE any command tamper injection). Used for C7 safety checks.

        Returns
        -------
        Verdict
            Anomaly verdict for this time step.
        """
        dt = max(1.0, t - self._t_prev) if self._t_prev >= 0 else 1.0
        self._t_prev = t

        current_cmd_a, pump_cmd_lpm = delivered_cmds
        req_current_a, req_pump_lpm = requested_cmds if requested_cmds is not None else delivered_cmds

        # Update first-order lag model of delivered pump command (tau = params.TAU_PUMP_SECONDS)
        tau_pump = params.TAU_PUMP_SECONDS
        if self._first_step:
            self._cmd_pump_flow = float(pump_cmd_lpm)
        else:
            self._cmd_pump_flow = float(np.clip(
                self._cmd_pump_flow + (pump_cmd_lpm - self._cmd_pump_flow) / tau_pump * dt,
                0.0, 50.0
            ))

        # Track pump command changes for actuator grace period
        if abs(pump_cmd_lpm - self._last_pump_cmd) > 0.1:
            self._last_pump_cmd_change_t = t
            self._last_pump_cmd = pump_cmd_lpm

        # ------------------------------------------------------------------
        # Step 0: Resolve valid sensor values (with fallback to last-valid)
        # ------------------------------------------------------------------
        valid_flags, resolved = self._resolve_signals(measured)

        # ------------------------------------------------------------------
        # Step 0b: On the very first step, initialize the twin temperature
        # from the measured sensor values so that the guard starts in sync
        # with the actual session (e.g., hot_day starts at 35C not 25C).
        # ------------------------------------------------------------------
        if self._first_step:
            self._first_step = False
            # Estimate initial pack temperature from valid temp sensors
            init_temps = []
            if valid_flags.get("temp_cell_a", False):
                init_temps.append(resolved["temp_cell_a"])
            if valid_flags.get("temp_cell_b", False):
                init_temps.append(resolved["temp_cell_b"] - 0.25)
            if init_temps:
                init_T = float(np.mean(init_temps))
                self._twin.T_batt = init_T
                self._twin.T_cool_out = resolved.get("coolant_temp_out",
                                                      resolved.get("coolant_temp_in", init_T))

        # ------------------------------------------------------------------
        # Step 1: Update twin with delivered current command, flow input,
        # measured ambient, measured coolant inlet.
        # Flow input (A2): do not blindly trust measured pump_flow_lpm.
        # If measured flow disagrees with first-order lag model of delivered
        # pump command by > Rflow_threshold, drive twin with command model.
        # ------------------------------------------------------------------
        thr_flow = self._thr.get("Rflow_threshold", 1.0)
        flow_discrepancy = abs(resolved["pump_flow_lpm"] - self._cmd_pump_flow)
        flag_flow_mismatch = False

        if resolved["pump_flow_lpm"] > self._cmd_pump_flow + thr_flow:
            # Measured flow is higher than commanded model (e.g. coordinated replay)
            twin_flow = self._cmd_pump_flow
            flag_flow_mismatch = True
        else:
            twin_flow = resolved["pump_flow_lpm"]

        twin_amb    = resolved["ambient_temp"]
        twin_inlet  = resolved["coolant_temp_in"]
        twin_i_pack = current_cmd_a   # delivered current to plant

        self._twin.step_estimator(
            i_pack_meas=float(twin_i_pack),
            pump_flow_meas=float(twin_flow),
            T_amb_meas=float(twin_amb),
            T_cool_in_meas=float(twin_inlet),
            dt=float(dt)
        )
        twin_T = self._twin.T_batt
        twin_T_out_pred = self._twin.T_cool_out
        twin_v_pack = self._twin.v_pack

        # Collect residuals (filled in by individual checks below)
        residuals: Dict[str, float] = {}

        # ------------------------------------------------------------------
        # Run all check families and collect findings
        # ------------------------------------------------------------------
        findings: List[Dict] = []

        if flag_flow_mismatch:
            findings.append({
                "check": "C6_flow_spoof",
                "diagnosis": "COMMAND_RESPONSE_MISMATCH",
                "alarm": "warning",
                "is_attack": True,
                "blamed": ["pump_flow_lpm"],
                "reasons": [
                    f"A2: Measured flow {resolved['pump_flow_lpm']:.2f} L/min exceeds command model "
                    f"{self._cmd_pump_flow:.2f} L/min by {flow_discrepancy:.2f} L/min > threshold {thr_flow:.2f} L/min"
                ],
                "instant": False,
            })
        else:
            self._persist_decrement("C6_flow_spoof")

        # C1: Validity and flatline
        self._check_c1(t, measured, valid_flags, findings, residuals)

        # C2: Redundancy (temp_cell_a vs temp_cell_b)
        self._check_c2(t, resolved, valid_flags, findings, residuals)

        # C3: Thermal consistency (temp vs twin)
        self._check_c3(t, resolved, valid_flags, twin_T, findings, residuals)

        # C4: Coolant energy balance (when flow >= 3 L/min)
        self._check_c4(t, resolved, valid_flags, twin_T_out_pred, findings, residuals)

        # C5: Electrical consistency (pack_voltage vs twin)
        self._check_c5(t, resolved, valid_flags, twin_v_pack, findings, residuals)

        # C6: Actuator echo and current echo
        self._check_c6(t, resolved, valid_flags, current_cmd_a, pump_cmd_lpm, findings, residuals)

        # C7: Command safety envelope evaluated on requested_cmds
        self._check_c7(t, resolved, valid_flags, req_current_a, req_pump_lpm, twin_T, findings, residuals)

        # ------------------------------------------------------------------
        # Apply persistence filter to findings (except instant-class ones)
        # ------------------------------------------------------------------
        active_findings = self._apply_persistence(t, findings)

        # ------------------------------------------------------------------
        # Aggregate into a single Verdict
        # ------------------------------------------------------------------
        verdict = self._aggregate(t, active_findings, residuals)

        # Store recent state for trust queries
        self._last_valid_flags = valid_flags
        self._last_active_findings = active_findings
        self._last_resolved = resolved
        self._last_verdict = verdict

        return verdict

    def trusted_temperature(self) -> float:
        """
        Return the guard's trusted battery temperature estimate in Celsius.

        Logic:
          - Both temps trusted -> average of temp_a and (temp_b - 0.25).
          - temp_a blamed -> temp_b - 0.25.
          - temp_b blamed -> temp_a.
          - Both suspect/bad -> twin temperature + 1.6 C conservative margin
            (justified by max twin error 1.56 C from results_twin_error.csv).
        """
        trust = self.sensor_trust()
        t_a_ok = (trust.get("temp_cell_a") == "ok")
        t_b_ok = (trust.get("temp_cell_b") == "ok")

        t_a_val = self._last_resolved.get("temp_cell_a", self._twin.T_batt)
        t_b_val = self._last_resolved.get("temp_cell_b", self._twin.T_batt + 0.25) - 0.25

        if t_a_ok and t_b_ok:
            return float(0.5 * (t_a_val + t_b_val))
        elif t_a_ok and not t_b_ok:
            return float(t_a_val)
        elif t_b_ok and not t_a_ok:
            return float(t_b_val)
        else:
            return float(self._twin.T_batt + 1.6)

    def sensor_trust(self) -> Dict[str, str]:
        """
        Return sensor trust mapping: dict of channel -> 'ok' | 'suspect' | 'bad'.
        Evaluated for: temp_cell_a, temp_cell_b, pump_flow_lpm, pack_current, pack_voltage.
        """
        channels = ["temp_cell_a", "temp_cell_b", "pump_flow_lpm", "pack_current", "pack_voltage"]
        trust: Dict[str, str] = {}

        for ch in channels:
            # Check 1: Validity (NaN/inf/out-of-range)
            if not self._last_valid_flags.get(ch, True):
                trust[ch] = "bad"
                continue

            # Check 2: Flatline on noisy signals
            hist = self._history.get(ch, [])
            if len(hist) >= FLATLINE_WINDOW and len(set(hist)) == 1:
                trust[ch] = "bad"
                continue

            # Check 3: Explicitly blamed in active findings
            blamed = any(ch in f.get("blamed", []) for f in self._last_active_findings)
            if blamed:
                trust[ch] = "bad"
                continue

            # Check 4: Accumulating persistence counter or soft suspicion
            suspect = any(
                (ch in k or (ch == "temp_cell_a" and "C3_a" in k) or (ch == "temp_cell_b" and "C3_b" in k))
                and count > 0 for k, count in self._persist.items()
            )
            if suspect:
                trust[ch] = "suspect"
                continue

            trust[ch] = "ok"

        return trust

    # ======================================================================
    # SIGNAL RESOLUTION
    # ======================================================================

    def _is_valid(self, key: str, val: Any) -> bool:
        """Return True if val is a finite number within the physical range for key."""
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(fval):
            return False
        lo, hi = SIGNAL_RANGES.get(key, (-1e9, 1e9))
        return lo <= fval <= hi

    def _resolve_signals(
        self,
        measured: Dict[str, Any]
    ) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """
        For every signal:
        - Check validity (finite, in range).
        - If valid, update last_valid and return the value.
        - If invalid, return the last valid value as a fallback.
        Update flatline history for noisy signals.
        """
        valid_flags: Dict[str, bool] = {}
        resolved: Dict[str, float] = {}

        all_keys = list(SIGNAL_RANGES.keys())

        for key in all_keys:
            raw = measured.get(key)
            ok = self._is_valid(key, raw)
            valid_flags[key] = ok
            if ok:
                fval = float(raw)
                self._last_valid[key] = fval
                resolved[key] = fval
            else:
                resolved[key] = self._last_valid.get(key, 0.0)

            # Update flatline history for noisy signals
            if key in NOISY_SIGNALS:
                h = self._history.setdefault(key, [])
                h.append(resolved[key])
                if len(h) > FLATLINE_WINDOW:
                    h.pop(0)

        return valid_flags, resolved

    # ======================================================================
    # C1: VALIDITY AND FLATLINE
    # ======================================================================

    def _check_c1(
        self,
        t: float,
        measured: Dict[str, Any],
        valid_flags: Dict[str, bool],
        findings: List[Dict],
        residuals: Dict[str, float]
    ) -> None:
        """
        C1: NaN / inf / zero / out-of-range detection and flatline detection.
        HW faults are instant (no persistence needed for NaN/zero/out-of-range).
        Flatlines require 10 consecutive identical values.
        """
        for key, ok in valid_flags.items():
            if not ok:
                raw = measured.get(key)
                try:
                    fval = float(raw) if raw is not None else float("nan")
                except (TypeError, ValueError):
                    fval = float("nan")

                if math.isnan(fval) or math.isinf(fval):
                    reason = f"C1: {key} is NaN/inf (value={fval})"
                    mode = "nan_inf"
                elif fval == 0.0 and key not in ("pump_flow_lpm",):
                    reason = f"C1: {key} is zero (hw fault)"
                    mode = "zero"
                else:
                    lo, hi = SIGNAL_RANGES.get(key, (-1e9, 1e9))
                    reason = f"C1: {key}={fval:.3f} is out of range [{lo}, {hi}]"
                    mode = "range"

                findings.append({
                    "check": f"C1_hw_{key}",
                    "diagnosis": "SENSOR_HW_FAULT",
                    "alarm": "warning",
                    "is_attack": False,
                    "blamed": [key],
                    "reasons": [reason],
                    "instant": True,  # no persistence needed
                })
                residuals[f"hw_fault_{key}"] = 1.0
                continue

            # Flatline check for noisy signals
            if key in NOISY_SIGNALS:
                h = self._history.get(key, [])
                if len(h) >= FLATLINE_WINDOW:
                    if len(set(h)) == 1:
                        findings.append({
                            "check": f"C1_flat_{key}",
                            "diagnosis": "SENSOR_FROZEN",
                            "alarm": "warning",
                            "is_attack": None,
                            "blamed": [key],
                            "reasons": [
                                f"C1: {key} has been identical for {FLATLINE_WINDOW} "
                                f"consecutive seconds (value={h[-1]:.4f})"
                            ],
                            "instant": False,
                        })
                        residuals[f"flatline_{key}"] = 1.0

    # ======================================================================
    # C2: REDUNDANCY CHECK
    # ======================================================================

    def _check_c2(
        self,
        t: float,
        resolved: Dict[str, float],
        valid_flags: Dict[str, bool],
        findings: List[Dict],
        residuals: Dict[str, float]
    ) -> None:
        """
        C2: Redundancy check between temp_cell_a and temp_cell_b.
        Learn the fixed offset R3 = (A - B) during first 60 s.
        Monitor deviations from this offset with CUSUM plus instant threshold
        for jumps > 5 C.
        """
        if not (valid_flags.get("temp_cell_a", False) and valid_flags.get("temp_cell_b", False)):
            return

        ta = resolved["temp_cell_a"]
        tb = resolved["temp_cell_b"]
        r3_raw = ta - tb

        # Learning phase: collect baseline offset
        if t <= CALIB_WINDOW_S:
            self._r3_baseline_vals.append(r3_raw)
            residuals["R3"] = r3_raw
            return

        # After learning, compute mean offset if not yet done
        if self._r3_offset is None:
            if self._r3_baseline_vals:
                self._r3_offset = float(np.mean(self._r3_baseline_vals))
            else:
                self._r3_offset = 0.0

        # Residual relative to learned offset
        R3 = r3_raw - self._r3_offset
        residuals["R3"] = R3

        # Determine which sensor disagrees more with the twin
        twin_T = self._twin.T_batt
        ta_err = abs(ta - twin_T)
        tb_err = abs(tb - 0.25 - twin_T)
        blamed_sensor = "temp_cell_a" if ta_err >= tb_err else "temp_cell_b"

        # Instant alarm for large jumps (no persistence)
        if abs(R3) > C2_JUMP_INSTANT_THRESHOLD_C:
            findings.append({
                "check": "C2_jump",
                "diagnosis": "SENSOR_SPOOF",
                "alarm": "critical",
                "is_attack": True,
                "blamed": [blamed_sensor],
                "reasons": [
                    f"C2: Redundancy jump R3={R3:.2f} C (offset={self._r3_offset:.2f} C) "
                    f"exceeds instant threshold {C2_JUMP_INSTANT_THRESHOLD_C} C"
                ],
                "instant": True,
            })
            return

        # CUSUM for slow drift
        thr = self._thr.get("R3_threshold_c", 2.0)
        cusum_fired = self._cusum_r3.update(R3)
        if cusum_fired or abs(R3) > thr:
            findings.append({
                "check": "C2_cusum",
                "diagnosis": "SENSOR_SPOOF",
                "alarm": "warning",
                "is_attack": True,
                "blamed": [blamed_sensor],
                "reasons": [
                    f"C2: CUSUM or threshold alarm on R3={R3:.2f} C "
                    f"(threshold={thr:.2f} C, S+={self._cusum_r3.s_pos:.2f})"
                ],
                "instant": False,
            })
        else:
            self._persist_decrement("C2_cusum")

    # ======================================================================
    # C3: THERMAL CONSISTENCY
    # ======================================================================

    def _check_c3(
        self,
        t: float,
        resolved: Dict[str, float],
        valid_flags: Dict[str, bool],
        twin_T: float,
        findings: List[Dict],
        residuals: Dict[str, float]
    ) -> None:
        """
        C3: Thermal consistency check.
        R1a = temp_cell_a - twin_T
        R1b = temp_cell_b - twin_T  (adjusted for the +0.25 C fixed offset)

        If one disagrees with the twin much more than the other, blame that one.
        If both disagree similarly, just give a warning without blaming.
        """
        valid_a = valid_flags.get("temp_cell_a", False)
        valid_b = valid_flags.get("temp_cell_b", False)

        thr = self._thr.get("R1_threshold_c", 3.0)
        diff_thr = self._thr.get("R1_diff_threshold_c", 2.0)

        R1a = None
        R1b = None

        if valid_a:
            R1a = resolved["temp_cell_a"] - twin_T
            residuals["R1a"] = R1a
        if valid_b:
            # Sensor B has a nominal +0.25 C offset relative to true temperature
            R1b = resolved["temp_cell_b"] - 0.25 - twin_T
            residuals["R1b"] = R1b

        # Check each sensor independently
        a_bad = R1a is not None and abs(R1a) > thr
        b_bad = R1b is not None and abs(R1b) > thr

        if a_bad and b_bad:
            # Both deviate - could be twin drifting or environmental condition
            # Blame neither individually; just issue a warning
            findings.append({
                "check": "C3_both",
                "diagnosis": "SENSOR_SPOOF",
                "alarm": "warning",
                "is_attack": None,   # undetermined - could be model error
                "blamed": [],
                "reasons": [
                    f"C3: Both temp sensors deviate from twin (R1a={R1a:.2f} C, "
                    f"R1b={R1b:.2f} C, threshold={thr:.2f} C) - cause undetermined"
                ],
                "instant": False,
            })
        elif a_bad and not b_bad:
            r1b_str = f"{R1b:.2f}" if R1b is not None else "N/A"
            findings.append({
                "check": "C3_a",
                "diagnosis": "SENSOR_SPOOF",
                "alarm": "warning",
                "is_attack": True,
                "blamed": ["temp_cell_a"],
                "reasons": [
                    f"C3: temp_cell_a disagrees with twin (R1a={R1a:.2f} C, "
                    f"R1b={r1b_str} C, threshold={thr:.2f} C)"
                ],
                "instant": False,
            })
        elif b_bad and not a_bad:
            r1a_str = f"{R1a:.2f}" if R1a is not None else "N/A"
            findings.append({
                "check": "C3_b",
                "diagnosis": "SENSOR_SPOOF",
                "alarm": "warning",
                "is_attack": True,
                "blamed": ["temp_cell_b"],
                "reasons": [
                    f"C3: temp_cell_b disagrees with twin (R1b={R1b:.2f} C, "
                    f"R1a={r1a_str} C, threshold={thr:.2f} C)"
                ],
                "instant": False,
            })
        else:
            self._persist_decrement("C3_a")
            self._persist_decrement("C3_b")
            self._persist_decrement("C3_both")

    # ======================================================================
    # C4: COOLANT ENERGY BALANCE
    # ======================================================================

    def _check_c4(
        self,
        t: float,
        resolved: Dict[str, float],
        valid_flags: Dict[str, bool],
        twin_T_out: float,
        findings: List[Dict],
        residuals: Dict[str, float]
    ) -> None:
        """
        C4: Coolant energy balance check (only when measured flow >= 3 L/min).
        R2 = (T_out_measured - T_in_measured) - (twin_T_out - twin_T_in)
        """
        flow = resolved.get("pump_flow_lpm", 0.0)
        if flow < FLOW_THRESHOLD_FOR_C4:
            return
        if not (valid_flags.get("coolant_temp_out", False)
                and valid_flags.get("coolant_temp_in", False)):
            return

        t_out_meas = resolved["coolant_temp_out"]
        t_in_meas = resolved["coolant_temp_in"]
        twin_t_in = resolved["coolant_temp_in"]  # guard drives twin with measured inlet

        R2 = (t_out_meas - t_in_meas) - (twin_T_out - twin_t_in)
        residuals["R2"] = R2

        thr = self._thr.get("R2_threshold_kj", 0.5)
        if abs(R2) > thr:
            findings.append({
                "check": "C4_energy",
                "diagnosis": "SENSOR_SPOOF",
                "alarm": "warning",
                "is_attack": True,
                "blamed": ["coolant_temp_out"],
                "reasons": [
                    f"C4: Coolant energy balance R2={R2:.3f} K exceeds "
                    f"threshold={thr:.3f} K (flow={flow:.1f} L/min)"
                ],
                "instant": False,
            })
        else:
            self._persist_decrement("C4_energy")

    # ======================================================================
    # C5: ELECTRICAL CONSISTENCY
    # ======================================================================

    def _check_c5(
        self,
        t: float,
        resolved: Dict[str, float],
        valid_flags: Dict[str, bool],
        twin_v_pack: float,
        findings: List[Dict],
        residuals: Dict[str, float]
    ) -> None:
        """
        C5: Electrical check - pack voltage vs twin pack voltage.
        Ignored in the first 120 s. Uses CUSUM for slow drifts.
        """
        if t < VOLTAGE_IGNORE_S:
            return
        if not valid_flags.get("pack_voltage", False):
            return

        Rv = resolved["pack_voltage"] - twin_v_pack
        residuals["Rv"] = Rv

        thr = self._thr.get("Rv_threshold_v", 5.0)
        cusum_fired = self._cusum_rv.update(Rv)
        instant_fired = abs(Rv) > thr

        if instant_fired:
            # Instant threshold breach: clearly abnormal voltage, likely spoof
            findings.append({
                "check": "C5_voltage",
                "diagnosis": "SENSOR_SPOOF",
                "alarm": "warning",
                "is_attack": True,
                "blamed": ["pack_voltage"],
                "reasons": [
                    f"C5: Pack voltage residual Rv={Rv:.2f} V exceeds "
                    f"instant threshold={thr:.2f} V"
                ],
                "instant": False,
            })
        elif cusum_fired:
            # CUSUM accumulated drift: cause undetermined (model mismatch or slow drift)
            findings.append({
                "check": "C5_voltage",
                "diagnosis": "COMMAND_RESPONSE_MISMATCH",
                "alarm": "warning",
                "is_attack": None,  # Could be model-plant mismatch, not necessarily attack
                "blamed": ["pack_voltage"],
                "reasons": [
                    f"C5: Pack voltage CUSUM drift Rv={Rv:.2f} V "
                    f"(threshold={thr:.2f} V, CUSUM S+={self._cusum_rv.s_pos:.2f})"
                ],
                "instant": False,
            })
        else:
            self._persist_decrement("C5_voltage")

    # ======================================================================
    # C6: ACTUATOR ECHO AND CURRENT ECHO
    # ======================================================================

    def _check_c6(
        self,
        t: float,
        resolved: Dict[str, float],
        valid_flags: Dict[str, bool],
        current_cmd_a: float,
        pump_cmd_lpm: float,
        findings: List[Dict],
        residuals: Dict[str, float]
    ) -> None:
        """
        C6: Actuator echo (pump command vs measured flow) and current echo.

        Rflow = measured_flow - delivered_pump_command
          - Ignore for 15 s after any pump command change.
          - COMMAND_RESPONSE_MISMATCH if flow > 0 while pump_cmd = 0 (is_attack=True).
          - ACTUATOR_FAULT if flow < command by a large margin (is_attack=False).

        Rcur = measured_current - delivered_current_command
          - COMMAND_RESPONSE_MISMATCH if significant discrepancy (is_attack=None).
        """
        # --- Pump flow check ---
        time_since_cmd_change = t - self._last_pump_cmd_change_t
        if time_since_cmd_change >= ACTUATOR_GRACE_S and valid_flags.get("pump_flow_lpm", False):
            meas_flow = resolved["pump_flow_lpm"]
            Rflow = meas_flow - pump_cmd_lpm
            residuals["Rflow"] = Rflow
            thr_flow = self._thr.get("Rflow_threshold", 3.0)

            # Case 1: Flow sensor reads flow while pump commanded OFF -> likely spoof/echo attack
            if pump_cmd_lpm == 0.0 and meas_flow > 1.0:
                findings.append({
                    "check": "C6_echo_flow",
                    "diagnosis": "COMMAND_RESPONSE_MISMATCH",
                    "alarm": "warning",
                    "is_attack": True,
                    "blamed": ["pump_flow_lpm"],
                    "reasons": [
                        f"C6: Pump commanded OFF but measured flow={meas_flow:.2f} L/min "
                        f"(suggests sensor replay or tamper)"
                    ],
                    "instant": False,
                })
            # Case 2: Commanded flow is non-zero but actual flow is significantly low -> actuator fault
            elif pump_cmd_lpm > 1.0 and Rflow < -thr_flow:
                findings.append({
                    "check": "C6_pump_fault",
                    "diagnosis": "ACTUATOR_FAULT",
                    "alarm": "warning",
                    "is_attack": False,
                    "blamed": ["pump_flow_lpm"],
                    "reasons": [
                        f"C6: Pump commanded {pump_cmd_lpm:.1f} L/min but measured "
                        f"flow={meas_flow:.2f} L/min (Rflow={Rflow:.2f} L/min, "
                        f"threshold={thr_flow:.2f} L/min) - actuator fault"
                    ],
                    "instant": False,
                })
            else:
                self._persist_decrement("C6_echo_flow")
                self._persist_decrement("C6_pump_fault")

        # --- Current echo check ---
        if valid_flags.get("pack_current", False):
            meas_curr = resolved["pack_current"]
            Rcur = meas_curr - current_cmd_a
            residuals["Rcur"] = Rcur
            thr_cur = self._thr.get("Rcur_threshold", 10.0)

            # Check if pack is in CV saturation (current tapers below command to hold 4.20V/cell)
            v_pack = resolved.get("pack_voltage", 0.0)
            in_cv_taper = (v_pack >= params.PACK_N_SERIES * (params.CELL_V_MAX - 0.02) and Rcur < 0)

            if not in_cv_taper:
                cusum_cur_fired = self._cusum_rcur.update(Rcur)
                instant_cur_fired = abs(Rcur) > thr_cur
            else:
                self._cusum_rcur.reset()
                cusum_cur_fired = False
                instant_cur_fired = False

            if instant_cur_fired:
                findings.append({
                    "check": "C6_current",
                    "diagnosis": "COMMAND_RESPONSE_MISMATCH",
                    "alarm": "warning",
                    "is_attack": None,
                    "blamed": ["pack_current"],
                    "reasons": [
                        f"C6: Measured current={meas_curr:.1f} A vs command "
                        f"{current_cmd_a:.1f} A (Rcur={Rcur:.2f} A, "
                        f"threshold={thr_cur:.2f} A)"
                    ],
                    "instant": False,
                })
            elif cusum_cur_fired:
                findings.append({
                    "check": "C6_current",
                    "diagnosis": "COMMAND_RESPONSE_MISMATCH",
                    "alarm": "warning",
                    "is_attack": None,
                    "blamed": ["pack_current"],
                    "reasons": [
                        f"C6: Current CUSUM accumulated drift Rcur={Rcur:.2f} A "
                        f"(threshold={thr_cur:.2f} A)"
                    ],
                    "instant": False,
                })
            else:
                self._persist_decrement("C6_current")

    # ======================================================================
    # C7: COMMAND SAFETY ENVELOPE
    # ======================================================================

    def _check_c7(
        self,
        t: float,
        resolved: Dict[str, float],
        valid_flags: Dict[str, bool],
        current_cmd_a: float,
        pump_cmd_lpm: float,
        twin_T: float,
        findings: List[Dict],
        residuals: Dict[str, float]
    ) -> None:
        """
        C7: Command safety envelope checks.

        (a) Current must not exceed 128 A * 1.02 = 130.56 A and must not
            exceed the derating curve evaluated on the TRUSTED temperature
            (median of valid temperature sensors + twin temperature).
        (b) Pump must be >= 5 L/min whenever current > 20 A.
        (c) Look-ahead: simulate twin 180 s forward; flag if predicted
            pack temperature exceeds 45 C.
        """
        reasons = []
        blame = []
        fire_c7 = False

        # Gather valid temperature sensors plus twin T for trusted temperature
        valid_temps = []
        if valid_flags.get("temp_cell_a", False):
            valid_temps.append(resolved["temp_cell_a"])
        if valid_flags.get("temp_cell_b", False):
            valid_temps.append(resolved["temp_cell_b"] - 0.25)  # remove offset
        valid_temps.append(twin_T)  # always include twin
        trusted_temp = float(np.median(valid_temps))
        residuals["trusted_temp"] = trusted_temp

        # (a1) Absolute current limit
        max_abs_current = params.PACK_RATED_CHARGE_CURRENT_A * 1.02
        if current_cmd_a > max_abs_current:
            fire_c7 = True
            blame.append("pack_current")
            reasons.append(
                f"C7a: Delivered current {current_cmd_a:.1f} A exceeds "
                f"absolute limit {max_abs_current:.1f} A (128A * 1.02)"
            )

        # (a2) Thermal derating curve: full current up to 40 C, linear to 0 A at 50 C
        if trusted_temp >= params.CELL_TEMP_WARN_C:
            derate = max(0.0, (params.CELL_TEMP_TRIP_C - trusted_temp) /
                              (params.CELL_TEMP_TRIP_C - params.CELL_TEMP_WARN_C))
            max_derated = params.PACK_RATED_CHARGE_CURRENT_A * derate
            if current_cmd_a > max_derated + 1.0:  # 1 A tolerance
                fire_c7 = True
                blame.append("pack_current")
                reasons.append(
                    f"C7a: Delivered current {current_cmd_a:.1f} A exceeds "
                    f"derating limit {max_derated:.1f} A at trusted_temp={trusted_temp:.1f} C"
                )

        # (b) Pump interlock: must have pump >= 5 L/min when current > 20 A
        if current_cmd_a > 20.0 and pump_cmd_lpm < 5.0:
            fire_c7 = True
            blame.append("pump_flow_lpm")
            reasons.append(
                f"C7b: Pump command {pump_cmd_lpm:.1f} L/min is below 5.0 L/min "
                f"while delivered current={current_cmd_a:.1f} A > 20 A"
            )

        # (c) Look-ahead: simulate twin 180 s ahead
        lookahead_T = self._run_lookahead(
            twin_state=self._twin,
            i_pack=current_cmd_a,
            pump_lpm=pump_cmd_lpm,
            T_amb=resolved["ambient_temp"],
            T_in=resolved["coolant_temp_in"],
            steps=LOOKAHEAD_S
        )
        residuals["lookahead_T"] = lookahead_T
        if lookahead_T > MAX_TEMP_LOOKAHEAD_C:
            fire_c7 = True
            reasons.append(
                f"C7c: Look-ahead of {LOOKAHEAD_S} s predicts pack temperature "
                f"{lookahead_T:.1f} C > {MAX_TEMP_LOOKAHEAD_C} C"
            )

        if fire_c7:
            findings.append({
                "check": "C7_safety",
                "diagnosis": "UNSAFE_COMMAND",
                "alarm": "critical",
                "is_attack": True,
                "blamed": blame,
                "reasons": reasons,
                "instant": False,
            })
        else:
            self._persist_decrement("C7_safety")

    # ======================================================================
    # LOOK-AHEAD SIMULATION (C7c)
    # ======================================================================

    def _run_lookahead(
        self,
        twin_state: Twin,
        i_pack: float,
        pump_lpm: float,
        T_amb: float,
        T_in: float,
        steps: int = LOOKAHEAD_S
    ) -> float:
        """Simulate twin forward holding commands; delegate to run_lookahead helper."""
        return run_lookahead(twin_state, i_pack, pump_lpm, T_amb, T_in, steps)

    # ======================================================================
    # PERSISTENCE FILTER
    # ======================================================================

    def _persist_decrement(self, check_key: str) -> None:
        """Decay the persistence counter for a given check (not below 0)."""
        if check_key in self._persist:
            self._persist[check_key] = max(0, self._persist[check_key] - 1)

    def _apply_persistence(
        self,
        t: float,
        findings: List[Dict]
    ) -> List[Dict]:
        """
        Apply a 5-second persistence filter to findings.

        Instant findings (NaN/zero/range) always pass through immediately.
        C2 jumps > 5 C also pass immediately (marked instant=True).
        All other findings must hold for PERSISTENCE_SECS consecutive seconds.
        """
        active = []
        # First, increment counters for all active checks
        active_check_keys = set()
        for f in findings:
            check_key = f["check"]
            active_check_keys.add(check_key)
            if f.get("instant", False):
                # Instant: always fire, reset counter
                self._persist[check_key] = PERSISTENCE_SECS
                active.append(f)
            else:
                self._persist[check_key] = self._persist.get(check_key, 0) + 1
                if self._persist[check_key] >= PERSISTENCE_SECS:
                    active.append(f)

        # Decay counters for checks that did NOT fire this step
        for key in list(self._persist.keys()):
            if key not in active_check_keys:
                self._persist[key] = max(0, self._persist[key] - 1)

        return active

    # ======================================================================
    # AGGREGATION INTO VERDICT
    # ======================================================================

    def _aggregate(
        self,
        t: float,
        active_findings: List[Dict],
        residuals: Dict[str, float]
    ) -> Verdict:
        """
        Combine active findings into a single Verdict.
        Priority order: UNSAFE_COMMAND > SENSOR_SPOOF > SENSOR_HW_FAULT >
        COMMAND_RESPONSE_MISMATCH > ACTUATOR_FAULT > SENSOR_FROZEN > NORMAL.
        Alarm level: critical > warning > none.
        """
        if not active_findings:
            return Verdict(t=t, residuals=residuals)

        # Priority ranking for diagnosis classes
        priority = {
            "UNSAFE_COMMAND": 7,
            "SENSOR_SPOOF": 6,
            "SENSOR_HW_FAULT": 5,
            "COMMAND_RESPONSE_MISMATCH": 4,
            "ACTUATOR_FAULT": 3,
            "SENSOR_FROZEN": 2,
            "NORMAL": 1,
        }

        # Alarm level ranking
        alarm_rank = {"critical": 2, "warning": 1, "none": 0}
        alarm_rank_rev = {2: "critical", 1: "warning", 0: "none"}

        # Pick the highest-priority diagnosis
        best = max(active_findings, key=lambda f: priority.get(f["diagnosis"], 0))
        diagnosis = best["diagnosis"]

        # Alarm = highest across all active findings
        max_alarm = max(alarm_rank.get(f.get("alarm", "none"), 0) for f in active_findings)
        alarm_level = alarm_rank_rev[max_alarm]

        # is_attack: if any finding is True, result is True; otherwise use best finding value
        is_attack_vals = [f.get("is_attack") for f in active_findings]
        if True in is_attack_vals:
            is_attack = True
        elif all(v is False for v in is_attack_vals):
            is_attack = False
        else:
            is_attack = best.get("is_attack")

        # Collect all blamed signals and reasons
        blamed = []
        reasons = []
        for f in active_findings:
            for sig in f.get("blamed", []):
                if sig not in blamed:
                    blamed.append(sig)
            reasons.extend(f.get("reasons", []))

        return Verdict(
            t=t,
            diagnosis=diagnosis,
            alarm_level=alarm_level,
            is_attack=is_attack,
            blamed_signals=blamed,
            reasons=reasons,
            residuals=residuals,
        )
