"""
PhysicsGuard - Closed-Loop Actuator Safety Gate
===============================================
Implements the SafetyGate class which sits on trusted hardware at the
actuator interface, operating like a cyber-physical interlock.

The gate enforces physical guarantees even when sensors or command buses
upstream have been compromised by a cyberattacker.

ISOLATION RULES:
  - SafetyGate NEVER reads ground-truth signals.
  - SafetyGate NEVER reads scenario names or labels.
  - SafetyGate NEVER accesses the plant object directly.
"""

from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np

from . import params
from .guard import (
    PhysicsGuard, Verdict, run_lookahead,
    LOOKAHEAD_S, MAX_TEMP_LOOKAHEAD_C, ACTUATOR_GRACE_S
)


class SafetyGate:
    """
    Cyber-physical interlock and virtual sensor gate.

    Operates in two stages per time step:
      1. sanitize(t, measured): Virtual sensor filtering ensuring the BMS
         controller only sees trusted temperatures.
      2. screen(t, requested_cmds, measured): Actuator command screening,
         interlocking pump and current, and performing lookahead bisection.
    """

    def __init__(self, guard: Optional[PhysicsGuard] = None):
        self.guard: Optional[PhysicsGuard] = guard

        # Operating modes: NOMINAL, DEGRADED, LIMP_HOME, SHUTDOWN
        self.mode: str = "NOMINAL"
        self.clean_seconds: int = 0
        self.deescalate_delay_s: int = 30  # N clean seconds before de-escalation

        # Actuator confirmation tracking
        self.pump_forced_t: Optional[float] = None

        # Step record & telemetry
        self.intervened: bool = False
        self.last_action: str = "none"
        self.last_reason: str = ""
        self.last_signals_affected: List[str] = []
        self.last_trusted_temp: float = params.T_AMBIENT_NOMINAL_C
        self.last_applied_vs_requested_curr: float = 0.0

        # Event log
        self.event_log: List[Dict[str, Any]] = []

    def set_guard(self, guard: PhysicsGuard) -> None:
        """Attach or update the guard instance."""
        self.guard = guard

    def sanitize(self, t: float, measured: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 1: Virtual sensor filtering.
        Substitutes untrusted temperature channels for the BMS controller.
        Keeps raw values untouched in the original dict for logging.
        """
        sanitized = dict(measured)
        self.intervened = False
        self.last_action = "none"
        self.last_reason = ""
        self.last_signals_affected = []

        if self.guard is None:
            self.last_trusted_temp = float(measured.get("temp_cell_a", params.T_AMBIENT_NOMINAL_C))
            return sanitized

        trusted_t = self.guard.trusted_temperature()
        self.last_trusted_temp = float(trusted_t)
        trust = self.guard.sensor_trust()

        ta_status = trust.get("temp_cell_a", "ok")
        tb_status = trust.get("temp_cell_b", "ok")

        ta_raw = measured.get("temp_cell_a")
        tb_raw = measured.get("temp_cell_b")

        # Case 1: Both sensors trusted -> normal passthrough
        if ta_status == "ok" and tb_status == "ok":
            return sanitized

        # Case 2: temp_cell_a blamed, temp_cell_b trusted -> substitute A with (B - 0.25)
        elif ta_status != "ok" and tb_status == "ok":
            tb_val = float(tb_raw) if tb_raw is not None and math.isfinite(float(tb_raw)) else trusted_t + 0.25
            sub_val = tb_val - 0.25
            sanitized["temp_cell_a"] = float(sub_val)
            self.intervened = True
            self.last_action = "virtual_sensor_sub_a"
            self.last_reason = (
                f"t={t:.0f}s: temp_cell_a is {ta_status} (raw={ta_raw}); "
                f"substituted with temp_cell_b - 0.25C ({sub_val:.2f} C)"
            )
            self.last_signals_affected.append("temp_cell_a")

        # Case 3: temp_cell_b blamed, temp_cell_a trusted -> substitute B with (A + 0.25)
        elif tb_status != "ok" and ta_status == "ok":
            ta_val = float(ta_raw) if ta_raw is not None and math.isfinite(float(ta_raw)) else trusted_t
            sub_val = ta_val + 0.25
            sanitized["temp_cell_b"] = float(sub_val)
            self.intervened = True
            self.last_action = "virtual_sensor_sub_b"
            self.last_reason = (
                f"t={t:.0f}s: temp_cell_b is {tb_status} (raw={tb_raw}); "
                f"substituted with temp_cell_a + 0.25C ({sub_val:.2f} C)"
            )
            self.last_signals_affected.append("temp_cell_b")

        # Case 4: Both suspect or bad -> substitute both with trusted temperature from twin (+ margin)
        else:
            sanitized["temp_cell_a"] = float(trusted_t)
            sanitized["temp_cell_b"] = float(trusted_t + 0.25)
            self.intervened = True
            self.last_action = "virtual_sensor_sub_both"
            self.last_reason = (
                f"t={t:.0f}s: both temp sensors untrusted (A={ta_status}, B={tb_status}); "
                f"substituted with twin model temperature + 1.6C margin ({trusted_t:.2f} C)"
            )
            self.last_signals_affected.extend(["temp_cell_a", "temp_cell_b"])

        if self.intervened:
            self._log_event(t, self.mode, self.last_action, self.last_reason, self.last_signals_affected)

        return sanitized

    def screen(
        self,
        t: float,
        requested_cmds: Tuple[float, float],
        measured: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Stage 2: Deterministic command screening at actuator boundary.

        Checks:
          - Current > 128*1.02 A -> clamp to derating limit on trusted temperature.
          - Current > 20 A while pump < 5 L/min -> force pump to nominal 10 L/min.
          - 180 s look-ahead predicts > 45 C -> reduce current until lookahead <= 44 C.
          - Actuator fault handling: derate only as needed, do not alarm attack.
          - Mode hysteresis: NOMINAL, DEGRADED, LIMP_HOME, SHUTDOWN.
        """
        req_curr, req_pump = requested_cmds
        appl_curr = float(req_curr)
        appl_pump = float(req_pump)

        action_reasons: List[str] = []
        action_names: List[str] = []

        trusted_t = self.last_trusted_temp

        # ------------------------------------------------------------------
        # 1. Absolute current limit and thermal derating envelope
        # ------------------------------------------------------------------
        max_abs_current = params.PACK_RATED_CHARGE_CURRENT_A * 1.02  # 130.56 A

        # Thermal derating envelope: full current up to 40 C, linear to 0 at 50 C
        if trusted_t <= params.CELL_TEMP_WARN_C:
            max_envelope_curr = params.PACK_RATED_CHARGE_CURRENT_A
        elif trusted_t >= params.CELL_TEMP_TRIP_C:
            max_envelope_curr = 0.0
        else:
            derate = (params.CELL_TEMP_TRIP_C - trusted_t) / (
                params.CELL_TEMP_TRIP_C - params.CELL_TEMP_WARN_C
            )
            max_envelope_curr = params.PACK_RATED_CHARGE_CURRENT_A * derate

        # Apply clamp if requested current exceeds absolute or derating limit
        limit_to_enforce = min(max_abs_current, max_envelope_curr)
        if appl_curr > limit_to_enforce + 0.01:
            orig = appl_curr
            appl_curr = float(limit_to_enforce)
            self.intervened = True
            action_names.append("clamp_current")
            action_reasons.append(
                f"current clamped {orig:.1f}A -> {appl_curr:.1f}A "
                f"(limit {limit_to_enforce:.1f}A at trusted_temp={trusted_t:.1f} C)"
            )
            self.last_signals_affected.append("pack_current")

        # ------------------------------------------------------------------
        # 2. Coolant pump interlock
        # ------------------------------------------------------------------
        # If charging current > 20 A while pump command < 5 L/min, force pump to nominal 10 L/min
        if appl_curr > 20.0 and appl_pump < 5.0:
            orig_p = appl_pump
            appl_pump = params.PUMP_NOMINAL_FLOW_LPM  # 10.0 L/min
            self.intervened = True
            action_names.append("force_pump")
            action_reasons.append(
                f"pump command forced {orig_p:.1f} -> {appl_pump:.1f} L/min "
                f"(pump below 5.0 L/min while current={appl_curr:.1f}A > 20A)"
            )
            self.last_signals_affected.append("pump_flow_lpm")
            if self.pump_forced_t is None:
                self.pump_forced_t = t

        # If pump was forced on but measured flow remains unconfirmed after grace period, drop to LIMP_HOME
        if self.pump_forced_t is not None:
            time_since_forced = t - self.pump_forced_t
            meas_flow = float(measured.get("pump_flow_lpm", 0.0))
            if time_since_forced >= ACTUATOR_GRACE_S:
                if meas_flow < 3.0:
                    # Flow not confirmed -> drop to LIMP_HOME (25 A)
                    if appl_curr > 25.0:
                        orig_c = appl_curr
                        appl_curr = 25.0
                        self.mode = "LIMP_HOME"
                        self.intervened = True
                        action_names.append("limp_home_unconfirmed_flow")
                        action_reasons.append(
                            f"pump flow not confirmed ({meas_flow:.2f} L/min) after {time_since_forced:.0f}s; "
                            f"dropped current {orig_c:.1f}A -> 25.0A"
                        )
                        self.last_signals_affected.append("pack_current")
                else:
                    self.pump_forced_t = None  # Flow confirmed
        else:
            self.pump_forced_t = None

        # ------------------------------------------------------------------
        # 3. Look-ahead thermal prediction and current bisection
        # ------------------------------------------------------------------
        if self.guard is not None and hasattr(self.guard, "_twin"):
            twin = self.guard._twin
            t_amb = float(measured.get("ambient_temp", params.T_AMBIENT_NOMINAL_C))
            t_in = float(measured.get("coolant_temp_in", params.T_COOLANT_INLET_NOMINAL_C))

            # Effective flow expected for look-ahead
            # If pump degradation or low flow, use conservative flow
            flow_for_lookahead = float(appl_pump)
            if self.guard._last_verdict and self.guard._last_verdict.diagnosis == "ACTUATOR_FAULT":
                # Actuator fault: actual pump flow is degraded
                meas_flow = float(measured.get("pump_flow_lpm", appl_pump))
                flow_for_lookahead = min(flow_for_lookahead, max(0.5, meas_flow))

            pred_T = run_lookahead(
                twin_state=twin,
                i_pack=appl_curr,
                pump_lpm=flow_for_lookahead,
                T_amb=t_amb,
                T_in=t_in,
                steps=LOOKAHEAD_S
            )

            # If look-ahead predicts pack temperature > 45 C, bisection search down until <= 44 C
            if pred_T > MAX_TEMP_LOOKAHEAD_C:
                orig_c = appl_curr
                target_T = 44.0  # target safe ceiling with 1 C margin

                # Bisection search between 0.0 A and appl_curr
                lo = 0.0
                hi = appl_curr
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    t_mid = run_lookahead(
                        twin_state=twin,
                        i_pack=mid,
                        pump_lpm=flow_for_lookahead,
                        T_amb=t_amb,
                        T_in=t_in,
                        steps=LOOKAHEAD_S
                    )
                    if t_mid > target_T:
                        hi = mid
                    else:
                        lo = mid

                appl_curr = float(lo)
                self.intervened = True
                action_names.append("lookahead_derate")
                action_reasons.append(
                    f"look-ahead predicted {pred_T:.1f} C > 45.0 C; "
                    f"reduced current {orig_c:.1f}A -> {appl_curr:.1f}A (predicts <= 44.0 C)"
                )
                self.last_signals_affected.append("pack_current")

                if self.mode not in ("SHUTDOWN", "LIMP_HOME"):
                    self.mode = "DEGRADED"

        # ------------------------------------------------------------------
        # 4. Mode Escalation and Hysteresis De-escalation
        # ------------------------------------------------------------------
        last_verdict = getattr(self.guard, "_last_verdict", None)

        # Check for unmitigated physical emergencies requiring SHUTDOWN:
        # 1. Trusted temperature reaching emergency trip limit
        # 2. Lookahead predicting > 45 C even when current is at 0 A
        is_thermal_runaway = (trusted_t >= params.CELL_TEMP_TRIP_C - 1.0)
        unmitigatable_heat = (appl_curr == 0.0 and locals().get("pred_T", 0.0) > MAX_TEMP_LOOKAHEAD_C)

        if is_thermal_runaway or unmitigatable_heat:
            if self.mode != "SHUTDOWN":
                self.mode = "SHUTDOWN"
                self.intervened = True
                action_names.append("emergency_shutdown")
                action_reasons.append(
                    f"thermal hazard (trusted_t={trusted_t:.1f}C, pred_T={locals().get('pred_T', 0.0):.1f}C); "
                    f"forced SHUTDOWN (0A)"
                )
                self.clean_seconds = 0

        # Mode escalation based on active interventions and guard findings:
        elif self.pump_forced_t is not None and (t - self.pump_forced_t >= ACTUATOR_GRACE_S) and float(measured.get("pump_flow_lpm", 0.0)) < 3.0:
            if self.mode != "SHUTDOWN":
                self.mode = "LIMP_HOME"

        elif self.guard is not None and all(self.guard.sensor_trust().get(ch) != "ok" for ch in ["temp_cell_a", "temp_cell_b"]):
            # Both temperature sensors untrusted -> LIMP_HOME
            if self.mode not in ("SHUTDOWN", "LIMP_HOME"):
                self.mode = "LIMP_HOME"
                action_names.append("limp_home_redundancy_loss")
                action_reasons.append("both temperature sensors untrusted; operating in LIMP_HOME")

        elif self.intervened or (last_verdict is not None and last_verdict.diagnosis != "NORMAL"):
            # Actuator fault, command clamp, pump force, or sensor substitution -> DEGRADED
            if self.mode == "NOMINAL":
                self.mode = "DEGRADED"
            if last_verdict is not None and last_verdict.diagnosis == "ACTUATOR_FAULT":
                action_names.append("operator_warning_actuator_fault")
            self.clean_seconds = 0

        else:
            # Clean step: track hysteresis counter for de-escalation
            if not self.intervened:
                self.clean_seconds += 1
                if self.clean_seconds >= self.deescalate_delay_s:
                    if self.mode == "SHUTDOWN":
                        self.mode = "LIMP_HOME"
                    elif self.mode == "LIMP_HOME":
                        self.mode = "DEGRADED"
                    elif self.mode == "DEGRADED":
                        self.mode = "NOMINAL"
                    self.clean_seconds = 0
            else:
                self.clean_seconds = 0

        # Enforce Mode Constraints
        if self.mode == "SHUTDOWN":
            if appl_curr > 0.0:
                appl_curr = 0.0
                self.intervened = True
            appl_pump = max(appl_pump, params.PUMP_NOMINAL_FLOW_LPM)
        elif self.mode == "LIMP_HOME":
            if appl_curr > 25.0:
                orig_c = appl_curr
                appl_curr = 25.0
                self.intervened = True
                action_names.append("limp_home_clamp")
                action_reasons.append(f"clamped current {orig_c:.1f}A -> 25.0A in LIMP_HOME mode")
            appl_pump = max(appl_pump, params.PUMP_NOMINAL_FLOW_LPM)

        # ------------------------------------------------------------------
        # 5. Telemetry & Event Logging
        # ------------------------------------------------------------------
        self.last_applied_vs_requested_curr = float(appl_curr - req_curr)

        if action_names:
            self.last_action = "; ".join(action_names)
            self.last_reason = f"t={t:.0f}s: " + "; ".join(action_reasons)
            self._log_event(t, self.mode, self.last_action, self.last_reason, self.last_signals_affected)

        return float(appl_curr), float(appl_pump)

    def _log_event(
        self,
        t: float,
        mode: str,
        action: str,
        reason: str,
        signals: List[str]
    ) -> None:
        """Record an intervention event in the gate event log."""
        self.event_log.append({
            "t": float(t),
            "mode": mode,
            "action": action,
            "reason": reason,
            "signals_affected": list(set(signals)),
        })
