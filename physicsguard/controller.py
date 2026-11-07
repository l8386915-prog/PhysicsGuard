"""
PhysicsGuard - Closed-Loop Charger BMS Controller
=================================================
Implements the closed-loop Charge Control and Battery Management System (BMS)
logic for DC fast-charging sessions.

The controller operates as an independent cyber-physical entity:
  - It receives ONLY measured sensor signals from the vehicle/charger instrumentation.
  - It NEVER accesses true plant state or unobserved plant parameters.
  - It outputs commanded charging current and cooling pump flow.

Control Rules:
  1. Base Current Request:
     128.0 A while measured soc_estimate < 0.80 (80%), otherwise 0.0 A.
  2. Thermal Derating (Cell Temperature A):
     Full current (100%) below 40.0 C.
     Linear derating down to 0.0 A at 50.0 C (datasheet charge trip limit).
     Current = requested * clip((50.0 - temp_cell_a) / 10.0, 0.0, 1.0).
  3. Coolant Pump Command:
     10.0 L/min while charging (current_cmd > 0), 3.0 L/min when idle (current_cmd == 0).
"""

import math
from typing import Dict, Tuple, Any


def _is_finite_number(val: Any) -> bool:
    """Check if value is a valid, finite real number."""
    if val is None:
        return False
    try:
        f = float(val)
        return math.isfinite(f)
    except (ValueError, TypeError):
        return False


class ChargerBMS:
    """
    Closed-loop Charger & BMS controller.
    
    Processes measured sensor readings and outputs (current_cmd_a, pump_cmd_lpm).
    
    Safety & Cutoff Rules:
      1. Termination Latch:
         Stops charging only after measured soc_estimate >= 0.80 on 3 consecutive
         samples, then latches at 0.0 A for the remainder of the session.
      2. Thermal Derating & NaN Safety:
         Reads temp_cell_a if finite; otherwise falls back to temp_cell_b if finite;
         otherwise assumes 45.0 C (which derates charging current by 50%).
         Full current below 40.0 C; linearly derated to 0.0 A at 50.0 C.
         A NaN in any other sensor channel will never crash the controller.
      3. Coolant Pump Command:
         10.0 L/min while actively charging (current_cmd > 0), 3.0 L/min when idle.
    """

    def __init__(
        self,
        nominal_current_a: float = 128.0,
        soc_cutoff: float = 0.80,
        temp_derate_start_c: float = 40.0,
        temp_derate_stop_c: float = 50.0,
        pump_charging_lpm: float = 10.0,
        pump_idle_lpm: float = 3.0
    ):
        self.nominal_current_a = float(nominal_current_a)
        self.soc_cutoff = float(soc_cutoff)
        self.temp_derate_start_c = float(temp_derate_start_c)
        self.temp_derate_stop_c = float(temp_derate_stop_c)
        self.pump_charging_lpm = float(pump_charging_lpm)
        self.pump_idle_lpm = float(pump_idle_lpm)

        # Cutoff state tracking
        self.consecutive_cutoff_samples: int = 0
        self.latched_stop: bool = False

    def reset(self) -> None:
        """Reset internal latch states for a new simulation session."""
        self.consecutive_cutoff_samples = 0
        self.latched_stop = False

    def compute_commands(self, measured: Dict[str, Any]) -> Tuple[float, float]:
        """
        Compute actuation commands from measured sensor readings only.
        
        Parameters
        ----------
        measured : dict
            Dictionary containing measured sensor readings from plant.measure()
            or tampered sensor signals.
            
        Returns
        -------
        tuple of (float, float)
            (current_cmd_a, pump_cmd_lpm)
        """
        # 1. SOC cutoff logic with 3 consecutive samples latch
        soc_raw = measured.get("soc_estimate") if isinstance(measured, dict) else None
        if not self.latched_stop:
            if _is_finite_number(soc_raw):
                if float(soc_raw) >= self.soc_cutoff:
                    self.consecutive_cutoff_samples += 1
                    if self.consecutive_cutoff_samples >= 3:
                        self.latched_stop = True
                else:
                    self.consecutive_cutoff_samples = 0
            else:
                self.consecutive_cutoff_samples = 0

        if self.latched_stop:
            req_current = 0.0
        else:
            req_current = self.nominal_current_a

        # 2. NaN-safe temperature determination:
        # Use temp_cell_a if finite, else temp_cell_b if finite, else assume 45 C
        temp_a_raw = measured.get("temp_cell_a") if isinstance(measured, dict) else None
        temp_b_raw = measured.get("temp_cell_b") if isinstance(measured, dict) else None

        if _is_finite_number(temp_a_raw):
            effective_temp = float(temp_a_raw)
        elif _is_finite_number(temp_b_raw):
            effective_temp = float(temp_b_raw)
        else:
            effective_temp = 45.0  # Assumes 45 C -> 50% derating

        # 3. Thermal derating calculation:
        # Full current <= 40 C, linear down to 0 A at 50 C
        if effective_temp <= self.temp_derate_start_c:
            derate_factor = 1.0
        elif effective_temp >= self.temp_derate_stop_c:
            derate_factor = 0.0
        else:
            derate_factor = (self.temp_derate_stop_c - effective_temp) / (
                self.temp_derate_stop_c - self.temp_derate_start_c
            )

        current_cmd_a = float(req_current * derate_factor)

        # 4. Pump command: 10 L/min while charging, 3 L/min when idle
        if current_cmd_a > 0.0:
            pump_cmd_lpm = self.pump_charging_lpm
        else:
            pump_cmd_lpm = self.pump_idle_lpm

        return current_cmd_a, pump_cmd_lpm

    def step(self, measured: Dict[str, Any]) -> Tuple[float, float]:
        """Alias for compute_commands."""
        return self.compute_commands(measured)

    def __call__(self, measured: Dict[str, Any]) -> Tuple[float, float]:
        """Callable interface for compute_commands."""
        return self.compute_commands(measured)

