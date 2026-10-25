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

from typing import Dict, Tuple


class ChargerBMS:
    """
    Closed-loop Charger & BMS controller.
    
    Processes measured sensor signals and outputs (current_cmd_a, pump_cmd_lpm).
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
        self.nominal_current_a = nominal_current_a
        self.soc_cutoff = soc_cutoff
        self.temp_derate_start_c = temp_derate_start_c
        self.temp_derate_stop_c = temp_derate_stop_c
        self.pump_charging_lpm = pump_charging_lpm
        self.pump_idle_lpm = pump_idle_lpm

    def compute_commands(self, measured: Dict[str, float]) -> Tuple[float, float]:
        """
        Compute actuation commands from measured sensor readings only.
        
        Parameters
        ----------
        measured : dict
            Dictionary containing measured sensor readings from plant.measure()
            or tampered sensor signals. Must include 'soc_estimate' and 'temp_cell_a'.
            
        Returns
        -------
        tuple of (float, float)
            (current_cmd_a, pump_cmd_lpm)
        """
        soc_est = measured.get("soc_estimate", 0.0)
        temp_a = measured.get("temp_cell_a", 25.0)

        # 1. Base requested current: 128 A while measured soc_estimate < 0.80, else 0 A
        if soc_est < self.soc_cutoff:
            req_current = self.nominal_current_a
        else:
            req_current = 0.0

        # 2. Thermal derating using measured temp_cell_a:
        # Full current below 40 C, linearly down to 0 A at 50 C
        if temp_a <= self.temp_derate_start_c:
            derate_factor = 1.0
        elif temp_a >= self.temp_derate_stop_c:
            derate_factor = 0.0
        else:
            derate_factor = (self.temp_derate_stop_c - temp_a) / (
                self.temp_derate_stop_c - self.temp_derate_start_c
            )

        current_cmd_a = float(req_current * derate_factor)

        # 3. Pump command: 10 L/min while charging, 3 L/min when idle
        if current_cmd_a > 0.0:
            pump_cmd_lpm = self.pump_charging_lpm
        else:
            pump_cmd_lpm = self.pump_idle_lpm

        return current_cmd_a, pump_cmd_lpm

    def step(self, measured: Dict[str, float]) -> Tuple[float, float]:
        """Alias for compute_commands."""
        return self.compute_commands(measured)

    def __call__(self, measured: Dict[str, float]) -> Tuple[float, float]:
        """Callable interface for compute_commands."""
        return self.compute_commands(measured)
