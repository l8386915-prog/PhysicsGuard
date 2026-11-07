"""
PhysicsGuard - Digital Twin Model (The Simplified Observer)
============================================================
The Digital Twin is an engineering surrogate model of the physical system,
used by the PhysicsGuard cybersecurity framework.

Model Specifications:
  1. Nominal 1-RC electrical state: Vp with dVp/dt = -Vp/(Rp*Cp) + I_cell/Cp
     using nominal Rp, Cp; V_cell = OCV_table(soc) + I_cell*R0 + Vp;
     heat per cell = I_cell^2*R0 + Vp^2/Rp, with R0 = CELL_R0_25C_OHM (0.030).
  2. Still NO temperature dependence, NO ageing, NO OCV bump, constant
     nominal cold-plate UA (no flow exponent).
  3. Uses strictly nominal parameters (no random manufacturing perturbations).
  4. Deterministic: zero sensor noise.
  5. Estimator mode: ingests measured inputs (current, flow, ambient, coolant inlet)
     and predicts expected physical battery temperature and coolant outlet temperature.
"""

from typing import Dict, Any, Optional
import numpy as np
from . import params


class Twin:
    """
    Simplified nominal physics model of the battery pack and cooling system.
    
    Can run autonomously or as an online state estimator fed by measured inputs.
    """

    def __init__(
        self,
        initial_soc: float = params.SOC_INITIAL,
        initial_temp_c: float = params.T_AMBIENT_NOMINAL_C,
        r0_nominal: float = params.CELL_R0_25C_OHM
    ):
        """
        Initialize the Digital Twin.

        Parameters
        ----------
        initial_soc : float, default 0.20
            Initial State of Charge.
        initial_temp_c : float, default 25.0 C
            Initial estimated pack temperature.
        r0_nominal : float
            Constant DC internal resistance per cell [Ohm].
            Default is R0 = CELL_R0_25C_OHM (0.030 Ohm).
        """
        # Pack architecture (identical nominal dimensions)
        self.n_series: int = params.PACK_N_SERIES         # 89 cells in series
        self.n_parallel: int = params.PACK_N_PARALLEL     # 32 strings in parallel
        self.n_cells: int = params.PACK_TOTAL_CELLS       # 2,848 cells total
        self.q_cell_coulombs: float = params.CELL_CAPACITY_COULOMBS

        # Nominal parameters (constant, no perturbations, no temperature scaling)
        self.r0: float = float(r0_nominal)
        self.rp: float = float(params.CELL_RP_OHM)
        self.cp: float = float(params.CELL_CP_F)
        self.c_pack: float = params.PACK_TOTAL_HEAT_CAPACITY_J_PER_K
        self.ua_cold_plate: float = params.COOLING_UA_COLD_PLATE_W_PER_K
        self.ua_ambient: float = params.CASING_UA_AMBIENT_W_PER_K
        self.rho_coolant: float = params.COOLANT_DENSITY_KG_PER_M3
        self.cp_coolant: float = params.COOLANT_SPECIFIC_HEAT_J_PER_KG_K
        self.tau_pump: float = params.TAU_PUMP_SECONDS

        # Dynamic State Estimates
        self.soc: float = float(initial_soc)
        self.vp: float = 0.0                          # Polarization capacitor voltage [V]
        self.T_batt: float = float(initial_temp_c)
        self.pump_flow_lpm: float = 0.0
        self.T_cool_in: float = params.T_COOLANT_INLET_NOMINAL_C
        self.T_cool_out: float = params.T_COOLANT_INLET_NOMINAL_C
        self.T_amb: float = params.T_AMBIENT_NOMINAL_C

        # Output Signals
        self.i_pack: float = 0.0
        self.i_cell: float = 0.0
        self.v_cell: float = params.evaluate_ocv(self.soc)
        self.v_pack: float = self.v_cell * self.n_series
        self.p_cmd_kw: float = 0.0

        # Thermal rate estimates [W]
        self.q_heat: float = 0.0
        self.q_cool: float = 0.0
        self.q_amb: float = 0.0
        self.time_s: float = 0.0

    def step_estimator(
        self,
        i_pack_meas: float,
        pump_flow_meas: float,
        T_amb_meas: float,
        T_cool_in_meas: float = params.T_COOLANT_INLET_NOMINAL_C,
        dt: float = params.SIMULATION_DT_S
    ) -> Dict[str, float]:
        """
        Step the Twin forward in Estimator Mode using sensor inputs.
        
        Given measured charging current, pump flow, and environment temperatures,
        update the internal state estimates and return predicted physical signals.
        
        Parameters
        ----------
        i_pack_meas : float
            Measured total pack charging current [A].
        pump_flow_meas : float
            Measured coolant volumetric flow rate [L/min].
        T_amb_meas : float
            Measured ambient air temperature [deg C].
        T_cool_in_meas : float, default 25.0 C
            Measured coolant inlet fluid temperature [deg C].
        dt : float, default 1.0 s
            Integration step size in seconds.
            
        Returns
        -------
        dict
            Predicted physical signals without sensor noise.
        """
        self.i_pack = float(i_pack_meas)
        self.pump_flow_lpm = float(pump_flow_meas)
        self.T_amb = float(T_amb_meas)
        self.T_cool_in = float(T_cool_in_meas)

        # 1. Electrical Calculation: 1-RC nominal model
        self.i_cell = self.i_pack / self.n_parallel
        ocv = params.evaluate_ocv(self.soc)
        self.v_cell = ocv + self.i_cell * self.r0 + self.vp
        self.v_pack = self.v_cell * self.n_series
        self.p_cmd_kw = (self.v_pack * self.i_pack) / 1000.0

        # 2. Dynamic States Update: Polarization voltage & SOC Coulomb counting
        # dVp/dt = -Vp / (Rp * Cp) + I_cell / Cp
        d_vp = (-self.vp / (self.rp * self.cp) + self.i_cell / self.cp) * dt
        self.vp += d_vp

        # dSOC/dt = I_cell / Q_cell_coulombs
        d_soc = (self.i_cell * dt) / self.q_cell_coulombs
        self.soc = float(np.clip(self.soc + d_soc, 0.0, 1.0))

        # 3. Joule & Polarization Heat Generation:
        # heat per cell = I_cell^2 * R0 + Vp^2 / Rp
        q_cell = (self.i_cell ** 2) * self.r0 + (self.vp ** 2) / self.rp
        self.q_heat = self.n_cells * q_cell

        # 4. Coolant Heat Removal (Effectiveness-NTU method with constant nominal UA, no flow exponent)
        flow_m3_s = (self.pump_flow_lpm / 1000.0) / 60.0
        m_dot = self.rho_coolant * flow_m3_s
        c_dot = m_dot * self.cp_coolant

        # If flow is below 1.0 L/min, coolant is stagnant: no heat removal,
        # and the outlet temperature relaxes toward the pack temperature with
        # a 90 s time constant (stagnant fluid warms up toward pack temperature).
        if self.pump_flow_lpm < 1.0:
            self.q_cool = 0.0
            self.T_cool_out += (self.T_batt - self.T_cool_out) / 90.0 * dt
        else:
            ntu = self.ua_cold_plate / c_dot
            effectiveness = 1.0 - np.exp(-ntu)
            self.q_cool = effectiveness * c_dot * (self.T_batt - self.T_cool_in)
            self.T_cool_out = self.T_cool_in + self.q_cool / c_dot

        # 5. Ambient Enclosure Loss
        self.q_amb = self.ua_ambient * (self.T_batt - self.T_amb)

        # 6. Thermal State Integration
        # First Law: C_pack * dT/dt = Q_heat - Q_cool - Q_amb
        dT_dt = (self.q_heat - self.q_cool - self.q_amb) / self.c_pack
        self.T_batt += dT_dt * dt
        self.time_s += dt

        return self.get_predicted_signals()

    def step(
        self,
        dt: float = params.SIMULATION_DT_S,
        pump_flow_cmd_lpm: float = params.PUMP_NOMINAL_FLOW_LPM,
        T_cool_in_c: float = params.T_COOLANT_INLET_NOMINAL_C,
        T_amb_c: float = params.T_AMBIENT_NOMINAL_C,
        current_override_a: Optional[float] = None
    ) -> None:
        """
        Autonomous step method matching the Plant interface.
        """
        self.T_cool_in = float(T_cool_in_c)
        self.T_amb = float(T_amb_c)

        # Flow lag
        d_flow = (pump_flow_cmd_lpm - self.pump_flow_lpm) / self.tau_pump
        self.pump_flow_lpm = float(np.clip(self.pump_flow_lpm + d_flow * dt, 0.0, 50.0))

        # Autonomous charging current command
        if current_override_a is not None:
            i_pack = float(current_override_a)
        else:
            if self.soc >= params.SOC_TARGET:
                i_pack = 0.0
            else:
                # CC clamp logic with 1-RC headroom
                i_cell_rated = params.CELL_RATED_CHARGE_CURRENT_A
                ocv = params.evaluate_ocv(self.soc)
                v_headroom = params.CELL_V_MAX - ocv - self.vp
                i_cell_max = max(0.0, v_headroom / self.r0) if v_headroom > 0 else 0.0
                i_cell = min(i_cell_rated, i_cell_max)
                i_pack = i_cell * self.n_parallel

        # Step estimator with computed command
        self.step_estimator(
            i_pack_meas=i_pack,
            pump_flow_meas=self.pump_flow_lpm,
            T_amb_meas=self.T_amb,
            T_cool_in_meas=self.T_cool_in,
            dt=dt
        )

    def measure(self) -> Dict[str, float]:
        """
        Returns clean nominal model signals (same keys as Plant.measure(), but no noise).
        """
        return self.get_predicted_signals()

    def get_predicted_signals(self) -> Dict[str, float]:
        """
        Return the 10 predicted sensor channels from the Digital Twin.
        """
        return {
            "pack_voltage": float(self.v_pack),
            "pack_current": float(self.i_pack),
            "soc_estimate": float(self.soc),
            "temp_cell_a": float(self.T_batt),
            "temp_cell_b": float(self.T_batt),  # No spatial offset or noise in twin
            "coolant_temp_in": float(self.T_cool_in),
            "coolant_temp_out": float(self.T_cool_out),
            "pump_flow_lpm": float(self.pump_flow_lpm),
            "ambient_temp": float(self.T_amb),
            "charger_power_cmd_kw": float(self.p_cmd_kw),
        }

    def get_true_state(self) -> Dict[str, float]:
        """
        Return full state dictionary matching Plant.get_true_state().
        """
        return {
            "time_s": self.time_s,
            "true_battery_temp": self.T_batt,
            "true_soc": self.soc,
            "true_pack_voltage": self.v_pack,
            "true_pack_current": self.i_pack,
            "true_cell_voltage": self.v_cell,
            "true_cell_current": self.i_cell,
            "true_vp": self.vp,
            "true_q_heat": self.q_heat,
            "true_q_cool": self.q_cool,
            "true_q_amb": self.q_amb,
            "true_pump_flow_lpm": self.pump_flow_lpm,
            "true_coolant_temp_in": self.T_cool_in,
            "true_coolant_temp_out": self.T_cool_out,
            "true_ambient_temp": self.T_amb,
            "true_charger_power_kw": self.p_cmd_kw,
        }

