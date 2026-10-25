"""
PhysicsGuard - Digital Twin Model (The Simplified Observer)
============================================================
The Digital Twin is an engineering surrogate model of the physical system,
used in subsequent phases by the PhysicsGuard cyberattack anomaly detector.

Key Differences from the Physical Plant:
  1. Simplified 0-RC (Rint) equivalent circuit: no polarization capacitor (Vp = 0).
  2. Constant DC internal resistance: R_dc = R0 + Rp, with no temperature dependence
     and no electrochemical ageing factor.
  3. Uses strictly nominal parameters (no random manufacturing perturbations).
  4. Deterministic: zero sensor noise.
  5. Estimator mode: ingests measured/commanded signals (current, flow, ambient)
     and predicts expected physical battery temperature and coolant outlet temperature.

Beginner Mechanical Engineering Note:
-------------------------------------
Why R_dc = R0 + Rp?
In an RC circuit, a capacitor behaves as an open-circuit to steady direct current (DC).
Therefore, all current passes through both R0 and Rp in steady state. A simplified
0-RC model lumps these together: R_int = R0 + Rp.

Twin Honesty:
-------------
R0 + Rp is the steady-state effective heating resistance at constant current, and
it was NOT tuned by repeated test runs. Also note that at 35 C the plant's real
resistance is lower, so the twin slightly overestimates heat.
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
        r0_nominal: float = params.CELL_R0_25C_OHM + params.CELL_RP_OHM
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
            Default is R0 + Rp = 0.030 + 0.015 = 0.045 Ohm.
        """
        # Pack architecture (identical nominal dimensions)
        self.n_series: int = params.PACK_N_SERIES         # 89 cells in series
        self.n_parallel: int = params.PACK_N_PARALLEL     # 32 strings in parallel
        self.n_cells: int = params.PACK_TOTAL_CELLS       # 2,848 cells total
        self.q_cell_coulombs: float = params.CELL_CAPACITY_COULOMBS

        # Nominal parameters (constant, no perturbations, no temperature scaling)
        # NOTE: R0 + Rp is the steady-state effective heating resistance at constant current,
        # and it was NOT tuned by repeated test runs. Also note that at 35 C the plant's real
        # resistance is lower, so the twin slightly overestimates heat.
        self.r0: float = float(r0_nominal)
        self.c_pack: float = params.PACK_TOTAL_HEAT_CAPACITY_J_PER_K
        self.ua_cold_plate: float = params.COOLING_UA_COLD_PLATE_W_PER_K
        self.ua_ambient: float = params.CASING_UA_AMBIENT_W_PER_K
        self.rho_coolant: float = params.COOLANT_DENSITY_KG_PER_M3
        self.cp_coolant: float = params.COOLANT_SPECIFIC_HEAT_J_PER_KG_K
        self.tau_pump: float = params.TAU_PUMP_SECONDS

        # Dynamic State Estimates
        self.soc: float = float(initial_soc)
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

        # 1. Electrical Calculation
        # Cell current = pack current / 32 strings
        self.i_cell = self.i_pack / self.n_parallel

        # Cell terminal voltage: 0-RC simplified model (V = OCV + I*R0)
        ocv = params.evaluate_ocv(self.soc)
        self.v_cell = ocv + self.i_cell * self.r0
        self.v_pack = self.v_cell * self.n_series
        self.p_cmd_kw = (self.v_pack * self.i_pack) / 1000.0

        # 2. State of Charge Update (Coulomb Counting)
        d_soc = (self.i_cell * dt) / self.q_cell_coulombs
        self.soc = float(np.clip(self.soc + d_soc, 0.0, 1.0))

        # 3. Joule Heat Generation
        # Cell heat = I_cell^2 * R0 (constant R0, no polarization state Vp)
        q_cell = (self.i_cell ** 2) * self.r0
        self.q_heat = self.n_cells * q_cell

        # 4. Coolant Heat Removal (Effectiveness-NTU method with nominal parameters)
        flow_m3_s = (self.pump_flow_lpm / 1000.0) / 60.0
        m_dot = self.rho_coolant * flow_m3_s
        c_dot = m_dot * self.cp_coolant

        if c_dot < 1e-4:
            self.q_cool = 0.0
            self.T_cool_out = self.T_cool_in
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
                # CC clamp logic
                i_cell_rated = params.CELL_RATED_CHARGE_CURRENT_A
                ocv = params.evaluate_ocv(self.soc)
                v_headroom = params.CELL_V_MAX - ocv
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
            "true_vp": 0.0,
            "true_q_heat": self.q_heat,
            "true_q_cool": self.q_cool,
            "true_q_amb": self.q_amb,
            "true_pump_flow_lpm": self.pump_flow_lpm,
            "true_coolant_temp_in": self.T_cool_in,
            "true_coolant_temp_out": self.T_cool_out,
            "true_ambient_temp": self.T_amb,
            "true_charger_power_kw": self.p_cmd_kw,
        }
