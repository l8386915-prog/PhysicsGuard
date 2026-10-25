"""
PhysicsGuard - Physical Plant Model (The "Real" World)
======================================================
This module models the detailed physical dynamics of an Electric Vehicle (EV)
DC Fast-Charging session, including:
  1. High-fidelity 1-RC Thevenin equivalent circuit for battery electrochemistry.
  2. Arrhenius temperature-dependent internal resistance and cell ageing.
  3. Coulomb-counting State of Charge (SOC) tracking.
  4. Non-linear cell voltage clamping (CC-CV transition protection at 4.20 V).
  5. Lumped thermal energy conservation with Joule heating and polarization loss.
  6. Effectiveness-NTU heat exchanger model for the liquid cold-plate cooling loop.
  7. Fluid pump dynamic first-order response lag.
  8. Realistic noisy sensor measurements (10 sensor channels).
  9. Seeded +/-10% parameter perturbation functionality.

All units follow the International System of Units (SI):
  - Current: Amperes [A] (positive into battery for charging)
  - Voltage: Volts [V]
  - Temperature: Degrees Celsius [deg C]
  - Energy & Heat: Joules [J], Watts [W]
  - Flow: Liters per minute [L/min]
"""

from typing import Dict, Any, Optional
import numpy as np
from . import params


def get_perturbed_parameters(seed: Optional[int] = None, fraction: float = 0.10) -> Dict[str, float]:
    """
    Generate physical parameters perturbed randomly by +/-fraction (default 10%).
    
    This ensures that the 'Plant' (real physical vehicle) is never identical to
    the 'Twin' (nominal digital model), simulating real-world manufacturing
    tolerances and modeling uncertainties.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for repeatable perturbation.
    fraction : float, default 0.10
        Maximum fractional perturbation (+/- 10%).
        
    Returns
    -------
    dict
        Dictionary of perturbed physical parameters.
    """
    rng = np.random.default_rng(seed)
    
    def perturb(val: float) -> float:
        # Uniform variation between (1 - fraction) and (1 + fraction)
        delta = rng.uniform(-fraction, fraction)
        return float(val * (1.0 + delta))

    return {
        "r0_25": perturb(params.CELL_R0_25C_OHM),
        "rp": perturb(params.CELL_RP_OHM),
        "cp": perturb(params.CELL_CP_F),
        "c_pack": perturb(params.PACK_TOTAL_HEAT_CAPACITY_J_PER_K),
        "ua_cold_plate": perturb(params.COOLING_UA_COLD_PLATE_W_PER_K),
        "ua_ambient": perturb(params.CASING_UA_AMBIENT_W_PER_K),
        "rho_coolant": perturb(params.COOLANT_DENSITY_KG_PER_M3),
        "cp_coolant": perturb(params.COOLANT_SPECIFIC_HEAT_J_PER_KG_K),
        "tau_pump": perturb(params.TAU_PUMP_SECONDS),
    }


class Plant:
    """
    Detailed physical simulator of the EV battery pack and cooling system.
    
    Models cell electrochemistry, polarization relaxation, fluid heat removal,
    thermal energy storage, and noisy instrumentation.
    """

    def __init__(
        self,
        perturb: bool = False,
        seed: Optional[int] = None,
        age_factor: float = 1.0,
        initial_soc: float = params.SOC_INITIAL,
        initial_temp_c: float = params.T_AMBIENT_NOMINAL_C,
        sensor_seed: Optional[int] = None
    ):
        """
        Initialize the Battery Pack and Cooling Loop Plant.

        Parameters
        ----------
        perturb : bool, default False
            If True, randomly perturb physical parameters by +/-10%.
        seed : int, optional
            Seed for parameter perturbation.
        age_factor : float, default 1.0
            Resistance multiplier to represent cell electrochemical ageing.
        initial_soc : float, default 0.20 (20%)
            Initial State of Charge.
        initial_temp_c : float, default 25.0 C
            Initial battery pack uniform temperature in Celsius.
        sensor_seed : int, optional
            Seed for Gaussian sensor measurement noise generation.
        """
        # Pack geometric and electrical architecture
        self.n_series: int = params.PACK_N_SERIES         # 89 cells in series
        self.n_parallel: int = params.PACK_N_PARALLEL     # 32 strings in parallel
        self.n_cells: int = params.PACK_TOTAL_CELLS       # 2,848 cells total
        self.q_cell_coulombs: float = params.CELL_CAPACITY_COULOMBS  # 10,620 C per cell

        # Apply nominal or perturbed physical parameters
        if perturb:
            p = get_perturbed_parameters(seed=seed, fraction=0.10)
            self.r0_25: float = p["r0_25"]
            self.rp: float = p["rp"]
            self.cp: float = p["cp"]
            self.c_pack: float = p["c_pack"]
            self.ua_cold_plate: float = p["ua_cold_plate"]
            self.ua_ambient: float = p["ua_ambient"]
            self.rho_coolant: float = p["rho_coolant"]
            self.cp_coolant: float = p["cp_coolant"]
            self.tau_pump: float = p["tau_pump"]
        else:
            self.r0_25 = params.CELL_R0_25C_OHM
            self.rp = params.CELL_RP_OHM
            self.cp = params.CELL_CP_F
            self.c_pack = params.PACK_TOTAL_HEAT_CAPACITY_J_PER_K
            self.ua_cold_plate = params.COOLING_UA_COLD_PLATE_W_PER_K
            self.ua_ambient = params.CASING_UA_AMBIENT_W_PER_K
            self.rho_coolant = params.COOLANT_DENSITY_KG_PER_M3
            self.cp_coolant = params.COOLANT_SPECIFIC_HEAT_J_PER_KG_K
            self.tau_pump = params.TAU_PUMP_SECONDS

        # Temperature sensitivity of R0 and ageing factor
        self.r0_alpha: float = params.CELL_R0_TEMP_COEFF  # 0.02 1/K
        self.age_factor: float = age_factor

        # Sensor random number generator (reproducible noise)
        self.sensor_rng = np.random.default_rng(sensor_seed)

        # Dynamic State Variables
        self.soc: float = float(initial_soc)          # State of Charge [0.0 to 1.0]
        self.vp: float = 0.0                          # Polarization capacitor voltage [V]
        self.T_batt: float = float(initial_temp_c)    # Battery pack temperature [deg C]
        self.pump_flow_lpm: float = 0.0               # Actual coolant volumetric flow [L/min]
        self.T_cool_in: float = params.T_COOLANT_INLET_NOMINAL_C   # Coolant inlet [deg C]
        self.T_cool_out: float = params.T_COOLANT_INLET_NOMINAL_C  # Coolant outlet [deg C]
        self.T_amb: float = params.T_AMBIENT_NOMINAL_C             # Ambient temperature [deg C]

        # Electrical variables at current time step
        self.i_pack: float = 0.0                      # Total pack current [A]
        self.i_cell: float = 0.0                      # Single cell current [A]
        self.v_cell: float = self.get_cell_ocv(self.soc)  # Single cell voltage [V]
        self.v_pack: float = self.v_cell * self.n_series    # Total pack voltage [V]
        self.p_cmd_kw: float = 0.0                    # Charger power command [kW]

        # Commanded inputs & actuator limits
        self.applied_current_cmd: float = 0.0
        self.last_applied_current_cmd: float = 0.0
        self.max_pump_flow_lpm: float = 50.0

        # Thermal rate variables [W]
        self.q_heat: float = 0.0                      # Heat generated in pack [W]
        self.q_cool: float = 0.0                      # Heat removed by coolant [W]
        self.q_amb: float = 0.0                       # Heat lost to ambient air [W]

        # Cumulative Energy Accounting [Joules] for First-Law energy balance verification
        self.initial_T_batt: float = float(initial_temp_c)
        self.cum_heat_gen_J: float = 0.0              # Total heat generated sum(Q_heat * dt)
        self.cum_heat_cool_J: float = 0.0             # Total heat removed sum(Q_cool * dt)
        self.cum_heat_amb_J: float = 0.0              # Total ambient loss sum(Q_amb * dt)
        self.time_s: float = 0.0                      # Elapsed simulation time [s]

    def get_cell_ocv(self, soc: float) -> float:
        """
        Physical plant cell open-circuit voltage with structural mismatch.
        
        ASSUMPTION: Real cells differ from the nominal LUT:
            OCV = nominal_OCV(SOC) + 0.010 * sin(3 * pi * SOC) [V].
        """
        soc_clamped = float(np.clip(soc, 0.0, 1.0))
        nominal_ocv = params.evaluate_ocv(soc_clamped)
        # Structural mismatch: 0.010 * sin(3 * pi * soc)
        return float(nominal_ocv + 0.010 * np.sin(3.0 * np.pi * soc_clamped))

    def get_cell_r0(self, temp_c: float) -> float:
        """
        Calculate cell ohmic resistance R0 at a given temperature and age.
        
        Textbook Equation:
            R0(T) = R0_25 * exp(-0.02 * (T - 25)) * age_factor
        """
        return self.r0_25 * np.exp(-self.r0_alpha * (temp_c - 25.0)) * self.age_factor

    def compute_charging_current(self, pump_cmd_lpm: float = params.PUMP_NOMINAL_FLOW_LPM) -> float:
        """
        Determine allowable charging current under CC-CV protocol and safety clamps.
        
        Constant Current (CC) of 4.0 A per cell (128 A pack) from SOC 20% to 80%.
        If cell terminal voltage approaches 4.20 V, clamp current so V_cell <= 4.20 V.
        If SOC >= 80% (0.80), charging terminates (current = 0).
        
        Returns
        -------
        float
            Total pack charging current in Amperes [A].
        """
        # 1. Termination condition: charging stops once target SOC (80%) is reached
        if self.soc >= params.SOC_TARGET:
            return 0.0

        # 2. Maximum allowable current by cell datasheet rating (4.0 A per cell)
        i_cell_rated = params.CELL_RATED_CHARGE_CURRENT_A

        # 3. Voltage clamp protection calculation:
        # Cell terminal voltage: V_cell = OCV(SOC) + I_cell * R0 + Vp
        # To guarantee V_cell <= 4.20 V:
        #   I_cell * R0 <= 4.20 - OCV(SOC) - Vp
        #   I_cell <= (4.20 - OCV - Vp) / R0
        r0 = self.get_cell_r0(self.T_batt)
        ocv = self.get_cell_ocv(self.soc)
        voltage_headroom = params.CELL_V_MAX - ocv - self.vp

        if voltage_headroom <= 0.0:
            i_cell_clamped = 0.0
        else:
            i_cell_clamped = voltage_headroom / r0

        # Choose the minimum between rated current and voltage-clamped current
        i_cell = float(np.clip(i_cell_rated, 0.0, i_cell_clamped))

        # Pack current = cell current * parallel strings (32 strings)
        return i_cell * self.n_parallel

    def step(
        self,
        dt: float = params.SIMULATION_DT_S,
        current_cmd_a: Optional[float] = None,
        pump_cmd_lpm: float = params.PUMP_NOMINAL_FLOW_LPM,
        T_cool_in_c: float = params.T_COOLANT_INLET_NOMINAL_C,
        T_amb_c: float = params.T_AMBIENT_NOMINAL_C,
        pump_flow_cmd_lpm: Optional[float] = None,
        current_override_a: Optional[float] = None
    ) -> None:
        """
        Advance the physical simulation forward by time step dt.
        
        Parameters
        ----------
        dt : float, default 1.0 s
            Discrete integration time step in seconds.
        current_cmd_a : float, optional
            Commanded total pack charging current in Amperes [A].
            The plant follows this commanded current, except for the hard clamp
            that keeps cell terminal voltage <= 4.20 V.
        pump_cmd_lpm : float, default 10.0 L/min
            Pump volumetric flow command in Liters per minute [L/min].
        T_cool_in_c : float
            Inlet coolant fluid temperature in Celsius.
        T_amb_c : float
            Surrounding ambient air temperature in Celsius.
        pump_flow_cmd_lpm : float, optional
            Alias for pump_cmd_lpm (backwards compatibility).
        current_override_a : float, optional
            Alias for current_cmd_a (backwards compatibility).
        """
        if pump_flow_cmd_lpm is not None:
            pump_cmd_lpm = pump_flow_cmd_lpm
        if current_override_a is not None:
            current_cmd_a = current_override_a

        self.T_cool_in = float(T_cool_in_c)
        self.T_amb = float(T_amb_c)

        # ----------------------------------------------------------------------
        # 1. PUMP FLOW DYNAMICS (First-order mechanical/fluid lag)
        # ----------------------------------------------------------------------
        # Equation: d(Flow)/dt = (Flow_cmd - Flow) / tau_pump
        # Real fluid pumps cannot jump instantaneously to target flow rate.
        d_flow = (pump_cmd_lpm - self.pump_flow_lpm) / self.tau_pump
        max_flow = getattr(self, "max_pump_flow_lpm", 50.0)
        self.pump_flow_lpm = float(np.clip(self.pump_flow_lpm + d_flow * dt, 0.0, max_flow))

        # ----------------------------------------------------------------------
        # 2. CHARGING CONTROL & ELECTRICAL STATES
        # ----------------------------------------------------------------------
        r0 = self.get_cell_r0(self.T_batt)
        ocv = self.get_cell_ocv(self.soc)

        # Existing hard clamp calculation that keeps cell voltage <= 4.20 V:
        # V_cell = OCV + I_cell * R0 + Vp <= 4.20 V  ==>  I_cell <= (4.20 - OCV - Vp) / R0
        voltage_headroom = params.CELL_V_MAX - ocv - self.vp
        max_i_cell = max(0.0, voltage_headroom / r0) if voltage_headroom > 0.0 else 0.0
        max_i_pack = max_i_cell * self.n_parallel

        if current_cmd_a is not None:
            self.applied_current_cmd = float(current_cmd_a)
            self.last_applied_current_cmd = float(current_cmd_a)
            # Follow commanded current, subject to cell voltage <= 4.20 V clamp
            self.i_pack = float(np.clip(current_cmd_a, 0.0, max_i_pack))
        else:
            self.i_pack = self.compute_charging_current(pump_cmd_lpm)
            self.applied_current_cmd = self.i_pack
            self.last_applied_current_cmd = self.i_pack

        # Cell-level current (pack current divided evenly across 32 strings)
        self.i_cell = self.i_pack / self.n_parallel

        # Terminal voltage per cell: V = OCV(SOC) + I_cell * R0 + Vp
        self.v_cell = ocv + self.i_cell * r0 + self.vp
        # Total pack voltage across all 89 cells in series
        self.v_pack = self.v_cell * self.n_series

        # Charger electrical power delivered: P = V_pack * I_pack [kW]
        self.p_cmd_kw = (self.v_pack * self.i_pack) / 1000.0

        # ----------------------------------------------------------------------
        # 3. POLARIZATION DYNAMICS & STATE OF CHARGE
        # ----------------------------------------------------------------------
        # Differential equation: dVp/dt = -Vp / (Rp * Cp) + I_cell / Cp
        # Models diffusion overvoltage in the porous battery electrodes.
        d_vp = (-self.vp / (self.rp * self.cp) + self.i_cell / self.cp) * dt
        self.vp += d_vp

        # State of Charge by Coulomb Counting:
        # dSOC/dt = I_cell / Q_cell_coulombs
        d_soc = (self.i_cell * dt) / self.q_cell_coulombs
        self.soc = float(np.clip(self.soc + d_soc, 0.0, 1.0))

        # ----------------------------------------------------------------------
        # 4. THERMAL HEAT GENERATION (Joule Heating + Polarization Loss)
        # ----------------------------------------------------------------------
        # Heat generated per single cell [Watts]:
        #   q_cell = I_cell^2 * R0 + Vp^2 / Rp
        q_cell = (self.i_cell ** 2) * r0 + (self.vp ** 2) / self.rp
        # Total pack heat generation across all 2,848 cells:
        self.q_heat = self.n_cells * q_cell

        # ----------------------------------------------------------------------
        # 5. COOLANT HEAT REMOVAL (Effectiveness-NTU Method with UA Flow Scaling)
        # ----------------------------------------------------------------------
        # Convert flow from Liters/min to m^3/s:
        #   V_dot [m^3/s] = (L/min) / (1000 L/m^3 * 60 s/min)
        flow_m3_s = (self.pump_flow_lpm / 1000.0) / 60.0
        # Mass flow rate [kg/s]: m_dot = rho * V_dot
        m_dot = self.rho_coolant * flow_m3_s
        # Fluid heat capacity rate [W/K]: C_dot = m_dot * cp
        c_dot = m_dot * self.cp_coolant

        # If pump is turned off or flow is negligible, no heat is removed by fluid (UA = 0)
        if c_dot < 1e-4 or self.pump_flow_lpm <= 0.0:
            self.q_cool = 0.0
            self.T_cool_out = self.T_cool_in
        else:
            # Structural mismatch: cold-plate UA scales with flow:
            # UA = UA_nominal * (flow_lpm / 10.0) ** 0.6 (UA = 0 when flow is 0).
            # At 10 L/min results are unchanged.
            ua_scaled = self.ua_cold_plate * ((self.pump_flow_lpm / 10.0) ** params.UA_FLOW_EXPONENT)
            # Number of Transfer Units: NTU = UA / C_dot
            ntu = ua_scaled / c_dot
            # Thermal effectiveness: epsilon = 1 - exp(-NTU)
            effectiveness = 1.0 - np.exp(-ntu)
            # Heat transfer rate [W]: Q_cool = epsilon * C_dot * (T_batt - T_cool_in)
            self.q_cool = effectiveness * c_dot * (self.T_batt - self.T_cool_in)
            # Fluid temperature at heat-exchanger outlet [deg C]:
            self.T_cool_out = self.T_cool_in + self.q_cool / c_dot

        # ----------------------------------------------------------------------
        # 6. CASING HEAT LOSS TO SURROUNDING AMBIENT AIR
        # ----------------------------------------------------------------------
        # Natural convection & radiation loss from aluminum pack enclosure:
        self.q_amb = self.ua_ambient * (self.T_batt - self.T_amb)

        # ----------------------------------------------------------------------
        # 7. LUMPED THERMAL ENERGY BALANCE
        # ----------------------------------------------------------------------
        # First Law of Thermodynamics:
        #   C_pack * dT/dt = Q_heat - Q_cool - Q_amb
        dT_dt = (self.q_heat - self.q_cool - self.q_amb) / self.c_pack
        self.T_batt += dT_dt * dt

        # Cumulative energy accounting [Joules] for First Law verification
        self.cum_heat_gen_J += self.q_heat * dt
        self.cum_heat_cool_J += self.q_cool * dt
        self.cum_heat_amb_J += self.q_amb * dt
        self.time_s += dt

    def measure(self) -> Dict[str, float]:
        """
        Sample the 10 physical sensors with realistic zero-mean Gaussian noise.
        
        Sensors:
          1.  pack_voltage          [V]     (noise std = 0.08 V)
          2.  pack_current          [A]     (noise std = 0.15 A)
          3.  soc_estimate          [-]     (noise std = 0.002)
          4.  temp_cell_a           [deg C] (noise std = 0.15 C)
          5.  temp_cell_b           [deg C] (offset +0.25 C from A, noise std = 0.15 C)
          6.  coolant_temp_in       [deg C] (noise std = 0.10 C)
          7.  coolant_temp_out      [deg C] (noise std = 0.10 C)
          8.  pump_flow_lpm         [L/min] (noise std = 0.08 L/min)
          9.  ambient_temp          [deg C] (noise std = 0.10 C)
          10. charger_power_cmd_kw  [kW]    (noise std = 0.05 kW)
          
        Returns
        -------
        dict
            Mapping from sensor name to measured numerical value.
        """
        # Sensor A reads battery temperature with instrumentation noise
        noise_ta = self.sensor_rng.normal(0.0, 0.15)
        temp_a = float(self.T_batt + noise_ta)

        # Sensor B is placed at a slightly different physical location (+0.25 C offset)
        # and has its own independent sensor noise channel
        noise_tb = self.sensor_rng.normal(0.0, 0.15)
        temp_b = float((self.T_batt + 0.25) + noise_tb)

        # Pack voltage & current measurements
        v_pack_meas = float(self.v_pack + self.sensor_rng.normal(0.0, 0.08))
        i_pack_meas = float(self.i_pack + self.sensor_rng.normal(0.0, 0.15))

        # SOC estimation (BMS Kalman filter / Coulomb counter estimate with small noise)
        soc_meas = float(np.clip(self.soc + self.sensor_rng.normal(0.0, 0.002), 0.0, 1.0))

        # Coolant temperatures
        t_in_meas = float(self.T_cool_in + self.sensor_rng.normal(0.0, 0.10))
        t_out_meas = float(self.T_cool_out + self.sensor_rng.normal(0.0, 0.10))

        # Pump flow (must be non-negative)
        flow_meas = float(max(0.0, self.pump_flow_lpm + self.sensor_rng.normal(0.0, 0.08)))

        # Ambient air temperature
        t_amb_meas = float(self.T_amb + self.sensor_rng.normal(0.0, 0.10))

        # Charger commanded power in kW:
        # charger_power_cmd_kw = applied_current_cmd * true pack voltage / 1000 + noise (std 0.05 kW)
        p_cmd_kw_val = (self.applied_current_cmd * self.v_pack) / 1000.0
        p_cmd_meas = float(p_cmd_kw_val + self.sensor_rng.normal(0.0, 0.05))

        return {
            "pack_voltage": v_pack_meas,
            "pack_current": i_pack_meas,
            "soc_estimate": soc_meas,
            "temp_cell_a": temp_a,
            "temp_cell_b": temp_b,
            "coolant_temp_in": t_in_meas,
            "coolant_temp_out": t_out_meas,
            "pump_flow_lpm": flow_meas,
            "ambient_temp": t_amb_meas,
            "charger_power_cmd_kw": p_cmd_meas,
        }

    def get_true_state(self) -> Dict[str, float]:
        """
        Return the uncorrupted 'ground truth' physics state for simulation logging.
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
