"""
PhysicsGuard - System Parameters
================================
All physical constants, cell specifications, pack dimensions, coolant
properties, and simulation defaults are defined in this single file.

Every parameter includes its numerical value, SI unit, and citation
(either a specific SOURCE or marked explicitly as an ASSUMPTION).

Target System:
EV DC Fast Charger + Battery Pack + Liquid Cooling Loop
Equivalent to Tata Nexon EV class (30 kWh, 320 V nominal).
"""

import numpy as np

# ==============================================================================
# 1. BATTERY CELL SPECIFICATIONS (Samsung INR18650-30Q)
# ==============================================================================
# Cell chemistry: Nickel-Manganese-Cobalt (NMC) cylindrical cell.
# High-drain Li-ion cell commonly used in EV applications.

# Nominal electrical capacity of a single cell in Ampere-hours [Ah]
# SOURCE: Samsung SDI INR18650-30Q Official Datasheet (Specification 2.95 Ah)
CELL_CAPACITY_AH: float = 2.95

# Electrical capacity in Coulombs [C = A*s]
# 1 Ampere-hour = 3600 Ampere-seconds (Coulombs)
CELL_CAPACITY_COULOMBS: float = CELL_CAPACITY_AH * 3600.0  # 10,620 C

# Nominal open-circuit cell voltage [V]
# SOURCE: Samsung SDI INR18650-30Q Datasheet
CELL_NOMINAL_VOLTAGE_V: float = 3.6

# Mass of a single bare cylindrical cell [kg]
# SOURCE: Samsung SDI INR18650-30Q Datasheet (max 48.0 g = 0.048 kg)
CELL_MASS_KG: float = 0.048

# Maximum rated continuous charging current per cell [A]
# 4.0 A corresponds to ~1.36 C fast charge rate.
# SOURCE: Samsung SDI INR18650-30Q Datasheet (Standard charge: 1.5 A, Max charge: 4.0 A)
CELL_RATED_CHARGE_CURRENT_A: float = 4.0

# Temperature thresholds for charging safety [deg C]
# Normal operating ceiling warning at 45 C, emergency charge disconnect at 50 C.
# SOURCE: Samsung SDI INR18650-30Q Datasheet (Operating temperature: charge 0 to 50 C)
CELL_TEMP_WARN_C: float = 45.0
CELL_TEMP_TRIP_C: float = 50.0

# Upper cut-off voltage clamp per cell [V]
# SOURCE: Samsung SDI INR18650-30Q Datasheet
CELL_V_MAX: float = 4.20

# Lower discharge cut-off voltage per cell [V]
# SOURCE: Samsung SDI INR18650-30Q Datasheet
CELL_V_MIN: float = 3.00


# ==============================================================================
# 2. BATTERY PACK ARCHITECTURE
# ==============================================================================
# Configuration: 89 Series x 32 Parallel (89s32p)
# Total cells = 89 * 32 = 2,848 cells.
# Pack Nominal Voltage = 89 * 3.6 V = 320.4 V
# Pack Total Capacity = 32 * 2.95 Ah = 94.4 Ah
# Pack Nominal Energy = 320.4 V * 94.4 Ah = 30.245 kWh (~30 kWh class)
# ASSUMPTION: Modelled after Tata Nexon EV class pack (~30 kWh, ~320 V bus).
PACK_N_SERIES: int = 89
PACK_N_PARALLEL: int = 32
PACK_TOTAL_CELLS: int = PACK_N_SERIES * PACK_N_PARALLEL  # 2848 cells

PACK_CAPACITY_AH: float = CELL_CAPACITY_AH * PACK_N_PARALLEL      # 94.4 Ah
PACK_CAPACITY_COULOMBS: float = CELL_CAPACITY_COULOMBS * PACK_N_PARALLEL  # 339,840 C
PACK_NOMINAL_ENERGY_KWH: float = (PACK_N_SERIES * CELL_NOMINAL_VOLTAGE_V * PACK_CAPACITY_AH) / 1000.0


# ==============================================================================
# 3. EQUIVALENT CIRCUIT MODEL (ECM) PARAMETERS PER CELL
# ==============================================================================
# The Thevenin 1-RC equivalent circuit represents cell terminal voltage:
#   V_cell = OCV(SOC) + I_cell * R0(T) + V_polarization
# Note: Charging current is defined as POSITIVE into the cell.

# Ohmic DC series resistance R0 at reference temperature (25 deg C) [Ohm]
# ASSUMPTION: Typical DC internal resistance for 18650-30Q.
# (Datasheet gives only 1 kHz AC impedance <= 26 mOhm; DC resistance is slightly higher).
CELL_R0_25C_OHM: float = 0.030

# Temperature sensitivity coefficient of R0 [1/K]
# Arrhenius-type empirical relation: R0(T) = R0_25 * exp(-0.02 * (T - 25))
# ASSUMPTION: Standard textbook approximation for Li-ion electrolyte conductivity.
CELL_R0_TEMP_COEFF: float = 0.02

# Polarization RC pair representing diffusion / charge-transfer dynamics:
# Polarization resistance Rp [Ohm]
# ASSUMPTION: Chosen from typical 1-RC parameter identification studies on NMC cells.
CELL_RP_OHM: float = 0.015

# Polarization capacitance Cp [Farads]
# ASSUMPTION: Chosen such that polarization time constant tau_p = Rp * Cp = 30 seconds.
CELL_CP_F: float = 2000.0


# ==============================================================================
# 4. THERMAL PROPERTIES
# ==============================================================================
# Cell specific heat capacity is ~900 J/(kg*K).
# With mass = 0.048 kg, single cell thermal capacitance = 0.048 * 900 = 43.2 J/K.
# ASSUMPTION: 43.0 J/K per cell.
CELL_HEAT_CAPACITY_J_PER_K: float = 43.0

# Battery module & pack casing structural overhead factor.
# The cold plate, busbars, wiring, and aluminium enclosure add ~30% thermal inertia.
# ASSUMPTION: 1.30 multiplier applied to total cell heat capacity.
PACK_THERMAL_MASS_FACTOR: float = 1.30

# Total lumped battery pack heat capacity [J/K]
# C_pack = N_cells * C_cell * 1.30
PACK_TOTAL_HEAT_CAPACITY_J_PER_K: float = (
    PACK_TOTAL_CELLS * CELL_HEAT_CAPACITY_J_PER_K * PACK_THERMAL_MASS_FACTOR
)  # ~159,203 J/K


# ==============================================================================
# 5. COOLING LOOP PROPERTIES (50/50 Ethylene Glycol-Water)
# ==============================================================================
# Coolant liquid: 50% Water / 50% Ethylene Glycol mixture by volume.
# SOURCE: Marmara University Mechanical Engineering Thesis, Table 3.8.
COOLANT_DENSITY_KG_PER_M3: float = 1077.0         # Density [kg/m^3]
COOLANT_SPECIFIC_HEAT_J_PER_KG_K: float = 3480.0  # Specific heat capacity cp [J/(kg*K)]

# Cold-plate overall heat transfer conductance UA [W/K]
# Characterizes convective and conductive thermal transfer between cells and coolant channels.
# ASSUMPTION: 150.0 W/K for liquid cold plate sized for 30 kWh pack.
COOLING_UA_COLD_PLATE_W_PER_K: float = 150.0

# Convective heat transfer scaling exponent with coolant flow rate [-]
# ASSUMPTION: Convective heat transfer rises with flow (UA = UA_nominal * (flow_lpm / 10.0) ** 0.6).
UA_FLOW_EXPONENT: float = 0.6

# Nominal liquid pump volumetric flow rate [Liters per minute, L/min]
# ASSUMPTION: 10.0 L/min standard EV coolant pump duty cycle during fast charging.
PUMP_NOMINAL_FLOW_LPM: float = 10.0

# Ambient natural convection and radiation thermal conductance UA_amb [W/K]
# Characterizes heat lost from the external pack aluminium casing to ambient air.
# ASSUMPTION: 5.0 W/K.
CASING_UA_AMBIENT_W_PER_K: float = 5.0

# Coolant loop inlet temperature [deg C]
# Supplied by vehicle radiator / chiller loop.
# ASSUMPTION: 25.0 deg C.
T_COOLANT_INLET_NOMINAL_C: float = 25.0

# Ambient air temperature [deg C]
# ASSUMPTION: 25.0 deg C room temperature / standard testing conditions.
T_AMBIENT_NOMINAL_C: float = 25.0

# Pump dynamic response time constant tau_pump [seconds]
# Real physical pump motors have mechanical inertia and fluid flow develops with a first-order lag.
# ASSUMPTION: tau = 3.0 s.
TAU_PUMP_SECONDS: float = 3.0


# ==============================================================================
# 6. OPEN CIRCUIT VOLTAGE (OCV) LOOKUP TABLE
# ==============================================================================
# State of Charge (SOC) ranges from 0.0 (0%) to 1.0 (100%).
# Monotonic NMC open-circuit voltage profile from 3.00 V to 4.20 V.
# ASSUMPTION: 10-point piecewise linear curve (to be calibrated with NASA battery dataset).
SOC_LUT_POINTS = np.array([
    0.00,  # 0%   Empty cell cutoff
    0.10,  # 10%
    0.20,  # 20%  Start of fast charge window
    0.30,  # 30%
    0.40,  # 40%
    0.50,  # 50%  Midpoint
    0.60,  # 60%
    0.70,  # 70%
    0.80,  # 80%  End of fast charge window
    1.00   # 100% Fully charged
], dtype=float)

OCV_LUT_VOLTS = np.array([
    3.00,  # 0%   Empty cell cutoff
    3.30,  # 10%
    3.50,  # 20%
    3.60,  # 30%
    3.66,  # 40%
    3.73,  # 50%  Flat plateau typical of NMC chemistry
    3.80,  # 60%
    3.90,  # 70%
    4.02,  # 80%
    4.20   # 100% Fully charged cutoff
], dtype=float)


def evaluate_ocv(soc: float) -> float:
    """
    Interpolate Open Circuit Voltage (OCV) as a function of State of Charge (SOC).
    
    Parameters
    ----------
    soc : float
        State of charge fraction, clamped between 0.0 and 1.0.
        
    Returns
    -------
    float
        Open circuit voltage in Volts [V].
    """
    # Clamp input to valid range [0, 1] to avoid out-of-bounds extrapolation
    soc_clamped = float(np.clip(soc, 0.0, 1.0))
    # Standard 1D piecewise linear interpolation
    return float(np.interp(soc_clamped, SOC_LUT_POINTS, OCV_LUT_VOLTS))


# ==============================================================================
# 7. SIMULATION & FAST-CHARGING PROFILE CONTROLS
# ==============================================================================
# Discrete simulation time step [seconds]
SIMULATION_DT_S: float = 1.0

# Fast-charge session initial and target SOC targets
SOC_INITIAL: float = 0.20  # Fast charge begins at 20% SOC
SOC_TARGET: float = 0.80   # Fast charge terminates at 80% SOC

# Pack charging current at rated cell current [A]
# 4.0 A per cell * 32 parallel cells = 128.0 A total pack charging current.
PACK_RATED_CHARGE_CURRENT_A: float = CELL_RATED_CHARGE_CURRENT_A * PACK_N_PARALLEL  # 128.0 A
