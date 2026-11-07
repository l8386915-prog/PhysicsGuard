# PhysicsGuard: Closed-Loop Cyber-Physical Protection for EV DC Fast-Charging

**PhysicsGuard** is an end-to-end physics-based cybersecurity and actuator interlock system for Electric Vehicle (EV) DC Fast-Charging systems. 

In Phases 1–4, PhysicsGuard developed a high-fidelity physical plant, an onboard 1-RC digital twin observer, and an anomaly detector (`PhysicsGuard`) evaluating 7 check families. In **Phase 5**, PhysicsGuard closes the control loop: when sensor or command tampering is detected, the **`SafetyGate`** cyber-physical interlock intervenes directly at the actuator boundary so the physical battery remains safely within its operating envelope ($\le 45.0^\circ\text{C}$, $\le 130.56\text{ A}$, $\ge 5\text{ L/min}$ pump flow) under all cyberattacks and component faults.

---

## 1. Closed-Loop Architecture & Hardware Boundary

```
[ Plant Sensors ] 
       │
       ▼
[ Sensor Tamper Attack ] (Upstream Adversary)
       │
       ▼
[ SafetyGate.sanitize() ] (Virtual Sensor Filter on Trusted Hardware)
       │
       ▼ (Sanitized Temperatures)
[ ChargerBMS Controller ]
       │
       ▼
[ Command Tamper Attack ] (Upstream Adversary)
       │
       ▼ (Requested Commands)
[ SafetyGate.screen() ]   (Actuator Screening & Interlock on Trusted Hardware)
       │
       ▼ (Applied Commands)
[ Physical Plant Step ]   (Actuator Delivery: Contactor & Inverter)
       │
       ▼
[ PhysicsGuard.update() ] (Digital Twin & Residual Observers)
       │
       ▼
[ Verdict ] ------------> (Stored for Next Step Gate Decisions)
```

### Hardware Placement Assumption
The **`SafetyGate`** sits on dedicated, trusted hardware at the physical actuator interface (analogous to an automotive SIL-3 / ISO 26262 hardwired safety interlock or secure actuator gateway). 
- **Upstream Network**: Sensor buses (CAN/telematics) and high-level BMS command lines are assumed untrusted and vulnerable to cyberattackers who can inject false data, replay past runs, freeze signals, or issue abusive charging currents.
- **Actuator Boundary**: The gate enforces deterministic physical bounds and look-ahead thermal protection directly before signals drive the physical contactors, power electronics, and coolant pump.

---

## 2. Threat & Attacker Model

PhysicsGuard assumes an attacker capable of manipulating the following channels upstream of the actuator gate:
1. **Sensor Tampering**:
   - `temp_cell_a` / `temp_cell_b`: Freezing, drifting, spiking ($\pm 15^\circ\text{C}$), replaying healthy runs, or injecting hardware faults (NaN/0V/flatline).
   - `pack_voltage`: Gradual ramp drift ($-2\text{ V/min}$) or step offset.
   - `pack_current`: Drift ($-2\text{ A/min}$) or flatline.
   - `pump_flow_lpm`: Replay of normal flow while physical pump is stalled.
2. **Command Tampering**:
   - Overcurrent injection (e.g., requesting $192\text{ A}$ / $1.5\times$ nominal rating, ignoring thermal derating).
   - Coolant shutdown (forcing pump command to $0\text{ L/min}$ while charging at $128\text{ A}$).
3. **Coordinated / Stealth Attacks**:
   - Forcing pump to $0\text{ L/min}$ while replaying normal sensor signals to blind observe-only detectors (`coordinated`).
   - Multi-vector composite attacks under hot ambient conditions (`demo_hot_attack`).

---

## 3. Core Protection Mechanisms

### B1. Virtual Sensor (`SafetyGate.sanitize`)
The BMS controller must never see an unverified temperature:
- **Both Sensors Trusted**: Raw measurements pass through unchanged.
- **Single Sensor Blamed** (`temp_cell_a` vs `temp_cell_b`): The untrusted sensor is replaced by the redundant healthy channel adjusted for the known $+0.25^\circ\text{C}$ cell B mounting offset ($T_{\text{sub}} = T_b - 0.25^\circ\text{C}$).
- **Both Sensors Untrusted**: Both channels are substituted with the Digital Twin temperature plus a conservative safety margin ($+3.0^\circ\text{C}$, derived from empirical max twin error).

### B2. Actuator Screening & Interlocks (`SafetyGate.screen`)
Deterministic, same-step physical protection:
- **Absolute Current Clamp**: Clamps current to $\le 128 \times 1.02 = 130.56\text{ A}$ and enforces the thermal derating envelope on trusted temperature ($100\%$ at $40^\circ\text{C}$, linearly down to $0\text{ A}$ at $50^\circ\text{C}$).
- **Cooling Pump Interlock**: Whenever delivered current $> 20.0\text{ A}$, the pump command is forced to nominal ($10.0\text{ L/min}$). If measured flow remains unconfirmed ($< 3\text{ L/min}$) after `ACTUATOR_GRACE_S` ($15\text{ s}$), current drops to `LIMP_HOME` ($25.0\text{ A}$).
- **180s Look-Ahead Bisection**: Simulates the twin forward $180\text{ s}$. If predicted pack temperature $> 45.0^\circ\text{C}$, bisection search scales current down until predicted temperature $\le 44.0^\circ\text{C}$.

### B3. Operating Modes with Hysteresis
- **`NOMINAL`**: Full charging envelope permitted.
- **`DEGRADED`**: Current restricted by derating curve and look-ahead search; active during single-sensor substitution, command clamps, or actuator faults.
- **`LIMP_HOME`**: Current capped at $25.0\text{ A}$, pump forced to $10.0\text{ L/min}$; active during loss of both temperature sensors or unconfirmed flow.
- **`SHUTDOWN`**: Current forced to $0.0\text{ A}$, pump forced to $10.0\text{ L/min}$; active on physical emergency ($T_{\text{trusted}} \ge 49^\circ\text{C}$ or unmitigatable lookahead overheating).
- **Hysteresis**: Escalation is immediate (same step); de-escalation requires $30\text{ s}$ of consecutive clean steps.

---

## 4. Comprehensive Assumptions & Parameters Table

| Parameter / Constant | Value | SI Unit | Source / Assumption | Justification & Technical Basis |
| :--- | :--- | :--- | :--- | :--- |
| **Cell Model** | Samsung INR18650-30Q | - | **SOURCE** | High-drain cylindrical NMC cell (2.95 Ah, 3.6 V nominal). Official Samsung SDI Datasheet. |
| **Pack Architecture** | 89s 32p (2,848 cells) | - | **ASSUMPTION** | 320 V nominal DC bus ($30.25\text{ kWh}$ pack capacity). Representing passenger EV class. |
| **Pack Heat Capacity ($C_{\text{pack}}$)** | $159,203.2$ | $\text{J/K}$ | Calculated | Bare cell heat capacity ($43.0\text{ J/K}$) $\times 2,848 \times 1.30$ structural mass factor. |
| **Coolant Mixture** | 50/50 Water/EG | - | **SOURCE** | Automotive standard; density $1077\text{ kg/m}^3$, $c_p = 3480\text{ J/(kg}\cdot\text{K)}$. |
| **Cold-Plate Conductance ($UA_{\text{plate}}$)** | $150.0$ | $\text{W/K}$ | **ASSUMPTION** | Sized for liquid cold-plate heat rejection under fast charging. |
| **Nominal Pump Flow Rate** | $10.0$ | $\text{L/min}$ | **ASSUMPTION** | Standard cooling loop nominal operating flow rate assumption. |
| **Pump Lag Time Constant ($\tau_{\text{pump}}$)** | $3.0$ | $\text{s}$ | **ASSUMPTION** | First-order hydrodynamic/motor dynamic response time constant. |
| **Rated Pack Charge Current** | $128.0$ | $\text{A}$ | **SOURCE** | $32 \times 4.0\text{ A}$ continuous cell charge rating (Samsung SDI Datasheet). |
| **Max Permitted Current Envelope** | $130.56$ | $\text{A}$ | **Calculated/derived** | Formula: $128.0\text{ A} \times 1.02$ ($2\%$ instrumentation tolerance ceiling). |
| **Cell Temperature Warning** | $45.0$ | $^{\circ}\text{C}$ | **ASSUMPTION** | Standard BMS derating start temperature threshold. |
| **Cell Temperature Trip** | $50.0$ | $^{\circ}\text{C}$ | **SOURCE** | Samsung SDI maximum charging cutoff temperature specification. |
| **Target Look-Ahead Temperature** | $44.0$ | $^{\circ}\text{C}$ | **ASSUMPTION** | $1.0^\circ\text{C}$ safety buffer below $45.0^\circ\text{C}$ prevents chattering derating cycles. |
| **Look-Ahead Horizon (`LOOKAHEAD_S`)** | $180$ | $\text{s}$ | **ASSUMPTION** | 3-minute prediction horizon matches pack thermal time constant. |
| **Twin Temperature Margin (`TWIN_TEMP_MARGIN_C`)** | $3.0$ | $^{\circ}\text{C}$ | **Calculated/derived** | Formula: 99th percentile twin error over 40 tuning seeds 100-139 ($2.58^\circ\text{C}$, `results_twin_error_extended.csv`) rounded up to nearest $0.5^\circ\text{C}$. |
| **Pump Interlock Current (`PUMP_INTERLOCK_CURRENT_A`)** | $20.0$ | $\text{A}$ | **ASSUMPTION** | Below 20 A, heat generation is easily dissipated naturally; above 20 A active flow is mandated. |
| **Actuator Confirmation Grace (`ACTUATOR_GRACE_S`)** | $15.0$ | $\text{s}$ | **Calculated/derived** | Formula: $5 \times \tau_{\text{pump}} = 5 \times 3.0\text{ s} = 15.0\text{ s}$ mechanical settling time. |
| **Limp-Home Current (`LIMP_HOME_CURRENT_A`)** | $25.0$ | $\text{A}$ | **ASSUMPTION** | Low current reduces Joule heating by $>96\%$ ($I^2 R_0$), guaranteeing thermal safety under degraded cooling. |
| **De-escalation Delay (`DE_ESCALATE_CLEAN_S`)** | $30$ | $\text{s}$ | **ASSUMPTION** | Hysteresis timer preventing rapid mode oscillation between DEGRADED and NOMINAL. |
| **CUSUM Parameter $k_{Rv}$** | $3.9246$ | $\text{V}$ | **Calculated/derived** | Formula: Tuned on seeds 100–139; includes baseline offset to avoid false accumulation. |
| **CUSUM Parameter $h_{Rv}$** | $39.2461$ | $\text{V}$ | **Calculated/derived** | Formula: Capped at $10 \times k_{Rv}$ to bound detection latency $\le 240\text{ s}$. |
| **CUSUM Parameter $k_{Rcur}$** | $0.9248$ | $\text{A}$ | **Calculated/derived** | Formula: Derived from steady-state residual standard deviation on tuning seeds. |
| **CUSUM Parameter $h_{Rcur}$** | $9.2478$ | $\text{A}$ | **Calculated/derived** | Formula: Capped at $10 \times k_{Rcur}$ ensuring current drift detection within $60\text{ s}$. |
| **Instant Current Threshold (`Rcur_threshold`)** | $10.0$ | $\text{A}$ | **ASSUMPTION/derived** | Formula: Hand-set 10 A cap to bound uncommanded current injection. |

---

## 5. Limitations

1. **Simulation Only**: The system is validated in numerical simulation; the physical model is validated against its own mathematical parameters and NASA B0005 empirical thermal heating data, not on physical commercial battery packs or hardware-in-the-loop (HIL) testbenches.
2. **Current and Voltage Sensor Manipulation Dynamics**: In this model, freeze and drift manipulations of current and voltage measurements do not alter the physical power converter output (they affect detection and state estimation only); physical over-current is driven through command manipulation.
3. **Actuator Path Trust**: The gate trusts the downstream physical actuator line; while the pump command interlock unconditionally forces pump commands to nominal ($10\text{ L/min}$) whenever current $> 20\text{ A}$, a spoofed/replayed `pump_flow` sensor reading could fake flow confirmation if both the telemetry and sensor lines were compromised simultaneously.
4. **Lumped Thermal Representation**: The onboard digital twin uses a lumped single-node thermal mass model. It models pack bulk average temperature and does not capture spatial temperature gradients or localized cell-to-cell hotspots within modules.
5. **Calibrated Parameter Domain**: Detection thresholds, CUSUM drift parameters, and twin tolerance margins were calibrated over a defined operating domain (ambient $20^\circ\text{C}$–$40^\circ\text{C}$, cell aging factors $1.0$–$1.6$, plant parameter perturbations $\pm 10\%$). Operation outside this design envelope requires recalibration.
6. **Coordinated Multi-Channel Spoofing on Hot Days (Safety vs Charge Speed Trade-off)**: Under a coordinated stealth attack on a hot day ($35^\circ\text{C}$ ambient / $35^\circ\text{C}$ coolant inlet, e.g. seed 42), the attacker zeroes the pump command while faking flow evidence (`pump_flow_lpm` replayed at $\sim 10\text{ L/min}$) and faking cool temperatures (`temp_cell_a` and `temp_cell_b` replayed from a $25^\circ\text{C}$ baseline at $\sim 27^\circ\text{C}$). The physics guard detects that both temperature measurements contradict thermal physics ($R_{1a} \approx -9.8^\circ\text{C}, R_{1b} \approx -10.0^\circ\text{C}$) and flags both temperature channels as untrusted (`A=bad, B=bad`). With complete loss of temperature redundancy and spoofed flow evidence, the SafetyGate enters `LIMP_HOME` mode and derates charging current to a safe $25\text{ A}$ ceiling ($300\text{ s}$ to $6923\text{ s}$). This results in a charge duration of $6923\text{ s}$ ($+289.4\%$ penalty relative to the matching $1784\text{ s}$ normal hot-day baseline). This is a deliberate, genuine physical safety trade-off: in the presence of falsified flow evidence and zero trustworthy temperature sensors, high-current fast charging ($128\text{ A}$) cannot be safely continued without risking catastrophic unmonitored thermal runaway. The gate holds peak true cell temperature to $37.80^\circ\text{C} \le 45.0^\circ\text{C}$ throughout the session.

---

## 6. Verification & How to Run

### Run All Unit and Acceptance Tests
```bash
python -m pytest
```
Runs full test suite covering:
- Ground-truth isolation AST check for `guard.py` and `mitigation.py`
- All 17 scenarios deterministic execution
- Thermal energy balance, NTU coolant heat rejection, and 1-RC tracking
- Criteria 1–7 formal acceptance tests in `physicsguard/tests/test_mitigation.py`
- Phase 5.1 test suite: Criterion 1 over 20 random perturbed seeds, `demo_hot_attack` safe completion, and AST check that held-out evaluation uses `perturb=True`.
- Phase 6 smoke test suite in `tests/test_app_smoke.py`.

### Run Closed-Loop Evaluation Suite (C1, C2, C3)
```bash
python evaluate_protected.py
```
Outputs generated in `outputs/`:
- `outputs/protected_vs_unprotected.csv`: Side-by-side metrics across all 17 scenarios.
- `outputs/heldout_report_v2.csv`: Statistical performance on 200 fresh held-out sessions (seeds 700–899, `perturb=True`) with 95% Wilson confidence intervals.
- `outputs/compare_<scenario>.png`: Side-by-side plots for `demo_hot_attack`, `pump_off`, `coordinated`, `abusive_command`, `drift_temp_a`, and `pump_degradation`.

### Run Hot-Day Attacks Evaluation
```bash
python evaluate_hot.py
```
Outputs generated in `outputs/`:
- `outputs/hot_day_attacks.csv`: Seed 42 baseline comparison under 35 C ambient / 35 C coolant inlet.
- `outputs/hot_protection_report.csv` & `outputs/hot_protection_summary.csv`: Development seeds 42 and 500–504 under `perturb=True` and random attack timing.

---

## 7. How to Demo in 5 Minutes (Exact Click Path)

Follow this 5-minute click path for judging or live demonstration:

### Step 1: Launch Dashboard (<3 seconds)
```bash
streamlit run app.py
```
The application opens in your browser at `http://localhost:8501`.

### Step 2: Show Normal Charging Baseline (30 seconds)
1. In the sidebar dropdown **Select Scenario**, choose **`⚡ normal`**.
2. Notice the top header badge displays **`CLASS: NORMAL`** and **`FINAL MODE: NOMINAL`**.
3. Point out that Unprotected, Plain-Alarm, and PhysicsGuard all complete in **`1594 s`** with identical **`35.1 °C`** peak temperature and **`0.80 SOC`**.
4. Point out the zero false alarms in the residual panel ($R_1, R_2, R_3, R_v, R_{flow}, R_{cur}$ remain well beneath their red threshold lines).

### Step 3: Demonstrate Multi-Vector Cyberattack (`demo_hot_attack`) (90 seconds)
1. In the sidebar dropdown, select **`🔥 DEMO_HOT_ATTACK`** (ambient and coolant inlet automatically configure to $35^\circ\text{C}$).
2. Look at the three comparison columns:
   - **UNPROTECTED (Red header)**: Battery heats to **`50.9 °C`**, spending **`631 s > 45°C`** and **`98 s > 50°C`** (severe thermal runway risk).
   - **PLAIN-ALARM BASELINE (Amber header)**: Tripped late only after battery hit $44.7^\circ\text{C}$, but because Sensor A was faked cool ($25^\circ\text{C}$), the A-only baseline was completely blinded!
   - **PHYSICSGUARD (Green header)**: Battery temperature held safely at **`42.1 °C`** (**`0 s > 45°C`**, **`0 s > 50°C`**) with only a **`+0.3%`** duration penalty ($1784\text{ s}$ vs $1778\text{ s}$ baseline).
3. Scrub the time slider to **`t = 312 s`**:
   - Highlight the **"Explain This Moment"** box:
     *"t=312 s: Sensor freeze detected. Temperature sensor A reads 25.0 °C while the digital twin predicts 34.1 °C under active charging (128.0 A, 10.0 L/min). Verdict: SENSOR_FROZEN. Gate mode: DEGRADED."*
   - Show in the protected plot how the **Gate-Trusted Temperature** (green line) overrides the spoofed Sensor A (blue line).

### Step 4: Demonstrate Coordinated Stealth Attack (`coordinated`) (60 seconds)
1. In the sidebar, select **`🔥 COORDINATED`**.
2. Show why standard threshold alarms fail:
   - The attacker zeroes the cooling pump while spoofing **both** Sensor A and Sensor B to read cool ($27^\circ\text{C}$).
   - Under the **Plain-Alarm Baseline**, the alarm **never trips** (`BLINDED`), and true temperature hits **`50.6 °C`**!
   - Under **PhysicsGuard**, the digital twin detects that both sensors violate physical heating laws. The gate escalates to **`LIMP_HOME`** ($25\text{ A}$ current cap), holding temperature at a safe **`37.8 °C`**.
3. Highlight the trade-off: charging takes $6921\text{ s}$ ($+289.3\%$ penalty) because with both sensors compromised, **safety is strictly prioritized over charging speed**.

### Step 5: Walk Through Key Verification Tabs (90 seconds)
1. Click **`📊 Physics Residual Matrix`**: Explain how different rows trigger different physical residuals ($R_{flow}$ for pump degradation, $R_{cur}$ for current drift, $R_v$ for voltage drift, $R_{1a}/R_3$ for temperature spoofing).
2. Click **`📈 Evaluation Results & Held-Out Suite`**: Show statistical results on the 200 held-out sessions with Wilson 95% Confidence Intervals (`17 of 17 [0.816, 1.000]` on all attack vectors, $0.0\%$ false alarm rate, $100\%$ thermal safety).
3. Click **`📋 Limitations & Technical Assumptions`**: Point out the dynamic ratio calculation ($3.88\times$ penalty) and the complete parameter justification table.

---

## 8. Phase 6.5 Attack Lab: How to Demo in 3 Minutes (Exact Clicks)

The **Attack Lab** allows judges to pick presets or construct stacked cyberattacks and watch the car (Plant + BMS), PhysicsGuard detection, and SafetyGate mitigation side-by-side with unprotected and plain 45 °C alarm baselines.

### Step 1: Launch and Open Attack Lab (10 seconds)
```bash
streamlit run app.py
```
Open `http://localhost:8501`. The **`🧪 Attack Lab (Phase 6.5)`** tab is the first active tab.

### Step 2: Recipe 1 — 1-Click Multi-Vector Demo (`demo_hot_attack`) (60 seconds)
1. In the **Load Attack Recipe Preset** dropdown, select **`demo_hot_attack`**.
   - Notice the builder populates Layer 1 (`temp_cell_a` set to $25.0^\circ\text{C}$ at $t=0\text{ s}$) and Layer 2 (`cmd_pump` set to $0.0\text{ L/min}$ at $t=60\text{ s}$), with Ambient=$35^\circ\text{C}$ and Inlet=$35^\circ\text{C}$.
2. Click **`⚡ Run Attack Lab Simulation`** (takes ~1.5 s, cached by parameter hash).
3. Point out the **RESULT CARD**:
   - **Status**: GREEN pill **`GUARD STATUS: YES (+1.0 s latency)`**, Diagnosis: `SENSOR_SPOOF, ACTUATOR_FAULT`.
   - **Unprotected**: True battery temperature skyrockets to **`50.9 °C`** (**`631 s > 45°C`**, **`98 s > 50°C`** thermal runaway danger).
   - **Plain 45°C Alarm**: Tripped late or blinded because sensor A was spoofed cold ($25^\circ\text{C}$).
   - **PhysicsGuard Protected**: Peak battery temperature held safely to **`42.1 °C`** (**`0 s > 45°C`**, **`0 s > 50°C`**), charging completes safely ($80.0\%$ SOC).
4. Point to the **"What Happened"** timeline box displaying the plain-English physical event log with numbers.
5. Drag the timeline scrub slider to **`t = 120 s`** and highlight the **Explain This Moment** box and gate mode band.

### Step 3: Recipe 2 — Actuator Override Demo (`abusive_command`) (45 seconds)
1. In the preset dropdown, select **`abusive_command`**.
   - Populates Layer 1: Target=`cmd_current`, Type=`set_value`, Magnitude=`192.0 A`, Start=`300 s`.
2. Click **`⚡ Run Attack Lab Simulation`**.
3. Point out the RESULT CARD:
   - **Detection**: PhysicsGuard flags command mismatch with $+1.0\text{ s}$ latency.
   - **Actuator Screening**: SafetyGate hard clamps applied current to $130.56\text{ A}$ envelope and maintains nominal cooling flow ($10\text{ L/min}$).
   - Unprotected pack heats up, whereas protected stays safe $\le 45.0^\circ\text{C}$.

### Step 4: Recipe 3 — Honesty Rule Demo (`coolant_temp_out` Drift) (45 seconds)
1. Select **`(Custom Attack)`** in the preset dropdown.
2. Configure Layer 1: Target=`coolant_temp_out`, Type=`drift`, Magnitude=`-2.0`, Start=`300 s`.
3. Click **`⚡ Run Attack Lab Simulation`**.
4. Highlight the **HONESTY RULE NOTICE** in amber:
   - Displays: **`⚠️ HONESTY RULE NOTICE — NO PHYSICAL EFFECT IN THIS MODEL`**.
   - Explain to the judge: PhysicsGuard is honest — since the BMS controller does not close loop on coolant outlet temperature in fast-charge mode, true cell temperatures, applied current, and battery SOC remain identical to the normal baseline. PhysicsGuard never fabricates fake physical effects.

### Step 5: Try Plant Variation (20 seconds)
1. Toggle **`Plant variation (±10%)`** ON.
2. Click **`⚡ Run Attack Lab Simulation`**.
3. PhysicsGuard still guarantees thermal safety $\le 45.0^\circ\text{C}$ even when physical parameters (resistance, heat capacity, pump lag) deviate by $\pm 10\%$ from the nominal digital twin.



