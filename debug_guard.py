"""Quick diagnostic to see what verdicts each scenario produces."""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from physicsguard import params
from physicsguard.plant import Plant
from physicsguard.controller import ChargerBMS
from physicsguard.scenarios import build_scenario
from physicsguard.guard import PhysicsGuard, Verdict
import numpy as np

def run_diag(sc_name, max_steps=700):
    sc = build_scenario(sc_name)
    plant_kw = dict(sc.plant_kwargs)
    init_soc = plant_kw.pop("initial_soc", params.SOC_INITIAL)
    init_temp = plant_kw.pop("initial_temp_c", sc.inlet_c if sc.inlet_c else params.T_AMBIENT_NOMINAL_C)
    plant = Plant(perturb=False, initial_soc=init_soc, initial_temp_c=init_temp, sensor_seed=42, **plant_kw)
    ctrl = ChargerBMS()
    guard = PhysicsGuard()
    plant.sensor_rng = np.random.default_rng(42)
    
    amb = float(sc.ambient_c) if sc.ambient_c else params.T_AMBIENT_NOMINAL_C
    inlet = float(sc.inlet_c) if sc.inlet_c else params.T_COOLANT_INLET_NOMINAL_C
    
    meas = (sc.sensor_tamper(plant.time_s, plant.measure().copy()) if sc.sensor_tamper else plant.measure().copy())
    
    start_s = sc.start_s if sc.start_s else 0.0
    first_alarm_t = None
    first_alarm_diag = None
    
    for step in range(max_steps):
        t = plant.time_s
        req_c, req_p = ctrl.compute_commands(meas)
        ac, ap = (sc.command_tamper(t, (req_c, req_p)) if sc.command_tamper else (req_c, req_p))
        if sc.plant_hook: sc.plant_hook(t, plant)
        plant.step(dt=1.0, current_cmd_a=ac, pump_cmd_lpm=ap, T_cool_in_c=inlet, T_amb_c=amb)
        t_post = plant.time_s
        meas_raw = plant.measure()
        meas = (sc.sensor_tamper(t_post, meas_raw.copy()) if sc.sensor_tamper else meas_raw.copy())
        
        v = guard.update(t_post, dict(meas), (float(ac), float(ap)))
        
        if v.alarm_level in ("warning", "critical") and t_post >= start_s and first_alarm_t is None:
            first_alarm_t = t_post
            first_alarm_diag = v.diagnosis
        
        # Print key info around start time 
        if abs(t_post - start_s) < 3 or (t_post > start_s and t_post < start_s + 30 and v.alarm_level != "none"):
            print(f"  t={t_post:.0f}: diag={v.diagnosis}, alarm={v.alarm_level}, is_atk={v.is_attack}, res={v.residuals}")
    
    print(f"  -> First alarm: t={first_alarm_t}, diag={first_alarm_diag}, lag={first_alarm_t - start_s if first_alarm_t else 'N/A'}")
    return first_alarm_t, first_alarm_diag

scenarios_to_check = [
    "freeze_temp_a", "spike_temp_a", "drift_temp_a", "replay_temp_a",
    "abusive_command", "demo_hot_attack", "normal"
]

for sc_name in scenarios_to_check:
    print(f"\n=== {sc_name} ===")
    run_diag(sc_name)
