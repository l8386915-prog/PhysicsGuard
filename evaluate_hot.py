"""
PhysicsGuard - Hot Day Attacks Evaluation Pipeline (evaluate_hot.py)
=====================================================================
Evaluates hot-day attacks under 35 C ambient / 35 C coolant inlet:
  - Task 4: Seed 42 baseline (unprotected vs protected) for:
      pump_off, abusive_command, coordinated, demo_hot_attack.
    Saved to outputs/hot_day_attacks.csv.
  - Addition D: Development set evaluation on seed 42 + perturbed seeds 500-504
    with perturb=True and random start_s in [200, 600].
    Saved to outputs/hot_protection_report.csv and outputs/hot_protection_summary.csv.
"""

import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physicsguard import params
from physicsguard.plant import Plant
from physicsguard.controller import ChargerBMS
from physicsguard.simulate import run_session
from physicsguard.guard import PhysicsGuard
from physicsguard.mitigation import SafetyGate
from physicsguard.attacks import pump_off, abusive_command, coordinated, demo_hot_attack

OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

HOT_ATTACK_SCENARIOS = ["pump_off", "abusive_command", "coordinated", "demo_hot_attack"]


def clean_df(df: pd.DataFrame, latched: bool = False) -> pd.DataFrame:
    """Drop t=0 pre-step sample and last 2 stop-latch zero samples."""
    d = df[df["time_s"] > 0].copy()
    if latched and len(d) > 2:
        d = d.iloc[:-2].copy()
    return d


def run_hot_evaluation():
    print("=" * 80)
    print("TASK 4: Hot-Day Attacks (35 C Ambient / 35 C Inlet) on Seed 42")
    print("=" * 80)

    seed_42_rows = []
    for sc_name in HOT_ATTACK_SCENARIOS:
        start_s = 60.0 if sc_name == "demo_hot_attack" else 300.0
        if sc_name == "demo_hot_attack":
            sc_unpr = demo_hot_attack()
            sc_prot = demo_hot_attack()
        elif sc_name == "pump_off":
            sc_unpr = pump_off(start_s=start_s)
            sc_prot = pump_off(start_s=start_s)
        elif sc_name == "abusive_command":
            sc_unpr = abusive_command(start_s=start_s)
            sc_prot = abusive_command(start_s=start_s)
        else:  # coordinated
            sc_unpr = coordinated(start_s=start_s)
            sc_prot = coordinated(start_s=start_s)

        for s_obj in [sc_unpr, sc_prot]:
            s_obj.ambient_c = 35.0
            s_obj.inlet_c = 35.0
            s_obj.plant_kwargs["initial_temp_c"] = 35.0

        # Unprotected
        p_u = Plant(perturb=False, sensor_seed=42, **dict(sc_unpr.plant_kwargs))
        c_u = ChargerBMS()
        df_u = run_session(p_u, c_u, scenario=sc_unpr, gate=None, guard=None, seed=42)
        df_u_c = clean_df(df_u, getattr(c_u, "latched_stop", False))

        # Protected
        p_p = Plant(perturb=False, sensor_seed=42, **dict(sc_prot.plant_kwargs))
        c_p = ChargerBMS()
        gate = SafetyGate()
        guard = PhysicsGuard()
        gate.set_guard(guard)
        df_p = run_session(p_p, c_p, scenario=sc_prot, gate=gate, guard=guard, seed=42)
        df_p_c = clean_df(df_p, getattr(c_p, "latched_stop", False))

        for mode_lbl, df_sub in [("unprotected", df_u_c), ("protected", df_p_c)]:
            pk_t = float(df_sub["true_battery_temp"].max())
            s45 = int((df_sub["true_battery_temp"] > 45.0).sum())
            s50 = int((df_sub["true_battery_temp"] > 50.0).sum())
            fin_soc = float(df_sub["true_soc"].iloc[-1])
            chg_time = float(df_sub["time_s"].iloc[-1])
            completed = bool(fin_soc >= 0.80)

            seed_42_rows.append({
                "scenario": sc_name,
                "mode": mode_lbl,
                "peak_temp_c": round(pk_t, 2),
                "seconds_above_45C": s45,
                "seconds_above_50C": s50,
                "charge_completed": completed,
                "final_soc": round(fin_soc, 3),
                "charge_time_s": round(chg_time, 1)
            })

    df_hot_42 = pd.DataFrame(seed_42_rows)
    csv_hot_42 = OUTPUTS_DIR / "hot_day_attacks.csv"
    df_hot_42.to_csv(csv_hot_42, index=False)
    print(f"Saved Task 4 -> {csv_hot_42}")
    print(df_hot_42.to_string(index=False))

    print("\n" + "=" * 80)
    print("ADDITION D: Hot Perturbed Evaluation (Seeds 42, 500-504 from Dev Set, perturb=True)")
    print("=" * 80)

    eval_seeds = [42, 500, 501, 502, 503, 504]
    rng_hot = np.random.default_rng(5555)
    report_rows = []

    for seed in eval_seeds:
        for sc_name in HOT_ATTACK_SCENARIOS:
            start_s = 60.0 if (sc_name == "demo_hot_attack" and seed == 42) else float(round(rng_hot.uniform(200.0, 600.0)))
            if sc_name == "demo_hot_attack":
                sc = demo_hot_attack()
                sc.start_s = start_s
            elif sc_name == "pump_off":
                sc = pump_off(start_s=start_s)
            elif sc_name == "abusive_command":
                sc = abusive_command(start_s=start_s)
            else:  # coordinated
                sc = coordinated(start_s=start_s)

            sc.ambient_c = 35.0
            sc.inlet_c = 35.0
            sc.plant_kwargs["initial_temp_c"] = 35.0

            # Unprotected run
            p_u = Plant(perturb=True, seed=seed, sensor_seed=seed, **dict(sc.plant_kwargs))
            c_u = ChargerBMS()
            df_u = run_session(p_u, c_u, scenario=sc, gate=None, guard=None, seed=seed)
            df_u_c = clean_df(df_u, getattr(c_u, "latched_stop", False))

            # Protected run
            p_p = Plant(perturb=True, seed=seed, sensor_seed=seed, **dict(sc.plant_kwargs))
            c_p = ChargerBMS()
            gate = SafetyGate()
            guard = PhysicsGuard()
            gate.set_guard(guard)
            df_p = run_session(p_p, c_p, scenario=sc, gate=gate, guard=guard, seed=seed)
            df_p_c = clean_df(df_p, getattr(c_p, "latched_stop", False))

            for mode_lbl, df_sub in [("unprotected", df_u_c), ("protected", df_p_c)]:
                pk_t = float(df_sub["true_battery_temp"].max())
                s45 = int((df_sub["true_battery_temp"] > 45.0).sum())
                s50 = int((df_sub["true_battery_temp"] > 50.0).sum())
                fin_soc = float(df_sub["true_soc"].iloc[-1])
                chg_time = float(df_sub["time_s"].iloc[-1])
                completed = bool(fin_soc >= 0.80)

                report_rows.append({
                    "seed": seed,
                    "scenario": sc_name,
                    "mode": mode_lbl,
                    "start_s": start_s,
                    "peak_temp_c": round(pk_t, 2),
                    "seconds_above_45C": s45,
                    "seconds_above_50C": s50,
                    "charge_completed": completed,
                    "final_soc": round(fin_soc, 3),
                    "charge_time_s": round(chg_time, 1)
                })

    df_rep = pd.DataFrame(report_rows)
    csv_rep = OUTPUTS_DIR / "hot_protection_report.csv"
    df_rep.to_csv(csv_rep, index=False)
    print(f"Saved hot protection report -> {csv_rep}")

    # Compute Summary: Median and Worst Case per scenario and mode
    summary_rows = []
    for sc_name in HOT_ATTACK_SCENARIOS:
        for mode_lbl in ["unprotected", "protected"]:
            sub = df_rep[(df_rep["scenario"] == sc_name) & (df_rep["mode"] == mode_lbl)]
            pk_med = float(sub["peak_temp_c"].median())
            pk_worst = float(sub["peak_temp_c"].max())
            s45_med = float(sub["seconds_above_45C"].median())
            s45_worst = int(sub["seconds_above_45C"].max())
            s50_med = float(sub["seconds_above_50C"].median())
            s50_worst = int(sub["seconds_above_50C"].max())
            comp_rate = float(sub["charge_completed"].mean())
            t_med = float(sub["charge_time_s"].median())
            t_worst = float(sub["charge_time_s"].max())

            summary_rows.append({
                "scenario": sc_name,
                "mode": mode_lbl,
                "peak_temp_median_c": round(pk_med, 2),
                "peak_temp_worst_c": round(pk_worst, 2),
                "seconds_above_45C_median": round(s45_med, 1),
                "seconds_above_45C_worst": s45_worst,
                "seconds_above_50C_median": round(s50_med, 1),
                "seconds_above_50C_worst": s50_worst,
                "completion_rate": round(comp_rate, 3),
                "charge_time_median_s": round(t_med, 1),
                "charge_time_worst_s": round(t_worst, 1),
                "sample_count": len(sub)
            })

    df_sum = pd.DataFrame(summary_rows)
    csv_sum = OUTPUTS_DIR / "hot_protection_summary.csv"
    df_sum.to_csv(csv_sum, index=False)
    print(f"Saved hot protection summary -> {csv_sum}")
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    run_hot_evaluation()
