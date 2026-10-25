"""
PhysicsGuard - Automatic Threshold Calibration
===============================================
Derives all PhysicsGuard thresholds and CUSUM parameters automatically
from a TUNING SET only (seeds 100 to 139).

Tuning set conditions (all must give NORMAL):
  - Normal with random +/-10% parameters
  - Aged cell (age_factor 1.2 to 1.6)
  - Hot day (ambient 30-40 C, inlet 30-38 C)

Rules for threshold derivation:
  - Instant threshold = 1.5 * max absolute residual seen
  - CUSUM k = 1.0 * residual standard deviation
  - CUSUM h = 1.5 * largest CUSUM value seen (from replay with k)

Outputs:
  - outputs/thresholds.json
  - Printed table of residual statistics (mean, std, max per residual)
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Ensure local package imports succeed
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
PARENT_DIR = ROOT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from physicsguard import params
    from physicsguard.plant import Plant
    from physicsguard.controller import ChargerBMS
    from physicsguard.simulate import run_session
    from physicsguard.scenarios import build_scenario
    from physicsguard.guard import PhysicsGuard, Verdict, CALIB_WINDOW_S
except ImportError:
    import params
    from plant import Plant
    from controller import ChargerBMS
    from simulate import run_session
    from scenarios import build_scenario
    from guard import PhysicsGuard, Verdict, CALIB_WINDOW_S


TUNING_SEEDS = list(range(100, 140))  # 40 seeds
AGE_FACTOR_RANGE = (1.2, 1.6)
HOT_AMBIENT_RANGE = (30.0, 40.0)
HOT_INLET_RANGE = (30.0, 38.0)


def _make_tuning_session(seed: int):
    """
    Create a tuning session: randomly pick one of three conditions.
    Returns (plant, controller, run_kwargs).
    """
    rng = random.Random(seed)
    kind = rng.choice(["normal", "aged", "hot"])

    if kind == "normal":
        plant = Plant(perturb=True, seed=seed,
                      initial_soc=params.SOC_INITIAL,
                      initial_temp_c=params.T_AMBIENT_NOMINAL_C,
                      sensor_seed=seed)
        return plant, ChargerBMS(), {
            "ambient_c": params.T_AMBIENT_NOMINAL_C,
            "inlet_c": params.T_COOLANT_INLET_NOMINAL_C,
        }

    elif kind == "aged":
        af = rng.uniform(*AGE_FACTOR_RANGE)
        plant = Plant(perturb=True, seed=seed, age_factor=af,
                      initial_soc=params.SOC_INITIAL,
                      initial_temp_c=params.T_AMBIENT_NOMINAL_C,
                      sensor_seed=seed)
        return plant, ChargerBMS(), {
            "ambient_c": params.T_AMBIENT_NOMINAL_C,
            "inlet_c": params.T_COOLANT_INLET_NOMINAL_C,
        }

    else:  # hot
        amb = rng.uniform(*HOT_AMBIENT_RANGE)
        inlet = rng.uniform(*HOT_INLET_RANGE)
        plant = Plant(perturb=True, seed=seed,
                      initial_soc=params.SOC_INITIAL,
                      initial_temp_c=inlet,
                      sensor_seed=seed)
        return plant, ChargerBMS(), {
            "ambient_c": amb,
            "inlet_c": inlet,
        }


def _compute_cusum_max(residuals: List[float], k: float) -> float:
    """Simulate CUSUM with given k and return maximum value of S+ and S-."""
    s_pos = 0.0
    s_neg = 0.0
    max_s = 0.0
    for r in residuals:
        s_pos = max(0.0, s_pos + r - k)
        s_neg = max(0.0, s_neg - r - k)
        max_s = max(max_s, s_pos, s_neg)
    return max_s


def run_calibration(max_steps: int = 3000) -> Dict[str, float]:
    """
    Run all tuning scenarios and collect residuals for threshold derivation.

    Parameters
    ----------
    max_steps : int
        Maximum simulation steps per tuning session.

    Returns
    -------
    dict
        Calibrated threshold dictionary.
    """
    # Storage for all residuals across all tuning runs
    all_residuals: Dict[str, List[float]] = {
        "R1a": [], "R1b": [], "R2": [], "R3": [], "Rv": [],
        "Rflow": [], "Rcur": [],
    }

    print("=" * 70)
    print("PhysicsGuard - Automatic Threshold Calibration")
    print(f"Tuning seeds: {TUNING_SEEDS[0]} to {TUNING_SEEDS[-1]} ({len(TUNING_SEEDS)} runs)")
    print("=" * 70)

    for seed in TUNING_SEEDS:
        plant, controller, run_kwargs = _make_tuning_session(seed)
        guard = PhysicsGuard()

        step = 0
        # Initial measurement
        meas_raw = plant.measure()
        meas = meas_raw.copy()

        while step < max_steps:
            t = plant.time_s
            req_curr, req_pump = controller.compute_commands(meas)
            appl_curr, appl_pump = req_curr, req_pump  # No tampering in tuning

            plant.step(
                dt=1.0,
                current_cmd_a=appl_curr,
                pump_cmd_lpm=appl_pump,
                T_cool_in_c=run_kwargs.get("inlet_c", params.T_COOLANT_INLET_NOMINAL_C),
                T_amb_c=run_kwargs.get("ambient_c", params.T_AMBIENT_NOMINAL_C),
            )
            t_post = plant.time_s
            meas_raw = plant.measure()
            meas = meas_raw.copy()

            # Call guard to compute residuals (only after calibration window)
            verdict = guard.update(t_post, meas, (appl_curr, appl_pump))

            # Collect residuals from verdict
            if t_post > CALIB_WINDOW_S:
                for key in all_residuals.keys():
                    val = verdict.residuals.get(key)
                    if val is not None and np.isfinite(val):
                        all_residuals[key].append(float(val))

            step += 1
            # Stop early if SOC target reached
            if plant.soc >= params.SOC_TARGET:
                break

        print(f"  seed {seed:3d}: {step:4d} steps, SOC={plant.soc:.3f}")

    # ------------------------------------------------------------------
    # Compute statistics and derive thresholds
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Residual Statistics Table")
    print(f"{'Residual':<12} {'Mean':>10} {'Std':>10} {'Max|r|':>10} {'N':>8}")
    print("-" * 52)

    thresholds: Dict[str, float] = {}
    residual_stats: Dict[str, Dict] = {}

    for key, vals in all_residuals.items():
        if not vals:
            print(f"  {key:<12} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'0':>8}")
            continue

        arr = np.array(vals)
        mean_r = float(np.mean(arr))
        std_r = float(np.std(arr))
        max_abs = float(np.max(np.abs(arr)))
        n = len(arr)

        residual_stats[key] = {"mean": mean_r, "std": std_r, "max_abs": max_abs, "n": n}
        print(f"  {key:<12} {mean_r:>10.4f} {std_r:>10.4f} {max_abs:>10.4f} {n:>8d}")

        # Instant threshold = 1.5 * max absolute residual
        inst_thr = 1.5 * max_abs
        # CUSUM k = 1.0 * std
        cusum_k = 1.0 * std_r
        # Compute max CUSUM value reached during tuning with this k
        cusum_max_seen = _compute_cusum_max(vals, k=cusum_k)
        # CUSUM h = 1.5 * max CUSUM seen
        cusum_h = 1.5 * cusum_max_seen if cusum_max_seen > 0 else 3.0 * std_r

        thresholds[key] = {
            "inst_threshold": inst_thr,
            "cusum_k": cusum_k,
            "cusum_h": cusum_h,
        }

    print("=" * 70)

    # ------------------------------------------------------------------
    # Map to the flat threshold dict used by guard.py
    # ------------------------------------------------------------------
    flat_thr: Dict[str, float] = {}

    def _get(key, sub):
        return thresholds.get(key, {}).get(sub, None)

    # R1 thermal consistency thresholds
    r1a_inst = _get("R1a", "inst_threshold")
    r1b_inst = _get("R1b", "inst_threshold")
    if r1a_inst is not None and r1b_inst is not None:
        flat_thr["R1_threshold_c"] = max(r1a_inst, r1b_inst)
    elif r1a_inst is not None:
        flat_thr["R1_threshold_c"] = r1a_inst
    else:
        flat_thr["R1_threshold_c"] = 3.0

    # Differential R1 threshold (difference between R1a and R1b for blaming one sensor)
    r1a_std = residual_stats.get("R1a", {}).get("std", 0.5)
    r1b_std = residual_stats.get("R1b", {}).get("std", 0.5)
    flat_thr["R1_diff_threshold_c"] = max(1.5 * (r1a_std + r1b_std), 2.0)

    # R2 coolant energy balance
    r2_inst = _get("R2", "inst_threshold")
    flat_thr["R2_threshold_kj"] = r2_inst if r2_inst is not None else 0.5

    # R3 redundancy
    r3_inst = _get("R3", "inst_threshold")
    flat_thr["R3_threshold_c"] = r3_inst if r3_inst is not None else 2.0
    flat_thr["cusum_k_R3"] = _get("R3", "cusum_k") or 0.3
    flat_thr["cusum_h_R3"] = _get("R3", "cusum_h") or 3.0

    # Rv voltage residual
    rv_inst = _get("Rv", "inst_threshold")
    flat_thr["Rv_threshold_v"] = rv_inst if rv_inst is not None else 5.0
    flat_thr["cusum_k_Rv"] = _get("Rv", "cusum_k") or 0.5
    flat_thr["cusum_h_Rv"] = _get("Rv", "cusum_h") or 5.0

    # Rflow actuator echo
    rflow_inst = _get("Rflow", "inst_threshold")
    flat_thr["Rflow_threshold"] = rflow_inst if rflow_inst is not None else 3.0

    # Rcur current echo
    rcur_inst = _get("Rcur", "inst_threshold")
    flat_thr["Rcur_threshold"] = rcur_inst if rcur_inst is not None else 15.0

    # Print the derived thresholds
    print("\nDerived Thresholds:")
    print("-" * 52)
    for k, v in sorted(flat_thr.items()):
        print(f"  {k:<28}: {v:.4f}")
    print("=" * 70)

    # Save to JSON
    outputs_dir = ROOT_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    json_path = outputs_dir / "thresholds.json"
    with open(json_path, "w") as f:
        json.dump(flat_thr, f, indent=2)
    print(f"\nSaved thresholds -> {json_path}")

    return flat_thr


if __name__ == "__main__":
    run_calibration()
