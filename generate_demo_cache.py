"""
generate_demo_cache.py - Pre-record fallback CSVs and metadata for all 17 scenarios
===================================================================================
Generates offline cached runs (seed 42) into demo_cache/ so that the Streamlit
app starts immediately and has 100% offline fallback guarantees.
"""

import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sim_engine import (
    ALL_SCENARIOS, MAIN_DEMO_SCENARIOS,
    run_full_simulation, save_to_cache, DEMO_CACHE_DIR
)


def main():
    print("=" * 78)
    print("PHYSICSGUARD DEMO CACHE GENERATOR (Seed 42)")
    print(f"Target Directory: {DEMO_CACHE_DIR}")
    print("=" * 78)

    DEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    for idx, name in enumerate(ALL_SCENARIOS, 1):
        t0 = time.time()
        print(f"[{idx:2d}/{len(ALL_SCENARIOS):2d}] Simulating {name:27s} ...", end="", flush=True)

        res = run_full_simulation(name, seed=42)
        save_to_cache(res, name, cache_dir=DEMO_CACHE_DIR)

        elapsed = time.time() - t0
        k = res["kpis"]
        print(
            f" done ({elapsed:4.1f}s) | "
            f"Unpr: {k['peak_t_unpr']:5.1f}C -> Prot: {k['peak_t_prot']:5.1f}C | "
            f"Chg: {k['dur_prot']:5.0f}s ({k['penalty_pct']:+5.1f}%) | "
            f"Badge: {k['overall_badge']} | Mode: {k['final_mode']}"
        )

    total_time = time.time() - t_start
    print("=" * 78)
    print(f"Successfully generated demo cache for all {len(ALL_SCENARIOS)} scenarios in {total_time:.1f}s.")
    print("=" * 78)


if __name__ == "__main__":
    main()
