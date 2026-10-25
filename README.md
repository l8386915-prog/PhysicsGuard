# PhysicsGuard: EV DC Fast-Charging Physics Simulator & Digital Twin

Please see root [README.md](../README.md) for full project documentation and textbook equations.

## Quick Reference
- `params.py`: Physical parameters, sources, and assumptions.
- `plant.py`: High-fidelity physical simulator with 1-RC electrochemistry and noisy sensors.
- `twin.py`: Simplified digital twin for state estimation and physics checks.
- `controller.py`: Closed-loop ChargerBMS controller.
- `simulate.py`: Closed-loop session simulator orchestrating plant, controller, and tamper functions.
- `run_normal.py`: Runs a normal charge from 20% to 80% SOC and saves artifacts to `outputs/`.
- `mismatch_study.py`: Runs all 10 mismatch scenarios and saves `results_twin_error.csv` to `outputs/`.
- `tests/test_sanity.py`: Pytest suite validating energy balance, coolant heat, dt invariance, twin independence, and cyberattack impact.
