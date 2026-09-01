# Release v0.9.0-beta — 5-DoF Physical Dynamics & PINN Diagnostics (Pre-Release)

This pre-release marks the transition of the **Dynamic Coin Toss** research project from a foundational chaotic baseline into a fully realized physical simulation framework. It introduces 3D spatial kinematics, Physics-Informed Neural Networks (PINNs), and high-resolution diagnostic telemetry to map the boundary where mechanical determinism degrades into statistical randomness.

---

## Key Highlights & Architectural Additions

### 1. Diaconis-Holmes-Montgomery (DHM) Physics Engine (`/coin_toss_dhm_research`)
* **Deterministic Baseline Integration:** Implemented the core DHM framework (`dhm_coin_physics.py` and `physics_engine.py`) to simulate coin flips governed by initial angular velocity, vertical impulse, and launch angle.
* **Physics-Informed Neural Network (PINN):** Integrated a specialized neural architecture (`pinn_model.py`) that enforces physical constraints during training to predict binary outcomes (Heads vs. Tails) based directly on initial state vectors.
* **Pipeline Automation:** Added `master_simulation.py` for end-to-end execution of physical generation, model training, and accuracy evaluation.

### 2. 5 Degrees of Freedom (5-DoF) Kinematic Suite (`/dhm-research-test-with-dof`)
* **Complex 3D Dynamics:** Expanded the physical simulation engine (`physics_engine.py`) to model 5 Degrees of Freedom, incorporating precession, nutation, torque-free rotation, and spatial translation in a viscous medium.
* **Covariance & Error Analytics:** Added dedicated diagnostic modules (`diagnostic_covariance.py`, `metrics_error.py`, `metrics_efficiency.py`, `metrics_progress.py`) to track how minor variations in initial conditions correlate with exponential model failure.
* **Unified Runner & Visualization:** Introduced `run_all.py` and `animated_visualizer.py` to concurrently execute the physical engine, run the predictive model (`model_predictor.py`), and render real-time Matplotlib readouts of system trajectories.

### 3. Baseline Isolation
* Re-isolated the original TensorFlow/LSTM logistic map simulation (`animated_simulation.py` / `try_chaos_theory.py`) to the root directory, establishing a pure mathematical baseline for comparing artificial chaotic maps against real Newtonian mechanics.

---

## Module Breakdown

| Module | Purpose | Core Dependencies |
| :--- | :--- | :--- |
| `coin_toss_dhm_research/` | 2D/3D DHM physics modeling and PINN classification | `tensorflow`, `numpy`, `scipy` |
| `dhm-research-test-with-dof/` | 5-DoF spatial kinematics, covariance analysis, and diagnostic UI | `matplotlib`, `numpy`, `tensorflow` |
| `animated_simulation.py` | Standalone mathematical baseline for chaotic trajectory decay | `matplotlib`, `numpy` |

---

## Known Limitations & Experimental Scope

* **Computational Overhead in 5-DoF Diagnostics:** Launching the full diagnostic suite via `run_all.py` spawns multiple concurrent Matplotlib rendering loops. High-frequency updates may cause frame drops on systems with limited multi-threading capabilities.
* **Sensitivity Under High Spin Rates:** The PINN model in `coin_toss_dhm_research` experiences rapid loss divergence when initial angular velocity exceeds extreme thresholds, accurately reflecting physical chaos but requiring further hyperparameter tuning for stable convergence.
* **Aerodynamic Approximations:** Current air resistance parameters use a constant drag coefficient; future iterations will model non-linear turbulent drag forces across flipping surfaces.

---

## Quick Start for Testing

To run the new 5-DoF diagnostic suite locally:

```bash
git clone -b v0.9.0-beta [https://github.com/rbksh/Dynamic-Coin-Toss.git](https://github.com/rbksh/Dynamic-Coin-Toss.git)
cd Dynamic-Coin-Toss/dhm-research-test-with-dof
pip install -r requirements.txt
python run_all.py