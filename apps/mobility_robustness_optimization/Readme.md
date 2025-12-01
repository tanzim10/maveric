# Mobility Robustness Optimization Application

**Version:** 1.0
**Date:** December 1, 2025

---

## 1. Overview

This application is a comprehensive, modular system designed to optimize cellular network mobility parameters to minimize handover failures and Radio Link Failures (RLF). The core objective is to find optimal **Hysteresis (HYST)** and **Time-To-Trigger (TTT)** values that maximize network uptime by reducing service interruptions during User Equipment (UE) mobility while maintaining reliable cell attachments.

The system leverages a **Bayesian Digital Twin (BDT)** RF model to predict received power at UE locations, enabling rapid local simulations of different mobility scenarios. The optimization workflow supports two distinct solving approaches—a simple random search and a Reinforcement Learning (RL) based method—allowing flexible trade-offs between speed and solution quality.

---

## 2. Key Features

- **Modular Architecture:** Each major component—BDT model management, MRO optimization, Simple MRO solver, and RL-based solver—is encapsulated in its own module for clarity and maintainability.
- **Mobility-Aware Optimization:** Optimizes handover parameters (HYST and TTT) to minimize service interruptions caused by handovers and Radio Link Failures.
- **Bayesian Digital Twin Integration:** Uses pre-trained BDT models to perform local RF power predictions, eliminating backend API latency during optimization.
- **Dual Solving Approaches:**
  - **Simple MRO:** Fast random search optimization with configurable epochs.
  - **Reinforced MRO:** PPO-based Reinforcement Learning for more sophisticated parameter tuning.
- **Multi-Objective MRO Metric:** Balances:
  - **Operational Time:** Maximizes effective UE operational time.
  - **Handover Efficiency:** Minimizes seamless handover interruptions.
  - **RLF Mitigation:** Penalizes Radio Link Failures heavily.
- **Model Persistence:** Save and load trained BDT models for reuse across sessions.
- **Incremental Learning:** Update existing BDT models with new observations without retraining from scratch.

---

## 3. System Architecture

The MRO application follows a clear pipeline structure where BDT models are trained/updated first, then used for optimization.

```mermaid
flowchart TD
    A[UE Data with Rx Power] --> B[Step 1: Train/Update BDT]
    B --> C [BDT Model Pickle]
    C --> D [Step 2: Solve for Optimal HYST & TTT]
    D --> E {Solving Approach} using Simple MRO | RL MRO
    E --> F [Optimal HYST & TTT]
    F --> G [Perform Attachment on Data using Optimal HYST & TTT]
    G --> H [Step 3: Visualize Results]
    H --> I [Scatter Plots & SINR Analysis]
```

---

## 4. Directory Structure

```
mobility_robustness_optimization/
│
├── mobility_robustness_optimization.py  # Base class with BDT management and shared utilities
├── simple_mro.py                        # Simple random search-based MRO solver
├── mro_rl.py                            # Reinforcement Learning-based MRO solver
│
├── tests/
│   └── test_mobility_robustness_optimization.py  # Unit tests
│
└── (Required Data - typically in notebooks/data/mro_data/):
    ├── mro_topology.csv                           # Cell tower layout
    ├── UE_data_with_rxpower_20UE_50ticks.csv          # Example UE mobility data
    ├── ue_data_with_rxpower_20UE_100ticks_train.csv   # Example BDT Training Data
    ├── UE_data_with_rxpower_20UE_100ticks_update.csv  # Example BDT Testing Data
    ├── UE_data_20UE_100ticks.csv                      # Example UE Mobility data (without Rx power)
    └── digital_twins.pkl                              # Saved BDT models (generated)
```

---

## 5. Prerequisites

- Go to project root

  ```bash
  cd path/to/maveric
  ```

- **Python 3.9-3.10:** Create venv and activate:

  Example: Ensure shell has `python3.10`

  ```bash
  python3.10 -m venv .venv
  source .venv/bin/activate
  python --version  # should report Python 3.10.x
  ```

- Configure Python Path

  ```bash
  # from maveric root
  export PYTHONPATH="$(pwd)":$PYTHONPATH
  ```

- Upgrade Pip

  ```bash
  pip install --upgrade pip
  ```

- **Required Python Packages:**

  ```bash
  # from maveric root
  pip install -r apps/requirements.txt
  ```

- **Required Data:** Ensure you have the MRO data available:

  ```bash
  # from maveric root
  cd notebooks/data
  unzip mro_data.zip  # if not already extracted
  ```

  This creates the `notebooks/data/mro_data/` directory with required CSV files.

---

## 6. Application Workflow & Usage

The application can be used either through the Python classes directly or via Jupyter notebooks. Below are examples using both approaches.

### **Approach: Using Jupyter Notebook (Recommended)**

See `notebooks/mro.ipynb` for a complete interactive walkthrough.

---

## 7. Detailed Module Breakdown

### **mobility_robustness_optimization.py (Base Class)**

- **Purpose:** Abstract base class providing core BDT management and utility functions.
- **Key Methods:**
  - `train_or_update_rf_twins(new_data)`: Trains new BDT models or updates existing ones with new observations.
  - `save_bdt(file_relative_path)`: Saves trained BDT models to a pickle file.
  - `load_bdt(file_relative_path)`: Loads BDT models from a pickle file.
  - `solve()`: Abstract method implemented by subclasses.
  - `_predictions(pred_data)`: Uses BDT to predict received power at UE locations.
  - `_prepare_train_or_update_data(df)`: Preprocesses data for BDT training.
  - `_add_sinr_column(df)`: Computes SINR for each UE-cell pair.

### **simple_mro.py (Simple MRO Solver)**

- **Purpose:** Implements random search optimization to find optimal HYST and TTT.
- **Logic:**
  - Generates random HYST and TTT values within valid ranges.
  - Evaluates MRO metric for each combination.
  - Tracks all evaluations and returns the best performing parameters.
- **Advantages:** Fast, simple to understand, good for initial exploration.
- **Method:** `solve(n_epochs)` - Runs optimization for specified number of epochs.

### **mro_rl.py (Reinforced MRO Solver)**

- **Purpose:** Uses Proximal Policy Optimization (PPO) to learn optimal HYST and TTT.
- **Logic:**
  - Creates a custom Gymnasium environment (`ReinforcedMROEnv`).
  - Trains a PPO agent using the MRO metric as reward signal.
  - Converges to optimal parameters through iterative learning.
- **Advantages:** More sophisticated, can discover complex patterns, potentially better solutions.
- **Method:** `solve(total_timesteps)` - Trains RL agent for specified timesteps.

### **Key Utility Functions**

- `calculate_mro_metric(data)`: Computes the MRO metric measuring effective operational time after accounting for handover and RLF delays.
- `_count_handovers(df)`: Counts seamless cell-to-cell handovers.
- `_count_rlf(df)`: Counts Radio Link Failures.

---

## 8. Understanding the MRO Metric

The MRO metric quantifies the **effective operational time** for UEs, accounting for service interruptions:

```
MRO Metric = T - (n_s × t_s + n_f × t_nas)
```

Where:
- **T**: Total simulation time (based on number of ticks)
- **n_s**: Number of seamless handovers (cell-to-cell switches)
- **t_s**: Time penalty per seamless handover (50 ms)
- **n_f**: Number of Radio Link Failures
- **t_nas**: Time penalty per RLF (1000 ms)

**Goal:** Maximize this metric by finding HYST and TTT values that minimize unnecessary handovers and RLF events.

---

## 9. Data Format Requirements

### **Topology CSV**

Must contain cell tower information:

```csv
cell_id,cell_lat,cell_lon,cell_carrier_freq_mhz,cell_az_deg
cell_1,45.0,-73.0,2100,120
cell_2,46.0,-74.0,2100,240
```

### **UE Data with Rx Power (for BDT Training)**

Must include received power measurements in **cartesian format** (UEs × Cells):

```csv
latitude,longitude,cell_id,cell_rxpwr_dbm
90.412,23.810,cell_1,-85.3
90.412,23.810,cell_2,-92.1
90.413,23.811,cell_1,-87.5
90.413,23.811,cell_2,-90.2
```

**Required Columns:** `latitude`, `longitude`, `cell_id`, `cell_rxpwr_dbm`

---

## 10. Tips and Best Practices

1. **Start with Simple MRO:** Use Simple MRO for quick initial optimization, then try Reinforced MRO for refinement.

2. **Adjust Epochs/Timesteps:**
   - Simple MRO: 100-500 epochs for good coverage
   - Reinforced MRO: 1000-10000 timesteps for convergence

3. **Incremental Updates:** When new observations arrive, use `train_or_update_rf_twins()` with existing models rather than retraining from scratch.

4. **Model Persistence:** Always save BDT models after training to avoid recomputation.

5. **GPU Acceleration:** Reinforced MRO automatically uses CUDA if available for faster RL training.

6. **Data Quality:** Ensure UE data includes diverse mobility patterns (stationary, pedestrian, cyclist, car) for robust optimization.

7. **Mobility Alpha Parameter:** Optionally use the mobility model parameter regression to learn the Gauss-Markov alpha from your data:
   ```python
   from radp.digital_twin.mobility.param_regression import get_predicted_alpha
   alpha = get_predicted_alpha(ue_data, alpha0=0.5, seed=42)
   mobility_model_params["ue_tracks_generation"]["params"]["gauss_markov_params"]["alpha"] = alpha
   ```

---

## 11. Troubleshooting

**Issue:** `ValueError: Bayesian Digital Twins are not trained`
**Solution:** Train or load BDT models before calling `solve()`.

**Issue:** `KeyError` when training BDT
**Solution:** Ensure input data has required columns: `latitude`, `longitude`, `cell_id`, `cell_rxpwr_dbm`.

**Issue:** Poor optimization results
**Solution:**
- Increase number of epochs/timesteps
- Ensure BDT models are well-trained with sufficient data
- Check that mobility model parameters match your use case

**Issue:** Slow RL training
**Solution:**
- Reduce `total_timesteps` for faster but potentially suboptimal results
- Use GPU by ensuring PyTorch CUDA is available
- Consider using Simple MRO as a faster alternative

---

## 12. References

- **Notebook Demo:** `notebooks/mro.ipynb`
- **Unit Tests:** `apps/mobility_robustness_optimization/tests/test_mobility_robustness_optimization.py`
- **RADP Library:** `notebooks/radp_library.py` (visualization utilities)
- **Digital Twin Documentation:** See `radp/digital_twin/` for BDT implementation details

---

## 13. Version History

**v1.0 (December 1, 2025)**
- Initial release with Simple MRO and Reinforced MRO solvers
- BDT training and incremental update support
- Comprehensive visualization utilities
- Full test coverage
