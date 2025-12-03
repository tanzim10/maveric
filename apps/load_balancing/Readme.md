# Load Balancing Application using Reinforcement Learning

**Version:** 1.0
**Date:** December 3, 2025

---

## 1. Overview

This application is a comprehensive, modular pipeline designed to train and deploy a Reinforcement Learning (RL) agent for dynamic cellular network load balancing. The core objective is to optimize network performance by intelligently redistributing user traffic across cells through dynamic antenna tilt adjustments, preventing congestion hotspots while maintaining balanced resource utilization and Quality of Service (QoS).

The system leverages a **Bayesian Digital Twin (BDT)** RF model, enabling the RL training loop to perform rapid, local RF simulations. This decouples the agent's learning process from backend latency, allowing efficient training on multi-day traffic patterns. The workflow is orchestrated through `main_app.py` with a clear command-line interface.

## 2. Key Features

- **Modular Pipeline:** Each major stage—data preprocessing, BDT model management, RL training, inference, and visualization—is encapsulated in its own Python module for clarity and maintainability.
- **Time-Aware Load Balancing:** The RL agent learns a policy dependent on the hour of the day (tick 0-23), with actions adjusting antenna tilts (`cell_el_deg`) to dynamically redistribute traffic loads.
- **Multi-Day Training & Testing:** Uses distinct datasets for training and testing, allowing the agent to learn from recurring daily traffic patterns and be evaluated on unseen data.
- **Local BDT-based Simulation:** The RL environment uses a pre-trained BDT model to run local RF simulations, providing immediate reward feedback without API calls.
- **Multi-Objective Reward Function:** Balances:
  - **Load Distribution:** Rewards even distribution of traffic across cells to prevent hotspots.
  - **Network Coverage:** Penalizes UEs in weak coverage zones.
  - **QoS:** Scores based on SINR of connected UEs.
  - **Utilization Efficiency:** Optimizes overall network resource usage.
- **Comparative Visualization:** Generates side-by-side plots comparing baseline and optimized load distribution scenarios.

---

## 3. System Architecture

The application follows a linear pipeline, where the output of one stage becomes the input for the next.

```mermaid
flowchart TD
        A[Raw UE Data (Multi-Day)] --> B[Step 1: Preprocess Data]
        B --> C[Gym-Ready UE Data]
        C --> D[Step 2: Train BDT]
        D --> E[BDT Model Pickle]
        E --> F[Step 3: Train RL Agent]
        F --> G[Trained RL Agent (.zip)]
        G --> H[Step 4: Inference]
        H --> I[Console Output (Tilt Config)]
        I --> J[Step 5: Visualize]
        J --> K[Comparison Plot (.png)]
```

---

## 4. Directory Structure

```
load_balancing/
│
├── main_app.py                     # Main orchestrator script
├── bdt_manager.py                  # Manages BDT model training and Docker communication
├── data_preprocessor.py            # Prepares UE data for the Gym environment
├── rl_trainer.py                   # RL training logic
├── cco_rl_env.py                   # Custom Gymnasium environment for load balancing
├── rl_predictor.py                 # Inference using the trained RL agent
├── cco_visualizer.py               # Generates comparison plots
│
├── data/                           # Static inputs required by the pipeline
│   ├── topology.csv                    # Cell tower layout
│   ├── config.csv                      # Initial cell tower configuration
│   └── dummy_ue_training_data.csv      # Training data for the BDT model
│
├── generated_data/                 # Day-wise UE datasets (raw + processed)
│   └── Day_*/
│       ├── ue_data_per_tick/           # Raw UE location data per hour (input)
│       │   ├── generated_ue_data_for_load_balancing_0.csv
│       │   └── ... (up to 23)
│       └── ue_data_gym_ready/          # Preprocessed UE data for RL (output)
│           ├── ue_data_gym_ready_0.csv
│           └── ... (up to 23)
│
└── (Generated Outputs)/
        ├── bdt_model_map.pickle            # Trained BDT model artifact
        ├── load_balancing_agent.zip        # Trained RL agent
        ├── rl_training_logs/               # RL training logs and checkpoints
        └── plots/                          # Visualization outputs
```

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
  python --version  # should report Python 3.10.16
  ```

- Configure Python Path

  ```bash
  # from maveric root
  export PYTHONPATH="$(pwd)":$PYTHONPATH
  ```

- **Docker:** BDT model training runs inside a Docker container. Ensure Docker daemon is running.

  Note: Refer [maveric/README.md ### Booting up RADP](../../README.md#booting-up-radp) for host GPU utilization.

  ```bash
  # Set dev port first if using dev mode
  cp .env-dev .env
  ```

  ```bash
  # from maveric root
  docker build -t radp radp
  docker compose -f dc.yml -f dc-dev.yml up -d --build
  ```

- **Required Python Packages:**

  ```bash
  # from maveric root
  pip install -r radp/client/requirements.txt
  pip install -r apps/requirements.txt
  ```

- **Required Data to Train Upon:** Have these following data dir in the app dir, As described above [see Directory Structure](#4-directory-structure):

  - `generated_data/`
  - `data/`

Note: If needed, these datasets can be generated with the utilities documented in [`radp/digital_twin/traffic_load/Readme.md`](../../radp/digital_twin/traffic_load/Readme.md)

    - generate `./generated_data/`
    - copy `generated_data/` to load_balancing rApp dir
    - mkdir `./data/` inside load_balancing rApp dir and copy `topology.csv`, `config.csv` and `dummy_ue_training_data.csv` there

---

## 6. Application Workflow & Usage

The application is run as a pipeline, with each step triggered by a specific flag to `main_app.py`.

> cd apps/load_balancing

### **Step 1: Preprocess UE Data**

Prepares raw, per-hour UE location data for simulation. Pass `--train-days` and `--test-day` as desired.

```bash
python main_app.py --preprocess-data --train-days 0 1 2 --test-day 3
```

- **Input:** `generated_data/Day_*/ue_data_per_tick/`
- **Output:** `generated_data/Day_*/ue_data_gym_ready/`

---

### **Step 2: Train the Bayesian Digital Twin (BDT)**

Trains the RF simulation model using a backend service in Docker.

- **Prerequisites:** Docker container (e.g., `radp_dev-training-1` for dev mode, `radp_prod-training-1` for prod) must be running.

```bash
python main_app.py --train-bdt --bdt-model-id "bdt_load_balance_v1" --container "radp_dev-training-1"
```

- **Inputs:** `data/topology.csv`, `data/dummy_ue_training_data.csv`
- **Output:** `bdt_model_map.pickle`

---

### **Step 3: Train the RL Load Balancing Agent**

Trains the PPO agent using preprocessed data and the BDT model.

```bash
python main_app.py --train-rl --train-days 0 1 2 --total-timesteps 24000
```

- **Inputs:** `bdt_model_map.pickle`, `generated_data/Day_*/ue_data_gym_ready/`, `data/topology.csv`, `data/config.csv`
- **Outputs:** `load_balancing_agent.zip`, `rl_training_logs/`

---

### **Step 4: Run Inference**

Uses the trained agent to predict the optimal cell tilt configuration for a specific hour.

```bash
python main_app.py --infer --tick <T>
```

- **Inputs:** `load_balancing_agent.zip`, `data/topology.csv`
- **Output:** Console table of predicted optimal tilt angles for each cell.

---

### **Step 5: Visualize the Results**

Generates a side-by-side plot comparing network state before and after load balancing optimization.

```bash
python main_app.py --visualize --test-day <D> --tick <T>
```

- **Inputs:** `load_balancing_agent.zip`, `bdt_model_map.pickle`, `data/topology.csv`, `data/config.csv`, `generated_data/Day_<D>/ue_data_gym_ready/`
- **Output:** `.png` image in `plots/` directory.

---

### **Full Pipeline Example**

```bash
# 1. Prepare UE data for training (days 0-2) and testing (day 3)
python main_app.py --preprocess-data --train-days 0 1 2 --test-day 3

# 2. Train the core RF simulation model (ensure Docker container is running)
python main_app.py --train-bdt --bdt-model-id "bdt_load_balance_v1" --container "radp_dev-training-1"

# 3. Train the RL agent on the first 3 days of data
python main_app.py --train-rl --train-days 0 1 2 --total-timesteps 24000

# 4. Predict the optimal configuration for peak hour (e.g., 2 PM / 14:00)
python main_app.py --infer --tick 14

# 5. Visualize the impact of the optimization on the test data for that hour
python main_app.py --visualize --test-day 3 --tick 14
```

---

## 7. Detailed Module Breakdown

### **data_preprocessor.py**

- **Function:** Prepares raw UE data for the RL environment.
- **Logic:** Reads per-tick CSV files, renames `lon` to `loc_x` and `lat` to `loc_y`, saves to `ue_data_gym_ready/`.

### **bdt_manager.py**

- **Function:** Manages backend-intensive training of the RF model.
- **Logic:** Uses a client (e.g., `radp_client`) to send topology and training data to a backend service. Downloads the trained model from Docker.

### **rl_trainer.py & cco_rl_env.py**

- **Function:** Orchestrates PPO agent training for load balancing.
- **Logic:** Loads BDT model and preprocessed UE data, initializes custom environment, trains agent with multi-objective reward focusing on load distribution.

### **rl_predictor.py**

- **Function:** Uses the trained agent for immediate recommendations.
- **Logic:** Loads PPO agent, predicts best tilt configuration for a target tick, outputs a human-readable table.

### **cco_visualizer.py**

- **Function:** Qualitative assessment of RL agent's performance.
- **Logic:** Simulates baseline and optimized scenarios, generates side-by-side plots showing load distribution improvements and UE signal strength.
