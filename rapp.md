# maveric_platform_rapp - Developer Guide

**Version:** 0.4.2 <br>
**Last Updated:** October 2025

---

## Table of Contents

0. [Quick Start Guide](#0-quick-start-guide)
1. [Overview & Introduction](#1-overview--introduction)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Concepts](#3-core-concepts)
4. [API Reference](#4-api-reference)
5. [Data Models](#5-data-models)
6. [Event-Driven Architecture](#6-event-driven-architecture)
7. [User Flows](#7-user-flows)
8. [Configuration & Deployment](#8-configuration--deployment)
9. [Development Guide](#9-development-guide)
10. [Appendix](#10-appendix)

---

## 0. Quick Start Guide

### For Developers

**Prerequisites:** Docker, docker-compose

**1. Start All Services (Recommended):**

The **maveric_platform_dev** repository includes docker-compose files that start all infrastructure and application services together.

```bash
# Navigate to parent platform repository
cd to/maveric_platform_dev

# Start infrastructure and all application services (includes rApp API + Worker)
docker compose -f docker-compose.infra.yml -f docker-compose.apps.yml up -d --build

# Verify services are running
docker compose ps
```

**Alternative: Run rApp Separately:**

If you need to run only the rApp service independently:

```bash
# Navigate to rapp repository
cd path/to/maveric_platform_rapp

# Build and run using Docker Compose (see docker-compose.yml template in Section 9.2.4)
docker compose up -d --build

# Or build the image manually
docker build -t maveric/rapp-engine:latest .
```

**2. Initialize Database:**

```bash
# Run Alembic migrations (recommended)
docker exec -it rapp alembic upgrade head
```

> **Note:** For detailed database setup options including manual SQL migrations, see Section 9.1.

**3. Verify Setup:**

```bash
# Check API health
curl http://localhost:8004/

# List available rApps
curl -H "X-API-Key: dev-secret-key" \
  http://localhost:8004/v1/tenants/11111111-1111-1111-1111-111111111111/rapps

# View API logs
docker compose logs -f rapp

# View Worker logs
docker compose logs -f rapp-worker
```

**Alternative: Local Development (Without Docker):**

If you prefer running Python directly for faster iteration:

```bash
# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment - use the existing .env file (already pushed for local development)
# The .env file contains all necessary environment variables
# Review and adjust values if needed (e.g., ensure services point to localhost)
source .env

# Terminal 1: API Server
uvicorn app.main:app --reload --port 8004

# Terminal 2: Worker
python -m app.workers.rapp_worker
```

### For Users (API Quick Reference)

**Train a Model:**

```bash
curl -X POST \
  "http://localhost:8004/v1/tenants/{tenant_id}/rapps/es/train" \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "rapp_model_id": "es-model-v1",
    "bdt_id": "BDT-001",
    "baseline_id": "baseline-1",
    "dataset_id": "training-data-1",
    "params": {"total_timesteps": 50000}
  }'
```

**Check Training Status:**

```bash
curl "http://localhost:8004/v1/tenants/{tenant_id}/rapps/es/models/es-model-v1" \
  -H "X-API-Key: your_key"
```

**Run Inference:**

```bash
curl -X POST \
  "http://localhost:8004/v1/tenants/{tenant_id}/rapps/es/models/es-model-v1/infer" \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "baseline_id": "baseline-1",
    "bdt_id": "BDT-001",
    "ue_dataset_id": "test-data-1",
    "tick": 12
  }'
```

**Next Steps:**

- Read [Section 1: Overview](#1-overview--introduction) for conceptual understanding
- Check [Section 4: API Reference](#4-api-reference) for complete API documentation
- See [Section 7: User Flows](#7-user-flows) for detailed workflows

---

## 1. Overview & Introduction

### 1.1 About maveric_platform_rapp

`maveric_platform_rapp` is a microservice that implements the **rApp Engine** for the Maveric platform. It provides machine learning-powered optimization capabilities for Radio Access Network (RAN) management through four specialized rApps (non real time Applications):

- **ES (Energy Saving)** - Optimizes cell on/off states and antenna tilt to reduce energy consumption while maintaining coverage
- **LB (Load Balancing)** - Adjusts antenna tilt angles to balance user equipment (UE) load across cells
- **MRO (Mobility Robustness Optimization)** - Optimizes handover parameters (hysteresis, time-to-trigger) to reduce handover failures
- **CCO (Coverage Capacity Optimization)** - Uses differential gradient policy to optimize antenna tilt for coverage and capacity

The service operates asynchronously, accepting training and inference requests via REST API. Training requests are processed through Kafka-based event handlers with background workers, while inference requests use a thread pool executor for faster execution.

### 1.2 Platform Context

The rApp Engine is part of the Maveric platform's Near-Real-Time RAN Intelligent Controller (Near-RT RIC) architecture, following O-RAN Alliance specifications:

```
┌─────────────────────────────────────────────────────────┐
│                    Maveric Platform                     │
│                                                         │
│  ┌──────────────┐    ┌─────────────┐   ┌────────────┐   │
│  │  SMO Sim     │───▶│  rApp Engine│◀──│  Data Sim  │   │
│  │  (Baseline,  │    │  (This Repo)│   │  (UE Data, │   │
│  │   Topology,  │    │             │   │   Traffic) │   │
│  │   Config)    │    │  Training & │   │            │   │
│  └──────────────┘    │  Inference  │   └────────────┘   │
│                      │             │                    │
│  ┌──────────────┐    │  ES/LB/MRO  │   ┌────────────┐   │
│  │     BDT      │───▶│     CCO     │   │     S3     │   │
│  │  (Bayesian   │    │             │   │  (Models,  │   │
│  │   Digital    │    └─────────────┘   │   Data)    │   │
│  │   Twin)      │                      └────────────┘   │
│  └──────────────┘                                       │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  PostgreSQL (Metadata) + Kafka (Events)         │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Key Platform Interactions:**

1. **SMO Sim** provides baseline configurations, network topology, and cell configurations
2. **BDT Service** provides trained Bayesian Digital Twin models for RF propagation prediction
3. **Data Sim** generates or provides real UE traffic data for training and inference
4. **S3/MinIO** stores all artifacts (trained models, datasets, BDT pickles)
5. **PostgreSQL** persists metadata (model registry, inference runs, training jobs)
6. **Kafka** enables asynchronous, decoupled processing of training and inference requests

### 1.3 Core Capabilities

**Training Capabilities:**

- Train reinforcement learning models (ES, LB using PPO from stable-baselines3)
- Train optimization-based models (MRO using grid search, CCO using differential gradient policy)
- Support for custom hyperparameters per rApp type
- Automatic artifact management (model persistence to S3)
- Multi-day training data support with tick-aware processing

**Inference Capabilities:**

- Real-time inference using trained models
- Tick-based predictions (0-23 hourly ticks)
- Automatic data loading from S3 (topology, config, UE datasets)
- Coverage and optimization metric calculation
- Result visualization (plot data generation)
- CCO inference caching and reuse (avoid redundant optimization)

**Operational Features:**

- Multi-tenant isolation via PostgreSQL Row-Level Security (RLS)
- Async processing via Kafka event-driven architecture
- Fault tolerance with retry mechanisms
- Comprehensive structured logging (JSON format)
- Prometheus metrics and OpenTelemetry tracing
- S3 and MinIO compatibility for object storage
- IAM role assumption for AWS S3 access

### 1.4 Key Features

1. **Multi-Tenancy**: Complete tenant isolation at database and storage layers
2. **Scalability**: Kafka-based event processing allows horizontal scaling of workers
3. **Flexibility**: Support for both S3 and local/EFS storage modes
4. **Robustness**: Comprehensive error handling, validation, and retry logic
5. **Observability**: Structured logging, Prometheus metrics, OpenTelemetry traces
6. **Security**: API key authentication, RLS policies, no credential logging

### 1.5 Technology Stack

**Core Framework:**

- **FastAPI** - Modern async web framework with automatic OpenAPI documentation
- **Python 3.10+** - Primary programming language
- **Pydantic** - Data validation and settings management

**Data Storage:**

- **PostgreSQL** - Relational database for metadata (rapp_models, inference_runs, training_jobs)
- **MongoDB** - Document store for additional metadata (optional)
- **S3/MinIO** - Object storage for models, datasets, and artifacts

**Event Processing:**

- **Kafka** - Event streaming platform for async job processing
- **Threading** - Python thread pools for worker execution

**Machine Learning:**

- **stable-baselines3** - Reinforcement learning library (PPO algorithm)
- **PyTorch** - Deep learning framework (backend for stable-baselines3)
- **GPyTorch** - Gaussian Process library for Bayesian Digital Twin
- **Gymnasium** - RL environment interface (successor to OpenAI Gym)
- **NumPy/Pandas** - Data manipulation and analysis

**Observability:**

- **OpenTelemetry** - Distributed tracing
- **Prometheus** - Metrics collection
- **Structured Logging** - JSON logs for centralized aggregation

**Deployment:**

- **Docker** - Containerization
- **Kubernetes** - Container orchestration (implied by platform architecture)

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

The rApp Engine follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT (External)                       │
│                   (Portal, xApp, Admin Tool)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS + JWT/API Key
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       API LAYER (FastAPI)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │   Routes     │─▶│  Endpoints   │─▶│  Request/Response  │     │
│  │  (Security)  │  │  (rapps.py)  │  │  Validation        │     │
│  └──────────────┘  └──────────────┘  └────────────────────┘     │
└────────────────────────────┬────────────────────────────────────┘
                             │ Publish Events (Training Only)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVENT BUS (Kafka)                            │
│  Topics: maveric.rapp.train.v1 (training only)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ Consume Events
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WORKER LAYER                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  rapp_worker.py (Kafka Consumer - Training)              │   │
│  │  ├─ Training Handler  ├─ Data Loader                     │   │
│  │                                                          │   │
│  │  inference_runner.py (Thread Pool - Inference)           │   │
│  │  ├─ Inference Handler  ├─ Artifact Manager               │   │
│  │  └─ Status Updater     └─ Error Handler                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ Invoke
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BUSINESS LOGIC (Services)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │ Inference    │  │ Data Loader  │  │ Metrics Builder    │     │
│  │ Runner       │  │ (S3 Utils)   │  │ (Validation)       │     │
│  └──────────────┘  └──────────────┘  └────────────────────┘     │
└────────────────────────────┬────────────────────────────────────┘
                             │ Train/Infer
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RADP LIBRARY (radplib/)                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │   ES    │  │   LB    │  │   MRO   │  │   CCO   │             │
│  │ (RL/PPO)│  │ (RL/PPO)│  │ (Grid)  │  │ (dGPCO) │             │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Dependencies: BDT, Digital Twin, Cell Selection        │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┴─────────────────────┐
        ▼                                          ▼
┌──────────────────┐                    ┌──────────────────────┐
│  PERSISTENCE     │                    │  OBJECT STORAGE      │
│  (PostgreSQL)    │                    │  (S3/MinIO)          │
│                  │                    │                      │
│  - rapp_models   │                    │  - Trained models    │
│  - inference_runs│                    │  - BDT pickles       │
│  - training_jobs │                    │  - Datasets (CSV)    │
│                  │                    │  - Artifacts (ZIP)   │
└──────────────────┘                    └──────────────────────┘
```

### 2.2 Component Interaction Flow

**Training Request Flow:**

1. Client sends `POST /tenants/{tenant_id}/rapps/{rapp_id}/train` with payload
2. API endpoint validates request, authenticates client
3. Creates `rapp_models` record with status=`queued`
4. Publishes training event to `maveric.rapp.train.v1` Kafka topic
5. Returns `202 Accepted` with `rapp_model_id`
6. Worker consumes event from Kafka
7. Worker downloads BDT model and dataset from S3
8. Worker invokes `radplib` training function for specified rApp
9. Training runs (RL iterations, optimization epochs)
10. Worker saves trained model to S3 and local cache
11. Worker updates `rapp_models` status to `ready` with metrics
12. Client polls `GET /rapps/{rapp_id}/models/{rapp_model_id}` until status=`ready`

**Inference Request Flow:**

1. Client sends `POST /tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}/infer` with tick/dataset
2. API endpoint validates request, creates `inference_runs` record with status=`queued`
3. Submits inference job to **thread pool executor** with 4 workers (not Kafka - asynchronous via background threads)
4. Returns `202 Accepted` with `run_id` and `Location` header
5. Thread pool worker processes the job in background
6. Worker updates status to `running`, downloads trained model, BDT, baseline, and UE dataset from S3
7. Worker invokes `radplib` inference function
8. Inference generates predictions, calculates optimization metrics
9. Worker generates plot data for visualization
10. Worker updates `inference_runs` with results, status=`completed`
11. Client polls `GET /tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}/infer/{run_id}` until status=`completed`

### 2.3 Multi-Tenancy

The rApp Engine enforces strict tenant isolation at multiple layers:

**Database Layer (PostgreSQL RLS):**

```sql
-- All tables have RLS enabled
ALTER TABLE rapp_models ENABLE ROW LEVEL SECURITY;

-- Policy restricts access to current tenant
CREATE POLICY rapp_models_rls ON rapp_models
USING (tenant_id = current_setting('app.current_tenant')::uuid)
WITH CHECK (tenant_id = current_setting('app.current_tenant')::uuid);
```

**Application Layer:**

```python
# Before every database query
set_current_tenant(db, tenant_id)

# This sets the session variable for RLS
db.execute(text("SET app.current_tenant = :tenant_id"), {"tenant_id": tenant_id})
```

**Storage Layer (S3 Prefixing):**

```
s3://{bucket}/tenants/{tenant_id}/
  ├── models/
  │   ├── bdt/{bdt_id}.pickle
  │   └── rapps/{rapp_id}/{rapp_model_id}.zip
  ├── datasets/
  │   └── {dataset_id}.csv
  └── baselines/
      └── {baseline_id}/
          ├── topology.csv
          └── config.csv
```

> S3_PREFIX is configurable (default: empty). Production typically uses `S3_PREFIX="tenants/"` resulting in paths like `s3://bucket/tenants/{tenant_id}/...`\*

**Security Guarantees:**

- Tenant A cannot access Tenant B's models, data, or inference results
- All API endpoints require `tenant_id` in path
- JWT tokens (future) will include tenant claim validation
- Worker processes respect tenant boundaries when loading artifacts

### 2.4 Data Flow Diagram

**Training Data Flow:**

```
UE Dataset (CSV)      Topology (CSV)      BDT Model (pickle)
      │                    │                      │
      └────────────────────┴──────────────────────┘
                           │
                           ▼
                    S3 / MinIO Storage
                           │
                           ▼
                      Data Loader ─────┐
                           │           │
                           ▼           │
                    Preprocessor       │
                           │           │
                           ▼           │
         ┌─────────────────────────────┤
         │                             │
         ▼                             ▼
  RL Environment (ES/LB)        Optimization (MRO/CCO)
         │                             │
         ▼                             ▼
    PPO Training                  Grid/dGPCO
         │                             │
         └─────────────┬───────────────┘
                       │
                       ▼
              Trained Model (ZIP)
                       │
                       ▼
              S3 / MinIO + Local Cache
                       │
                       ▼
          rapp_models.status = 'ready'
```

**Inference Data Flow:**

```
Trained Model  +  BDT Model  +  UE Dataset  +  Topology/Config
      │               │              │                 │
      └───────────────┴──────────────┴─────────────────┘
                           │
                           ▼
                    Data Loader (S3)
                           │
                           ▼
           Thread Pool Executor (per rApp)
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    Predictions      Coverage Calc    Plot Generation
          │                │                │
          └────────────────┴────────────────┘
                           │
                           ▼
                  InferenceResult
                   (plot, metrics, text)
                           │
                           ▼
          inference_runs.result = {...}
```

---

## 3. Core Concepts

### 3.1 rApp Types

The rApp Engine supports four distinct Radio Application types, each addressing specific RAN optimization objectives:

#### 3.1.1 ES - Energy Saving

**Purpose:** Minimize energy consumption by optimizing cell on/off states and antenna tilt angles while maintaining acceptable coverage and QoS.

**Algorithm:** Reinforcement Learning with Proximal Policy Optimization (PPO)

**Key Parameters:**

- `train_days`: Days of training data (default: [0])
- `total_timesteps`: RL training iterations (default: 50,000)
- `learning_rate`: PPO learning rate (default: 5e-2)
- `ent_coef`: Entropy coefficient for exploration (default: 0.001)

**Action Space:**

- Per cell: ON/OFF state + tilt angle (0-20 degrees in 2-degree increments)

**Reward Function:**

```python
reward = 0.2 * cco_score + 0.1 * load_balance_score + 1.0 * energy_saving_score
```

Where:

- `cco_score`: Coverage quality (weak coverage penalty, over-coverage penalty)
- `load_balance_score`: Negative standard deviation of UE counts per cell
- `energy_saving_score`: `(1 - active_cells/total_cells) * 100`

**Output:**

- `TextMetricsES`: List of cell configurations with `cell_id`, `el_degree`, `on_off` (boolean)
- `optimization_metric`: Weighted reward score (scaled to integer)

#### 3.1.2 LB - Load Balancing

**Purpose:** Distribute UE load evenly across cells by adjusting antenna tilt angles to influence cell selection.

**Algorithm:** Reinforcement Learning with Proximal Policy Optimization (PPO)

**Key Parameters:**

- `train_days`: Days of training data
- `total_timesteps`: RL training iterations
- `learning_rate`: PPO learning rate

**Action Space:**

- Per cell: Tilt angle adjustment (0-20 degrees)

**Reward Function:**

```python
reward = 1.0 * cco_score + 2.0 * load_balance_score
```

Where:

- `cco_score`: Coverage quality metric
- `load_balance_score`: Negative std dev of UE counts (higher is better)

**Output:**

- `TextMetricsLB`: List of cell configurations with `cell_id`, `el_degree`
- `tick`: Hour of day (0-23)
- `optimization_metric`: Weighted reward score

#### 3.1.3 MRO - Mobility Robustness Optimization

**Purpose:** Optimize handover parameters (hysteresis, time-to-trigger) to minimize handover failures, ping-pong handovers, and radio link failures.

**Algorithm:** Grid Search over parameter space

**Key Parameters:**

- `hyst_values`: Hysteresis values to test (dB)
- `ttt_values`: Time-to-trigger values to test (ms)
- `rlf_threshold`: Radio link failure SINR threshold (dB)

**Parameter Space:**

- `hyst`: 0-10 dB (typically 0.5 dB steps)
- `ttt`: 0-5120 ms (typically 40, 64, 80, 100, 128, 160, 256, 320, 480, 512, 640, 1024, 1280, 2560, 5120 ms)

**Evaluation Metric:**

```python
mro_metric = (
    successful_handovers
    - penalties_for_failures
    - penalties_for_ping_pong
    - penalties_for_rlf
)
```

**Output:**

- `TextMetricsMRO`: `hyst` (float), `ttt` (int)
- `optimization_metric`: MRO metric score (higher is better)

#### 3.1.4 CCO - Coverage Capacity Optimization

**Purpose:** Optimize antenna tilt angles to maximize coverage while minimizing interference and maintaining capacity.

**Algorithm:** Differential Gradient Policy-based Coverage Optimization (dGPCO)

**Key Parameters:**

- `num_epochs`: Optimization iterations (default: 500)
- `learning_rate`: Gradient step size (default: 0.1)
- `lambda_`: Regularization parameter (default: 0.01)
- `weak_coverage_threshold`: RSRP threshold for weak coverage (dBm, default: -95)
- `over_coverage_threshold`: RSRP threshold for over-coverage (dBm, default: -65)
- `epsilon`: Convergence tolerance (default: 1e-6)
- `opt_delta`: Tilt adjustment constraints (tuple, default: (2, 0))

**Objective Function:**

```python
objective = (
    weak_coverage_penalty
    + over_coverage_penalty
    - regularization_term
)
```

**CCO Inference Caching:**

- Training produces an `input_signature` hash (topology + config + UE data)
- If inference snapshot matches training signature → return cached results immediately
- If snapshot differs → apply trained tilts in zero-epoch replay (fast)
- If `allow_reoptimization=True` → run full dGPCO again (slow)

**Output:**

- `TextMetricsCCO`: List of cell configurations with `cell_id`, `el_degree`
- `optimization_metric`: Final CCO objective value (rounded)

### 3.2 Bayesian Digital Twin (BDT)

The Bayesian Digital Twin is a Gaussian Process-based RF propagation model that predicts received signal strength (RSRP, RSRQ, SINR) at any geographic location from a cell site.

**Purpose:**

- Predict RF metrics without running expensive ray-tracing simulations
- Enable what-if analysis for cell configuration changes (e.g., tilt adjustments)
- Provide attachment predictions for UE devices

**How It Works:**

1. **Training (done by BDT service):**

   - Collect RF measurements from the network or simulator
   - Train Gaussian Process Regression models per cell
   - Serialize trained models to pickle files

2. **Usage (in rApp service):**
   - Load BDT pickle file from S3
   - Given UE location and cell configuration, predict RF metrics
   - Used during RL training (ES, LB) to evaluate actions
   - Used during inference to validate predictions

**Model Structure:**

```python
bdt_model_map: Dict[str, BayesianDigitalTwin]
# Key: cell_id
# Value: Trained GP model
```

**Prediction API:**

```python
attachment_df = bayesian_digital_twin.predict_attachment(
    ue_data_df,        # DataFrame with ue_id, loc_x, loc_y
    cell_configs_df,   # DataFrame with cell_id, cell_el_deg, hTx, ...
    qos_sinr_threshold # Minimum SINR for attachment (dB)
)
# Returns: DataFrame with serving_cell_id, cell_rxpower_dbm, sinr_db
```

**Integration Points:**

- `radplib/dependencies/bayesian/bayesian_engine.py` - BDT prediction engine
- `radplib/dependencies/radp/digital_twin/rf/` - RF propagation utilities
- `radplib/es/rl/energy_env.py` - ES environment uses BDT for step() evaluation
- `radplib/lb/rl/cco_rl_env.py` - LB environment uses BDT for step() evaluation
- `radplib/cco/manager.py` - CCO uses BDT for coverage evaluation

### 3.3 Training vs Inference

#### 3.3.1 Training

**Purpose:** Learn optimal policies/parameters from historical data.

**Process:**

1. Load historical UE traffic data (multiple days)
2. Load network topology and cell configurations
3. Load BDT model for RF predictions
4. For RL (ES/LB):
   - Initialize Gymnasium environment
   - Train PPO agent for `total_timesteps` iterations
   - Save trained policy to ZIP file
5. For Optimization (MRO/CCO):
   - Run grid search or gradient-based optimization
   - Save optimal parameters or tilt configuration
6. Persist trained model to S3 and update database

**Output Artifacts:**

- ES/LB: `{rapp_model_id}.zip` (PPO policy network weights)
- MRO: `mro_result.json` (optimal hyst, ttt)
- CCO: `cco_result.json` (optimal tilts per cell)

**Training Metrics:**

```python
{
    "optimization_metric": int,     # Final score
    "training_duration_sec": float, # Time taken
    "num_epochs": int,              # Iterations
    "learning_rate": float,         # Hyperparameter
    # ... additional rApp-specific metrics
}
```

#### 3.3.2 Inference

**Purpose:** Generate real-time predictions using a trained model for a specific network snapshot.

**Process:**

1. Load trained model from S3
2. Load current network snapshot:
   - Topology (cell locations, azimuths)
   - Config (current tilt angles)
   - UE dataset (user locations, traffic patterns)
   - BDT model (RF prediction)
3. For RL (ES/LB):
   - Load PPO policy
   - Run single episode for specified tick
   - Generate cell configuration predictions
4. For Optimization (MRO):
   - Return trained hyst/ttt directly
5. For CCO:
   - Check input signature hash
   - If cached → return immediately
   - Else → run zero-epoch replay or full optimization
6. Calculate optimization metric
7. Generate plot data (UE positions, cell coverage)
8. Return structured result

**Output:**

```python
{
    "plot": PlotData,               # Visualization data
    "optimization_metric": int,     # Score
    "text": TextMetrics*,           # Actionable parameters
    "metrics": Dict[str, Any]       # Additional metadata
}
```

### 3.4 Optimization Metrics

Each rApp calculates an `optimization_metric` to quantify the quality of its predictions:

#### ES Optimization Metric

```python
# Source: app/services/utils/metrics_builder.py:calculate_optimization_metric_es()

# Calculate component scores
energy_saving_score = (1.0 - (num_active_cells / total_cells)) * 100  # 0-100
cco_score = CcoEngine.get_cco_objective_value(...)  # Network coverage utility
load_balance_score = -ue_counts.std()  # Negative of std dev (0 or negative)

# Apply weights and scale
reward = 0.2 * cco_score + 0.1 * load_balance_score + 1.0 * energy_saving_score
metric = int(round(reward * 100))
```

**Range:** Can be negative to large positive values (typically -1000 to +20000), higher is better

**Component Breakdown:**

- **energy_saving_score (0-100):** Percentage of cells turned OFF. Weight = 1.0

  - 100 = all cells OFF (maximum energy savings)
  - 0 = all cells ON (no energy savings)

- **cco_score (network coverage utility):** Coverage quality from `CcoEngine`. Weight = 0.2

  - Higher = better coverage (fewer weak/over-coverage areas)
  - Can range from negative (poor coverage) to positive hundreds/thousands (excellent coverage)

- **load_balance_score:** Negative of UE count standard deviation. Weight = 0.1
  - 0 = perfect balance (all cells serve equal UEs)
  - Negative = imbalanced load (some cells overloaded)

**Interpretation:**

The final metric heavily weights energy savings (1.0) over coverage (0.2) and load balance (0.1), encouraging aggressive cell shutoff while maintaining acceptable coverage.

#### LB Optimization Metric

```python
# Source: app/services/utils/metrics_builder.py:calculate_optimization_metric_lb()

# Calculate component scores
cco_score = CcoEngine.get_cco_objective_value(...)  # Network coverage utility
load_balance_score = -ue_counts.std()  # Negative of std dev (0 or negative)

# Apply weights and scale
reward = 1.0 * cco_score + 2.0 * load_balance_score
metric = int(round(reward * 100))
```

**Range:** Can be negative to large positive values (typically -2000 to +10000), higher is better

**Component Breakdown:**

- **cco_score (network coverage utility):** Coverage quality from `CcoEngine`. Weight = 1.0

  - Higher = better coverage (fewer weak/over-coverage areas)
  - Can range from negative (poor coverage) to positive hundreds/thousands (excellent coverage)

- **load_balance_score:** Negative of UE count standard deviation. Weight = 2.0
  - 0 = perfect balance (all cells serve equal UEs)
  - Negative = imbalanced load (some cells overloaded)
  - More heavily weighted than coverage (2.0 vs 1.0)

**Interpretation:**

The final metric prioritizes load balancing (weight 2.0) over coverage quality (weight 1.0). A higher score indicates better UE distribution across cells while maintaining good coverage.

**Example:**

```python
# If cco_score = 500 (good coverage) and load_balance_score = -5 (slight imbalance):
metric = int(round((1.0 * 500 + 2.0 * (-5)) * 100))
       = int(round((500 - 10) * 100))
       = 49000
```

#### MRO Optimization Metric

**Actually Used (in Training):**

```python
# Calculate total operational cellular time remaining after handover losses
# Source: app/radplib/mro/utils/utils.py:calculate_mro_metric()

ts = 0.050  # Handover interruption time (50ms)
t_nas = 1.000  # Radio Link Failure interruption time (1000ms)

T = total_ticks * 1  # Total operational time in seconds (1 second per tick)
ns_handover_count = count_of_successful_handovers(data)
nf_handover_count = count_of_radio_link_failures(data)

# Subtract interruption time from total time
metric = T - (ns_handover_count * ts + nf_handover_count * t_nas)
metric = int(round(metric))  # Rounded to integer seconds
```

**Range:** Can be negative to positive (in seconds), higher is better

**Calculation:**

- **T:** Total operational time based on number of ticks (1 tick = 1 second)
- **ns_handover_count:** Number of successful handovers (each causes 50ms interruption)
- **nf_handover_count:** Number of Radio Link Failures (each causes 1000ms interruption)
- **Result:** Effective operational time after accounting for all interruption delays

**Interpretation:**

- **Higher values:** Better mobility performance (fewer handovers and RLFs)
- **Lower values:** More interruptions due to frequent handovers or RLFs
- **Negative values:** Total interruption time exceeds operational time (severe mobility issues)

**Example:** If simulation runs for 100 ticks (100 seconds) with 20 successful handovers and 2 RLFs:

```
metric = 100 - (20 * 0.050 + 2 * 1.000) = 100 - 3 = 97 seconds
```

**Note on Unused Heuristic:**

The function `app/services/utils/metrics_builder.py:calculate_optimization_metric_mro()` contains an alternative heuristic formula that calculates a 0-100 score based on hysteresis and TTT parameters:

```python
# UNUSED - Placeholder for future enhancements
hyst_score = min(hyst * 5.0, 50.0)
ttt_score = max(0, 50.0 - (abs(ttt - 480) / 20.0))
metric = int(round(hyst_score + ttt_score))
```

This function is **not currently used** in training or inference. It serves as a placeholder for future scoring enhancements.

#### CCO Optimization Metric

```python
# Final objective value from optimization history
# Source: app/services/utils/metrics_builder.py:calculate_optimization_metric_cco()
objective_history = inference_output.get("cco_objective_per_epoch") or []
metric = int(round(objective_history[-1]))  # Last epoch's objective value
```

**Range:** Higher is better (maximizes network coverage utility)

**Calculation:**

The CCO objective represents the **network coverage utility** - a weighted combination of soft weak coverage and soft over-coverage scores:

```python
# From CcoEngine.get_cco_objective_value()
network_coverage_utility = (
    lambda_ * soft_weak_coverage + (1 - lambda_) * soft_over_coverage
)

# Where:
# soft_weak_coverage = 1000 * tanh(0.05 * growth_rate * (rsrp - weak_threshold))
# soft_over_coverage = 1000 * tanh(0.05 * growth_rate * (sinr - over_threshold))
```

**Interpretation:**

- **Higher values:** Better coverage quality (good signal strength, minimal weak/over-coverage areas)
- **Lower values:** Worse coverage quality (more areas with weak signal or over-coverage issues)
- **Negative values:** Significant coverage problems (many weak or over-covered areas)

The optimization process seeks to **maximize** this utility value by adjusting cell antenna tilt angles.

### 3.5 Tick-Based Processing

Many rApps operate on an hourly granularity with a **tick** system:

**Tick Definition:**

- Tick 0 = 00:00-01:00 (midnight to 1 AM)
- Tick 1 = 01:00-02:00
- ...
- Tick 23 = 23:00-24:00 (11 PM to midnight)

**Usage:**

- Training data is organized by tick (traffic patterns vary by hour)
- Inference requests specify a tick to predict for specific hour
- ES/LB actions can vary by tick (e.g., turn off cells at night)

**Example:**

```python
# Inference request for evening peak hour
{
    "baseline_id": "baseline-v1",
    "bdt_id": "BDT-001",
    "ue_dataset_id": "dataset-peak-traffic",
    "tick": 18  # 6 PM - 7 PM
}
```

---

## 4. API Reference

### 4.1 Authentication

All API endpoints (except root `/`) require authentication via API key:

**Header:**

```http
X-API-Key: <your_api_key>
```

**Configuration:**

```bash
export API_KEY=your_secret_key_here
```

**Future Enhancement:** JWT bearer tokens with tenant claims will be supported:

```http
Authorization: Bearer <jwt_token>
```

### 4.2 Base URL

```
http://localhost:8004/v1/tenants/{tenant_id}/rapps
```

**Path Parameters:**

- `tenant_id` (UUID, required) - Tenant identifier for multi-tenancy

### 4.3 Response Envelope

All successful responses use a standardized envelope format:

```json
{
    "success": true,
    "timestamp": "2024-10-17T12:00:00Z",
    "data": { ... },
    "message": "Optional success message",
    "errors": []
}
```

**Error Response:**

```json
{
  "success": false,
  "timestamp": "2024-10-17T12:00:00Z",
  "data": null,
  "message": "Error description",
  "errors": [
    {
      "type": "validation_error",
      "loc": ["body", "tick"],
      "msg": "tick must be between 0 and 23"
    }
  ]
}
```

### 4.4 API Endpoints Summary

| Endpoint                                                                        | Method | Description                         | Key Request Fields                                               |
| ------------------------------------------------------------------------------- | ------ | ----------------------------------- | ---------------------------------------------------------------- |
| `/v1/tenants/{tenant_id}/rapps`                                                 | GET    | List all available rApp types       | None (path: `tenant_id`)                                         |
| `/v1/tenants/{tenant_id}/rapps/{rapp_id}/train`                                 | POST   | Train a new rApp model (async)      | `rapp_model_id`, `bdt_id`, `baseline_id`, `dataset_id`, `params` |
| `/v1/tenants/{tenant_id}/rapps/{rapp_id}/models`                                | GET    | List trained models with pagination | Query: `limit`, `offset`, `status`                               |
| `/v1/tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}`                | GET    | Get specific model details          | None (path params only)                                          |
| `/v1/tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}`                | DELETE | Delete a trained model              | None (path params only)                                          |
| `/v1/tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}/infer`          | POST   | Start inference job (async)         | `baseline_id`, `bdt_id`, `ue_dataset_id`, `tick`                 |
| `/v1/tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}/infer/{run_id}` | GET    | Get inference status and results    | None (path params only)                                          |

### 4.5 Endpoint Details

#### 4.5.1 List Available rApps

**GET** `/v1/tenants/{tenant_id}/rapps`

Returns all available rApp types (ES, LB, MRO, CCO) with descriptions.

**Response:** `200 OK`

```json
{
  "data": [
    {
      "rapp_id": "es",
      "name": "Energy Saving",
      "description": "Optimizes cell on/off states and tilt angles"
    },
    {
      "rapp_id": "lb",
      "name": "Load Balancing",
      "description": "Adjusts antenna tilt to balance UE load"
    },
    {
      "rapp_id": "mro",
      "name": "Mobility Robustness Optimization",
      "description": "Optimizes handover parameters"
    },
    {
      "rapp_id": "cco",
      "name": "Coverage Capacity Optimization",
      "description": "Optimizes antenna tilt for coverage"
    }
  ]
}
```

---

#### 4.5.2 Train rApp Model

**POST** `/v1/tenants/{tenant_id}/rapps/{rapp_id}/train`

Initiates asynchronous training of a new rApp model. Returns `202 Accepted` immediately; training runs in background worker.

**Request Body:**

```json
{
  "rapp_model_id": "es-model-v1.0",
  "bdt_id": "BDT-001",
  "baseline_id": "baseline-fall-2024",
  "dataset_id": "dataset-training-week1",
  "params": {
    "total_timesteps": 3000, // for es|lb|cco
    "mro_type": "rl", // for mro only
    "n_epochs": 10 // for mro only
  }
}
```

**Key Fields:**

- `rapp_model_id` (string, required) - Unique identifier for trained model
- `bdt_id` (string, required) - Bayesian Digital Twin model ID (must exist in S3)
- `baseline_id` (string, required) - Baseline network configuration ID
- `dataset_id` (string, required) - Training dataset ID (UE traffic for ES/LB/CCO, UE mobility for MRO)
- `params` (object, required) - Training hyperparameters (rApp-specific)

**Training Parameters:**

**ES/LB/CCO:**

- `total_timesteps` (int, required) - Training steps (default: 30000 for ES, 24000 for LB)
- `train_days` (array, optional) - Days to train on (default: [0,1,2,3])
- `bdt_filename` (string, optional) - BDT model filename
- **CCO-specific:** `num_epochs`, `lambda_`, `weak_coverage_threshold`, `over_coverage_threshold`, `learning_rate`

**MRO:**

- `mro_type` (string, required) - Algorithm: "simple" or "rl" (default: "simple")
- `n_epochs` (int, required) - Training epochs (default: 100)
- `bdt_filename`, `agent_filename` (optional)

**Response:** `202 Accepted`

```json
{
  "data": {
    "rapp_model_id": "es-model-v1.0",
    "status": "queued",
    "created_at": "2024-10-17T12:00:00Z"
  }
}
```

**Status Flow:** `queued` → `training` → `ready` (or `failed`)

**Validation:**

- `rapp_id` must be valid enum: `es`, `lb`, `mro`, `cco`
- `tick` must be 0-23
- All IDs (bdt_id, baseline_id, dataset_id) must exist in S3
- Model artifacts stored in S3 and local filesystem

---

#### 4.5.3 List Trained Models

**GET** `/v1/tenants/{tenant_id}/rapps/{rapp_id}/models`

Returns paginated list of trained models for a specific rApp type.

**Query Parameters:**

- `limit` (int, optional) - Results per page (default: 20, max: 100)
- `offset` (int, optional) - Pagination offset (default: 0)
- `status` (string, optional) - Filter: `queued`, `training`, `ready`, `failed`

**Response:** `200 OK`

```json
{
  "data": {
    "items": [
      {
        "rapp_model_id": "es-model-v1.0",
        "rapp_id": "es",
        "baseline_id": "baseline-fall-2024",
        "status": "ready",
        "metrics": {
          "optimization_metric": 85,
          "training_duration_sec": 1234.56
        },
        "artifacts_uri": [
          "/app/var/models/.../es-model-v1.0.zip",
          "s3://bucket/.../es-model-v1.0.zip"
        ],
        "created_at": "2024-10-17T10:00:00Z",
        "updated_at": "2024-10-17T11:30:00Z"
      }
    ],
    "total": 1
  }
}
```

---

#### 4.5.4 Get Model Details

**GET** `/v1/tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}`

Retrieves detailed information about a specific trained model including configuration, metrics, and artifact locations.

**Response:** `200 OK`

```json
{
  "data": {
    "rapp_model_id": "es-model-v1.0",
    "status": "ready",
    "config": { "train_days": [0, 1, 2], "total_timesteps": 50000 },
    "metrics": { "optimization_metric": 85, "final_reward": 0.85 },
    "artifacts_uri": ["/app/var/models/.../es-model-v1.0.zip"]
  }
}
```

**Error:** `404 Not Found` - Model does not exist or belongs to another tenant

---

#### 4.5.5 Delete Trained Model

**DELETE** `/v1/tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}`

Deletes trained model and its artifacts from database and S3 storage.

**Response:** `204 No Content`

**Errors:**

- `404 Not Found` - Model does not exist
- `409 Conflict` - Model currently in use (active inference jobs)

---

#### 4.5.6 Start Inference

**POST** `/v1/tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}/infer`

Initiates asynchronous inference job using a trained model. Returns `202 Accepted` immediately; inference runs in background.

**Request Body:**

```json
{
  "baseline_id": "baseline-fall-2024",
  "bdt_id": "BDT-001",
  "ue_dataset_id": "dataset-test-day1",
  "tick": 12
}
```

**Key Fields:**

- `baseline_id` (string, required) - Baseline configuration (topology, cell configs)
- `bdt_id` (string, required) - Bayesian Digital Twin model ID
- `ue_dataset_id` (string, required) - UE dataset for inference
- `tick` (int, required) - Hour of day (0-23)

**Response:** `202 Accepted`

```json
{
  "data": {
    "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "rapp_model_id": "es-model-v1.0",
    "status": "queued",
    "created_at": "2024-10-17T12:00:00Z"
  }
}
```

**Response Header:**

```http
Location: /v1/tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}/infer/{run_id}
```

**Status Flow:** `queued` → `running` → `completed` (or `failed`)

---

#### 4.5.7 Get Inference Status and Results

**GET** `/v1/tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}/infer/{run_id}`

Retrieves status and results of an inference job. Poll this endpoint until status is `completed` or `failed`.

**Response (running):** `200 OK`

```json
{
  "data": {
    "run_id": "a1b2c3d4...",
    "status": "running",
    "result": null,
    "error": null
  }
}
```

**Response (completed):** `200 OK`

```json
{
  "data": {
    "run_id": "a1b2c3d4...",
    "status": "completed",
    "result": {
      "plot": {
        "groups": [
          { "title": "Cell 1 Coverage", "data": [{ "x": 100.5, "y": 200.3 }] }
        ]
      },
      "optimization_metric": 85,
      "text": {
        "tick": "12",
        "items": [
          { "cell_id": "cell-001", "el_degree": 10.0, "on_off": true },
          { "cell_id": "cell-002", "el_degree": 12.0, "on_off": false }
        ]
      },
      "metrics": {
        "cco_score": 92.5,
        "load_balance_score": -5.3,
        "energy_saving_score": 65.0
      }
    }
  }
}
```

**Result Structure:**

- `plot` - Visualization data (groups with x/y coordinates)
- `optimization_metric` - Overall optimization score
- `text.items` - Cell-specific recommendations (tilt angles, on/off states)
- `metrics` - Additional performance metrics (CCO score, load balance, energy saving)

---

## 5. Data Models

This section summarizes the key data models used in the rApp Engine. See source code for complete definitions.

### 5.1 Pydantic Models (app/models/rapp.py)

Pydantic models define the request/response schema for the API with automatic validation.

#### 5.1.1 Core Enums

- **RAppID:** `mro`, `cco`, `es`, `lb`
- **TrainStatus:** `queued`, `training`, `ready`, `failed`
- **InferenceStatus:** `queued`, `running`, `completed`, `failed`

#### 5.1.2 Request Models

**RAppTrainRequest** - Training job submission

- `rapp_model_id`: Client-supplied model identifier (string)
- `bdt_id`, `baseline_id`, `dataset_id`: Resource references (strings)
- `params`: Dict of rApp-specific hyperparameters (optional)

**InferenceRequest** - Inference job submission

- `baseline_id`, `bdt_id`, `ue_dataset_id`: Resource references (strings)
- `tick`: Hour of day (integer, 0-23)

#### 5.1.3 Response Models

**RAppModelSummary** - Model metadata

- Identifiers: `rapp_model_id`, `rapp_id`, `baseline_id`
- Status: `status` (TrainStatus enum)
- Results: `metrics` (JSON), `artifacts_uri` (list of URIs)
- Timestamps: `created_at`, `updated_at`

**InferenceResponse** - Inference job status and results

- Identifiers: `run_id` (UUID), `rapp_model_id`, `rapp_id`
- Status: `status` (InferenceStatus enum)
- Result: `result` (InferenceResult object, present when status=`completed`)
- Error: `error` (string, present when status=`failed`)

#### 5.1.4 Result Models

**InferenceResult** - Complete inference output

- `plot`: Visualization data (PlotData with groups of x/y points)
- `optimization_metric`: Integer score (higher is better)
- `text`: rApp-specific configuration parameters (TextMetrics\*)
- `metrics`: Additional metadata (optional dict)

**TextMetrics** - rApp-specific actionable outputs

- **MRO:** `hyst` (hysteresis in dB), `ttt` (time-to-trigger in ms)
- **CCO/LB:** List of `{cell_id, el_degree}` tilt configurations
- **ES:** List of `{cell_id, el_degree, on_off}` cell states

### 5.2 Database Schemas (PostgreSQL)

All tables use Row-Level Security (RLS) policies to enforce tenant isolation. See `db/migrations/` directory for complete schema definitions.

#### 5.2.1 Core Tables

**rapp_models** - Trained model metadata

- Primary key: `id` (UUID)
- Identifiers: `tenant_id`, `rapp_model_id` (unique per tenant), `rapp_id`, `baseline_id`
- Status: `status` (train_status enum)
- Data: `config` (JSONB training params), `metrics` (JSONB results), `artifacts_uri` (text array)
- Timestamps: `created_at`, `updated_at`
- Index: `(tenant_id, rapp_id, status)`

**inference_runs** - Inference job records

- Primary key: `id` (UUID, also used as `run_id`)
- Identifiers: `tenant_id`, `rapp_id`, `rapp_model_id`, `baseline_id`
- Request: `request` (JSONB), `ue_dataset_id`, `tick`
- Result: `result` (JSONB), `result_uri` (text array), `status`, `error`
- Timestamps: `created_at`, `updated_at`
- Index: `(tenant_id, created_at DESC)`

**training_jobs** - Shared training tracking (optional)

- Cross-service table for BDT, rApp, and Data Sim training
- Key fields: `tenant_id`, `kind` (service type), `model_ref`, `status`, `idempotency_key`
- Supports idempotent training job submission

#### 5.2.2 Row-Level Security

All tables enforce RLS using the session variable `app.current_tenant`:

```sql
CREATE POLICY <table>_rls ON <table>
USING (tenant_id = current_setting('app.current_tenant')::uuid);
```

This ensures queries automatically filter by the current tenant. Set the session variable before every query:

```sql
SET app.current_tenant = '<tenant-uuid>';
```

---

## 6. Event-Driven Architecture

### 6.1 Kafka Integration

The rApp Engine uses Apache Kafka for asynchronous processing of **training** requests. (Note: Inference uses a thread pool executor instead - see Section 6.3 for details.)

#### 6.1.1 Topics

**maveric.rapp.train.v1**

Training events published by API, consumed by workers.

**Message Format:**

```json
{
  "tenant_id": "11111111-1111-1111-1111-111111111111",
  "rapp_id": "es",
  "rapp_model_id": "es-model-v1.0",
  "bdt_id": "BDT-001",
  "baseline_id": "baseline-fall-2024",
  "dataset_id": "dataset-training-week1",
  "params": {
    "total_timesteps": 50000
  },
  "created_at": "2024-10-17T12:00:00Z"
}
```

**Note on Inference:**

Inference does **not** use Kafka. Instead, the API server handles inference requests using a **thread pool executor** (`InferenceJobRunner`). This provides faster response times for inference operations which are typically shorter-lived than training jobs.

The inference flow uses:

- `app/services/inference_runner.py` - Thread pool executor with configurable workers
- Direct database updates for status tracking
- No message queue overhead

See Section 7.2.3 for details on the thread pool implementation.

#### 6.1.2 Producer (app/event_handlers/kafka_handler.py)

**Configuration:**

```python
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
```

**Publishing Events:**

```python
def publish_training_event(tenant_id, rapp_id, payload):
    topic = "maveric.rapp.train.v1"
    event = {
        "tenant_id": str(tenant_id),
        "rapp_id": rapp_id,
        **payload,
        "created_at": datetime.utcnow().isoformat()
    }
    future = producer.send(topic, value=event)
    future.get(timeout=10)  # Block until sent
    logger.info(f"Published training event for {rapp_id} model {payload['rapp_model_id']}")
```

#### 6.1.3 Consumer (app/workers/rapp_worker.py)

**Configuration:**

```python
consumer = KafkaConsumer(
    "maveric.rapp.train.v1",
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    group_id="rapp-worker-group",
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True
)
```

**Message Processing:**

```python
for message in consumer:
    topic = message.topic
    event = message.value

    try:
        if topic == "maveric.rapp.train.v1":
            handle_training_event(event)
    except Exception as exc:
        logger.exception(f"Error processing {topic} event: {exc}")
        # Update status to 'failed' with error message
```

### 6.2 Worker Architecture

The rApp worker is a background process that consumes Kafka events and executes **training** jobs. Training jobs are distributed via Kafka for scalability and fault tolerance.

**Key Components:**

- **Kafka Consumer:** Subscribes to `maveric.rapp.train.v1` topic
- **Thread Pool:** Executes training jobs concurrently (default: 4 workers)
- **Job Handlers:** Load data from S3, invoke radplib training functions, upload artifacts, update database

**Training Job Flow:**

1. Consume event from Kafka
2. Update `rapp_models.status` to `training`
3. Download BDT model, dataset, and baseline from S3
4. Invoke rApp-specific training function (radplib)
5. Upload trained model to S3
6. Update database with status=`ready`, metrics, and artifacts
7. On error: Update status=`failed` with error message

**Note on Inference:** Unlike training, inference requests are handled by a thread pool executor within the API server (see Section 6.3), not by this Kafka worker, for lower latency.

For detailed implementation including code examples and configuration, see **Section 7.5 (Workers)**.

### 6.3 Inference Thread Pool Executor

Unlike training which uses Kafka for job distribution, inference requests are handled by a **ThreadPoolExecutor** within the API server process for faster response times.

**Implementation** (app/services/inference_runner.py):

```python
class InferenceJobRunner:
    """Execute inference jobs off the request thread."""

    def __init__(self, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="rapp-infer"
        )
```

**Key Characteristics:**

- **4 worker threads** by default (configurable)
- **In-process execution** - no external worker processes needed
- **Immediate submission** - job submitted to queue synchronously, runs async
- **Future-based** - uses Python's `concurrent.futures` for job tracking
- **Database-backed status** - updates `inference_runs` table directly

**Why Thread Pool Instead of Kafka?**

1. **Lower latency** - No message broker overhead
2. **Simpler deployment** - No separate worker process required
3. **Faster execution** - Inference typically completes in seconds, not minutes
4. **Resource efficiency** - Shares API server resources

**Job Flow:**

1. API endpoint receives POST request
2. Creates `inference_runs` record (status=`queued`)
3. Submits `InferenceJob` to thread pool: `INFERENCE_JOB_RUNNER.submit(job)`
4. Returns `202 Accepted` immediately
5. Background thread picks up job from queue
6. Updates status to `running`, executes inference, updates status to `completed`/`failed`

### 6.4 Error Handling and Retries

**Kafka Consumer:**

- Auto-commit enabled: Messages are committed after successful processing
- On exception: Status updated to `failed`, message acknowledged to avoid reprocessing

**S3 Operations:**

- Boto3 built-in retries (exponential backoff)
- Custom retry logic for transient errors (5xx responses)
- IAM role assumption with automatic credential refresh

**Database Operations:**

- Connection pooling (SQLAlchemy)
- Transaction rollback on error
- RLS enforcement on every query

**Worker Crash Recovery:**

- Kafka consumer group rebalancing
- Unacknowledged messages re-consumed by other workers
- Jobs stuck in `training` or `running` status can be detected via monitoring

---

## 7. User Flows

### 7.1 Training Flow (End-to-End)

**Actors:** Client, API, Kafka, Worker, RadPLib, S3, PostgreSQL

**Sequence:**

1. **Client** sends `POST /tenants/{tenant_id}/rapps/es/train`:

   ```json
   {
     "rapp_model_id": "es-model-v1.0",
     "bdt_id": "BDT-001",
     "baseline_id": "baseline-fall-2024",
     "dataset_id": "dataset-training-week1",
     "params": { "total_timesteps": 50000 }
   }
   ```

   _Note: For ES/LB/CCO, typically only `total_timesteps` is passed. For MRO, pass `mro_type` and `n_epochs`. See Section 4.4.2 for all available parameters._

2. **API** (rapps.py:train_rapp_model):

   - Validates request (Pydantic)
   - Sets current tenant (RLS)
   - Inserts record into `rapp_models` (status=`queued`)
   - Publishes event to `maveric.rapp.train.v1` Kafka topic
   - Returns `202 Accepted` with `rapp_model_id`

3. **Kafka** delivers event to `rapp-worker-group`

4. **Worker** (rapp_worker.py):

   - Consumes event
   - Spawns thread to handle training job
   - Updates `rapp_models.status` to `training`

5. **Worker Thread**:

   - Downloads BDT pickle from S3: `tenants/{tenant_id}/models/bdt/BDT-001.pickle`
   - Downloads dataset from S3: `tenants/{tenant_id}/datasets/dataset-training-week1.csv`
   - Downloads topology/config from S3: `tenants/{tenant_id}/baselines/baseline-fall-2024/*.csv`

6. **RadPLib** (es/manager.py):

   - Loads BDT model
   - Preprocesses UE data
   - Initializes TickAwareEnergyEnv
   - Creates PPO agent (stable-baselines3)
   - Trains for 50,000 timesteps (may take 30-60 minutes)
   - Saves trained policy to `/app/var/models/{tenant_id}/rapps/es/es-model-v1.0.zip`

7. **Worker**:

   - Uploads model to S3: `tenants/{tenant_id}/models/rapps/es/es-model-v1.0.zip`
   - Calculates training metrics (duration, final reward, etc.)
   - Updates `rapp_models`:
     - `status` = `ready`
     - `metrics` = `{"optimization_metric": 85, "training_duration_sec": 1800}`
     - `artifacts_uri` = [local_path, s3_uri]

8. **Client** polls `GET /tenants/{tenant_id}/rapps/es/models/es-model-v1.0` every 10 seconds:
   - Initially: `{"status": "queued", ...}`
   - Then: `{"status": "training", ...}`
   - Finally: `{"status": "ready", "metrics": {...}, "artifacts_uri": [...]}`

**Total Time:** Typically 30-120 minutes depending on rApp type and dataset size.

### 7.2 Inference Flow (End-to-End)

**Actors:** Client, API, Kafka, Worker, RadPLib, S3, PostgreSQL

**Sequence:**

1. **Client** sends `POST /tenants/{tenant_id}/rapps/es/models/es-model-v1.0/infer`:

   ```json
   {
     "baseline_id": "baseline-fall-2024",
     "bdt_id": "BDT-001",
     "ue_dataset_id": "dataset-test-day1",
     "tick": 12
   }
   ```

2. **API** (rapps.py:infer_model):

   - Validates request
   - Generates `run_id` (UUID)
   - Inserts record into `inference_runs` (status=`queued`)
   - Submits job to thread pool executor (`INFERENCE_JOB_RUNNER`)
   - Returns `202 Accepted` with `run_id` and `Location` header

3. **Thread Pool Worker** (inference_runner.py):

   - Picks up job from thread pool queue
   - Updates `inference_runs.status` to `running`
   - Downloads artifacts:
     - Trained model (or uses cached)
     - BDT model (or uses cached)
     - UE dataset
     - Topology/config

4. **RadPLib** (services/real_inference.py):

   - Loads ES PPO policy
   - Creates environment with tick=12 UE data
   - Runs single episode (inference)
   - Predicts cell ON/OFF states and tilt angles
   - Calculates coverage using BDT
   - Computes optimization metric

5. **Worker**:

   - Generates plot data (UE positions colored by serving cell)
   - Builds InferenceResult:
     ```json
     {
         "plot": {...},
         "optimization_metric": 85,
         "text": {"tick": "12", "items": [{"cell_id": "cell-001", "el_degree": 10.0, "on_off": true}, ...]},
         "metrics": {"cco_score": 92.5, "load_balance_score": -5.3, "energy_saving_score": 65.0}
     }
     ```
   - Updates `inference_runs` with result, status=`completed`

6. **Client** polls `GET /tenants/{tenant_id}/rapps/es/models/es-model-v1.0/infer/{run_id}` every 5 seconds:
   - Initially: `{"status": "running", "result": null}`
   - Finally: `{"status": "completed", "result": {...}}`

**Total Time:** Typically 10-60 seconds depending on dataset size and rApp complexity.

### 7.3 Data Upload Flow

**S3 Bucket Structure:**

```
s3://{bucket}/tenants/
└── {tenant_id}/
    ├── baselines/
    │   └── {baseline_id}/
    │       ├── topology.csv
    │       └── config.csv
    ├── datasets/
    │   └── {dataset_id}/
    │       └── synthetic_dataset.csv or synthetic_dataset_mobility.csv
    └── models/
        ├── bdt/
        │   └── {bdt_id}.pickle
        └── rapps/
            └── {rapp_id}/
                └── {rapp_model_id}.zip or {rapp_id}_result.json
```

**Upload Methods:**

1. API endpoint
2. Direct S3 upload (via AWS CLI, SDK)
3. Presigned URL upload (via Portal/UI)

### 7.4 Model Management Flow

**Listing Models:**

```bash
GET /tenants/{tenant_id}/rapps/es/models?status=ready&limit=20
```

Returns paginated list of models with metadata.

**Deleting Models:**

```bash
DELETE /tenants/{tenant_id}/rapps/es/models/es-model-v1.0
```

- Soft delete: Mark status as `deleted` (retain metadata)
- Hard delete: Remove from database and S3

**Model Versioning:**

- Client controls `rapp_model_id` (e.g., `es-model-v1.0`, `es-model-v1.1`)
- No automatic versioning; client responsible for unique IDs

---

## 8. Configuration & Deployment

### 8.1 Environment Variables

**Required Variables:**

| Variable                       | Description                  | Example                                         |
| ------------------------------ | ---------------------------- | ----------------------------------------------- |
| `DATABASE_URL`                 | PostgreSQL connection string | `postgresql://user:pass@localhost:5432/rapp_db` |
| `KAFKA_BOOTSTRAP_SERVERS`      | Kafka broker addresses       | `localhost:9092`                                |
| `S3_BUCKET` or `S3_BUCKET_ARN` | S3 bucket for artifacts      | `arn:aws:s3:::my-bucket`                        |
| `API_KEY`                      | API authentication key       | `your-secret-key`                               |

**Optional Variables:**

| Variable                         | Description                          | Default               |
| -------------------------------- | ------------------------------------ | --------------------- |
| `APP_NAME`                       | Application name                     | `rapp-engine`         |
| `APP_ENV`                        | Environment                          | `development`         |
| `LOG_LEVEL`                      | Logging level                        | `INFO`                |
| `MONGODB_URL`                    | MongoDB connection (optional)        | None                  |
| `S3_PREFIX`                      | S3 key prefix (typically `tenants/`) | `""` (empty)          |
| `S3_REGION`                      | AWS region                           | `us-east-1`           |
| `S3_ENDPOINT_URL`                | MinIO endpoint                       | None                  |
| `S3_ASSUME_ROLE_ARN`             | IAM role to assume                   | None                  |
| `S3_ASSUME_ROLE_EXTERNAL_ID`     | External ID for role                 | None                  |
| `S3_ASSUME_ROLE_SESSION_NAME`    | Session name                         | `maveric-rapp-worker` |
| `RAPP_WORKER_MODEL_BASE_DIR`     | Model cache directory                | `/app/var/models`     |
| `RAPP_WORKER_LOCAL_DATASET_ROOT` | Dataset cache directory              | `/app/var/datasets`   |

### 8.2 Docker & Container Image

The rApp Engine uses a single Dockerfile to build container images that run as either API Server or Worker.

#### 8.2.1 Dockerfile

**Base:** Python 3.11 slim variant (~150MB)

**Key Features:**

- Non-root user (appuser) for security
- System dependencies: gcc, g++, libpq-dev, curl
- Python dependencies: FastAPI, SQLAlchemy, Kafka, boto3, pandas, PyTorch, stable-baselines3
- Health check on port 8001 (30s interval)
- Default command: `uvicorn app.main:app` (API server)

**Build:**

```bash
docker build -t maveric/rapp-engine:1.0.0 .
```

#### 8.2.2 Running Containers

**API Server:**

```bash
docker run -d --name rapp -p 8004:8001 \
  -e DATABASE_URL="postgresql://..." \
  -e KAFKA_BOOTSTRAP_SERVERS="kafka:9092" \
  -e S3_ENDPOINT_URL="http://minio:9000" \
  -e API_KEY="secret-key" \
  maveric/rapp-engine:1.0.0
```

**Worker:**

```bash
docker run -d --name rapp-worker \
  -e DATABASE_URL="postgresql://..." \
  -e KAFKA_BOOTSTRAP_SERVERS="kafka:9092" \
  -e S3_ENDPOINT_URL="http://minio:9000" \
  -v /data/models:/app/var/models \
  maveric/rapp-engine:1.0.0 \
  python -m app.workers.rapp_worker
```

**API vs Worker:**

| Aspect      | API Server               | Worker                              |
| ----------- | ------------------------ | ----------------------------------- |
| Command     | `uvicorn app.main:app`   | `python -m app.workers.rapp_worker` |
| Purpose     | HTTP API requests        | Training/inference jobs             |
| Ports       | 8001 (HTTP)              | None                                |
| Resources   | CPU: 1-2, Memory: 2-4 GB | CPU: 4-8, Memory: 16-32 GB          |
| Volumes     | Not required             | Model cache (`/app/var/models`)     |
| Workload    | Short-lived (ms-seconds) | Long-running (minutes-hours)        |
| GPU Support | No                       | Optional (for faster training)      |

#### 8.2.3 Docker Compose for Local Development

See `docker-compose.yml` in the repository for a complete local development setup including:

- PostgreSQL 14
- Kafka + Zookeeper
- MinIO (S3-compatible storage)
- rApp API server
- rApp worker

**Quick Start:**

```bash
docker-compose up -d              # Start all services
docker-compose logs -f rapp       # View API logs
docker-compose logs -f rapp-worker # View worker logs
docker-compose down               # Stop all services
```

#### 8.2.4 Optimization Tips

- Use multi-stage builds to reduce image size (~200MB savings)
- Copy `requirements.txt` before application code for better layer caching
- Use `.dockerignore` to exclude unnecessary files (tests, docs, .git)

### 8.3 Database Setup

**PostgreSQL Initialization:**

1. Create database:

   ```sql
   CREATE DATABASE rapp_db;
   ```

2. Run migrations (in order):

   ```bash
   psql -U user -d rapp_db -f db/migrations/00_enums.sql
   psql -U user -d rapp_db -f db/migrations/00_shared_training_jobs.sql
   psql -U user -d rapp_db -f db/migrations/40_rapp_models.sql
   psql -U user -d rapp_db -f db/migrations/41_inference_runs.sql
   ```

3. Verify RLS policies:
   ```sql
   SELECT tablename, policyname FROM pg_policies WHERE schemaname = 'public';
   ```

**MongoDB Setup (Optional):**

No schema required; collections created dynamically.

### 8.3 S3 Configuration

**IAM Role for Kubernetes Deployment:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::my-bucket", "arn:aws:s3:::my-bucket/*"]
    }
  ]
}
```

**MinIO for Local Development:**

```bash
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

**Environment:**

```bash
export S3_ENDPOINT_URL=http://localhost:9000
export S3_BUCKET=rapp-artifacts
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
```

### 8.4 Kafka Setup

> **Note:** Kafka and Zookeeper services are included in the complete docker-compose.yml in Section 9.2.4.

**Create Topics:**

```bash
# Only training topic is needed - inference uses thread pool, not Kafka
kafka-topics --create --topic maveric.rapp.train.v1 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

---

## 9. Development Guide

### 9.1 Local Setup

**Prerequisites:**

- Python 3.10+
- PostgreSQL 14+
- Kafka 3.0+
- MinIO (or AWS S3 access)

**Steps:**

1. Clone repository:

   ```bash
   git clone https://github.com/maveric/maveric_platform_rapp.git
   cd maveric_platform_rapp
   ```

2. Create virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up `.env` file:

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Run database migrations:

   ```bash
   psql -U user -d rapp_db -f db/migrations/00_enums.sql
   # ... repeat for other migrations
   ```

6. Start API server:

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8004
   ```

7. Start worker (separate terminal):
   ```bash
   python -m app.workers.rapp_worker
   ```

### 9.2 Running Tests

```bash
pytest tests/ -v --cov=app --cov-report=html
```

**Test Structure:**

- `tests/unit/` - Unit tests for individual functions
- `tests/integration/` - Integration tests (API + DB + Kafka)
- `tests/e2e/` - End-to-end scenarios

### 9.3 Debugging

**Enable Debug Logging:**

```bash
export LOG_LEVEL=DEBUG
```

**View Kafka Messages:**

```bash
kafka-console-consumer --bootstrap-server localhost:9092 --topic maveric.rapp.train.v1 --from-beginning
```

**Check PostgreSQL RLS:**

```sql
SET app.current_tenant = '11111111-1111-1111-1111-111111111111';
SELECT * FROM rapp_models;  -- Should only show tenant's models
```

**Common Issues:**

1. **S3 Access Denied:**

   - Check IAM role permissions
   - Verify `S3_BUCKET_ARN` is correct
   - Ensure role trust policy allows EKS service account

2. **Kafka Connection Refused:**

   - Verify `KAFKA_BOOTSTRAP_SERVERS`
   - Check if Kafka is running: `docker ps | grep kafka`

3. **Training Stuck in "queued":**

   - Check worker logs: `docker logs rapp-worker`
   - Verify worker is consuming from correct topics

4. **Database Permission Denied:**
   - Ensure RLS session variable is set
   - Check user has SELECT/INSERT/UPDATE permissions

### 9.4 Code Organization Principles

**Separation of Concerns:**

- **API Layer**: Request validation, authentication, response formatting
- **Services Layer**: Business logic, orchestration
- **Workers Layer**: Async job processing
- **RadPLib**: Core ML/optimization algorithms (reusable, testable)
- **Database Layer**: CRUD operations, RLS enforcement

**Dependency Injection:**

- Use FastAPI's `Depends()` for database sessions, configuration
- Avoid global state where possible

**Error Handling:**

- Use specific exception types
- Log with context (tenant_id, rapp_id, model_id)
- Return user-friendly error messages in API responses

**Testing:**

- Unit tests: Pure functions in radplib, services
- Integration tests: API endpoints with test database
- E2E tests: Full workflows with Docker Compose

---

## 10. Appendix

### 10.1 Glossary

**Terms:**

- **BDT (Bayesian Digital Twin):** Gaussian Process-based RF propagation model for predicting signal strength
- **CCO (Coverage Capacity Optimization):** rApp for optimizing cell coverage via antenna tilt adjustment
- **Cell El Deg (Elevation Degree):** Antenna tilt angle, typically 0-20 degrees
- **dGPCO:** Differential Gradient Policy-based Coverage Optimization algorithm
- **ES (Energy Saving):** rApp for minimizing energy consumption by cell on/off and tilt optimization
- **Hysteresis (Hyst):** Handover parameter that prevents ping-pong handovers (dB)
- **LB (Load Balancing):** rApp for distributing UE load evenly across cells
- **MRO (Mobility Robustness Optimization):** rApp for optimizing handover parameters
- **Near-RT RIC:** Near-Real-Time RAN Intelligent Controller (O-RAN architecture component)
- **O-RAN:** Open Radio Access Network (industry standard for interoperable RAN)
- **PPO (Proximal Policy Optimization):** Reinforcement learning algorithm used for ES and LB
- **rApp (Non Real Time Application):** Application running on Near-RT RIC for RAN optimization
- **RAN (Radio Access Network):** Cellular network infrastructure connecting UEs to core network
- **RLS (Row-Level Security):** PostgreSQL feature for tenant data isolation
- **RSRP (Reference Signal Received Power):** Signal strength metric (dBm)
- **SINR (Signal-to-Interference-plus-Noise Ratio):** Signal quality metric (dB)
- **SMO (Service Management and Orchestration):** O-RAN component for network management
- **Tick:** Hour of day (0-23), used for time-aware predictions
- **TTT (Time-to-Trigger):** Handover parameter specifying delay before triggering handover (ms)
- **UE (User Equipment):** Mobile device (smartphone, IoT device, etc.)
- **xApp:** RAN application running on Non-RT RIC (longer timescale than rApp)

### 10.2 Troubleshooting

**Problem:** Training fails with "BDT model not found"

**Solution:**

- Verify BDT model exists in S3: `aws s3 ls s3://bucket/tenants/{tenant_id}/models/bdt/`
- Check `bdt_id` matches exactly (case-sensitive)
- Ensure worker has S3 read permissions

---

**Problem:** Inference returns poor optimization metric

**Solution:**

- Check if UE dataset matches training data distribution
- Verify BDT model is trained on same network topology
- Review training metrics to ensure model converged

---

**Problem:** Worker not consuming Kafka messages

**Solution:**

- Check consumer group status: `kafka-consumer-groups --bootstrap-server localhost:9092 --group rapp-worker-group --describe`
- Verify topics exist: `kafka-topics --list --bootstrap-server localhost:9092`
- Check worker logs for connection errors

---

**Problem:** Database connection pool exhausted

**Solution:**

- Increase pool size in `sql_handler.py`: `pool_size=50`
- Check for connection leaks (missing `db.close()`)
- Monitor active connections: `SELECT count(*) FROM pg_stat_activity;`

---

**Problem:** S3 uploads slow or timing out

**Solution:**

- Use multipart upload for large files
- Check network bandwidth to S3 region
- Consider using EFS for local caching in Kubernetes

---

**Problem:** CCO inference returns cached results when it shouldn't

**Solution:**

- Disable caching by passing `allow_reoptimization=True`
- Clear cache: Delete trained model and retrain
- Verify input data has actually changed (check signature hash)

---

**End of Documentation**

For questions or support, please contact the Maveric platform team or open an issue in the repository.
