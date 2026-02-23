# Maveric Platform Development Repository Documentation

## Table of Contents
1. [Platform Overview](#platform-overview)
2. [Architecture](#architecture)
3. [Submodule Documentation](#submodule-documentation)
4. [Database Schema](#database-schema)
5. [API Endpoints](#api-endpoints)
6. [Authentication & Authorization](#authentication--authorization)
7. [User Flows](#user-flows)
8. [Development Setup](#development-setup)

---

## Platform Overview

The Maveric Platform is a **multi-tenant SaaS platform** designed for RIC (RAN Intelligent Controller) Algorithm Development. It provides a comprehensive suite of tools for network topology generation, UE (User Equipment) data simulation, Bayesian Digital Twin (BDT) modeling, and Radio Application (rApp) training and inference.

### Key Features
- **Multi-tenant Architecture**: Complete tenant isolation with Row Level Security (RLS)
- **Microservices Design**: Modular services orchestrated through a unified Gateway
- **Digital Twin Modeling**: Bayesian Digital Twin (BDT) for network behavior prediction
- **Radio Applications**: Support for MRO (Mobility Robustness Optimization), CCO (Coverage and Capacity Optimization), ES (Energy Saving), and LB (Load Balancing)
- **Synthetic Data Generation**: Topology and UE traffic/mobility simulation utilities
- **Async Job Processing**: Training and inference jobs with status tracking
- **Cloud-Native Storage**: S3/MinIO for artifact management
- **RESTful API**: OpenAPI 3.0.3 compliant with JWT authentication

### Technology Stack
- **Backend Framework**: FastAPI (Python)
- **Database**: PostgreSQL with GORM auto-migrations and Row Level Security
- **Authentication**: Supabase GoTrue (JWT tokens)
- **Object Storage**: S3/MinIO
- **Message Queue**: Kafka (for async job orchestration)
- **Containerization**: Docker & Docker Compose
- **API Specification**: OpenAPI 3.0.3

---

## Architecture

### System Components

The platform consists of the following microservices (each maintained in separate submodules):

1. **Gateway** ([maveric_platform_gateway](../submodule/maveric_platform_gateway/))
   - Central orchestrator and API gateway
   - Handles tenant management and authentication
   - Manages unified database schema migrations
   - Routes requests to specialized services

2. **Data Simulator** ([maveric_platform_data_sim](../submodule/maveric_platform_data_sim/))
   - Generates synthetic network topologies
   - Creates baseline configurations

3. **SMO Simulator** ([maveric_platform_smo_sim](../submodule/maveric_platform_smo_sim/))
   - Manages baseline network configurations
   - Generates simulated UE traffic and mobility data
   - Handles UE dataset management

4. **BDT Engine** ([maveric_platform_bdt_engine](../submodule/maveric_platform_bdt_engine/))
   - Trains Bayesian Digital Twin models
   - Provides probabilistic network behavior predictions
   - Manages BDT model artifacts and metrics

5. **rApp Engine** ([maveric_platform_rapp](../submodule/maveric_platform_rapp/))
   - Trains specialized Radio Applications (MRO, CCO, ES, LB)
   - Executes inference runs using trained models
   - Manages rApp model artifacts and configurations

### Infrastructure Services
- **PostgreSQL**: Primary relational database
- **Redis**: Caching layer
- **MongoDB**: Document storage (for specific workloads)
- **Kafka + Zookeeper**: Message queue for async job processing
- **MinIO**: S3-compatible object storage for artifacts
- **pgAdmin**: Database administration interface

### Network Architecture
All services communicate over a shared Docker network named `maveric`. Services are isolated but can communicate via service names as DNS aliases.

### Data Flow

```
User Request
    ↓
[JWT Authentication via Supabase]
    ↓
Gateway (Tenant Context Set)
    ↓
[Route to Specialized Service]
    ↓
Service Logic + Database Access (RLS enforced)
    ↓
[S3 Storage for Artifacts]
    ↓
[Kafka Job Queue for Async Tasks]
    ↓
Worker Processes Training/Inference
    ↓
Response to User
```

---

## Submodule Documentation

This repository orchestrates multiple microservices maintained as Git submodules. Each submodule has its own detailed documentation:

- **Gateway**: [submodule/maveric_platform_gateway/doc/gateway.md](../submodule/maveric_platform_gateway/doc/gateway.md)
- **Data Simulator**: [submodule/maveric_platform_data_sim/doc/data_sim.md](../submodule/maveric_platform_data_sim/doc/data_sim.md)
- **SMO Simulator**: [submodule/maveric_platform_smo_sim/doc/smo_sim.md](../submodule/maveric_platform_smo_sim/doc/smo_sim.md)
- **BDT Engine**: [submodule/maveric_platform_bdt_engine/doc/bdt_engine.md](../submodule/maveric_platform_bdt_engine/doc/bdt_engine.md)
- **rApp Engine**: [submodule/maveric_platform_rapp/doc/rapp.md](../submodule/maveric_platform_rapp/doc/rapp.md)

Refer to these documents for service-specific implementation details, class definitions, and function parameters.

---

## Database Schema

The platform uses PostgreSQL with Row Level Security (RLS) for multi-tenant data isolation. The schema is defined in [design/schemas.sql](../design/schemas.sql) and maintained by the Gateway service.

### Enums

#### `train_status`
Status of training jobs and models.
- `queued`: Job is queued and waiting to start
- `training`: Training is in progress
- `ready`: Training completed successfully
- `failed`: Training failed with errors

#### `source_type`
Origin of UE datasets.
- `real`: Real-world data uploaded by users
- `utils_traffic_load`: Generated via traffic-load utility
- `utils_mobility`: Generated via mobility utility

#### `rapp_id`
Supported Radio Application types.
- `mro`: Mobility Robustness Optimization
- `cco`: Coverage and Capacity Optimization
- `es`: Energy Saving
- `lb`: Load Balancing

### Core Tables

#### `tenants`
Represents organizational tenants in the multi-tenant system.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `tenant_id` | uuid | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique tenant identifier |
| `name` | text | NOT NULL | Tenant display name |
| `status` | text | NOT NULL, DEFAULT 'active' | Tenant status (active/suspended) |
| `created_at` | timestamptz | NOT NULL, DEFAULT now() | Creation timestamp |
| `updated_at` | timestamptz | NOT NULL, DEFAULT now() | Last update timestamp |

**Indexes**: None (primary key only)
**Triggers**: `tenants_updated_at` (auto-updates `updated_at`)
**RLS**: Enabled with tenant-scoped policy

#### `tenant_memberships`
Maps users to tenants with role-based access control.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `tenant_id` | uuid | REFERENCES tenants(tenant_id) ON DELETE CASCADE | Tenant reference |
| `user_id` | uuid | NOT NULL | Supabase auth.users.id |
| `role` | text | CHECK IN ('owner','admin','viewer') | User role within tenant |
| `status` | text | NOT NULL, DEFAULT 'active' | Membership status |
| `created_at` | timestamptz | NOT NULL, DEFAULT now() | Creation timestamp |
| `updated_at` | timestamptz | NOT NULL, DEFAULT now() | Last update timestamp |

**Primary Key**: `(tenant_id, user_id)`
**Triggers**: `tenant_memberships_updated_at`
**RLS**: Enabled with tenant-scoped policy

#### `baselines`
Network baseline configurations (topology, training data, config).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | uuid | PRIMARY KEY, DEFAULT gen_random_uuid() | Internal unique ID |
| `tenant_id` | uuid | NOT NULL | Tenant owner |
| `baseline_id` | text | NOT NULL | User-defined baseline identifier |
| `description` | text | | Baseline description |
| `url_to_topo_csv` | text | | S3 URL to topology CSV |
| `url_to_trainingdata_csv` | text | | S3 URL to training data CSV |
| `url_to_config_csv` | text | | S3 URL to configuration CSV |
| `created_by` | uuid | | Creator user ID |
| `created_at` | timestamptz | NOT NULL, DEFAULT now() | Creation timestamp |
| `updated_at` | timestamptz | NOT NULL, DEFAULT now() | Last update timestamp |

**Unique Constraint**: `(tenant_id, baseline_id)`
**Indexes**: `idx_baselines_tenant` on `(tenant_id)`
**Triggers**: `baselines_updated_at`
**RLS**: Enabled with tenant-scoped policy
**S3 Artifacts**: Stored under `s3://{bucket}/tenants/{tenant_id}/baselines/{baseline_id}/`

**Notes**:
- URLs can be absolute S3 URLs, ARN format, or relative object keys
- Artifacts are purged via boto3 during DELETE operations

#### `ue_datasets`
User Equipment datasets (real or synthetic).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | uuid | PRIMARY KEY, DEFAULT gen_random_uuid() | Internal unique ID |
| `tenant_id` | uuid | NOT NULL | Tenant owner |
| `dataset_id` | text | NOT NULL | User-defined dataset identifier |
| `source_type` | source_type | NOT NULL | Dataset origin type |
| `baseline_id` | text | | Associated baseline (if synthetic) |
| `url_to_trainingdata_csv` | text | | Legacy training data URL |
| `url_to_smo_ue_data_csv` | text | | Canonical UE data CSV URL |
| `stats` | jsonb | | Dataset statistics (JSON) |
| `created_by` | uuid | | Creator user ID |
| `created_at` | timestamptz | NOT NULL, DEFAULT now() | Creation timestamp |

**Unique Constraint**: `(tenant_id, dataset_id)`
**Indexes**: `idx_ue_tenant` on `(tenant_id)`
**RLS**: Enabled with tenant-scoped policy
**S3 Artifacts**: Stored under `s3://{bucket}/tenants/{tenant_id}/ue/{dataset_id}/`

**Notes**:
- `url_to_smo_ue_data_csv` is the canonical field for UE data
- `url_to_trainingdata_csv` maintained for backward compatibility
- Inference loaders derive `loc_x/loc_y/mock_ue_id` from `lon/lat/ue_id` when missing

#### `bdt_models`
Bayesian Digital Twin trained models.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | uuid | PRIMARY KEY, DEFAULT gen_random_uuid() | Internal unique ID |
| `tenant_id` | uuid | NOT NULL | Tenant owner |
| `bdt_id` | text | NOT NULL | User-defined BDT identifier |
| `baseline_id` | text | NOT NULL | Baseline used for training |
| `status` | train_status | NOT NULL | Training status |
| `details` | jsonb | | Training details (JSON) |
| `hyperparams` | jsonb | | Hyperparameters used |
| `metrics` | jsonb | | Training metrics |
| `artifacts_uri` | text[] | | Ordered list of artifact URIs |
| `created_by` | uuid | | Creator user ID |
| `created_at` | timestamptz | NOT NULL, DEFAULT now() | Creation timestamp |
| `updated_at` | timestamptz | NOT NULL, DEFAULT now() | Last update timestamp |

**Unique Constraint**: `(tenant_id, bdt_id)`
**Indexes**: `idx_bdt_tenant_status` on `(tenant_id, status)`
**Triggers**: `bdt_models_updated_at`
**RLS**: Enabled with tenant-scoped policy

**Artifacts URI Format**:
- Local disk first: `/app/var/models/{tenant_id}/bdt/{bdt_id}.pickle`
- Remote S3: `s3://{bucket}/{tenant_id}/bdt/{bdt_id}/{bdt_id}.pickle`
- Worker normalizes filenames to `*.pickle` for consistency

#### `rapp_models`
Radio Application trained models.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | uuid | PRIMARY KEY, DEFAULT gen_random_uuid() | Internal unique ID |
| `tenant_id` | uuid | NOT NULL | Tenant owner |
| `rapp_model_id` | text | NOT NULL | User-defined rApp model identifier |
| `rapp_id` | rapp_id | NOT NULL | rApp type (mro/cco/es/lb) |
| `baseline_id` | text | NOT NULL | Baseline used for training |
| `bdt_id` | text | | BDT model used (optional) |
| `dataset_id` | text | | UE dataset used (optional) |
| `status` | train_status | NOT NULL | Training status |
| `config` | jsonb | | Training configuration (JSON) |
| `metrics` | jsonb | | Training metrics/hyperparameters |
| `artifacts_uri` | text[] | | Ordered list of artifact URIs |
| `created_by` | uuid | | Creator user ID |
| `created_at` | timestamptz | NOT NULL, DEFAULT now() | Creation timestamp |
| `updated_at` | timestamptz | NOT NULL, DEFAULT now() | Last update timestamp |

**Unique Constraint**: `(tenant_id, rapp_model_id)`
**Indexes**: `idx_rapp_tenant_status` on `(tenant_id, rapp_id, status)`
**Triggers**: `rapp_models_updated_at`
**RLS**: Enabled with tenant-scoped policy

**Artifacts URI Format**:
- Local disk first: `/app/var/models/{tenant_id}/rapps/{rapp_id}/{rapp_model_id}.zip`
- Remote S3: `s3://{bucket}/{tenant_id}/models/rapps/{rapp_id}/{rapp_model_id}.zip`
- Inference resolves this list to rehydrate missing ZIPs before executing RadP

**Metrics Field**:
Stores hyperparameter snapshot recorded at training time (e.g., `num_epochs`, `lambda_`, `weak_coverage_threshold`, `learning_rate`, `epsilon`, `opt_delta`, `input_signature`).

#### `training_jobs`
Async training job orchestration (BDT, rApp, Utils).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | uuid | PRIMARY KEY, DEFAULT gen_random_uuid() | Internal unique ID |
| `tenant_id` | uuid | NOT NULL | Tenant owner |
| `kind` | text | CHECK IN ('bdt','rapp','utils') | Job type |
| `model_ref` | uuid | | Reference to model table ID |
| `status` | train_status | NOT NULL | Job status |
| `idempotency_key` | text | | Prevents duplicate jobs |
| `error` | text | | Error message (if failed) |
| `worker` | text | | Worker instance that processed job |
| `logs_uri` | text | | S3 URL to training logs |
| `created_by` | uuid | | Creator user ID |
| `started_at` | timestamptz | | Job start time |
| `finished_at` | timestamptz | | Job completion time |

**Unique Constraint**: `(tenant_id, idempotency_key)`
**Indexes**: `idx_jobs_tenant_status` on `(tenant_id, status, started_at)`
**RLS**: Enabled with tenant-scoped policy

**Notes**:
- Jobs are enqueued to Kafka and processed by background workers
- `idempotency_key` ensures duplicate requests don't create multiple jobs

#### `inference_runs`
rApp inference execution records.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | uuid | PRIMARY KEY, DEFAULT gen_random_uuid() | Internal unique ID |
| `tenant_id` | uuid | NOT NULL | Tenant owner |
| `rapp_id` | rapp_id | NOT NULL | rApp type |
| `rapp_model_id` | text | NOT NULL | rApp model identifier |
| `baseline_id` | text | | Baseline context (optional) |
| `bdt_id` | text | | BDT model used (optional) |
| `request` | jsonb | | Inference request payload |
| `ue_dataset_id` | text | | UE dataset for inference |
| `tick` | text | | Time tick for inference |
| `result` | jsonb | | Inference result (JSON) |
| `result_uri` | text[] | | S3 URLs to result artifacts |
| `status` | text | NOT NULL, DEFAULT 'queued' | Execution status |
| `error` | text | | Error message (if failed) |
| `created_by` | uuid | | Creator user ID |
| `created_at` | timestamptz | NOT NULL, DEFAULT now() | Creation timestamp |
| `updated_at` | timestamptz | NOT NULL, DEFAULT now() | Last update timestamp |

**Indexes**: `idx_infer_tenant` on `(tenant_id, created_at DESC)`
**RLS**: Enabled with tenant-scoped policy

**Notes**:
- Inference runs are async; clients poll the status
- Results include policy recommendations and metrics

### Row Level Security (RLS)

All tenant-scoped tables enforce RLS policies:
```sql
CREATE POLICY {table}_rls ON {table}
  USING (tenant_id = current_setting('app.current_tenant')::uuid)
  WITH CHECK (tenant_id = current_setting('app.current_tenant')::uuid);
```

The Gateway sets the `app.current_tenant` session variable based on JWT claims before executing queries, ensuring users can only access their tenant's data.

---

## API Endpoints

The platform exposes a unified RESTful API through the Gateway service. All endpoints (except `/health`) require JWT authentication via the `Authorization: Bearer <token>` header.

**Base URL**: `http://localhost:8000` (development)
**API Version**: 0.4.6
**Specification**: [design/openapi.yaml](../design/openapi.yaml)

### Health Check

#### `GET /health`
Check service health status.

**Authentication**: None

**Response** (`200 OK`):
```json
{
  "status": "ok"
}
```

---

### Tenant Management

#### `GET /tenants`
List all tenants accessible to the authenticated user.

**Authentication**: Required (JWT)

**Response** (`200 OK`):
```json
[
  {
    "tenant_id": "uuid",
    "name": "string",
    "status": "active"
  }
]
```

#### `GET /tenants/{tenant_id}`
Get details of a specific tenant.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Response** (`200 OK`):
```json
{
  "tenant_id": "uuid",
  "name": "string",
  "status": "active",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

#### `PATCH /tenants/{tenant_id}`
Update tenant details.

**Authentication**: Required (JWT, owner/admin role)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Request Body**:
```json
{
  "name": "string",
  "status": "active"
}
```

**Response** (`200 OK`):
```json
{
  "tenant_id": "uuid",
  "name": "string",
  "status": "active",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

#### `GET /tenants/{tenant_id}/members`
List tenant members.

**Authentication**: Required (JWT, admin role)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Response** (`200 OK`):
```json
[
  {
    "user_id": "uuid",
    "role": "owner",
    "status": "active",
    "created_at": "2025-01-01T00:00:00Z"
  }
]
```

#### `POST /tenants/{tenant_id}/members`
Add a member to the tenant.

**Authentication**: Required (JWT, owner role)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Request Body**:
```json
{
  "user_id": "uuid",
  "role": "admin"
}
```

**Response** (`201 Created`):
```json
{
  "user_id": "uuid",
  "role": "admin",
  "status": "active",
  "created_at": "2025-01-01T00:00:00Z"
}
```

---

### Baseline Management

#### `GET /tenants/{tenant_id}/baselines`
List all baselines for a tenant.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Response** (`200 OK`):
```json
[
  {
    "id": "uuid",
    "baseline_id": "string",
    "description": "string",
    "url_to_topo_csv": "s3://bucket/path/to/topology.csv",
    "url_to_trainingdata_csv": "s3://bucket/path/to/training.csv",
    "url_to_config_csv": "s3://bucket/path/to/config.csv",
    "created_at": "2025-01-01T00:00:00Z"
  }
]
```

#### `POST /tenants/{tenant_id}/baselines`
Create a new baseline.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Request Body**:
```json
{
  "baseline_id": "string",
  "description": "string",
  "url_to_topo_csv": "s3://bucket/path/to/topology.csv",
  "url_to_trainingdata_csv": "s3://bucket/path/to/training.csv",
  "url_to_config_csv": "s3://bucket/path/to/config.csv"
}
```

**Response** (`201 Created`):
```json
{
  "id": "uuid",
  "baseline_id": "string",
  "description": "string",
  "url_to_topo_csv": "s3://bucket/path/to/topology.csv",
  "url_to_trainingdata_csv": "s3://bucket/path/to/training.csv",
  "url_to_config_csv": "s3://bucket/path/to/config.csv",
  "created_at": "2025-01-01T00:00:00Z"
}
```

#### `GET /tenants/{tenant_id}/baselines/{baseline_id}`
Get baseline details.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier
- `baseline_id` (string): Baseline identifier

**Response** (`200 OK`):
```json
{
  "id": "uuid",
  "baseline_id": "string",
  "description": "string",
  "url_to_topo_csv": "s3://bucket/path/to/topology.csv",
  "url_to_trainingdata_csv": "s3://bucket/path/to/training.csv",
  "url_to_config_csv": "s3://bucket/path/to/config.csv",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

#### `DELETE /tenants/{tenant_id}/baselines/{baseline_id}`
Delete a baseline and its S3 artifacts.

**Authentication**: Required (JWT, admin role)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier
- `baseline_id` (string): Baseline identifier

**Response** (`204 No Content`): Empty body

---

### Topology Generation (Utils)

#### `POST /tenants/{tenant_id}/utils/topology/generate`
Generate a synthetic network topology.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Request Body**:
```json
{
  "baseline_id": "string",
  "description": "string",
  "num_cells": 100,
  "params": {
    "area_width": 1000.0,
    "area_height": 1000.0,
    "cell_radius": 50.0
  }
}
```

**Response** (`202 Accepted`):
```json
{
  "baseline_id": "string",
  "status": "queued",
  "message": "Topology generation started"
}
```

**Notes**:
- Generation is async; poll baseline status
- Generated artifacts are uploaded to S3

---

### UE Dataset Management

#### `GET /tenants/{tenant_id}/ue-data/datasets`
List all UE datasets for a tenant.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Response** (`200 OK`):
```json
[
  {
    "id": "uuid",
    "dataset_id": "string",
    "source_type": "utils_traffic_load",
    "baseline_id": "string",
    "url_to_smo_ue_data_csv": "s3://bucket/path/to/ue_data.csv",
    "stats": {},
    "created_at": "2025-01-01T00:00:00Z"
  }
]
```

#### `POST /tenants/{tenant_id}/ue-data/datasets`
Upload a real UE dataset.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Request Body**:
```json
{
  "dataset_id": "string",
  "source_type": "real",
  "baseline_id": "string",
  "url_to_smo_ue_data_csv": "s3://bucket/path/to/ue_data.csv"
}
```

**Response** (`201 Created`):
```json
{
  "id": "uuid",
  "dataset_id": "string",
  "source_type": "real",
  "baseline_id": "string",
  "url_to_smo_ue_data_csv": "s3://bucket/path/to/ue_data.csv",
  "created_at": "2025-01-01T00:00:00Z"
}
```

#### `GET /tenants/{tenant_id}/ue-data/datasets/{dataset_id}`
Get UE dataset details.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier
- `dataset_id` (string): Dataset identifier

**Response** (`200 OK`):
```json
{
  "id": "uuid",
  "dataset_id": "string",
  "source_type": "utils_traffic_load",
  "baseline_id": "string",
  "url_to_smo_ue_data_csv": "s3://bucket/path/to/ue_data.csv",
  "stats": {
    "num_ues": 1000,
    "num_days": 7
  },
  "created_at": "2025-01-01T00:00:00Z"
}
```

#### `DELETE /tenants/{tenant_id}/ue-data/datasets/{dataset_id}`
Delete a UE dataset and its S3 artifacts.

**Authentication**: Required (JWT, admin role)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier
- `dataset_id` (string): Dataset identifier

**Response** (`204 No Content`): Empty body

---

### Traffic Load Generation (Utils)

#### `POST /tenants/{tenant_id}/utils/traffic-load/generate`
Generate synthetic UE traffic data.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Request Body**:
```json
{
  "dataset_id": "string",
  "baseline_id": "string",
  "days": 7,
  "num_ues": 1000,
  "spatial_params": {
    "distribution": "uniform"
  },
  "time_params": {
    "peak_hours": [8, 18]
  }
}
```

**Response** (`200 OK`):
```json
{
  "id": "uuid",
  "dataset_id": "string",
  "source_type": "utils_traffic_load",
  "baseline_id": "string",
  "url_to_smo_ue_data_csv": "s3://bucket/tenants/{tenant_id}/ue/{dataset_id}/synthetic_dataset.csv",
  "stats": {
    "num_ues": 1000,
    "num_days": 7
  },
  "created_at": "2025-01-01T00:00:00Z"
}
```

**Notes**:
- Executes inline (no Kafka fallback)
- Loads baseline topology from S3 before invoking RADP spatial generator
- Returns created dataset descriptor on success

---

### Mobility Generation (Utils)

#### `POST /tenants/{tenant_id}/utils/mobility/generate`
Generate synthetic UE mobility patterns.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Request Body**:
```json
{
  "dataset_id": "string",
  "baseline_id": "string",
  "days": 7,
  "num_ues": 500,
  "mobility_model": "random_walk",
  "params": {
    "speed_kmh": 30,
    "pause_time_s": 60
  }
}
```

**Response** (`202 Accepted`):
```json
{
  "dataset_id": "string",
  "status": "queued",
  "message": "Mobility generation started"
}
```

**Notes**:
- Generation is async; poll dataset status
- Generated artifacts are uploaded to S3

---

### BDT (Bayesian Digital Twin) Management

#### `GET /tenants/{tenant_id}/bdt`
List all BDT models for a tenant.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Query Parameters** (optional):
- `status` (train_status): Filter by status (queued/training/ready/failed)

**Response** (`200 OK`):
```json
[
  {
    "id": "uuid",
    "bdt_id": "string",
    "baseline_id": "string",
    "status": "ready",
    "hyperparams": {
      "num_iterations": 1000,
      "learning_rate": 0.01
    },
    "metrics": {
      "accuracy": 0.95,
      "loss": 0.05
    },
    "artifacts_uri": [
      "/app/var/models/{tenant_id}/bdt/{bdt_id}.pickle",
      "s3://bucket/{tenant_id}/bdt/{bdt_id}/{bdt_id}.pickle"
    ],
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z"
  }
]
```

#### `POST /tenants/{tenant_id}/bdt/train`
Train a new BDT model.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Request Body**:
```json
{
  "bdt_id": "string",
  "baseline_id": "string",
  "hyperparams": {
    "num_iterations": 1000,
    "learning_rate": 0.01,
    "batch_size": 32
  },
  "idempotency_key": "string"
}
```

**Response** (`202 Accepted`):
```json
{
  "id": "uuid",
  "bdt_id": "string",
  "baseline_id": "string",
  "status": "queued",
  "hyperparams": {
    "num_iterations": 1000,
    "learning_rate": 0.01,
    "batch_size": 32
  },
  "created_at": "2025-01-01T00:00:00Z"
}
```

**Notes**:
- Training is async via Kafka worker
- Poll `/tenants/{tenant_id}/bdt/{bdt_id}` for status updates
- `idempotency_key` prevents duplicate training jobs

#### `GET /tenants/{tenant_id}/bdt/{bdt_id}`
Get BDT model details.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier
- `bdt_id` (string): BDT model identifier

**Response** (`200 OK`):
```json
{
  "id": "uuid",
  "bdt_id": "string",
  "baseline_id": "string",
  "status": "ready",
  "details": {},
  "hyperparams": {
    "num_iterations": 1000,
    "learning_rate": 0.01
  },
  "metrics": {
    "accuracy": 0.95,
    "loss": 0.05
  },
  "artifacts_uri": [
    "/app/var/models/{tenant_id}/bdt/{bdt_id}.pickle",
    "s3://bucket/{tenant_id}/bdt/{bdt_id}/{bdt_id}.pickle"
  ],
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

#### `DELETE /tenants/{tenant_id}/bdt/{bdt_id}`
Delete a BDT model and its artifacts.

**Authentication**: Required (JWT, admin role)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier
- `bdt_id` (string): BDT model identifier

**Response** (`204 No Content`): Empty body

**Notes**:
- Deletes local disk artifacts and S3 objects
- Cannot delete models currently in use by rApps

---

### rApp Management

#### `GET /tenants/{tenant_id}/rapps`
List available rApp types.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Response** (`200 OK`):
```json
[
  {
    "rapp_id": "mro",
    "name": "Mobility Robustness Optimization",
    "description": "Optimizes handover parameters to reduce failures"
  },
  {
    "rapp_id": "cco",
    "name": "Coverage and Capacity Optimization",
    "description": "Optimizes cell antenna parameters for coverage and capacity"
  },
  {
    "rapp_id": "es",
    "name": "Energy Saving",
    "description": "Reduces energy consumption by optimizing cell sleep patterns"
  },
  {
    "rapp_id": "lb",
    "name": "Load Balancing",
    "description": "Balances traffic load across cells"
  }
]
```

#### `POST /tenants/{tenant_id}/rapps/{rapp_id}/train`
Train a new rApp model.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier
- `rapp_id` (rapp_id): rApp type (mro/cco/es/lb)

**Request Body**:
```json
{
  "rapp_model_id": "string",
  "baseline_id": "string",
  "bdt_id": "string",
  "dataset_id": "string",
  "config": {
    "num_epochs": 100,
    "lambda_": 0.5,
    "weak_coverage_threshold": -100,
    "learning_rate": 0.001,
    "epsilon": 0.1,
    "opt_delta": 0.01
  },
  "idempotency_key": "string"
}
```

**Response** (`202 Accepted`):
```json
{
  "id": "uuid",
  "rapp_model_id": "string",
  "rapp_id": "mro",
  "baseline_id": "string",
  "bdt_id": "string",
  "dataset_id": "string",
  "status": "queued",
  "config": {
    "num_epochs": 100,
    "lambda_": 0.5,
    "weak_coverage_threshold": -100
  },
  "created_at": "2025-01-01T00:00:00Z"
}
```

**Notes**:
- Training is async via Kafka worker
- Requires trained BDT model and UE dataset
- Poll `/tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}` for status

#### `GET /tenants/{tenant_id}/rapps/{rapp_id}/models`
List all trained models for a specific rApp type.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier
- `rapp_id` (rapp_id): rApp type

**Query Parameters** (optional):
- `status` (train_status): Filter by status

**Response** (`200 OK`):
```json
[
  {
    "id": "uuid",
    "rapp_model_id": "string",
    "rapp_id": "mro",
    "baseline_id": "string",
    "bdt_id": "string",
    "dataset_id": "string",
    "status": "ready",
    "config": {},
    "metrics": {
      "num_epochs": 100,
      "final_loss": 0.02
    },
    "artifacts_uri": [
      "/app/var/models/{tenant_id}/rapps/mro/{rapp_model_id}.zip",
      "s3://bucket/{tenant_id}/models/rapps/mro/{rapp_model_id}.zip"
    ],
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z"
  }
]
```

#### `GET /tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}`
Get rApp model details.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier
- `rapp_id` (rapp_id): rApp type
- `rapp_model_id` (string): rApp model identifier

**Response** (`200 OK`):
```json
{
  "id": "uuid",
  "rapp_model_id": "string",
  "rapp_id": "mro",
  "baseline_id": "string",
  "bdt_id": "string",
  "dataset_id": "string",
  "status": "ready",
  "config": {
    "num_epochs": 100,
    "lambda_": 0.5
  },
  "metrics": {
    "num_epochs": 100,
    "final_loss": 0.02,
    "training_time_s": 3600
  },
  "artifacts_uri": [
    "/app/var/models/{tenant_id}/rapps/mro/{rapp_model_id}.zip",
    "s3://bucket/{tenant_id}/models/rapps/mro/{rapp_model_id}.zip"
  ],
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

#### `DELETE /tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}`
Delete a rApp model and its artifacts.

**Authentication**: Required (JWT, admin role)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier
- `rapp_id` (rapp_id): rApp type
- `rapp_model_id` (string): rApp model identifier

**Response** (`204 No Content`): Empty body

---

### rApp Inference

#### `POST /tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}/infer`
Execute inference using a trained rApp model.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier
- `rapp_id` (rapp_id): rApp type
- `rapp_model_id` (string): rApp model identifier

**Request Body**:
```json
{
  "ue_dataset_id": "string",
  "tick": "2025-01-01T00:00:00Z",
  "baseline_id": "string",
  "bdt_id": "string",
  "params": {
    "optimization_target": "minimize_failures"
  }
}
```

**Response** (`202 Accepted`):
```json
{
  "id": "uuid",
  "rapp_id": "mro",
  "rapp_model_id": "string",
  "baseline_id": "string",
  "bdt_id": "string",
  "ue_dataset_id": "string",
  "tick": "2025-01-01T00:00:00Z",
  "status": "queued",
  "created_at": "2025-01-01T00:00:00Z"
}
```

**Notes**:
- Inference is async; poll `/tenants/{tenant_id}/rapps/{rapp_id}/infer/{inference_id}` for results
- Requires compatible BDT model and UE dataset

#### `GET /tenants/{tenant_id}/rapps/{rapp_id}/infer/{inference_id}`
Get inference run status and results.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier
- `rapp_id` (rapp_id): rApp type
- `inference_id` (uuid): Inference run identifier

**Response** (`200 OK`):
```json
{
  "id": "uuid",
  "rapp_id": "mro",
  "rapp_model_id": "string",
  "baseline_id": "string",
  "bdt_id": "string",
  "ue_dataset_id": "string",
  "tick": "2025-01-01T00:00:00Z",
  "request": {},
  "result": {
    "optimized_params": {
      "cell_1": {"tilt": 5, "power": 40}
    },
    "predicted_improvement": 0.15
  },
  "result_uri": [
    "s3://bucket/{tenant_id}/inference/{inference_id}/results.json"
  ],
  "status": "completed",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

**Status Values**:
- `queued`: Inference queued for processing
- `running`: Inference in progress
- `completed`: Inference completed successfully
- `failed`: Inference failed (check `error` field)

---

### Training Jobs

#### `GET /tenants/{tenant_id}/jobs`
List training jobs for a tenant.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier

**Query Parameters** (optional):
- `kind` (string): Filter by job type (bdt/rapp/utils)
- `status` (train_status): Filter by status

**Response** (`200 OK`):
```json
[
  {
    "id": "uuid",
    "kind": "bdt",
    "model_ref": "uuid",
    "status": "ready",
    "worker": "bdt-worker-1",
    "started_at": "2025-01-01T00:00:00Z",
    "finished_at": "2025-01-01T00:05:00Z"
  }
]
```

#### `GET /tenants/{tenant_id}/jobs/{job_id}`
Get training job details.

**Authentication**: Required (JWT)

**Path Parameters**:
- `tenant_id` (uuid): Tenant identifier
- `job_id` (uuid): Job identifier

**Response** (`200 OK`):
```json
{
  "id": "uuid",
  "kind": "rapp",
  "model_ref": "uuid",
  "status": "training",
  "error": null,
  "worker": "rapp-worker-2",
  "logs_uri": "s3://bucket/{tenant_id}/logs/{job_id}.log",
  "started_at": "2025-01-01T00:00:00Z",
  "finished_at": null
}
```

---

## Authentication & Authorization

### JWT Authentication

The platform uses **Supabase GoTrue** for authentication. Users authenticate with Supabase and receive a JWT token containing:
- `sub`: User UUID
- `email`: User email
- `app_metadata.tenant_id`: Primary tenant ID
- `app_metadata.role`: User role (owner/admin/viewer)

### Authorization Flow

1. **User Authentication**: User logs in via Supabase (email/password, OAuth, etc.)
2. **Token Generation**: Supabase issues a JWT token
3. **Request Authorization**: Client includes token in `Authorization: Bearer <token>` header
4. **Gateway Validation**: Gateway validates JWT signature and expiry
5. **Tenant Context**: Gateway extracts `tenant_id` from token claims
6. **RLS Enforcement**: Gateway sets PostgreSQL session variable `app.current_tenant`
7. **Data Access**: Database RLS policies enforce tenant isolation

### Role-Based Access Control (RBAC)

| Role | Permissions |
|------|-------------|
| **owner** | Full access: CRUD on all resources, manage members, delete tenant |
| **admin** | Manage resources: CRUD on baselines/models/datasets, cannot manage members |
| **viewer** | Read-only: List and view resources, cannot create/update/delete |

### Row Level Security (RLS)

All tenant-scoped tables enforce RLS policies:
```sql
CREATE POLICY {table}_rls ON {table}
  USING (tenant_id = current_setting('app.current_tenant')::uuid)
  WITH CHECK (tenant_id = current_setting('app.current_tenant')::uuid);
```

This ensures:
- Users can only query data belonging to their tenant
- INSERT/UPDATE operations automatically scope to the user's tenant
- Cross-tenant data access is impossible at the database level

---

## User Flows

### Flow 1: New Tenant Onboarding

1. **User Registration**: User signs up via Supabase authentication
2. **Tenant Creation**: System creates a new tenant and assigns user as owner
3. **Tenant Membership**: User is added to `tenant_memberships` with role='owner'
4. **JWT Token**: User receives JWT with `tenant_id` in claims
5. **Access Platform**: User can now create baselines, datasets, and models

**API Calls**:
- Supabase: `POST /auth/v1/signup` → Returns JWT
- Platform: `GET /tenants` → Verifies tenant access

---

### Flow 2: Creating a Network Baseline

1. **Generate Topology**: User requests synthetic topology generation
   - API: `POST /tenants/{tenant_id}/utils/topology/generate`
   - Body: `{baseline_id, num_cells, params}`
   - Response: `{baseline_id, status: "queued"}`

2. **Async Processing**: Data Sim service generates topology CSV
   - Worker downloads baseline parameters
   - Executes RADP topology generator
   - Uploads topology CSV to S3

3. **Baseline Ready**: Baseline status changes to "ready"
   - API: `GET /tenants/{tenant_id}/baselines/{baseline_id}`
   - Response: `{baseline_id, url_to_topo_csv, status: "ready"}`

4. **User Verification**: User can download topology CSV from S3

**Key Components**:
- Gateway: Orchestrates request
- Data Sim: Generates topology
- S3: Stores topology artifacts
- PostgreSQL: Tracks baseline metadata

---

### Flow 3: Training a BDT Model

1. **Prepare Baseline**: Ensure baseline exists with topology and training data
   - API: `GET /tenants/{tenant_id}/baselines/{baseline_id}`

2. **Submit Training Job**: User requests BDT training
   - API: `POST /tenants/{tenant_id}/bdt/train`
   - Body: `{bdt_id, baseline_id, hyperparams, idempotency_key}`
   - Response: `{bdt_id, status: "queued"}`

3. **Job Queuing**: Gateway creates training job and publishes to Kafka
   - Insert into `training_jobs` table
   - Publish message to Kafka topic `bdt-training`

4. **Worker Processing**: BDT worker consumes Kafka message
   - Download baseline artifacts from S3
   - Execute Bayesian training algorithm
   - Upload trained model (`.pickle`) to S3 and local disk

5. **Status Updates**: Worker updates model status
   - `status: "training"` → Worker starts
   - `status: "ready"` → Training complete
   - `status: "failed"` → Training error (with error message)

6. **User Polling**: User polls for status updates
   - API: `GET /tenants/{tenant_id}/bdt/{bdt_id}`
   - Response: `{bdt_id, status: "ready", metrics, artifacts_uri}`

**Key Components**:
- Gateway: Receives request, creates job
- Kafka: Message queue for async jobs
- BDT Worker: Executes training
- PostgreSQL: Tracks job and model status
- S3: Stores model artifacts

---

### Flow 4: Generating Synthetic UE Data

1. **Select Baseline**: User chooses existing baseline with topology
   - API: `GET /tenants/{tenant_id}/baselines`

2. **Generate Traffic Load**: User requests synthetic UE data
   - API: `POST /tenants/{tenant_id}/utils/traffic-load/generate`
   - Body: `{dataset_id, baseline_id, days, num_ues, spatial_params, time_params}`
   - Response: `{dataset_id, url_to_smo_ue_data_csv, stats}`

3. **Inline Processing**: SMO Sim executes synchronously
   - Load baseline topology from S3
   - Invoke RADP spatial generator
   - Generate UE mobility and traffic patterns
   - Upload synthetic dataset to S3: `s3://{tenant_id}/ue/{dataset_id}/synthetic_dataset.csv`

4. **Dataset Ready**: Dataset is immediately available
   - API: `GET /tenants/{tenant_id}/ue-data/datasets/{dataset_id}`
   - Response: `{dataset_id, source_type: "utils_traffic_load", stats, url_to_smo_ue_data_csv}`

**Key Components**:
- Gateway: Routes request
- SMO Sim: Generates UE data inline
- S3: Stores UE dataset
- PostgreSQL: Tracks dataset metadata

---

### Flow 5: Training a rApp Model

1. **Prerequisites**: Ensure BDT model and UE dataset exist
   - API: `GET /tenants/{tenant_id}/bdt/{bdt_id}` → `status: "ready"`
   - API: `GET /tenants/{tenant_id}/ue-data/datasets/{dataset_id}`

2. **Submit rApp Training**: User requests rApp training
   - API: `POST /tenants/{tenant_id}/rapps/{rapp_id}/train`
   - Body: `{rapp_model_id, baseline_id, bdt_id, dataset_id, config, idempotency_key}`
   - Response: `{rapp_model_id, status: "queued"}`

3. **Job Queuing**: Gateway creates training job and publishes to Kafka
   - Insert into `training_jobs` table
   - Publish message to Kafka topic `rapp-training`

4. **Worker Processing**: rApp worker consumes Kafka message
   - Download BDT model from S3
   - Download UE dataset from S3
   - Load baseline topology
   - Execute rApp training (RADP algorithm)
   - Upload trained model (`.zip`) to S3 and local disk

5. **Status Updates**: Worker updates model status
   - `status: "training"` → Worker starts
   - `status: "ready"` → Training complete (with metrics)
   - `status: "failed"` → Training error

6. **User Polling**: User polls for status
   - API: `GET /tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}`
   - Response: `{rapp_model_id, status: "ready", metrics, artifacts_uri}`

**Key Components**:
- Gateway: Orchestrates request
- Kafka: Async job queue
- rApp Worker: Executes training
- BDT Engine: Provides Digital Twin predictions
- PostgreSQL: Tracks job and model status
- S3: Stores model artifacts

---

### Flow 6: Running rApp Inference

1. **Prepare Resources**: Ensure rApp model, BDT model, and UE dataset are ready
   - API: `GET /tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}` → `status: "ready"`
   - API: `GET /tenants/{tenant_id}/bdt/{bdt_id}` → `status: "ready"`
   - API: `GET /tenants/{tenant_id}/ue-data/datasets/{dataset_id}`

2. **Submit Inference Request**: User requests inference
   - API: `POST /tenants/{tenant_id}/rapps/{rapp_id}/models/{rapp_model_id}/infer`
   - Body: `{ue_dataset_id, tick, baseline_id, bdt_id, params}`
   - Response: `{id, status: "queued"}`

3. **Job Queuing**: Gateway creates inference run and publishes to Kafka
   - Insert into `inference_runs` table
   - Publish message to Kafka topic `rapp-inference`

4. **Worker Processing**: rApp worker consumes message
   - Download rApp model from S3 (if not cached locally)
   - Download BDT model from S3 (if needed)
   - Load UE dataset for specified tick
   - Execute inference (RADP algorithm)
   - Generate optimization recommendations
   - Upload results to S3

5. **Status Updates**: Worker updates inference status
   - `status: "running"` → Inference starts
   - `status: "completed"` → Inference complete (with results)
   - `status: "failed"` → Inference error

6. **User Polling**: User polls for results
   - API: `GET /tenants/{tenant_id}/rapps/{rapp_id}/infer/{inference_id}`
   - Response: `{id, status: "completed", result: {optimized_params, predicted_improvement}, result_uri}`

7. **Apply Recommendations**: User applies optimization policies to network
   - Results contain cell-specific parameter adjustments (e.g., antenna tilt, power)

**Key Components**:
- Gateway: Orchestrates request
- Kafka: Async job queue
- rApp Worker: Executes inference
- BDT Engine: Provides network state predictions
- PostgreSQL: Tracks inference run status
- S3: Stores inference results

---

### Flow 7: Managing Tenant Members

1. **List Members**: Owner views current team members
   - API: `GET /tenants/{tenant_id}/members`
   - Response: `[{user_id, role, status}]`

2. **Invite User**: Owner adds new member
   - Prerequisite: User must have Supabase account
   - API: `POST /tenants/{tenant_id}/members`
   - Body: `{user_id, role: "admin"}`
   - Response: `{user_id, role: "admin", status: "active"}`

3. **Update Role**: Owner changes member role
   - API: `PATCH /tenants/{tenant_id}/members/{user_id}`
   - Body: `{role: "viewer"}`
   - Response: `{user_id, role: "viewer"}`

4. **Remove Member**: Owner revokes access
   - API: `DELETE /tenants/{tenant_id}/members/{user_id}`
   - Response: `204 No Content`
   - Member loses access to tenant resources

**Key Components**:
- Gateway: Enforces RBAC (only owners can manage members)
- PostgreSQL: `tenant_memberships` table
- JWT Claims: User's new role reflected in subsequent tokens

---

## Development Setup

### Prerequisites
- Docker & Docker Compose
- Git
- Python 3.9+ (for local development)

### Initial Setup

1. **Clone Repository**:
   ```bash
   git clone https://github.com/CloudlyIO/maveric_platform_dev.git
   cd maveric_platform_dev
   ```

2. **Initialize Submodules**:
   ```bash
   git submodule update --init --recursive
   ```

3. **Create Docker Network** (one-time):
   ```bash
   docker network create maveric
   ```

### Running Infrastructure Services

Start PostgreSQL, Redis, MongoDB, Kafka, MinIO, and pgAdmin:
```bash
docker compose -f docker-compose.infra.yml up -d
```

**Access URLs**:
- PostgreSQL: `localhost:5432` (user: `postgres`, password: `postgres`, db: `maveric`)
- Redis: `localhost:6379`
- MongoDB: `localhost:27017`
- Kafka: `localhost:9092`
- MinIO: `http://localhost:9000` (console: `http://localhost:9001`)
- pgAdmin: `http://localhost:5050` (email: `admin@maveric.io`, password: `admin`)

### Running Application Services

Start all microservices (Gateway, Data Sim, SMO Sim, BDT Engine, rApp Engine):
```bash
docker compose -f docker-compose.infra.yml -f docker-compose.apps.yml up -d --build
```

**Service Ports**:
- Gateway: `http://localhost:8000`
- Data Sim: `http://localhost:8001`
- SMO Sim: `http://localhost:8002`
- BDT Engine: `http://localhost:8003`
- BDT Worker: (background process)
- rApp Engine: `http://localhost:8004`
- rApp Worker: (background process)

### Rebuild Single Service

Rebuild and restart a specific service without touching infrastructure:
```bash
docker compose -f docker-compose.infra.yml -f docker-compose.apps.yml up --no-deps --build <service-name>
```

Example (rebuild BDT worker):
```bash
docker compose -f docker-compose.infra.yml -f docker-compose.apps.yml up --no-deps --build bdt-worker
```

### Stop Services

Stop application services only:
```bash
docker compose -f docker-compose.infra.yml -f docker-compose.apps.yml stop gateway bdt-engine bdt-worker rapp rapp-worker smo-sim data-sim
```

Stop everything (apps + infra):
```bash
docker compose -f docker-compose.infra.yml -f docker-compose.apps.yml down
```

### Database Management

**Inspect Tables**:
```bash
docker exec -it postgres psql -U postgres -d maveric -c "\dt"
```

**Run Migrations**:
The Gateway service automatically runs GORM auto-migrations at startup using `db/migrations/schemas.sql` (mirrored from `design/schemas.sql`).

**Manual Schema Changes**:
See `datamigration.md` for Alembic-based migration workflow.

### Local Python Development (Optional)

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   Copy `.env.example` to `.env` and configure:
   ```env
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/maveric
   REDIS_URL=redis://localhost:6379/0
   S3_ENDPOINT=http://localhost:9000
   S3_ACCESS_KEY=minioadmin
   S3_SECRET_KEY=minioadmin
   KAFKA_BROKER=localhost:9092
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-key
   ```

4. **Run Service Locally**:
   ```bash
   cd submodule/maveric_platform_gateway
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Testing

**Postman Collection**:
Import `test/postman/maveric_platform.postman_collection.json` for pre-configured API requests.

**Contract Tests**:
Update contract tests to validate against each service's OpenAPI spec and the stitched gateway surface.

---

## Appendix: Key Files

- [README.md](../README.md): Project overview and Docker workflows
- [design/openapi.yaml](../design/openapi.yaml): OpenAPI 3.0.3 specification (API contract)
- [design/schemas.sql](../design/schemas.sql): PostgreSQL database schema (source of truth)
- [design/HLD.md](../design/HLD.md): High-Level Design document
- [design/LLD.md](../design/LLD.md): Low-Level Design document
- **docker-compose.infra.yml**: Infrastructure services (Postgres, Redis, Kafka, MinIO)
- **docker-compose.apps.yml**: Application services (Gateway, BDT, rApp, etc.)
- **datamigration.md**: Database migration workflow (Alembic)

---

## Glossary

- **BDT**: Bayesian Digital Twin - Probabilistic model of network behavior
- **rApp**: Radio Application - Optimization algorithm (MRO, CCO, ES, LB)
- **MRO**: Mobility Robustness Optimization - Reduces handover failures
- **CCO**: Coverage and Capacity Optimization - Optimizes cell parameters
- **ES**: Energy Saving - Reduces network energy consumption
- **LB**: Load Balancing - Distributes traffic across cells
- **UE**: User Equipment - Mobile devices (phones, IoT devices)
- **RIC**: RAN Intelligent Controller - Central controller for radio networks
- **RADP**: RIC Algorithm Development Platform - Framework for rApp development
- **RLS**: Row Level Security - PostgreSQL feature for tenant isolation
- **JWT**: JSON Web Token - Authentication token format
- **GORM**: Go Object-Relational Mapping library (used by Gateway)
- **MinIO**: S3-compatible object storage
- **Kafka**: Distributed message queue for async jobs

---

**Document Version**: 1.0
**Last Updated**: 2025-10-17
**Maintained By**: CloudlyIO Platform Team
