# RADP App Validation System

The RADP App Validation System provides comprehensive validation for application-specific parameters and configurations across all supported RADP applications.

## Supported Applications

| App Type | Name | Description |
|----------|------|-------------|
| `coverage_capacity_optimization` | CCO | Optimize cell coverage and capacity through RF parameter tuning |
| `energy_savings` | Energy Savings | Reduce network energy consumption through intelligent cell sleep |
| `load_balancing` | Load Balancing | Balance traffic load across cells for optimal performance |
| `mobility_robustness_optimization` | MRO | Optimize handover parameters to reduce failures |
| `example` | Example App | Basic example demonstrating RADP usage patterns |

## API Endpoints

### Validate App Configuration
```
POST /api/apps/validate
```

Validates app-specific parameters without file validation.

**Request Body:**
```json
{
  "app_type": "coverage_capacity_optimization",
  "train_days": [0, 1, 2, 3],
  "test_day": 4,
  "tick": 12,
  "total_timesteps": 50000,
  "cco_params": {
    "lambda_": 0.5,
    "weak_coverage_threshold": -110.0,
    "over_coverage_threshold": -5.0,
    "growth_rate": 1.2,
    "optimization_metric": "pixel"
  }
}
```

### Validate Full App Configuration
```
POST /api/apps/validate-full
```

Validates complete app configuration including configuration and topology files.

**Request Body:**
```json
{
  "app_type": "energy_savings",
  "topology_path": "/path/to/topology.csv",
  "config_path": "/path/to/config.json",
  "energy_params": {
    "power_reduction_target": 0.25,
    "sleep_threshold_hours": 2
  }
}
```

### Get App Schema
```
GET /api/apps/schema/{app_type}
```

Returns validation schema for a specific app type.

### List Supported Apps
```
GET /api/apps/supported
```

Returns list of all supported app types with descriptions.

## Validation Schemas

### Common Parameters (All Apps)

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `app_type` | string | enum | Required. Application type identifier |
| `train_days` | int[] | 0-30 | Training day numbers (no duplicates) |
| `test_day` | int | 0-30 | Test day (cannot overlap with train_days) |
| `tick` | int | 0-23 | Hour of day for simulation |
| `total_timesteps` | int | 1000-1000000 | RL training timesteps |
| `bdt_model_id` | string | pattern | Model identifier (alphanumeric, _, -) |
| `container` | string | pattern | Docker container name |

### CCO (Coverage Capacity Optimization)

**Parameters (`cco_params`):**
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `lambda_` | float | (0, 1) | Balance factor between weak/over coverage |
| `weak_coverage_threshold` | float | -150 to 0 | RSRP threshold for weak coverage (dBm) |
| `over_coverage_threshold` | float | -20 to 40 | SINR threshold for over coverage (dB) |
| `growth_rate` | float | 0.1 to 10.0 | Growth rate for soft coverage functions |
| `optimization_metric` | string | pixel, cell | Optimization granularity |

### Energy Savings

**Energy Parameters (`energy_params`):**
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `power_reduction_target` | float | 0.01 to 0.50 | Target power reduction (1% to 50%) |
| `sleep_threshold_hours` | int | 1 to 24 | Hours before cell sleep consideration |
| `min_active_cells` | int | 1 to 1000 | Minimum cells that must remain active |

**RL Parameters (`rl_params`):**
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `learning_rate` | float | 1e-6 to 1.0 | RL learning rate |
| `discount_factor` | float | 0.0 to 1.0 | Future reward discount factor |
| `epsilon` | float | 0.0 to 1.0 | Exploration rate |

### Load Balancing

**Parameters (`load_balance_params`):**
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `target_load_threshold` | float | 0.1 to 1.0 | Target load threshold (10% to 100%) |
| `handover_margin_db` | float | 0.0 to 10.0 | Handover margin in dB |
| `load_balancing_algorithm` | string | enum | Algorithm: round_robin, least_loaded, proportional |

### MRO (Mobility Robustness Optimization)

**MRO Parameters (`mro_params`):**
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `handover_failure_threshold` | float | 0.0 to 1.0 | Acceptable failure rate |
| `ping_pong_threshold` | int | 1 to 10 | Max back-and-forth handovers |
| `rlf_timeout_ms` | int | 100 to 10000 | Radio Link Failure timeout |

**Mobility Model Parameters (`mobility_model_params`):**
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `velocity_kmh` | float | 0.0 to 200.0 | UE velocity in km/h |
| `direction_change_probability` | float | 0.0 to 1.0 | Probability of direction change |
| `path_loss_model` | string | enum | Model: free_space, urban_macro, urban_micro, rural |

## Configuration Files

### Supported Formats
- **JSON** (`.json`)
- **YAML** (`.yaml`, `.yml`)  
- **CSV** (`.csv`)

### RF Configuration
```json
{
  "rf_config": {
    "frequency_bands": [1800, 2100, 2600],
    "tx_power_dbm": 43,
    "antenna_config": {
      "gain_dbi": 18
    }
  }
}
```

### UE Mobility Configuration
```json
{
  "ue_mobility": {
    "mobility_models": ["random_walk", "gauss_markov"],
    "velocity_range": {"min": 1, "max": 50}
  }
}
```

### Cell Configuration
```json
{
  "cell_config": {
    "deployment_pattern": "hexagonal",
    "inter_site_distance_m": 500,
    "cell_count": 19
  }
}
```

## Topology File Validation

**Required Columns:**
- `cell_id` - Unique cell identifier
- `cell_lat` - Cell latitude (-90 to 90)
- `cell_lon` - Cell longitude (-180 to 180)
- `cell_az_deg` - Azimuth angle (0 to 359)
- `cell_carrier_freq_mhz` - Frequency (400 to 6000 MHz)

**Optional Columns:**
- `cell_el_deg` - Electrical tilt (0 to 20 degrees)
- `cell_tx_power_dbm` - TX power (0 to 50 dBm)
- `cell_antenna_gain_dbi` - Antenna gain
- `sector_id` - Sector identifier

## Usage Examples

### Python Usage
```python
from api_manager.validators.app_validator import AppValidatorFactory
from api_manager.validators.config_validator import AppConfigValidator

# Validate app parameters only
validator = AppValidatorFactory.create_validator("coverage_capacity_optimization")
validator.validate(config_data)

# Validate complete configuration including files
full_validator = AppConfigValidator("energy_savings")
full_validator.validate(full_config_data)
```

### cURL Examples
```bash
# Validate CCO app configuration
curl -X POST http://localhost:5000/api/apps/validate \
  -H "Content-Type: application/json" \
  -d '{
    "app_type": "coverage_capacity_optimization",
    "train_days": [0, 1, 2],
    "cco_params": {"lambda_": 0.5}
  }'

# Get app schema
curl http://localhost:5000/api/apps/schema/energy_savings

# List supported apps
curl http://localhost:5000/api/apps/supported
```

## Error Handling

The validation system provides detailed error messages for debugging:

```json
{
  "error": "Validation failed",
  "error_type": "validation_error",
  "status_code": 400,
  "validation_errors": [
    {
      "field": "lambda_",
      "error": "Field 'lambda_' must be between 0 and 1 (exclusive)"
    },
    {
      "field": "test_day", 
      "error": "Test day 2 cannot be included in training days [0, 1, 2, 3]"
    }
  ]
}
```

## Integration Notes

- All validators inherit from `BaseValidator` for consistency
- Factory pattern allows easy addition of new app types
- Comprehensive file validation includes format, size, and content checks
- Cross-field validation ensures logical consistency
- Performance optimized for large configuration files and datasets