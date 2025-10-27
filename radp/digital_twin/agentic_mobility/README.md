# Agentic Mobility Generation

Natural language interface for RADP mobility parameter generation using LLM-powered workflows.

## Overview

Transform mobility simulation from parameter-driven JSON configs to natural language queries:

**Before:**
```json
{
  "ue_tracks_generation": {
    "params": {
      "num_ues": 100,
      "ue_class_distribution": {...},
      "lat_lon_boundaries": {...},
      "gauss_markov_params": {...}
    }
  }
}
```

**After:**
```python
"Generate 100 UEs in urban Tokyo during morning rush hour"
```

## Features

- **Natural Language Parsing**: Extract scenario intent from plain text
- **Intelligent Parameter Generation**: LLM-based parameter inference adapted to scenario type
- **Location Resolution**: Automatic geocoding to lat/lon bounds
- **Context-Aware Distribution**: Generate UE distributions from contextual clues
- **Self-Correction**: Automatic validation with retry logic (max 2 attempts)
- **RADP Integration**: Direct compatibility with existing mobility generators

## Quick Start

### 1. Install Dependencies

```bash
cd radp/digital_twin
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cd agentic_mobility
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Basic Usage

```python
from radp.digital_twin.agentic_mobility.api import generate_mobility_params

# Generate parameters from natural language
result = generate_mobility_params(
    "Generate 100 UEs in urban Tokyo during morning rush hour"
)

print(result["status"])  # "success"
print(result["radp_params"])  # RADP-compatible JSON
```

## Examples

### Example 1: Simple Urban Scenario
```python
query = "Generate 100 UEs in urban Tokyo"
result = generate_mobility_params(query)
```

### Example 2: Explicit Distribution
```python
query = "Create 50 UEs in suburban Chicago with 60% pedestrians, 30% cars, 10% cyclists"
result = generate_mobility_params(query)
```

### Example 3: Implicit Context
```python
query = "Generate 200 UEs in downtown New York mimicking a residential neighborhood"
result = generate_mobility_params(query)
```

### Example 4: Highway Scenario
```python
query = "500 UEs on highway I-95 near Boston"
result = generate_mobility_params(query)
```

## Architecture

### Components

1. **Query Parser (1.1)**: Parse natural language → QueryIntent
2. **Location Resolver (1.2)**: Location string → geographic bounds (parallel)
3. **Parameter Generator (1.3)**: Generate mobility parameters (parallel)
4. **Validation Chain (1.4)**: Validate physics + consistency
5. **Suggestion Generator (1.5)**: Generate corrections on failures

### Workflow

```
User Query → Parser → (Location Resolver | Parameter Generator) → Validation
                                                                      ↓
                                          Success ← [Retry?] → Suggestion
```

- **Parallel Execution**: Components 1.2 and 1.3 run simultaneously
- **Self-Correction**: Up to 2 retry attempts with LLM-generated suggestions
- **Max Retries**: Accept last parameters with warnings (POC behavior)

## Configuration

### Environment Variables (.env)

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-70b-versatile

# Optional
GEOCODING_CACHE_ENABLED=true
GEOCODING_CACHE_TTL=3600
MAX_RETRY_ATTEMPTS=2
RETRY_TIMEOUT_SECONDS=30
```

### Scenario Types

- `urban`: City/downtown environments
- `suburban`: Residential/office areas
- `rural`: Countryside/farmland
- `highway`: Freeways/motorways
- `mixed`: Combined environments

### Mobility Classes

- `stationary`: Non-moving entities (velocity = 0 m/s)
- `pedestrian`: Walking users (0.5-2.0 m/s)
- `cyclist`: Bicycle riders (3.0-8.0 m/s)
- `car`: Vehicles (5.0-30.0 m/s)

## Response Format

### Success Response

```python
{
    "status": "success",
    "radp_params": {
        "ue_tracks_generation": {
            "params": {
                "num_ticks": 50,
                "ue_class_distribution": {...},
                "lat_lon_boundaries": {...},
                "gauss_markov_params": {...}
            }
        }
    },
    "metadata": {
        "retry_count": 0,
        "query_intent": {...}
    }
}
```

### Success with Warnings

```python
{
    "status": "success_with_warnings",
    "radp_params": {...},
    "warnings": ["Highway cannot have pedestrians"],
    "failure_reasons": {
        "consistency_issues": [...]
    },
    "metadata": {
        "retry_count": 2,
        "validation_status": "failed_but_accepted"
    }
}
```

## API Reference

### `generate_mobility_params(query: str) -> Dict`

Convenience function for generating parameters.

**Args:**
- `query`: Natural language description

**Returns:**
- Dictionary with status, radp_params, and metadata

### `AgenticMobilityGenerator`

Main class for mobility generation.

**Methods:**
- `generate_from_query(user_query: str) -> Dict`: Generate parameters
- `visualize_graph(output_path: str)`: Visualize workflow graph

## Distribution Generation

The system intelligently generates UE distributions:

### Explicit Distribution
User specifies exact percentages:
```
"60% pedestrians, 30% cars, 10% cyclists"
→ {pedestrian: 0.6, car: 0.3, cyclist: 0.1}
```

### Implicit Distribution
User provides context clues:
```
"mimicking a residential area"
→ {stationary: 0.3, pedestrian: 0.35, cyclist: 0.1, car: 0.25}

"morning rush hour"
→ {pedestrian: 0.25, car: 0.55, cyclist: 0.15, stationary: 0.05}
```

### Default Distribution
No context provided, use scenario defaults:
```
"urban" → {pedestrian: 0.4, car: 0.4, cyclist: 0.15, stationary: 0.05}
```

## Validation

### Physics Checks (Critical)
- Alpha: 0.0-1.0
- Variance: ≥ 0.0
- Velocities within class ranges
- Distribution sums to 1.0

### Consistency Checks (Warnings)
- Highway → no pedestrians/cyclists
- Scenario-appropriate alpha values
- Reasonable distribution for scenario type

## Development

### Project Structure

```
radp/digital_twin/agentic_mobility/
├── api.py                  # Public API
├── config.py               # Configuration
├── models/                 # Data models
├── nodes/                  # LangGraph nodes
├── chains/                 # LangChain chains
├── prompts/                # Prompt templates
├── graph/                  # Workflow definition
├── formatters/             # RADP formatting
├── utils/                  # Utilities
└── defaults/               # Validation ranges
```

### Running Tests

```bash
# TODO: Add test instructions
pytest tests/digital_twin/agentic_mobility/
```

## Troubleshooting

### Issue: "GROQ_API_KEY not found"
**Solution**: Create `.env` file with your API key

### Issue: Geocoding fails
**Solution**: Check internet connection, Nominatim has rate limits (1 req/sec)

### Issue: Validation failures
**Solution**: System auto-retries with suggestions. After 2 retries, accepts parameters with warnings.

## Contributing

1. Follow existing code structure
2. Add type hints
3. Update prompts for new scenarios
4. Add tests for new features

## License

Part of the RADP (Radio Access Digital Platform) project.

## Contact

For issues or questions, contact the RADP development team.
