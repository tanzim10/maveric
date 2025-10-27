# Agentic Mobility Generation

Transform RADP's mobility generator from **parameter-driven JSON** to **natural language queries** using LLM-powered workflows.

## Overview

**Before:** Complex JSON configuration with 15+ parameters
**After:** `"Generate 100 UEs in urban Tokyo during morning rush hour"`

### What It Does

Converts natural language → Valid RADP mobility parameters → Mobility simulation DataFrame

```python
from radp.digital_twin.agentic_mobility.api import generate_mobility_params

result = generate_mobility_params("Generate 100 UEs in urban Tokyo")
# Returns: RADP-compatible JSON ready for UETracksGenerator
```

---

## Key Features

| Feature                                    | Description                                                                      |
| ------------------------------------------ | -------------------------------------------------------------------------------- |
| **🗣️ Natural Language Parsing**            | Extract scenario, location, UE count from plain text                             |
| **🌍 Auto-Geocoding**                      | Location names → lat/lon bounds (cached, 15min TTL)                              |
| **🧠 Context-Aware Generation**            | LLM infers realistic distributions from context (e.g., "rush hour" → high car %) |
| **📊 Distribution Source Tracking** ✨ NEW | Tracks if distribution was "parsed" from query or "predicted" by LLM             |
| **🔄 Self-Correction**                     | Auto-validates + retries with LLM suggestions (max 2 retries)                    |
| **⚡ Parallel Execution**                  | Location resolver + parameter agent run simultaneously (~35% faster)             |
| **✅ End-to-End Integration**              | Complete pipeline: NL → DataFrame (not just NL → JSON)                           |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r radp/digital_twin/requirements.txt
```

### 2. Configure API Key

```bash
# Create .env file in project root
cp radp/digital_twin/agentic_mobility/.env.example .env

# Edit .env and add:
GROQ_API_KEY=your_api_key_here
```

### 3. Basic Usage

```python
from radp.digital_twin.agentic_mobility.api import generate_mobility_params

# Generate parameters
result = generate_mobility_params(
    "Generate 100 UEs in urban Tokyo during morning rush hour"
)

print(result["status"])        # "success"
print(result["radp_params"])   # RADP JSON format
print(result["metadata"])      # Query intent + retry count + distribution source
```

### 4. End-to-End: Natural Language → DataFrame

```python
from radp.digital_twin.agentic_mobility.integration import AgenticMobilityIntegration

# Complete pipeline in one call
df, metadata = AgenticMobilityIntegration.generate_from_natural_language(
    "Generate 100 UEs in urban Tokyo"
)

print(df.head())  # pandas DataFrame with mobility tracks
# Columns: mock_ue_id, lon, lat, tick
```

---

## Examples

### Implicit Distribution (LLM-Predicted)

```python
query = "Generate 100 UEs in urban Tokyo during morning rush hour"
result = generate_mobility_params(query)

# Result includes distribution source tracking:
result["metadata"]["query_intent"]["ue_distribution"]
# {
#   "source": "predicted",  ← LLM generated this
#   "distribution": {
#     "stationary": 0.05,
#     "pedestrian": 0.25,
#     "cyclist": 0.15,
#     "car": 0.55  # High car % for rush hour!
#   }
# }
```

### Explicit Distribution (Parsed)

```python
query = "Create 50 UEs in Chicago with 60% pedestrians, 30% cars, 10% cyclists"
result = generate_mobility_params(query)

# Distribution source:
result["metadata"]["query_intent"]["ue_distribution"]
# {
#   "source": "parsed",  ← Extracted from user query
#   "distribution": {
#     "pedestrian": 0.6,
#     "car": 0.3,
#     "cyclist": 0.1
#   }
# }
```

### Context-Aware num_ues Guessing

```python
# No UE count specified - LLM guesses based on context
query1 = "Generate UEs in downtown Manhattan"
# → num_ues ≈ 200-300 (dense urban)

query2 = "Generate UEs in rural Kansas"
# → num_ues ≈ 30-50 (sparse rural)
```

---

## Architecture

```
Natural Language Query
         ↓
   [1.1] Parser (LLM) → QueryIntent
         ↓
   [1.2] Location Resolver (Geocoding) ∥ [1.3] Parameter Agent (LLM)
         ↓                                      ↓
         └──────────────┬──────────────────────┘
                        ↓
   [1.4] Validator → [1.5] Suggestion (if failed) → Retry
         ↓
   [2.1] RADP Formatter → JSON
         ↓
   [3] Integration Layer → UETracksGenerator → DataFrame
```

**Parallel Execution:** Components 1.2 and 1.3 run simultaneously
**Self-Correction:** Max 2 retries with LLM-generated suggestions
**Distribution Tracking:** Source field shows "parsed" or "predicted"

---

## Output Structure

```python
{
  "status": "success" | "success_with_warnings" | "failed",
  "radp_params": {
    "ue_tracks_generation": {
      "params": {
        "num_ticks": 50,
        "num_batches": 1,
        "simulation_duration": 3600,
        "simulation_time_interval_seconds": 0.01,
        "ue_class_distribution": {...},
        "lat_lon_boundaries": {...},
        "gauss_markov_params": {...}
      }
    }
  },
  "metadata": {
    "retry_count": 0,
    "query_intent": {
      "scenario_type": "urban",
      "location": "Tokyo",
      "num_ues": 100,
      "num_ticks": 50,
      "ue_distribution": {
        "source": "parsed" | "predicted",  ← NEW: Source tracking
        "distribution": {...}
      },
      "raw_query": "..."
    },
    "validation_warnings": null | [...]
  }
}
```

---

## Technology Stack

- **LangGraph**: Workflow orchestration with parallel execution & conditional routing
- **LangChain**: LLM chain implementations
- **Groq API**: LLM provider (llama-3.1-70b-versatile)
- **Pydantic**: Structured output validation
- **Geopy/Nominatim**: Free geocoding service (1 req/sec limit, cached)

---

## Performance

| Metric                          | Value                    |
| ------------------------------- | ------------------------ |
| **Query Parsing**               | ~2-3s                    |
| **Location Resolution**         | ~0.5-1s (cached: <0.01s) |
| **Parameter Generation**        | ~2-3s                    |
| **Validation**                  | <0.1s                    |
| **Total (no retry)**            | **~5-7s**                |
| **Total (with retry)**          | ~10-15s                  |
| **End-to-End (NL → DataFrame)** | ~7-12s                   |

---

## File Structure

```
radp/digital_twin/agentic_mobility/
├── api.py                      # Public API
├── integration.py              # End-to-end integration
├── models/                     # Pydantic models
│   ├── query_intent.py         # QueryIntent + DistributionSource enum
│   ├── generation_params.py    # GenParams
│   └── state.py                # LangGraph state
├── chains/                     # LangChain implementations
│   ├── parser_chain.py         # Component 1.1
│   ├── parameter_chain.py      # Component 1.3
│   ├── validation_chain.py     # Component 1.4
│   └── suggestion_chain.py     # Component 1.5
├── nodes/                      # LangGraph nodes
│   ├── query_parser.py
│   ├── location_resolver.py
│   ├── parameter_agent.py
│   ├── validation.py
│   └── suggestion.py
├── graph/                      # LangGraph workflow
│   └── workflow.py
├── formatters/
│   └── radp_formatter.py       # Component 2.1
├── utils/
│   ├── llm_client.py           # Groq API wrapper
│   ├── geocoding.py            # Geopy wrapper
│   └── validators.py           # Validation functions
├── prompts/                    # LLM prompts
├── tests/
│   └── test_integration.py
└── examples/
    ├── basic_usage.py
    └── end_to_end_example.py
```

---

## Advanced Usage

### Use with Existing RADP Code

```python
from radp.digital_twin.agentic_mobility.api import generate_mobility_params
from radp.digital_twin.mobility.ue_tracks_params import UETracksGenerationParams

# Generate parameters
result = generate_mobility_params("Generate 100 UEs in Tokyo")

# Use with existing RADP
params = UETracksGenerationParams(result["radp_params"])
# Continue with UETracksGenerator...
```

### Handle Warnings

```python
result = generate_mobility_params("Generate 1000 UEs on highway with pedestrians")

if result["status"] == "success_with_warnings":
    print("Warnings:", result["metadata"]["validation_warnings"])
    # Warnings: ["Highway scenarios typically should not include pedestrians"]
```

---

## Limitations

- **LLM Dependency**: Requires Groq API access
- **Geocoding Rate Limit**: Nominatim has 1 req/sec (mitigated by caching)
- **Single Mobility Model**: Only GaussMarkov (Level 2: multi-model support)
- **POC Behavior**: Accepts params after max retries (even with warnings)

---

## Documentation

- **IMPLEMENTATION_COMPLETE.md**: Full implementation details + parameter generation deep dive
- **IMPLEMENTED_ARCHITECTURE.md**: Visual architecture diagram
- **poc_plan.md**: Original POC plan (Level 1 scope)

---

## Support

**Examples**: `radp/digital_twin/agentic_mobility/examples/`
**Tests**: `radp/digital_twin/agentic_mobility/tests/`
**Configuration**: `.env.example` for required environment variables

---

**Last Updated**: 2025-10-27
