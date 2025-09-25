#!/usr/bin/env python3
"""
Integration test demonstrating the complete app validation workflow.
Tests real-world scenarios with actual app configurations.
"""

import sys
import os
import tempfile
import json
import yaml
import pandas as pd
import logging
from typing import Dict, Any

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../services'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../radp'))

from api_manager.validators.config_validator import AppConfigValidator
from api_manager.validators.app_validator import AppValidatorFactory
from api_manager.exceptions.validation_exception import ValidationException, FileValidationException

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_topology_file(temp_dir: str) -> str:
    """Create a sample topology file for testing"""
    topology_data = {
        'cell_id': [f'cell_{i}' for i in range(1, 8)],
        'cell_lat': [35.6762 + i*0.001 for i in range(7)],
        'cell_lon': [139.6503 + i*0.001 for i in range(7)],
        'cell_az_deg': [(i * 60) % 360 for i in range(7)],
        'cell_carrier_freq_mhz': [2100 if i % 2 == 0 else 1800 for i in range(7)],
        'cell_el_deg': [3 + (i % 3) for i in range(7)],
        'cell_tx_power_dbm': [43 if i % 2 == 0 else 40 for i in range(7)]
    }

    df = pd.DataFrame(topology_data)
    topology_path = os.path.join(temp_dir, "topology.csv")
    df.to_csv(topology_path, index=False)

    logger.info(f"✅ Created sample topology file with {len(df)} cells")
    return topology_path


def create_sample_config_file(temp_dir: str, config_type: str = "cco") -> str:
    """Create a sample configuration file for testing"""
    if config_type == "cco":
        config_data = {
            "rf_config": {
                "frequency_bands": [1800, 2100, 2600],
                "tx_power_dbm": 43,
                "antenna_config": {
                    "gain_dbi": 18,
                    "pattern_type": "omni"
                }
            },
            "ue_mobility": {
                "mobility_models": ["random_walk", "gauss_markov"],
                "velocity_range": {"min": 1, "max": 120}
            },
            "cell_config": {
                "deployment_pattern": "hexagonal",
                "inter_site_distance_m": 500,
                "cell_count": 7
            },
            "cco_specific": {
                "optimization_intervals_s": 300,
                "convergence_threshold": 0.01
            }
        }
    elif config_type == "energy":
        config_data = {
            "energy_optimization": {
                "optimization_window_hours": 24,
                "traffic_patterns": ["peak", "off_peak", "weekend"],
                "sleep_decision_algorithm": "ml_based"
            },
            "rl_training": {
                "episodes": 1000,
                "replay_buffer_size": 10000,
                "target_network_update_freq": 100
            }
        }
    else:
        config_data = {"generic_config": {"enabled": True}}

    config_path = os.path.join(temp_dir, f"{config_type}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    logger.info(f"✅ Created sample {config_type} configuration file")
    return config_path


def test_cco_app_integration():
    """Test complete CCO app validation workflow"""
    logger.info("🧪 Testing CCO App Integration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create supporting files
        topology_path = create_sample_topology_file(temp_dir)
        config_path = create_sample_config_file(temp_dir, "cco")

        # Create comprehensive CCO configuration
        cco_config = {
            "app_type": "coverage_capacity_optimization",
            "train_days": [0, 1, 2, 3, 4],
            "test_day": 5,
            "tick": 15,
            "container": "radp_dev-training-1",
            "total_timesteps": 75000,
            "bdt_model_id": "cco_integration_test_v1",
            "topology_path": topology_path,
            "config_path": config_path,
            "cco_params": {
                "lambda_": 0.6,
                "weak_coverage_threshold": -105.0,
                "over_coverage_threshold": -3.0,
                "growth_rate": 1.2,
                "optimization_metric": "cell"
            }
        }

        # Test app-specific validation first
        try:
            app_validator = AppValidatorFactory.create_validator("coverage_capacity_optimization")
            app_validator.validate(cco_config)
            logger.info("✅ CCO app-specific validation passed")
        except ValidationException as e:
            logger.error(f"❌ CCO app validation failed: {e}")
            return False

        # Test full configuration validation
        try:
            full_validator = AppConfigValidator("coverage_capacity_optimization")
            full_validator.validate(cco_config)
            logger.info("✅ CCO full configuration validation passed")
            return True
        except (ValidationException, FileValidationException) as e:
            logger.error(f"❌ CCO full validation failed: {e}")
            return False


def test_energy_savings_app_integration():
    """Test complete Energy Savings app validation workflow"""
    logger.info("🧪 Testing Energy Savings App Integration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create supporting files
        topology_path = create_sample_topology_file(temp_dir)
        config_path = create_sample_config_file(temp_dir, "energy")

        # Create comprehensive Energy Savings configuration
        energy_config = {
            "app_type": "energy_savings",
            "train_days": [0, 1, 2],
            "test_day": 3,
            "tick": 8,
            "total_timesteps": 50000,
            "bdt_model_id": "energy_savings_integration_test_v1",
            "topology_path": topology_path,
            "config_path": config_path,
            "energy_params": {
                "power_reduction_target": 0.30,  # 30% reduction
                "sleep_threshold_hours": 3,
                "min_active_cells": 2
            },
            "rl_params": {
                "learning_rate": 0.0005,
                "discount_factor": 0.99,
                "epsilon": 0.15
            }
        }

        # Test validation
        try:
            full_validator = AppConfigValidator("energy_savings")
            full_validator.validate(energy_config)
            logger.info("✅ Energy Savings full configuration validation passed")
            return True
        except (ValidationException, FileValidationException) as e:
            logger.error(f"❌ Energy Savings validation failed: {e}")
            return False


def test_multi_app_validation_workflow():
    """Test validation workflow for multiple app types"""
    logger.info("🧪 Testing Multi-App Validation Workflow...")

    test_configs = [
        {
            "name": "Load Balancing",
            "config": {
                "app_type": "load_balancing",
                "train_days": [0, 1],
                "test_day": 2,
                "total_timesteps": 30000,
                "load_balance_params": {
                    "target_load_threshold": 0.85,
                    "handover_margin_db": 2.5,
                    "load_balancing_algorithm": "proportional"
                }
            }
        },
        {
            "name": "MRO",
            "config": {
                "app_type": "mobility_robustness_optimization",
                "mro_params": {
                    "handover_failure_threshold": 0.05,
                    "ping_pong_threshold": 2,
                    "rlf_timeout_ms": 1500
                },
                "mobility_model_params": {
                    "velocity_kmh": 60.0,
                    "direction_change_probability": 0.2,
                    "path_loss_model": "urban_macro"
                }
            }
        },
        {
            "name": "Example App",
            "config": {
                "app_type": "example",
                "tick": 10,
                "example_params": {
                    "incremental_update": True,
                    "demo_mode": False
                }
            }
        }
    ]

    results = []

    for test_case in test_configs:
        try:
            validator = AppValidatorFactory.create_validator(test_case["config"]["app_type"])
            validator.validate(test_case["config"])
            logger.info(f"✅ {test_case['name']} validation passed")
            results.append(True)
        except ValidationException as e:
            logger.error(f"❌ {test_case['name']} validation failed: {e}")
            results.append(False)

    return all(results)


def test_validation_error_scenarios():
    """Test various error scenarios to ensure proper error handling"""
    logger.info("🧪 Testing Validation Error Scenarios...")

    error_scenarios = [
        {
            "name": "Invalid App Type",
            "config": {"app_type": "nonexistent_app"},
            "should_fail": True
        },
        {
            "name": "Test Day in Training Days",
            "config": {
                "app_type": "coverage_capacity_optimization",
                "train_days": [0, 1, 2, 3],
                "test_day": 2  # Overlaps with training
            },
            "should_fail": True
        },
        {
            "name": "Invalid Lambda Parameter",
            "config": {
                "app_type": "coverage_capacity_optimization",
                "cco_params": {"lambda_": 1.5}  # Out of range
            },
            "should_fail": True
        },
        {
            "name": "Invalid Velocity",
            "config": {
                "app_type": "mobility_robustness_optimization",
                "mobility_model_params": {"velocity_kmh": 500.0}  # Too fast
            },
            "should_fail": True
        }
    ]

    results = []

    for scenario in error_scenarios:
        try:
            if scenario["config"].get("app_type") == "nonexistent_app":
                # Special handling for factory test
                AppValidatorFactory.create_validator("nonexistent_app")
            else:
                validator = AppValidatorFactory.create_validator(scenario["config"]["app_type"])
                validator.validate(scenario["config"])

            if scenario["should_fail"]:
                logger.error(f"❌ {scenario['name']} should have failed but passed")
                results.append(False)
            else:
                logger.info(f"✅ {scenario['name']} passed as expected")
                results.append(True)

        except (ValidationException, ValueError) as e:
            if scenario["should_fail"]:
                logger.info(f"✅ {scenario['name']} correctly failed: {type(e).__name__}")
                results.append(True)
            else:
                logger.error(f"❌ {scenario['name']} should have passed but failed: {e}")
                results.append(False)

    return all(results)


def test_config_file_validation():
    """Test configuration file validation with different formats"""
    logger.info("🧪 Testing Configuration File Validation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test JSON config
        json_config = create_sample_config_file(temp_dir, "cco")
        json_test_data = {
            "app_type": "coverage_capacity_optimization",
            "config_path": json_config
        }

        # Test YAML config
        yaml_data = {
            "rf_config": {"frequency_bands": [2100], "tx_power_dbm": 43},
            "optimization": {"algorithm": "genetic", "generations": 100}
        }
        yaml_config = os.path.join(temp_dir, "test_config.yaml")
        with open(yaml_config, 'w') as f:
            yaml.dump(yaml_data, f)

        yaml_test_data = {
            "app_type": "energy_savings",
            "config_path": yaml_config
        }

        test_cases = [
            ("JSON Config", json_test_data),
            ("YAML Config", yaml_test_data)
        ]

        results = []

        for name, test_data in test_cases:
            try:
                validator = AppConfigValidator(test_data["app_type"])
                validator.validate(test_data)
                logger.info(f"✅ {name} validation passed")
                results.append(True)
            except (ValidationException, FileValidationException) as e:
                logger.error(f"❌ {name} validation failed: {e}")
                results.append(False)

        return all(results)


def main():
    """Run comprehensive integration tests"""
    logger.info("🚀 RADP App Validation Integration Test Suite")
    logger.info("=" * 70)

    test_functions = [
        ("CCO App Integration", test_cco_app_integration),
        ("Energy Savings App Integration", test_energy_savings_app_integration),
        ("Multi-App Validation Workflow", test_multi_app_validation_workflow),
        ("Validation Error Scenarios", test_validation_error_scenarios),
        ("Config File Validation", test_config_file_validation)
    ]

    results = []

    for test_name, test_func in test_functions:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.exception(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name:<35} {status}")

    logger.info(f"\nOverall: {passed}/{total} integration tests passed")

    if passed == total:
        logger.info("✅ All integration tests passed!")
        logger.info("🎉 App-specific validation system is working correctly!")
        return 0
    else:
        logger.error("❌ Some integration tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())