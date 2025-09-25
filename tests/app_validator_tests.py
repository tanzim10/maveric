#!/usr/bin/env python3
"""
Comprehensive tests for app-specific validation system.
Tests all application validators and configuration validation.
"""

import sys
import os
import tempfile
import json
import yaml
import pandas as pd
import unittest
from typing import Dict, Any
import logging

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../services'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../radp'))

from api_manager.validators.app_validator import (
    AppValidatorFactory, AppType, CCOAppValidator, EnergySavingsAppValidator,
    LoadBalancingAppValidator, MROAppValidator, ExampleAppValidator
)
from api_manager.validators.config_validator import (
    ConfigFileValidator, TopologyFileValidator, AppConfigValidator
)
from api_manager.exceptions.validation_exception import ValidationException, FileValidationException

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAppValidatorFactory(unittest.TestCase):
    """Test the app validator factory"""

    def test_create_valid_validators(self):
        """Test creating validators for all supported app types"""
        for app_type in AppType:
            validator = AppValidatorFactory.create_validator(app_type.value)
            self.assertIsNotNone(validator)
            logger.info(f"✅ Created validator for {app_type.value}")

    def test_invalid_app_type(self):
        """Test handling of invalid app types"""
        with self.assertRaises(ValidationException):
            AppValidatorFactory.create_validator("invalid_app_type")

    def test_get_supported_app_types(self):
        """Test getting list of supported app types"""
        supported_types = AppValidatorFactory.get_supported_app_types()
        self.assertEqual(len(supported_types), 5)
        self.assertIn("coverage_capacity_optimization", supported_types)


class TestCCOAppValidator(unittest.TestCase):
    """Test CCO (Coverage Capacity Optimization) validator"""

    def setUp(self):
        self.validator = CCOAppValidator()

    def test_valid_cco_config(self):
        """Test valid CCO configuration"""
        config = {
            "app_type": "coverage_capacity_optimization",
            "train_days": [0, 1, 2],
            "test_day": 3,
            "tick": 12,
            "total_timesteps": 50000,
            "cco_params": {
                "lambda_": 0.5,
                "weak_coverage_threshold": -110.0,
                "over_coverage_threshold": -5.0,
                "growth_rate": 1.5,
                "optimization_metric": "pixel"
            }
        }

        try:
            self.validator.validate(config)
            logger.info("✅ Valid CCO configuration passed validation")
        except ValidationException as e:
            self.fail(f"Valid CCO config failed validation: {e}")

    def test_invalid_lambda_bounds(self):
        """Test lambda parameter bounds validation"""
        # Test lambda = 0 (should fail - exclusive bounds)
        config = {
            "app_type": "coverage_capacity_optimization",
            "cco_params": {"lambda_": 0.0}
        }

        with self.assertRaises(ValidationException):
            self.validator.validate(config)

        # Test lambda = 1 (should fail - exclusive bounds)
        config["cco_params"]["lambda_"] = 1.0
        with self.assertRaises(ValidationException):
            self.validator.validate(config)

        # Test lambda = 1.5 (should fail - out of range)
        config["cco_params"]["lambda_"] = 1.5
        with self.assertRaises(ValidationException):
            self.validator.validate(config)

        logger.info("✅ Lambda bounds validation working correctly")

    def test_invalid_coverage_thresholds(self):
        """Test coverage threshold validation"""
        config = {
            "app_type": "coverage_capacity_optimization",
            "cco_params": {
                "weak_coverage_threshold": 10.0,  # Invalid - should be negative
                "over_coverage_threshold": -50.0   # Invalid - too low for SINR
            }
        }

        with self.assertRaises(ValidationException):
            self.validator.validate(config)


class TestEnergySavingsAppValidator(unittest.TestCase):
    """Test Energy Savings app validator"""

    def setUp(self):
        self.validator = EnergySavingsAppValidator()

    def test_valid_energy_config(self):
        """Test valid energy savings configuration"""
        config = {
            "app_type": "energy_savings",
            "total_timesteps": 48000,
            "energy_params": {
                "power_reduction_target": 0.25,  # 25% reduction
                "sleep_threshold_hours": 2,
                "min_active_cells": 10
            },
            "rl_params": {
                "learning_rate": 0.001,
                "discount_factor": 0.95,
                "epsilon": 0.1
            }
        }

        try:
            self.validator.validate(config)
            logger.info("✅ Valid Energy Savings configuration passed validation")
        except ValidationException as e:
            self.fail(f"Valid Energy Savings config failed validation: {e}")

    def test_invalid_power_reduction_target(self):
        """Test power reduction target validation"""
        config = {
            "app_type": "energy_savings",
            "energy_params": {
                "power_reduction_target": 0.75  # 75% - too aggressive
            }
        }

        with self.assertRaises(ValidationException):
            self.validator.validate(config)


class TestLoadBalancingAppValidator(unittest.TestCase):
    """Test Load Balancing app validator"""

    def setUp(self):
        self.validator = LoadBalancingAppValidator()

    def test_valid_load_balance_config(self):
        """Test valid load balancing configuration"""
        config = {
            "app_type": "load_balancing",
            "total_timesteps": 24000,
            "load_balance_params": {
                "target_load_threshold": 0.8,
                "handover_margin_db": 3.0,
                "load_balancing_algorithm": "least_loaded"
            }
        }

        try:
            self.validator.validate(config)
            logger.info("✅ Valid Load Balancing configuration passed validation")
        except ValidationException as e:
            self.fail(f"Valid Load Balancing config failed validation: {e}")

    def test_invalid_algorithm(self):
        """Test load balancing algorithm validation"""
        config = {
            "app_type": "load_balancing",
            "load_balance_params": {
                "load_balancing_algorithm": "invalid_algorithm"
            }
        }

        with self.assertRaises(ValidationException):
            self.validator.validate(config)


class TestMROAppValidator(unittest.TestCase):
    """Test MRO (Mobility Robustness Optimization) validator"""

    def setUp(self):
        self.validator = MROAppValidator()

    def test_valid_mro_config(self):
        """Test valid MRO configuration"""
        config = {
            "app_type": "mobility_robustness_optimization",
            "mro_params": {
                "handover_failure_threshold": 0.1,
                "ping_pong_threshold": 3,
                "rlf_timeout_ms": 1000
            },
            "mobility_model_params": {
                "velocity_kmh": 50.0,
                "direction_change_probability": 0.1,
                "path_loss_model": "urban_macro"
            }
        }

        try:
            self.validator.validate(config)
            logger.info("✅ Valid MRO configuration passed validation")
        except ValidationException as e:
            self.fail(f"Valid MRO config failed validation: {e}")

    def test_invalid_velocity(self):
        """Test velocity validation"""
        config = {
            "app_type": "mobility_robustness_optimization",
            "mobility_model_params": {
                "velocity_kmh": 300.0  # Too fast - unrealistic
            }
        }

        with self.assertRaises(ValidationException):
            self.validator.validate(config)


class TestCommonAppValidation(unittest.TestCase):
    """Test common app validation features"""

    def test_train_test_day_overlap(self):
        """Test that test day cannot be in training days"""
        validator = CCOAppValidator()

        config = {
            "app_type": "coverage_capacity_optimization",
            "train_days": [0, 1, 2, 3],
            "test_day": 2  # Overlaps with training days
        }

        with self.assertRaises(ValidationException) as context:
            validator.validate(config)

        self.assertIn("cannot be included in training days", str(context.exception))
        logger.info("✅ Train/test day overlap validation working correctly")

    def test_invalid_train_days(self):
        """Test training days validation"""
        validator = CCOAppValidator()

        # Test empty list
        config = {
            "app_type": "coverage_capacity_optimization",
            "train_days": []
        }

        with self.assertRaises(ValidationException):
            validator.validate(config)

        # Test duplicate days
        config["train_days"] = [0, 1, 2, 2]
        with self.assertRaises(ValidationException):
            validator.validate(config)

        # Test invalid day numbers
        config["train_days"] = [0, 1, 35]  # Day 35 is out of range
        with self.assertRaises(ValidationException):
            validator.validate(config)


class TestConfigFileValidator(unittest.TestCase):
    """Test configuration file validation"""

    def setUp(self):
        self.validator = ConfigFileValidator()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_valid_json_config(self):
        """Test valid JSON configuration file"""
        config_data = {
            "rf_config": {
                "frequency_bands": [1800, 2100],
                "tx_power_dbm": 43,
                "antenna_config": {"gain_dbi": 18}
            },
            "ue_mobility": {
                "mobility_models": ["random_walk"],
                "velocity_range": {"min": 1, "max": 50}
            }
        }

        config_path = os.path.join(self.temp_dir, "test_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        try:
            self.validator.validate({"config_path": config_path})
            logger.info("✅ Valid JSON configuration passed validation")
        except ValidationException as e:
            self.fail(f"Valid JSON config failed validation: {e}")

    def test_valid_yaml_config(self):
        """Test valid YAML configuration file"""
        config_data = {
            "cell_config": {
                "deployment_pattern": "hexagonal",
                "inter_site_distance_m": 500,
                "cell_count": 19
            }
        }

        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        try:
            self.validator.validate({"config_path": config_path})
            logger.info("✅ Valid YAML configuration passed validation")
        except ValidationException as e:
            self.fail(f"Valid YAML config failed validation: {e}")

    def test_invalid_frequency_bands(self):
        """Test invalid frequency band validation"""
        config_data = {
            "rf_config": {
                "frequency_bands": [100, 7000]  # Out of valid range
            }
        }

        config_path = os.path.join(self.temp_dir, "invalid_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        with self.assertRaises(FileValidationException):
            self.validator.validate({"config_path": config_path})

    def test_missing_config_file(self):
        """Test handling of missing configuration file"""
        with self.assertRaises(FileValidationException):
            self.validator.validate({"config_path": "/nonexistent/config.json"})

    def test_unsupported_format(self):
        """Test unsupported file format"""
        unsupported_path = os.path.join(self.temp_dir, "config.txt")
        with open(unsupported_path, 'w') as f:
            f.write("some config content")

        with self.assertRaises(FileValidationException):
            self.validator.validate({"config_path": unsupported_path})


class TestTopologyFileValidator(unittest.TestCase):
    """Test topology file validation"""

    def setUp(self):
        self.validator = TopologyFileValidator()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_valid_topology_file(self):
        """Test valid topology file"""
        topology_data = {
            'cell_id': ['cell_1', 'cell_2', 'cell_3'],
            'cell_lat': [35.6762, 35.6763, 35.6764],
            'cell_lon': [139.6503, 139.6504, 139.6505],
            'cell_az_deg': [0, 120, 240],
            'cell_carrier_freq_mhz': [2100, 2100, 1800],
            'cell_el_deg': [2, 3, 4],
            'cell_tx_power_dbm': [43, 43, 40]
        }

        df = pd.DataFrame(topology_data)
        topology_path = os.path.join(self.temp_dir, "topology.csv")
        df.to_csv(topology_path, index=False)

        try:
            self.validator.validate({"topology_path": topology_path})
            logger.info("✅ Valid topology file passed validation")
        except ValidationException as e:
            self.fail(f"Valid topology file failed validation: {e}")

    def test_invalid_coordinates(self):
        """Test invalid coordinate validation"""
        topology_data = {
            'cell_id': ['cell_1'],
            'cell_lat': [95.0],  # Invalid latitude
            'cell_lon': [200.0], # Invalid longitude
            'cell_az_deg': [0],
            'cell_carrier_freq_mhz': [2100]
        }

        df = pd.DataFrame(topology_data)
        topology_path = os.path.join(self.temp_dir, "invalid_topology.csv")
        df.to_csv(topology_path, index=False)

        with self.assertRaises(FileValidationException):
            self.validator.validate({"topology_path": topology_path})

    def test_duplicate_cell_ids(self):
        """Test duplicate cell ID validation"""
        topology_data = {
            'cell_id': ['cell_1', 'cell_1'],  # Duplicate
            'cell_lat': [35.6762, 35.6763],
            'cell_lon': [139.6503, 139.6504],
            'cell_az_deg': [0, 120],
            'cell_carrier_freq_mhz': [2100, 2100]
        }

        df = pd.DataFrame(topology_data)
        topology_path = os.path.join(self.temp_dir, "duplicate_topology.csv")
        df.to_csv(topology_path, index=False)

        with self.assertRaises(FileValidationException):
            self.validator.validate({"topology_path": topology_path})


def main():
    """Run all app validator tests"""
    logger.info("🧪 RADP App Validation Test Suite")
    logger.info("=" * 60)

    # Test suites
    test_classes = [
        TestAppValidatorFactory,
        TestCCOAppValidator,
        TestEnergySavingsAppValidator,
        TestLoadBalancingAppValidator,
        TestMROAppValidator,
        TestCommonAppValidation,
        TestConfigFileValidator,
        TestTopologyFileValidator
    ]

    total_tests = 0
    total_failures = 0

    for test_class in test_classes:
        logger.info(f"\n--- Running {test_class.__name__} ---")

        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        total_tests += result.testsRun
        total_failures += len(result.failures) + len(result.errors)

        if result.wasSuccessful():
            logger.info(f"✅ {test_class.__name__} - All tests passed")
        else:
            logger.error(f"❌ {test_class.__name__} - {len(result.failures + result.errors)} tests failed")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("APP VALIDATION TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {total_tests - total_failures}")
    logger.info(f"Failed: {total_failures}")

    if total_failures == 0:
        logger.info("✅ All app validation tests passed!")
        return 0
    else:
        logger.info(f"❌ {total_failures} app validation tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())