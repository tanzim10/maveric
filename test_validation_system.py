#!/usr/bin/env python3
"""
Test script to demonstrate the comprehensive validation system for RADP APIs.
This script tests both valid and invalid requests to show how validation works.
"""

import tempfile
import os
import json
from io import StringIO

# Import validation components
from services.api_manager.validators.training_validator import TrainingRequestValidator
from services.api_manager.validators.simulation_validator import SimulationRequestValidator
from services.api_manager.exceptions.validation_exception import ValidationException, FileValidationException


def test_training_validation():
    """Test training request validation"""
    print("🧪 Testing Training Request Validation...")
    
    validator = TrainingRequestValidator()
    
    # Test 1: Valid request
    try:
        valid_request = {
            "model_id": "test_model_123",
            "model_update": False,
            "params": {
                "maxiter": 100,
                "lr": 0.05,
                "stopping_threshold": 0.0001
            }
        }
        
        # Mock the model existence check
        import unittest.mock
        with unittest.mock.patch.object(validator, '_validate_model_id_availability'):
            validator.validate(valid_request)
        print("✅ Valid training request passed validation")
        
    except Exception as e:
        print(f"❌ Valid request failed: {e}")
    
    # Test 2: Invalid request - missing model_id
    try:
        invalid_request = {
            "model_update": False,
            "params": {}
        }
        validator.validate(invalid_request)
        print("❌ Invalid request should have failed but didn't")
        
    except ValidationException as e:
        print(f"✅ Invalid request correctly rejected: {e.message}")
        print(f"   Validation errors: {e.validation_errors}")
    
    # Test 3: Invalid training parameters
    try:
        invalid_params_request = {
            "model_id": "test_model",
            "params": {
                "maxiter": -5,  # Invalid: negative
                "lr": 2.0,      # Invalid: > 1.0
            }
        }
        
        with unittest.mock.patch.object(validator, '_validate_model_id_availability'):
            validator.validate(invalid_params_request)
        print("❌ Invalid params should have failed but didn't")
        
    except ValidationException as e:
        print(f"✅ Invalid training params correctly rejected: {e.message}")
        print(f"   Validation errors: {e.validation_errors}")


def test_simulation_validation():
    """Test simulation request validation"""
    print("\n🧪 Testing Simulation Request Validation...")
    
    validator = SimulationRequestValidator()
    
    # Test 1: Valid request with UE tracks generation
    try:
        valid_request = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {
                "ue_tracks_generation": {
                    "ue_class_distribution": {
                        "pedestrian": {
                            "count": 10,
                            "velocity": 1.5,
                            "velocity_variance": 0.5
                        }
                    },
                    "lat_lon_boundaries": {
                        "min_lat": 35.0,
                        "max_lat": 36.0,
                        "min_lon": 139.0,
                        "max_lon": 140.0
                    },
                    "gauss_markov_params": {
                        "alpha": 0.5,
                        "variance": 0.8
                    }
                }
            },
            "rf_prediction": {
                "model_id": "test_model"
            }
        }
        
        # Mock the model existence check
        import unittest.mock
        with unittest.mock.patch.object(validator, '_validate_model_exists'):
            validator.validate(valid_request)
        print("✅ Valid simulation request passed validation")
        
    except Exception as e:
        print(f"❌ Valid request failed: {e}")
    
    # Test 2: Invalid request - missing required fields
    try:
        invalid_request = {
            "simulation_time_interval_seconds": 0.01
            # Missing ue_tracks and rf_prediction
        }
        validator.validate(invalid_request)
        print("❌ Invalid request should have failed but didn't")
        
    except ValidationException as e:
        print(f"✅ Missing fields correctly rejected: {e.message}")
        print(f"   Validation errors: {e.validation_errors}")
    
    # Test 3: Invalid UE class
    try:
        invalid_ue_class_request = {
            "simulation_time_interval_seconds": 0.01,
            "ue_tracks": {
                "ue_tracks_generation": {
                    "ue_class_distribution": {
                        "invalid_class": {  # Invalid UE class
                            "count": 10,
                            "velocity": 1.0,
                            "velocity_variance": 0.5
                        }
                    },
                    "lat_lon_boundaries": {
                        "min_lat": 35.0, "max_lat": 36.0,
                        "min_lon": 139.0, "max_lon": 140.0
                    },
                    "gauss_markov_params": {
                        "alpha": 0.5,
                        "variance": 0.8
                    }
                }
            },
            "rf_prediction": {"model_id": "test"}
        }
        
        validator.validate(invalid_ue_class_request)
        print("❌ Invalid UE class should have failed but didn't")
        
    except ValidationException as e:
        print(f"✅ Invalid UE class correctly rejected: {e.message}")


def test_file_validation():
    """Test file validation capabilities"""
    print("\n🧪 Testing File Validation...")
    
    validator = TrainingRequestValidator()
    
    # Create temporary test files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ue_file:
        ue_file.write("cell_id,avg_rsrp,lon,lat,cell_el_deg\n")
        ue_file.write("cell_1,-80,139.699,35.644,5\n")
        ue_file.write("cell_2,-75,139.700,35.645,3\n")
        ue_file_path = ue_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as topo_file:
        topo_file.write("cell_lat,cell_lon,cell_id,cell_az_deg,cell_carrier_freq_mhz\n")
        topo_file.write("35.690,139.691,cell_1,0,2100\n")
        topo_file.write("35.691,139.692,cell_2,120,2100\n")
        topo_file_path = topo_file.name
    
    try:
        # Test valid files
        files = {
            "ue_training_data_file_path": ue_file_path,
            "topology_file_path": topo_file_path
        }
        
        validator.validate_training_files(files)
        print("✅ Valid training files passed validation")
        
    except Exception as e:
        print(f"❌ Valid files failed: {e}")
    
    finally:
        os.unlink(ue_file_path)
        os.unlink(topo_file_path)
    
    # Test invalid file - missing columns
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as bad_file:
        bad_file.write("cell_id,avg_rsrp\n")  # Missing required columns
        bad_file.write("cell_1,-80\n")
        bad_file_path = bad_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as topo_file:
        topo_file.write("cell_lat,cell_lon,cell_id,cell_az_deg,cell_carrier_freq_mhz\n")
        topo_file.write("35.690,139.691,cell_1,0,2100\n")
        topo_file_path = topo_file.name
    
    try:
        files = {
            "ue_training_data_file_path": bad_file_path,
            "topology_file_path": topo_file_path
        }
        
        validator.validate_training_files(files)
        print("❌ Invalid file should have failed but didn't")
        
    except FileValidationException as e:
        print(f"✅ Invalid file correctly rejected: {e.message}")
        
    finally:
        os.unlink(bad_file_path)
        os.unlink(topo_file_path)


def main():
    """Run all validation tests"""
    print("🚀 RADP API Validation System Test Suite")
    print("=" * 50)
    
    test_training_validation()
    test_simulation_validation()
    test_file_validation()
    
    print("\n" + "=" * 50)
    print("✅ Validation system testing completed!")
    print("\nThe comprehensive validation system includes:")
    print("• Request structure validation with detailed field-level errors")
    print("• File format and content validation with security checks")
    print("• Cross-validation between related data (e.g., cells in training data vs topology)")
    print("• Range validation for cellular network parameters (RSRP, frequencies, coordinates)")
    print("• Model existence validation for RF prediction")
    print("• Performance constraint validation (UE counts, simulation sizes)")


if __name__ == "__main__":
    main()