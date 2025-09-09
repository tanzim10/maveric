# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
App-specific API endpoints for RADP application management.
Provides REST endpoints for validating and managing different app types.
"""

from flask import request, jsonify
from typing import Dict, Any

from api_manager.validators.app_validator import AppValidatorFactory
from api_manager.validators.config_validator import AppConfigValidator
from api_manager.exceptions.validation_exception import ValidationException, FileValidationException


class AppEndpoints:
    """REST endpoints for app management and validation"""
    
    @staticmethod
    def validate_app_config():
        """
        POST /api/apps/validate
        Validate app-specific configuration parameters
        """
        try:
            # Get request data
            data = request.get_json()
            if not data:
                return jsonify({
                    "error": "Request body is required",
                    "error_type": "validation_error",
                    "status_code": 400
                }), 400
            
            # Get app type
            app_type = data.get("app_type")
            if not app_type:
                return jsonify({
                    "error": "app_type is required",
                    "error_type": "validation_error", 
                    "status_code": 400
                }), 400
            
            # Create and use app-specific validator
            validator = AppValidatorFactory.create_validator(app_type)
            validator.validate(data)
            
            return jsonify({
                "status": "success",
                "message": f"Configuration for {app_type} app is valid",
                "validated_fields": _get_validated_field_summary(data)
            }), 200
            
        except ValidationException as e:
            return jsonify(e.to_dict()), 400
        except Exception as e:
            return jsonify({
                "error": f"Internal server error: {str(e)}",
                "error_type": "server_error",
                "status_code": 500
            }), 500
    
    @staticmethod
    def validate_full_app_config():
        """
        POST /api/apps/validate-full
        Validate complete app configuration including files
        """
        try:
            # Get request data
            data = request.get_json()
            if not data:
                return jsonify({
                    "error": "Request body is required",
                    "error_type": "validation_error",
                    "status_code": 400
                }), 400
            
            # Get app type
            app_type = data.get("app_type")
            if not app_type:
                return jsonify({
                    "error": "app_type is required",
                    "error_type": "validation_error",
                    "status_code": 400
                }), 400
            
            # Create comprehensive validator
            validator = AppConfigValidator(app_type)
            validator.validate(data)
            
            return jsonify({
                "status": "success", 
                "message": f"Complete configuration for {app_type} app is valid",
                "validation_summary": _get_full_validation_summary(data)
            }), 200
            
        except (ValidationException, FileValidationException) as e:
            return jsonify(e.to_dict()), 400
        except Exception as e:
            return jsonify({
                "error": f"Internal server error: {str(e)}",
                "error_type": "server_error",
                "status_code": 500
            }), 500
    
    @staticmethod
    def get_app_schema():
        """
        GET /api/apps/schema/<app_type>
        Get validation schema for specific app type
        """
        app_type = request.view_args.get('app_type')
        
        try:
            # Validate app type exists
            validator = AppValidatorFactory.create_validator(app_type)
            
            # Get app-specific schema
            app_schema = validator.get_app_specific_schema()
            
            return jsonify({
                "app_type": app_type,
                "common_schema": validator.COMMON_APP_SCHEMA,
                "app_specific_schema": app_schema,
                "supported_file_formats": [".json", ".yaml", ".yml", ".csv"],
                "required_files": {
                    "topology": "topology.csv - Cell tower locations and configurations",
                    "config": "Optional configuration file for app-specific parameters"
                }
            }), 200
            
        except ValidationException as e:
            return jsonify(e.to_dict()), 400
        except Exception as e:
            return jsonify({
                "error": f"Internal server error: {str(e)}",
                "error_type": "server_error", 
                "status_code": 500
            }), 500
    
    @staticmethod
    def list_supported_apps():
        """
        GET /api/apps/supported
        List all supported app types and their descriptions
        """
        try:
            supported_apps = []
            app_descriptions = {
                "coverage_capacity_optimization": {
                    "name": "Coverage & Capacity Optimization (CCO)",
                    "description": "Optimize cell coverage and capacity through RF parameter tuning",
                    "key_parameters": ["lambda_", "weak_coverage_threshold", "over_coverage_threshold"]
                },
                "energy_savings": {
                    "name": "Energy Savings",
                    "description": "Reduce network energy consumption through intelligent cell sleep",
                    "key_parameters": ["power_reduction_target", "sleep_threshold_hours", "min_active_cells"]
                },
                "load_balancing": {
                    "name": "Load Balancing",
                    "description": "Balance traffic load across cells for optimal performance",
                    "key_parameters": ["target_load_threshold", "handover_margin_db", "load_balancing_algorithm"]
                },
                "mobility_robustness_optimization": {
                    "name": "Mobility Robustness Optimization (MRO)",
                    "description": "Optimize handover parameters to reduce failures and interruptions",
                    "key_parameters": ["handover_failure_threshold", "ping_pong_threshold", "rlf_timeout_ms"]
                },
                "example": {
                    "name": "Example Application",
                    "description": "Basic example demonstrating RADP usage patterns",
                    "key_parameters": ["incremental_update", "demo_mode"]
                }
            }
            
            for app_type in AppValidatorFactory.get_supported_app_types():
                app_info = app_descriptions.get(app_type, {
                    "name": app_type.title(),
                    "description": f"Application type: {app_type}",
                    "key_parameters": []
                })
                
                supported_apps.append({
                    "app_type": app_type,
                    **app_info
                })
            
            return jsonify({
                "supported_apps": supported_apps,
                "total_count": len(supported_apps)
            }), 200
            
        except Exception as e:
            return jsonify({
                "error": f"Internal server error: {str(e)}",
                "error_type": "server_error",
                "status_code": 500
            }), 500


def _get_validated_field_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of validated fields"""
    summary = {
        "common_fields": [],
        "app_specific_fields": [],
        "total_fields": 0
    }
    
    common_fields = [
        "app_type", "train_days", "test_day", "tick", 
        "container", "total_timesteps", "bdt_model_id"
    ]
    
    for field in common_fields:
        if field in data:
            summary["common_fields"].append(field)
            summary["total_fields"] += 1
    
    # App-specific field detection
    app_specific_keys = [
        "cco_params", "energy_params", "rl_params", "load_balance_params",
        "mro_params", "mobility_model_params", "example_params"
    ]
    
    for field in app_specific_keys:
        if field in data:
            summary["app_specific_fields"].append(field)
            summary["total_fields"] += 1
            
            # Count nested parameters
            if isinstance(data[field], dict):
                summary["total_fields"] += len(data[field])
    
    return summary


def _get_full_validation_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of full validation"""
    summary = {
        "validated_components": [],
        "file_validations": [],
        "parameter_count": 0
    }
    
    # Check what was validated
    if data.get("app_type"):
        summary["validated_components"].append("app_parameters")
    
    if data.get("config_path"):
        summary["validated_components"].append("configuration_file")
        summary["file_validations"].append("config")
    
    if data.get("topology_path"):
        summary["validated_components"].append("topology_file") 
        summary["file_validations"].append("topology")
    
    # Count total parameters
    for key, value in data.items():
        if isinstance(value, dict):
            summary["parameter_count"] += len(value)
        elif not key.endswith("_path"):
            summary["parameter_count"] += 1
    
    return summary