# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from enum import Enum

from api_manager.validators.base_validator import BaseValidator, SchemaValidator
from api_manager.exceptions.validation_exception import ValidationException


class AppType(Enum):
    """Supported RADP application types"""
    CCO = "coverage_capacity_optimization"
    ENERGY_SAVINGS = "energy_savings" 
    LOAD_BALANCING = "load_balancing"
    MRO = "mobility_robustness_optimization"
    EXAMPLE = "example"


class BaseAppValidator(BaseValidator):
    """Base validator for RADP applications"""
    
    # Common app configuration schema
    COMMON_APP_SCHEMA = {
        "app_type": {
            "required": True,
            "type": str,
            "enum": [app.value for app in AppType]
        },
        "train_days": {
            "required": False,
            "type": list,
            "validator": lambda val, field: _validate_day_list(val, field)
        },
        "test_day": {
            "required": False,
            "type": int,
            "min": 0,
            "max": 30  # Reasonable upper bound for simulation days
        },
        "tick": {
            "required": False,
            "type": int,
            "min": 0,
            "max": 23  # Hours in a day
        },
        "container": {
            "required": False,
            "type": str,
            "pattern": r"^[a-zA-Z0-9_-]+$"
        },
        "total_timesteps": {
            "required": False,
            "type": int,
            "min": 1000,
            "max": 1000000  # Reasonable RL training bounds
        },
        "bdt_model_id": {
            "required": False,
            "type": str,
            "min_length": 1,
            "max_length": 100,
            "pattern": r"^[a-zA-Z0-9_-]+$"
        }
    }
    
    def __init__(self):
        self.common_validator = SchemaValidator(self.COMMON_APP_SCHEMA)
    
    @abstractmethod
    def get_app_specific_schema(self) -> Dict[str, Any]:
        """Return app-specific validation schema"""
        pass
    
    def validate(self, data: Dict[str, Any]) -> None:
        """Validate common app parameters and delegate to app-specific validation"""
        # Validate common parameters
        self.common_validator.validate(data)
        
        # Validate app-specific parameters
        app_specific_schema = self.get_app_specific_schema()
        if app_specific_schema:
            app_validator = SchemaValidator(app_specific_schema)
            app_validator.validate(data)
        
        # Custom cross-field validation
        self._validate_cross_field_constraints(data)
    
    def _validate_cross_field_constraints(self, data: Dict[str, Any]) -> None:
        """Validate constraints that span multiple fields"""
        train_days = data.get("train_days", [])
        test_day = data.get("test_day")
        
        # Ensure test day is not in training days
        if test_day is not None and test_day in train_days:
            raise ValidationException(
                f"Test day {test_day} cannot be included in training days {train_days}",
                field="test_day"
            )


def _validate_day_list(day_list: List[int], field_name: str) -> None:
    """Validate list of training days"""
    if not isinstance(day_list, list):
        raise ValidationException(f"Field '{field_name}' must be a list", field=field_name)
    
    if len(day_list) == 0:
        raise ValidationException(f"Field '{field_name}' cannot be empty", field=field_name)
    
    for day in day_list:
        if not isinstance(day, int):
            raise ValidationException(f"All values in '{field_name}' must be integers", field=field_name)
        if day < 0 or day > 30:
            raise ValidationException(f"Day values in '{field_name}' must be between 0 and 30", field=field_name)
    
    # Check for duplicates
    if len(set(day_list)) != len(day_list):
        raise ValidationException(f"Field '{field_name}' cannot contain duplicate days", field=field_name)


class CCOAppValidator(BaseAppValidator):
    """Validator for Coverage and Capacity Optimization applications"""
    
    def get_app_specific_schema(self) -> Dict[str, Any]:
        return {
            "cco_params": {
                "required": False,
                "type": dict,
                "validator": lambda val, field: self._validate_cco_params(val, field)
            }
        }
    
    def _validate_cco_params(self, params: Dict[str, Any], field_name: str) -> None:
        """Validate CCO-specific parameters"""
        cco_schema = {
            "lambda_": {
                "required": False,
                "type": float,
                "min": 0.0,
                "max": 1.0,
                "validator": lambda val, field: self._validate_lambda_exclusive_bounds(val, field)
            },
            "weak_coverage_threshold": {
                "required": False,
                "type": float,
                "min": -150.0,  # Reasonable RSRP lower bound
                "max": 0.0      # RSRP upper bound
            },
            "over_coverage_threshold": {
                "required": False,
                "type": float,
                "min": -20.0,   # Reasonable SINR lower bound
                "max": 40.0     # Reasonable SINR upper bound
            },
            "growth_rate": {
                "required": False,
                "type": float,
                "min": 0.1,
                "max": 10.0
            },
            "optimization_metric": {
                "required": False,
                "type": str,
                "enum": ["pixel", "cell"]
            }
        }
        
        validator = SchemaValidator(cco_schema)
        validator.validate(params)
    
    def _validate_lambda_exclusive_bounds(self, value: float, field_name: str) -> None:
        """Validate lambda parameter with exclusive bounds (0 < lambda < 1)"""
        if value <= 0.0 or value >= 1.0:
            raise ValidationException(
                f"Field '{field_name}' must be between 0 and 1 (exclusive)",
                field=field_name
            )


class EnergySavingsAppValidator(BaseAppValidator):
    """Validator for Energy Savings applications"""
    
    def get_app_specific_schema(self) -> Dict[str, Any]:
        return {
            "energy_params": {
                "required": False,
                "type": dict,
                "validator": lambda val, field: self._validate_energy_params(val, field)
            },
            "rl_params": {
                "required": False,
                "type": dict,
                "validator": lambda val, field: self._validate_rl_params(val, field)
            }
        }
    
    def _validate_energy_params(self, params: Dict[str, Any], field_name: str) -> None:
        """Validate energy savings specific parameters"""
        energy_schema = {
            "power_reduction_target": {
                "required": False,
                "type": float,
                "min": 0.01,  # 1% minimum
                "max": 0.50   # 50% maximum realistic reduction
            },
            "sleep_threshold_hours": {
                "required": False,
                "type": int,
                "min": 1,
                "max": 24
            },
            "min_active_cells": {
                "required": False,
                "type": int,
                "min": 1,
                "max": 1000  # Reasonable upper bound
            }
        }
        
        validator = SchemaValidator(energy_schema)
        validator.validate(params)
    
    def _validate_rl_params(self, params: Dict[str, Any], field_name: str) -> None:
        """Validate RL training parameters"""
        rl_schema = {
            "learning_rate": {
                "required": False,
                "type": float,
                "min": 1e-6,
                "max": 1.0
            },
            "discount_factor": {
                "required": False,
                "type": float,
                "min": 0.0,
                "max": 1.0
            },
            "epsilon": {
                "required": False,
                "type": float,
                "min": 0.0,
                "max": 1.0
            }
        }
        
        validator = SchemaValidator(rl_schema)
        validator.validate(params)


class LoadBalancingAppValidator(BaseAppValidator):
    """Validator for Load Balancing applications"""
    
    def get_app_specific_schema(self) -> Dict[str, Any]:
        return {
            "load_balance_params": {
                "required": False,
                "type": dict,
                "validator": lambda val, field: self._validate_load_balance_params(val, field)
            }
        }
    
    def _validate_load_balance_params(self, params: Dict[str, Any], field_name: str) -> None:
        """Validate load balancing specific parameters"""
        lb_schema = {
            "target_load_threshold": {
                "required": False,
                "type": float,
                "min": 0.1,   # 10% minimum load
                "max": 1.0    # 100% maximum load
            },
            "handover_margin_db": {
                "required": False,
                "type": float,
                "min": 0.0,
                "max": 10.0  # Reasonable handover margin
            },
            "load_balancing_algorithm": {
                "required": False,
                "type": str,
                "enum": ["round_robin", "least_loaded", "proportional"]
            }
        }
        
        validator = SchemaValidator(lb_schema)
        validator.validate(params)


class MROAppValidator(BaseAppValidator):
    """Validator for Mobility Robustness Optimization applications"""
    
    def get_app_specific_schema(self) -> Dict[str, Any]:
        return {
            "mro_params": {
                "required": False,
                "type": dict,
                "validator": lambda val, field: self._validate_mro_params(val, field)
            },
            "mobility_model_params": {
                "required": False,
                "type": dict,
                "validator": lambda val, field: self._validate_mobility_model_params(val, field)
            }
        }
    
    def _validate_mro_params(self, params: Dict[str, Any], field_name: str) -> None:
        """Validate MRO specific parameters"""
        mro_schema = {
            "handover_failure_threshold": {
                "required": False,
                "type": float,
                "min": 0.0,
                "max": 1.0  # Percentage
            },
            "ping_pong_threshold": {
                "required": False,
                "type": int,
                "min": 1,
                "max": 10  # Number of back-and-forth handovers
            },
            "rlf_timeout_ms": {
                "required": False,
                "type": int,
                "min": 100,    # Minimum RLF timeout
                "max": 10000   # Maximum RLF timeout
            }
        }
        
        validator = SchemaValidator(mro_schema)
        validator.validate(params)
    
    def _validate_mobility_model_params(self, params: Dict[str, Any], field_name: str) -> None:
        """Validate mobility model parameters"""
        mobility_schema = {
            "velocity_kmh": {
                "required": False,
                "type": float,
                "min": 0.0,
                "max": 200.0  # Reasonable vehicle speed limit
            },
            "direction_change_probability": {
                "required": False,
                "type": float,
                "min": 0.0,
                "max": 1.0
            },
            "path_loss_model": {
                "required": False,
                "type": str,
                "enum": ["free_space", "urban_macro", "urban_micro", "rural"]
            }
        }
        
        validator = SchemaValidator(mobility_schema)
        validator.validate(params)


class ExampleAppValidator(BaseAppValidator):
    """Validator for example applications"""
    
    def get_app_specific_schema(self) -> Dict[str, Any]:
        return {
            "example_params": {
                "required": False,
                "type": dict,
                "validator": lambda val, field: self._validate_example_params(val, field)
            }
        }
    
    def _validate_example_params(self, params: Dict[str, Any], field_name: str) -> None:
        """Validate example app parameters"""
        example_schema = {
            "incremental_update": {
                "required": False,
                "type": bool
            },
            "demo_mode": {
                "required": False,
                "type": bool
            }
        }
        
        validator = SchemaValidator(example_schema)
        validator.validate(params)


class AppValidatorFactory:
    """Factory for creating app-specific validators"""
    
    _validators = {
        AppType.CCO: CCOAppValidator,
        AppType.ENERGY_SAVINGS: EnergySavingsAppValidator,
        AppType.LOAD_BALANCING: LoadBalancingAppValidator,
        AppType.MRO: MROAppValidator,
        AppType.EXAMPLE: ExampleAppValidator
    }
    
    @classmethod
    def create_validator(cls, app_type: str) -> BaseAppValidator:
        """Create appropriate validator for app type"""
        try:
            app_enum = AppType(app_type)
            validator_class = cls._validators.get(app_enum)
            
            if not validator_class:
                raise ValidationException(f"No validator available for app type: {app_type}")
            
            return validator_class()
            
        except ValueError:
            raise ValidationException(f"Unsupported app type: {app_type}")
    
    @classmethod
    def get_supported_app_types(cls) -> List[str]:
        """Get list of supported app types"""
        return [app.value for app in AppType]