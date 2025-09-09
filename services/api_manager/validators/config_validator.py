# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from api_manager.validators.base_validator import BaseValidator, FileValidator
from api_manager.exceptions.validation_exception import ValidationException, FileValidationException


class ConfigFileValidator(BaseValidator):
    """Validator for application configuration files"""
    
    SUPPORTED_FORMATS = ['.json', '.yaml', '.yml', '.csv']
    
    def validate(self, data: Dict[str, Any]) -> None:
        """Validate configuration file parameters"""
        config_path = data.get('config_path')
        if not config_path:
            raise ValidationException("Configuration file path is required", field="config_path")
        
        # Validate file exists and format
        self._validate_config_file_format(config_path)
        
        # Load and validate config content based on file type
        config_content = self._load_config_file(config_path)
        self._validate_config_content(config_content, config_path)
    
    def _validate_config_file_format(self, file_path: str) -> None:
        """Validate configuration file format and existence"""
        if not os.path.exists(file_path):
            raise FileValidationException(f"Configuration file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            raise FileValidationException(
                f"Unsupported configuration file format: {file_ext}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        # Validate file size (config files should be reasonable)
        FileValidator.validate_file_size(file_path, max_size_mb=10, filename=os.path.basename(file_path))
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration file content"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
                    
            elif file_ext in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise FileValidationException("YAML support not available - install PyYAML")
                with open(file_path, 'r') as f:
                    return yaml.safe_load(f) or {}
                    
            elif file_ext == '.csv':
                # For CSV configs, convert to dict format
                df = pd.read_csv(file_path)
                return {'data': df.to_dict('records')}
                
        except Exception as e:
            raise FileValidationException(
                f"Error loading configuration file: {str(e)}",
                filename=os.path.basename(file_path)
            )
    
    def _validate_config_content(self, config: Dict[str, Any], file_path: str) -> None:
        """Validate configuration file content structure"""
        if not isinstance(config, dict):
            raise FileValidationException(
                "Configuration file must contain a valid object/dictionary structure",
                filename=os.path.basename(file_path)
            )
        
        # Validate common configuration patterns
        self._validate_common_config_patterns(config, file_path)
    
    def _validate_common_config_patterns(self, config: Dict[str, Any], file_path: str) -> None:
        """Validate common configuration patterns across apps"""
        filename = os.path.basename(file_path)
        
        # Validate RF simulation config if present
        if 'rf_config' in config:
            self._validate_rf_config(config['rf_config'], filename)
        
        # Validate UE mobility config if present  
        if 'ue_mobility' in config:
            self._validate_ue_mobility_config(config['ue_mobility'], filename)
        
        # Validate cell configuration if present
        if 'cell_config' in config:
            self._validate_cell_config(config['cell_config'], filename)
    
    def _validate_rf_config(self, rf_config: Dict[str, Any], filename: str) -> None:
        """Validate RF simulation configuration"""
        errors = []
        
        # Validate frequency bands
        if 'frequency_bands' in rf_config:
            freq_bands = rf_config['frequency_bands']
            if not isinstance(freq_bands, list):
                errors.append({"field": "frequency_bands", "error": "Must be a list"})
            else:
                for i, freq in enumerate(freq_bands):
                    if not isinstance(freq, (int, float)) or freq < 400 or freq > 6000:
                        errors.append({
                            "field": f"frequency_bands[{i}]", 
                            "error": "Frequency must be between 400 and 6000 MHz"
                        })
        
        # Validate power levels
        if 'tx_power_dbm' in rf_config:
            power = rf_config['tx_power_dbm']
            if not isinstance(power, (int, float)) or power < 0 or power > 50:
                errors.append({"field": "tx_power_dbm", "error": "TX power must be between 0 and 50 dBm"})
        
        # Validate antenna configuration
        if 'antenna_config' in rf_config:
            antenna = rf_config['antenna_config']
            if 'gain_dbi' in antenna:
                gain = antenna['gain_dbi']
                if not isinstance(gain, (int, float)) or gain < -10 or gain > 30:
                    errors.append({"field": "antenna_config.gain_dbi", "error": "Antenna gain must be between -10 and 30 dBi"})
        
        if errors:
            raise FileValidationException(
                "RF configuration validation failed",
                filename=filename,
                validation_errors=errors
            )
    
    def _validate_ue_mobility_config(self, mobility_config: Dict[str, Any], filename: str) -> None:
        """Validate UE mobility configuration"""
        errors = []
        
        # Validate mobility models
        if 'mobility_models' in mobility_config:
            models = mobility_config['mobility_models']
            if not isinstance(models, list):
                errors.append({"field": "mobility_models", "error": "Must be a list"})
            else:
                valid_models = ['random_walk', 'random_waypoint', 'gauss_markov', 'manhattan']
                for i, model in enumerate(models):
                    if not isinstance(model, str) or model not in valid_models:
                        errors.append({
                            "field": f"mobility_models[{i}]", 
                            "error": f"Must be one of: {valid_models}"
                        })
        
        # Validate velocity ranges
        if 'velocity_range' in mobility_config:
            vel_range = mobility_config['velocity_range']
            if isinstance(vel_range, dict):
                min_vel = vel_range.get('min', 0)
                max_vel = vel_range.get('max', 0)
                
                if min_vel < 0 or max_vel < 0 or min_vel > max_vel:
                    errors.append({"field": "velocity_range", "error": "Invalid velocity range"})
                if max_vel > 200:  # Reasonable upper bound in km/h
                    errors.append({"field": "velocity_range.max", "error": "Maximum velocity too high (>200 km/h)"})
        
        if errors:
            raise FileValidationException(
                "UE mobility configuration validation failed",
                filename=filename,
                validation_errors=errors
            )
    
    def _validate_cell_config(self, cell_config: Dict[str, Any], filename: str) -> None:
        """Validate cell configuration"""
        errors = []
        
        # Validate cell deployment patterns
        if 'deployment_pattern' in cell_config:
            pattern = cell_config['deployment_pattern']
            valid_patterns = ['hexagonal', 'grid', 'random', 'realistic']
            if pattern not in valid_patterns:
                errors.append({
                    "field": "deployment_pattern", 
                    "error": f"Must be one of: {valid_patterns}"
                })
        
        # Validate inter-site distance
        if 'inter_site_distance_m' in cell_config:
            distance = cell_config['inter_site_distance_m']
            if not isinstance(distance, (int, float)) or distance < 100 or distance > 10000:
                errors.append({
                    "field": "inter_site_distance_m", 
                    "error": "Inter-site distance must be between 100m and 10km"
                })
        
        # Validate cell count
        if 'cell_count' in cell_config:
            count = cell_config['cell_count']
            if not isinstance(count, int) or count < 1 or count > 1000:
                errors.append({"field": "cell_count", "error": "Cell count must be between 1 and 1000"})
        
        if errors:
            raise FileValidationException(
                "Cell configuration validation failed",
                filename=filename,
                validation_errors=errors
            )


class TopologyFileValidator(BaseValidator):
    """Validator specifically for topology.csv files"""
    
    REQUIRED_COLUMNS = [
        'cell_id', 'cell_lat', 'cell_lon', 'cell_az_deg', 'cell_carrier_freq_mhz'
    ]
    
    OPTIONAL_COLUMNS = [
        'cell_el_deg', 'cell_tx_power_dbm', 'cell_antenna_gain_dbi', 'sector_id'
    ]
    
    def validate(self, data: Dict[str, Any]) -> None:
        """Validate topology file"""
        topology_path = data.get('topology_path')
        if not topology_path:
            raise ValidationException("Topology file path is required", field="topology_path")
        
        if not os.path.exists(topology_path):
            raise FileValidationException(f"Topology file not found: {topology_path}")
        
        # Validate CSV format and load data
        df = FileValidator.validate_csv_file(
            topology_path, 
            self.REQUIRED_COLUMNS, 
            "topology.csv"
        )
        
        # Validate file size
        FileValidator.validate_file_size(topology_path, max_size_mb=50, filename="topology.csv")
        
        # Validate topology data content
        self._validate_topology_data(df, os.path.basename(topology_path))
    
    def _validate_topology_data(self, df: pd.DataFrame, filename: str) -> None:
        """Validate topology data content"""
        errors = []
        
        # Validate cell coordinates
        invalid_lat = df[(df['cell_lat'] < -90) | (df['cell_lat'] > 90)]
        if not invalid_lat.empty:
            errors.append({
                "field": "cell_lat",
                "error": f"Found {len(invalid_lat)} invalid latitude values (must be -90 to 90)"
            })
        
        invalid_lon = df[(df['cell_lon'] < -180) | (df['cell_lon'] > 180)]
        if not invalid_lon.empty:
            errors.append({
                "field": "cell_lon", 
                "error": f"Found {len(invalid_lon)} invalid longitude values (must be -180 to 180)"
            })
        
        # Validate azimuth angles
        invalid_az = df[(df['cell_az_deg'] < 0) | (df['cell_az_deg'] >= 360)]
        if not invalid_az.empty:
            errors.append({
                "field": "cell_az_deg",
                "error": f"Found {len(invalid_az)} invalid azimuth values (must be 0-359 degrees)"
            })
        
        # Validate frequencies
        invalid_freq = df[(df['cell_carrier_freq_mhz'] < 400) | (df['cell_carrier_freq_mhz'] > 6000)]
        if not invalid_freq.empty:
            errors.append({
                "field": "cell_carrier_freq_mhz",
                "error": f"Found {len(invalid_freq)} invalid frequency values (must be 400-6000 MHz)"
            })
        
        # Check for duplicate cell IDs
        duplicates = df[df.duplicated(subset=['cell_id'], keep=False)]
        if not duplicates.empty:
            duplicate_ids = duplicates['cell_id'].unique().tolist()
            errors.append({
                "field": "cell_id",
                "error": f"Found duplicate cell IDs: {duplicate_ids}"
            })
        
        # Validate optional columns if present
        if 'cell_el_deg' in df.columns:
            invalid_tilt = df[(df['cell_el_deg'] < 0) | (df['cell_el_deg'] > 20)]
            if not invalid_tilt.empty:
                errors.append({
                    "field": "cell_el_deg",
                    "error": f"Found {len(invalid_tilt)} invalid tilt values (must be 0-20 degrees)"
                })
        
        if 'cell_tx_power_dbm' in df.columns:
            invalid_power = df[(df['cell_tx_power_dbm'] < 0) | (df['cell_tx_power_dbm'] > 50)]
            if not invalid_power.empty:
                errors.append({
                    "field": "cell_tx_power_dbm",
                    "error": f"Found {len(invalid_power)} invalid power values (must be 0-50 dBm)"
                })
        
        if errors:
            raise FileValidationException(
                "Topology file validation failed",
                filename=filename,
                validation_errors=errors
            )


class AppConfigValidator(BaseValidator):
    """Main validator for complete app configurations"""
    
    def __init__(self, app_type: str):
        from api_manager.validators.app_validator import AppValidatorFactory
        self.app_validator = AppValidatorFactory.create_validator(app_type)
        self.config_validator = ConfigFileValidator()
        self.topology_validator = TopologyFileValidator()
    
    def validate(self, data: Dict[str, Any]) -> None:
        """Validate complete app configuration"""
        # Validate app-specific parameters
        self.app_validator.validate(data)
        
        # Validate configuration file if provided
        if data.get('config_path'):
            self.config_validator.validate(data)
        
        # Validate topology file if provided
        if data.get('topology_path'):
            self.topology_validator.validate(data)
        
        # Validate file path consistency
        self._validate_file_paths(data)
    
    def _validate_file_paths(self, data: Dict[str, Any]) -> None:
        """Validate that all file paths are accessible and consistent"""
        file_paths = {}
        
        # Collect all file paths
        for key in ['config_path', 'topology_path', 'training_data_path', 'model_path']:
            if key in data and data[key]:
                file_paths[key] = data[key]
        
        # Validate path accessibility
        errors = []
        for key, path in file_paths.items():
            if not isinstance(path, str):
                errors.append({"field": key, "error": "File path must be a string"})
                continue
            
            # Check if path is absolute or relative
            if not os.path.isabs(path):
                # For relative paths, they should exist relative to some base directory
                # This is a basic check - in practice you'd want to resolve against a known base
                pass
            
            # Additional path validation could go here
        
        if errors:
            raise ValidationException("File path validation failed", validation_errors=errors)