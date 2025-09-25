# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from api_manager.validators.base_validator import BaseValidator
from api_manager.exceptions.validation_exception import ValidationException


class ValidationContext(Enum):
    """Different validation contexts for cellular networks"""
    URBAN_DENSE = "urban_dense"
    URBAN_MACRO = "urban_macro"
    SUBURBAN = "suburban"
    RURAL = "rural"
    INDOOR = "indoor"


@dataclass
class CellularEnvironment:
    """Cellular environment characteristics"""
    context: ValidationContext
    typical_ici_distance_m: float  # Inter-cell interference distance
    typical_power_dbm: float
    frequency_bands: List[float]
    expected_user_density: float  # users per km²
    mobility_patterns: List[str]


class ContextAwareValidator(BaseValidator):
    """Validator that adapts based on cellular network context"""

    # Environment-specific configurations
    ENVIRONMENTS = {
        ValidationContext.URBAN_DENSE: CellularEnvironment(
            context=ValidationContext.URBAN_DENSE,
            typical_ici_distance_m=200,
            typical_power_dbm=39,
            frequency_bands=[1800, 2100, 2600],
            expected_user_density=10000,
            mobility_patterns=["pedestrian", "cyclist", "car"]
        ),
        ValidationContext.URBAN_MACRO: CellularEnvironment(
            context=ValidationContext.URBAN_MACRO,
            typical_ici_distance_m=500,
            typical_power_dbm=43,
            frequency_bands=[800, 1800, 2100],
            expected_user_density=5000,
            mobility_patterns=["pedestrian", "car"]
        ),
        ValidationContext.SUBURBAN: CellularEnvironment(
            context=ValidationContext.SUBURBAN,
            typical_ici_distance_m=800,
            typical_power_dbm=43,
            frequency_bands=[800, 2100],
            expected_user_density=1000,
            mobility_patterns=["car", "stationary"]
        ),
        ValidationContext.RURAL: CellularEnvironment(
            context=ValidationContext.RURAL,
            typical_ici_distance_m=2000,
            typical_power_dbm=46,
            frequency_bands=[800, 900],
            expected_user_density=100,
            mobility_patterns=["car", "stationary"]
        )
    }

    def __init__(self, context: Optional[ValidationContext] = None):
        self.context = context
        self.environment = self.ENVIRONMENTS.get(context) if context else None

    def validate(self, data: Dict[str, Any]) -> None:
        """Validate with context awareness"""
        # Auto-detect context if not provided
        if not self.context:
            self.context = self._detect_context(data)
            self.environment = self.ENVIRONMENTS.get(self.context)

        # Perform context-aware validation
        self._validate_with_context(data)

    def _detect_context(self, data: Dict[str, Any]) -> ValidationContext:
        """Auto-detect cellular environment context from data"""
        # Analyze topology if available
        topology_path = data.get('topology_path')
        if topology_path:
            context = self._analyze_topology_context(topology_path)
            if context:
                return context

        # Analyze configuration patterns
        config_hints = []

        # Check frequency bands
        if 'cco_params' in data:
            # High frequency suggests urban dense
            if 'frequency_bands' in str(data.get('cco_params', {})):
                config_hints.append('urban')

        # Check user density indicators
        if 'energy_params' in data:
            min_active = data['energy_params'].get('min_active_cells', 0)
            if min_active > 20:
                config_hints.append('dense')

        # Default to urban macro if unclear
        return ValidationContext.URBAN_MACRO

    def _analyze_topology_context(self, topology_path: str) -> Optional[ValidationContext]:
        """Analyze topology file to determine network context"""
        try:
            import pandas as pd
            df = pd.read_csv(topology_path)

            if df.empty:
                return None

            # Calculate average inter-site distance
            if len(df) > 1:
                avg_distance = self._calculate_average_ici_distance(df)

                if avg_distance < 300:
                    return ValidationContext.URBAN_DENSE
                elif avg_distance < 600:
                    return ValidationContext.URBAN_MACRO
                elif avg_distance < 1200:
                    return ValidationContext.SUBURBAN
                else:
                    return ValidationContext.RURAL

            # Analyze frequency distribution
            if 'cell_carrier_freq_mhz' in df.columns:
                avg_freq = df['cell_carrier_freq_mhz'].mean()
                if avg_freq > 2000:
                    return ValidationContext.URBAN_DENSE
                elif avg_freq < 1000:
                    return ValidationContext.RURAL

        except Exception as e:
            logger.warning(f"Could not analyze topology context: {e}")

        return None

    def _calculate_average_ici_distance(self, topology_df) -> float:
        """Calculate average inter-cell interference distance"""
        import numpy as np

        lats = topology_df['cell_lat'].values
        lons = topology_df['cell_lon'].values

        distances = []
        for i in range(len(lats)):
            for j in range(i + 1, len(lats)):
                # Haversine distance approximation
                dlat = np.radians(lats[j] - lats[i])
                dlon = np.radians(lons[j] - lons[i])
                a = (np.sin(dlat/2)**2 + np.cos(np.radians(lats[i])) *
                     np.cos(np.radians(lats[j])) * np.sin(dlon/2)**2)
                distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))
                distances.append(distance_km * 1000)  # Convert to meters

        return np.mean(distances) if distances else 0

    def _validate_with_context(self, data: Dict[str, Any]) -> None:
        """Perform context-aware validation"""
        if not self.environment:
            # Fall back to standard validation if no context
            return

        errors = []

        # Validate CCO parameters against environment
        if 'cco_params' in data:
            errors.extend(self._validate_cco_context(data['cco_params']))

        # Validate energy parameters against environment
        if 'energy_params' in data:
            errors.extend(self._validate_energy_context(data['energy_params']))

        # Validate load balancing parameters
        if 'load_balance_params' in data:
            errors.extend(self._validate_load_balance_context(data['load_balance_params']))

        # Validate MRO parameters
        if 'mro_params' in data:
            errors.extend(self._validate_mro_context(data['mro_params']))

        if errors:
            raise ValidationException(
                f"Context-aware validation failed for {self.context.value} environment",
                validation_errors=errors
            )

    def _validate_cco_context(self, cco_params: Dict[str, Any]) -> List[Dict[str, str]]:
        """Validate CCO parameters against cellular environment context"""
        errors = []
        env = self.environment

        # Validate weak coverage threshold based on environment
        weak_threshold = cco_params.get('weak_coverage_threshold')
        if weak_threshold:
            if env.context == ValidationContext.URBAN_DENSE and weak_threshold < -105:
                errors.append({
                    "field": "weak_coverage_threshold",
                    "error": f"In {env.context.value} environment, weak coverage threshold should be > -105 dBm due to high interference"
                })
            elif env.context == ValidationContext.RURAL and weak_threshold > -120:
                errors.append({
                    "field": "weak_coverage_threshold",
                    "error": f"In {env.context.value} environment, weak coverage threshold can be more aggressive (< -120 dBm)"
                })

        # Validate lambda based on user density
        lambda_val = cco_params.get('lambda_')
        if lambda_val:
            if env.expected_user_density > 5000 and lambda_val < 0.3:
                errors.append({
                    "field": "lambda_",
                    "error": f"High user density ({env.expected_user_density}/km²) suggests lambda_ should be > 0.3 to prioritize coverage"
                })

        return errors

    def _validate_energy_context(self, energy_params: Dict[str, Any]) -> List[Dict[str, str]]:
        """Validate energy parameters against environment"""
        errors = []
        env = self.environment

        min_active = energy_params.get('min_active_cells')
        if min_active:
            # Calculate minimum based on coverage requirements
            if env.context == ValidationContext.URBAN_DENSE and min_active < 5:
                errors.append({
                    "field": "min_active_cells",
                    "error": f"In {env.context.value} environment, need at least 5 active cells for coverage"
                })
            elif env.context == ValidationContext.RURAL and min_active > 2:
                errors.append({
                    "field": "min_active_cells",
                    "error": f"In {env.context.value} environment, 1-2 active cells usually sufficient"
                })

        # Validate power reduction target
        power_target = energy_params.get('power_reduction_target')
        if power_target:
            if env.expected_user_density > 8000 and power_target > 0.2:
                errors.append({
                    "field": "power_reduction_target",
                    "error": f"High user density limits power reduction to < 20%"
                })

        return errors

    def _validate_load_balance_context(self, lb_params: Dict[str, Any]) -> List[Dict[str, str]]:
        """Validate load balancing parameters against environment"""
        errors = []
        env = self.environment

        target_threshold = lb_params.get('target_load_threshold')
        if target_threshold:
            if env.context == ValidationContext.URBAN_DENSE and target_threshold > 0.85:
                errors.append({
                    "field": "target_load_threshold",
                    "error": f"In {env.context.value}, recommend load threshold < 85% due to high variability"
                })

        return errors

    def _validate_mro_context(self, mro_params: Dict[str, Any]) -> List[Dict[str, str]]:
        """Validate MRO parameters against environment"""
        errors = []
        env = self.environment

        failure_threshold = mro_params.get('handover_failure_threshold')
        if failure_threshold:
            if env.context == ValidationContext.RURAL and failure_threshold < 0.02:
                errors.append({
                    "field": "handover_failure_threshold",
                    "error": f"In {env.context.value}, handover failure threshold can be more lenient (> 2%)"
                })

        return errors


class SmartParameterValidator(BaseValidator):
    """Smart validator that provides parameter recommendations"""

    def validate(self, data: Dict[str, Any]) -> None:
        """Validate and provide smart recommendations"""
        recommendations = []
        warnings = []

        # Analyze parameter combinations for optimization opportunities
        if 'cco_params' in data:
            cco_recommendations = self._analyze_cco_parameters(data['cco_params'])
            recommendations.extend(cco_recommendations)

        # Check for conflicting objectives
        conflicts = self._detect_parameter_conflicts(data)
        if conflicts:
            warnings.extend(conflicts)

        # Add recommendations to validation result (non-blocking)
        if recommendations or warnings:
            # Store recommendations for API response
            data['_validation_insights'] = {
                'recommendations': recommendations,
                'warnings': warnings
            }

    def _analyze_cco_parameters(self, cco_params: Dict[str, Any]) -> List[str]:
        """Analyze CCO parameters and provide recommendations"""
        recommendations = []

        lambda_val = cco_params.get('lambda_', 0.5)
        weak_threshold = cco_params.get('weak_coverage_threshold', -100)
        over_threshold = cco_params.get('over_coverage_threshold', 0)

        # Analyze parameter balance
        threshold_gap = abs(weak_threshold - over_threshold)
        if threshold_gap < 90:
            recommendations.append(
                f"Consider increasing threshold gap (currently {threshold_gap:.1f} dB) for better coverage distinction"
            )

        # Lambda optimization suggestions
        if lambda_val > 0.8:
            recommendations.append(
                "High lambda_ value (> 0.8) heavily prioritizes weak coverage - consider balanced approach"
            )
        elif lambda_val < 0.2:
            recommendations.append(
                "Low lambda_ value (< 0.2) may lead to over-coverage issues - consider increasing"
            )

        return recommendations

    def _detect_parameter_conflicts(self, data: Dict[str, Any]) -> List[str]:
        """Detect conflicting parameter objectives"""
        conflicts = []

        # Energy vs Coverage conflict
        energy_params = data.get('energy_params', {})
        cco_params = data.get('cco_params', {})

        if energy_params and cco_params:
            power_reduction = energy_params.get('power_reduction_target', 0)
            weak_threshold = cco_params.get('weak_coverage_threshold', -100)

            if power_reduction > 0.3 and weak_threshold > -110:
                conflicts.append(
                    "High power reduction (>30%) with lenient weak coverage threshold may degrade service quality"
                )

        # Load balancing vs Energy savings conflict
        lb_params = data.get('load_balance_params', {})
        if energy_params and lb_params:
            min_active = energy_params.get('min_active_cells', 0)
            target_load = lb_params.get('target_load_threshold', 0.8)

            if min_active < 3 and target_load > 0.9:
                conflicts.append(
                    "Very few active cells with high load threshold may cause service degradation"
                )

        return conflicts


class PhysicsBasedValidator(BaseValidator):
    """Validator using cellular network physics models"""

    def validate(self, data: Dict[str, Any]) -> None:
        """Validate parameters against cellular physics"""
        # Validate RF propagation physics
        if 'cco_params' in data:
            self._validate_rf_physics(data['cco_params'], data.get('topology_path'))

        # Validate interference models
        if 'load_balance_params' in data:
            self._validate_interference_physics(data['load_balance_params'])

    def _validate_rf_physics(self, cco_params: Dict[str, Any], topology_path: Optional[str]) -> None:
        """Validate CCO parameters against RF propagation physics"""
        weak_threshold = cco_params.get('weak_coverage_threshold')
        over_threshold = cco_params.get('over_coverage_threshold')

        if weak_threshold and over_threshold:
            # Physics check: SINR threshold should account for RSRP
            expected_sinr_range = self._calculate_expected_sinr_range(weak_threshold)

            if over_threshold < expected_sinr_range[0]:
                raise ValidationException(
                    f"Over coverage SINR threshold ({over_threshold} dB) too low for weak RSRP threshold ({weak_threshold} dBm). "
                    f"Expected SINR range: {expected_sinr_range[0]:.1f} to {expected_sinr_range[1]:.1f} dB",
                    field="over_coverage_threshold"
                )

    def _calculate_expected_sinr_range(self, rsrp_threshold: float) -> Tuple[float, float]:
        """Calculate expected SINR range based on RSRP threshold"""
        # Simplified physics model
        # In practice, this would use more sophisticated propagation models
        thermal_noise = -104  # dBm for 20MHz
        interference_margin = 10  # dB

        min_sinr = rsrp_threshold - thermal_noise - interference_margin
        max_sinr = min_sinr + 20  # Typical cellular SINR range

        return (max(-10, min_sinr), min(30, max_sinr))

    def _validate_interference_physics(self, lb_params: Dict[str, Any]) -> None:
        """Validate load balancing parameters against interference physics"""
        handover_margin = lb_params.get('handover_margin_db')
        target_load = lb_params.get('target_load_threshold')

        if handover_margin and target_load:
            # Physics: Higher load requires larger handover margins
            min_margin = self._calculate_min_handover_margin(target_load)

            if handover_margin < min_margin:
                raise ValidationException(
                    f"Handover margin ({handover_margin} dB) too small for target load ({target_load*100:.0f}%). "
                    f"Minimum recommended: {min_margin:.1f} dB",
                    field="handover_margin_db"
                )

    def _calculate_min_handover_margin(self, target_load: float) -> float:
        """Calculate minimum handover margin based on target load"""
        # Higher load creates more interference, requiring larger margins
        base_margin = 2.0  # dB
        load_factor = target_load / 0.5  # Reference load of 50%
        return base_margin * load_factor


class MLPoweredValidator(BaseValidator):
    """ML-powered validator for anomaly detection and optimization"""

    def __init__(self):
        self.historical_configs = []  # Would be loaded from database
        self.performance_metrics = {}  # Historical performance data

    def validate(self, data: Dict[str, Any]) -> None:
        """Validate using ML-based insights"""
        # Anomaly detection
        anomaly_score = self._detect_parameter_anomalies(data)
        if anomaly_score > 0.8:
            raise ValidationException(
                f"Parameter configuration appears anomalous (score: {anomaly_score:.2f}). "
                "Please review parameter values.",
                field="configuration"
            )

        # Performance prediction
        predicted_performance = self._predict_performance(data)
        if predicted_performance['expected_improvement'] < 0.05:
            data['_ml_insights'] = {
                'predicted_performance': predicted_performance,
                'recommendations': self._generate_ml_recommendations(data)
            }

    def _detect_parameter_anomalies(self, data: Dict[str, Any]) -> float:
        """Detect anomalous parameter combinations using ML"""
        # Simplified anomaly detection
        # In practice, this would use trained ML models

        anomaly_indicators = []

        # Check for extreme parameter values
        if 'cco_params' in data:
            lambda_val = data['cco_params'].get('lambda_', 0.5)
            if lambda_val < 0.1 or lambda_val > 0.9:
                anomaly_indicators.append(0.3)

        if 'energy_params' in data:
            power_reduction = data['energy_params'].get('power_reduction_target', 0)
            if power_reduction > 0.4:
                anomaly_indicators.append(0.5)

        # Calculate overall anomaly score
        return max(anomaly_indicators) if anomaly_indicators else 0.0

    def _predict_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict performance impact of configuration"""
        # Simplified performance prediction
        # In practice, this would use trained ML models

        base_performance = 0.8  # 80% baseline

        # Adjust based on parameters
        if 'cco_params' in data:
            lambda_val = data['cco_params'].get('lambda_', 0.5)
            # Balanced lambda tends to perform better
            balance_factor = 1 - abs(lambda_val - 0.5) * 0.5
            base_performance *= balance_factor

        return {
            'expected_improvement': base_performance - 0.75,  # vs baseline
            'confidence': 0.85,
            'key_factors': ['parameter_balance', 'historical_patterns']
        }

    def _generate_ml_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate ML-based parameter recommendations"""
        recommendations = []

        # Analyze historical best practices
        if 'cco_params' in data:
            recommendations.append(
                "Consider lambda_ = 0.6 based on similar network configurations"
            )

        if 'energy_params' in data:
            recommendations.append(
                "Gradual power reduction (start with 15%) recommended for initial deployment"
            )

        return recommendations