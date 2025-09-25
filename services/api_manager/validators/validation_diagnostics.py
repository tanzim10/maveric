# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import traceback
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from api_manager.exceptions.validation_exception import ValidationException, FileValidationException

logger = logging.getLogger(__name__)


class DiagnosticLevel(Enum):
    """Diagnostic detail levels"""
    MINIMAL = "minimal"      # Basic error info
    STANDARD = "standard"    # Error + context
    DETAILED = "detailed"    # Error + context + suggestions
    EXPERT = "expert"        # All info + technical details


@dataclass
class ValidationDiagnostic:
    """Detailed validation diagnostic information"""
    error_code: str
    error_type: str
    field_path: str
    user_message: str
    technical_message: str
    severity: str
    suggested_fixes: List[str]
    related_parameters: List[str]
    documentation_links: List[str]
    examples: List[Dict[str, Any]]
    validation_context: Dict[str, Any]


class ValidationDiagnosticsEngine:
    """Engine for generating detailed validation diagnostics"""

    # Error code mappings
    ERROR_CODES = {
        'LAMBDA_OUT_OF_BOUNDS': {
            'user_message': 'Lambda parameter must be between 0 and 1 (exclusive)',
            'technical_message': 'Lambda value controls coverage-capacity trade-off in CCO algorithms',
            'suggested_fixes': [
                'Use lambda_ = 0.5 for balanced optimization',
                'Use lambda_ = 0.7 to prioritize coverage over capacity',
                'Use lambda_ = 0.3 to prioritize capacity over coverage'
            ],
            'related_parameters': ['weak_coverage_threshold', 'over_coverage_threshold'],
            'documentation_links': [
                '/docs/cco-parameters.md#lambda-parameter',
                '/docs/coverage-optimization.md'
            ],
            'examples': [
                {'description': 'Urban dense network', 'lambda_': 0.65},
                {'description': 'Rural network', 'lambda_': 0.45}
            ]
        },
        'RSRP_THRESHOLD_INVALID': {
            'user_message': 'RSRP threshold is outside realistic cellular range',
            'technical_message': 'RSRP (Reference Signal Received Power) typically ranges from -150 to 0 dBm',
            'suggested_fixes': [
                'Use -110 dBm for urban environments',
                'Use -120 dBm for suburban environments',
                'Use -130 dBm for rural environments'
            ],
            'related_parameters': ['cell_tx_power_dbm', 'frequency_bands'],
            'documentation_links': ['/docs/rf-parameters.md#rsrp'],
            'examples': [
                {'environment': 'urban', 'weak_coverage_threshold': -110},
                {'environment': 'rural', 'weak_coverage_threshold': -120}
            ]
        },
        'TRAIN_TEST_DAY_OVERLAP': {
            'user_message': 'Test day cannot be included in training days',
            'technical_message': 'Data leakage prevention - test data must be separate from training data',
            'suggested_fixes': [
                'Choose test day outside training day range',
                'Use consecutive days: train on [0,1,2,3], test on [4]',
                'Use temporal split: train on [0,2,4], test on [1,3,5]'
            ],
            'related_parameters': ['train_days', 'test_day'],
            'documentation_links': ['/docs/ml-best-practices.md#data-splits'],
            'examples': [
                {'train_days': [0, 1, 2, 3], 'test_day': 4},
                {'train_days': [0, 2, 4, 6], 'test_day': 1}
            ]
        },
        'POWER_REDUCTION_AGGRESSIVE': {
            'user_message': 'Power reduction target may be too aggressive',
            'technical_message': 'High power reduction can degrade service quality and coverage',
            'suggested_fixes': [
                'Start with 15% reduction and monitor performance',
                'Implement gradual reduction over time',
                'Ensure minimum cell count maintains coverage'
            ],
            'related_parameters': ['min_active_cells', 'coverage_threshold'],
            'documentation_links': ['/docs/energy-savings.md#power-reduction'],
            'examples': [
                {'conservative': {'power_reduction_target': 0.15}},
                {'aggressive': {'power_reduction_target': 0.25}}
            ]
        }
    }

    def generate_diagnostic(
        self,
        exception: ValidationException,
        context: Dict[str, Any],
        level: DiagnosticLevel = DiagnosticLevel.STANDARD
    ) -> ValidationDiagnostic:
        """Generate detailed diagnostic for validation error"""

        # Determine error code
        error_code = self._classify_error(exception, context)
        error_info = self.ERROR_CODES.get(error_code, {})

        # Build diagnostic
        diagnostic = ValidationDiagnostic(
            error_code=error_code,
            error_type=type(exception).__name__,
            field_path=getattr(exception, 'field', 'unknown'),
            user_message=error_info.get('user_message', exception.message),
            technical_message=error_info.get('technical_message', ''),
            severity=self._determine_severity(exception, context),
            suggested_fixes=error_info.get('suggested_fixes', []),
            related_parameters=error_info.get('related_parameters', []),
            documentation_links=error_info.get('documentation_links', []),
            examples=error_info.get('examples', []),
            validation_context=self._build_validation_context(context, level)
        )

        return diagnostic

    def _classify_error(self, exception: ValidationException, context: Dict[str, Any]) -> str:
        """Classify error type based on message and context"""
        message = exception.message.lower()
        field = getattr(exception, 'field', '') or ''

        # Pattern matching for error classification
        if 'lambda_' in message and ('between 0 and 1' in message or 'exclusive' in message):
            return 'LAMBDA_OUT_OF_BOUNDS'
        elif 'rsrp' in message or 'coverage_threshold' in message:
            return 'RSRP_THRESHOLD_INVALID'
        elif 'test day' in message and 'training days' in message:
            return 'TRAIN_TEST_DAY_OVERLAP'
        elif 'power_reduction_target' in message or 'power reduction' in message:
            return 'POWER_REDUCTION_AGGRESSIVE'
        elif 'frequency' in message:
            return 'FREQUENCY_OUT_OF_RANGE'
        elif 'coordinate' in message or 'latitude' in message or 'longitude' in message:
            return 'INVALID_COORDINATES'
        else:
            return 'GENERIC_VALIDATION_ERROR'

    def _determine_severity(self, exception: ValidationException, context: Dict[str, Any]) -> str:
        """Determine error severity"""
        if isinstance(exception, FileValidationException):
            return 'high'  # File errors are typically blocking

        field = getattr(exception, 'field', '') or ''

        # Critical parameters
        if field in ['app_type', 'model_id', 'topology_path']:
            return 'high'

        # Important parameters
        if field in ['lambda_', 'weak_coverage_threshold', 'train_days']:
            return 'medium'

        return 'low'

    def _build_validation_context(self, context: Dict[str, Any], level: DiagnosticLevel) -> Dict[str, Any]:
        """Build validation context information"""
        validation_context = {
            'app_type': context.get('app_type', 'unknown'),
            'timestamp': time.time(),
            'diagnostic_level': level.value
        }

        if level in [DiagnosticLevel.DETAILED, DiagnosticLevel.EXPERT]:
            # Add detailed context
            validation_context.update({
                'parameter_count': self._count_parameters(context),
                'has_files': self._has_files(context),
                'configuration_complexity': self._assess_complexity(context)
            })

        if level == DiagnosticLevel.EXPERT:
            # Add expert-level context
            validation_context.update({
                'call_stack': traceback.format_stack()[-3:-1],  # Relevant stack frames
                'system_info': {
                    'validation_engine_version': '2.0',
                    'supported_apps': 5
                }
            })

        return validation_context

    def _count_parameters(self, context: Dict[str, Any]) -> int:
        """Count total parameters in configuration"""
        count = 0
        param_sections = ['cco_params', 'energy_params', 'load_balance_params', 'mro_params']

        for section in param_sections:
            if section in context and isinstance(context[section], dict):
                count += len(context[section])

        return count

    def _has_files(self, context: Dict[str, Any]) -> bool:
        """Check if configuration includes files"""
        file_keys = ['topology_path', 'config_path', 'training_data_path']
        return any(key in context for key in file_keys)

    def _assess_complexity(self, context: Dict[str, Any]) -> str:
        """Assess configuration complexity"""
        param_count = self._count_parameters(context)
        has_files = self._has_files(context)

        if param_count > 15 or has_files:
            return 'high'
        elif param_count > 8:
            return 'medium'
        else:
            return 'low'


class EnhancedValidationException(ValidationException):
    """Enhanced validation exception with diagnostic support"""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        diagnostic: Optional[ValidationDiagnostic] = None,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message, field, validation_errors)
        self.diagnostic = diagnostic
        self.suggestions = suggestions or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enhanced diagnostic info"""
        result = super().to_dict()

        if self.diagnostic:
            result['diagnostic'] = asdict(self.diagnostic)

        if self.suggestions:
            result['suggestions'] = self.suggestions

        return result


class ValidationHealthChecker:
    """Monitors validation system health and performance"""

    def __init__(self):
        self.performance_history: List[Dict[str, Any]] = []
        self.error_patterns: Dict[str, int] = {}
        self.validation_trends: Dict[str, List[float]] = {}

    def record_validation_event(
        self,
        app_type: str,
        validation_time: float,
        success: bool,
        error_codes: List[str] = None
    ):
        """Record validation event for health monitoring"""

        event = {
            'timestamp': time.time(),
            'app_type': app_type,
            'validation_time': validation_time,
            'success': success,
            'error_codes': error_codes or []
        }

        self.performance_history.append(event)

        # Track error patterns
        for error_code in (error_codes or []):
            self.error_patterns[error_code] = self.error_patterns.get(error_code, 0) + 1

        # Track validation time trends
        if app_type not in self.validation_trends:
            self.validation_trends[app_type] = []
        self.validation_trends[app_type].append(validation_time)

        # Keep only recent history (last 1000 events)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def get_health_report(self) -> Dict[str, Any]:
        """Generate validation system health report"""

        if not self.performance_history:
            return {'status': 'no_data', 'message': 'No validation events recorded'}

        recent_events = self.performance_history[-100:]  # Last 100 events

        # Calculate success rate
        success_rate = sum(1 for event in recent_events if event['success']) / len(recent_events)

        # Calculate average validation time
        avg_validation_time = sum(event['validation_time'] for event in recent_events) / len(recent_events)

        # Find most common errors
        top_errors = sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]

        # Assess overall health
        health_status = self._assess_health_status(success_rate, avg_validation_time)

        return {
            'status': health_status,
            'metrics': {
                'success_rate': success_rate,
                'avg_validation_time_ms': avg_validation_time * 1000,
                'total_validations': len(self.performance_history),
                'recent_validations': len(recent_events)
            },
            'error_analysis': {
                'top_error_codes': [{'code': code, 'count': count} for code, count in top_errors],
                'error_rate': 1 - success_rate
            },
            'performance_trends': self._analyze_performance_trends(),
            'recommendations': self._generate_health_recommendations(success_rate, avg_validation_time)
        }

    def _assess_health_status(self, success_rate: float, avg_time: float) -> str:
        """Assess overall validation system health"""

        if success_rate >= 0.95 and avg_time < 0.1:
            return 'excellent'
        elif success_rate >= 0.90 and avg_time < 0.2:
            return 'good'
        elif success_rate >= 0.80 and avg_time < 0.5:
            return 'fair'
        else:
            return 'poor'

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze validation performance trends"""
        trends = {}

        for app_type, times in self.validation_trends.items():
            if len(times) > 10:  # Need enough data for trend analysis
                recent_avg = sum(times[-10:]) / 10
                older_avg = sum(times[-20:-10]) / 10 if len(times) > 20 else recent_avg

                trend_direction = 'improving' if recent_avg < older_avg else 'degrading'
                trend_magnitude = abs(recent_avg - older_avg) / older_avg if older_avg > 0 else 0

                trends[app_type] = {
                    'trend_direction': trend_direction,
                    'trend_magnitude': trend_magnitude,
                    'recent_avg_ms': recent_avg * 1000,
                    'sample_size': len(times)
                }

        return trends

    def _generate_health_recommendations(self, success_rate: float, avg_time: float) -> List[str]:
        """Generate recommendations for improving validation health"""
        recommendations = []

        if success_rate < 0.90:
            recommendations.append("Low success rate detected - review common error patterns")
            recommendations.append("Consider improving validation error messages for better user experience")

        if avg_time > 0.2:
            recommendations.append("Validation time is high - consider enabling caching")
            recommendations.append("Large file validations may benefit from async processing")

        # Analyze error patterns
        top_error = max(self.error_patterns.items(), key=lambda x: x[1]) if self.error_patterns else None
        if top_error and top_error[1] > 10:
            recommendations.append(f"Most common error: {top_error[0]} - consider improving documentation")

        return recommendations


class SmartErrorReporter:
    """Generates intelligent error reports with actionable insights"""

    def __init__(self):
        self.diagnostics_engine = ValidationDiagnosticsEngine()
        self.health_checker = ValidationHealthChecker()

    def generate_enhanced_error_report(
        self,
        exception: ValidationException,
        context: Dict[str, Any],
        level: DiagnosticLevel = DiagnosticLevel.STANDARD
    ) -> Dict[str, Any]:
        """Generate comprehensive error report"""

        # Generate diagnostic
        diagnostic = self.diagnostics_engine.generate_diagnostic(exception, context, level)

        # Build enhanced error report
        error_report = {
            'error': exception.message,
            'error_type': type(exception).__name__,
            'status_code': getattr(exception, 'code', 400),
            'field': getattr(exception, 'field', None),
            'diagnostic': asdict(diagnostic) if level != DiagnosticLevel.MINIMAL else None,
            'timestamp': time.time()
        }

        # Add validation errors if present
        if hasattr(exception, 'validation_errors') and exception.validation_errors:
            error_report['validation_errors'] = exception.validation_errors

        # Add quick fixes for common issues
        if level in [DiagnosticLevel.DETAILED, DiagnosticLevel.EXPERT]:
            error_report['quick_fixes'] = self._generate_quick_fixes(exception, context)

        # Add expert-level technical details
        if level == DiagnosticLevel.EXPERT:
            error_report['technical_details'] = {
                'validation_stack': self._build_validation_stack(context),
                'parameter_analysis': self._analyze_parameter_interactions(context),
                'optimization_suggestions': self._suggest_optimizations(context)
            }

        return error_report

    def _generate_quick_fixes(self, exception: ValidationException, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate quick fix suggestions"""
        quick_fixes = []

        field = getattr(exception, 'field', '') or ''
        app_type = context.get('app_type', '')

        # Lambda parameter fixes
        if 'lambda_' in field:
            quick_fixes.append({
                'action': 'set_parameter',
                'parameter': 'cco_params.lambda_',
                'suggested_value': 0.5,
                'description': 'Use balanced coverage-capacity trade-off'
            })

        # Training day fixes
        if 'test_day' in field or 'train_days' in field:
            train_days = context.get('train_days', [])
            if train_days:
                next_day = max(train_days) + 1
                quick_fixes.append({
                    'action': 'set_parameter',
                    'parameter': 'test_day',
                    'suggested_value': next_day,
                    'description': f'Use day {next_day} for testing (after training days)'
                })

        # Power reduction fixes
        if 'power_reduction_target' in field:
            quick_fixes.append({
                'action': 'set_parameter',
                'parameter': 'energy_params.power_reduction_target',
                'suggested_value': 0.15,
                'description': 'Start with conservative 15% power reduction'
            })

        return quick_fixes

    def _build_validation_stack(self, context: Dict[str, Any]) -> List[str]:
        """Build validation execution stack for debugging"""
        stack = ['ValidationOrchestrator']

        app_type = context.get('app_type', '')
        if app_type:
            stack.append(f'AppValidator[{app_type}]')

        if context.get('topology_path'):
            stack.append('TopologyFileValidator')

        if context.get('config_path'):
            stack.append('ConfigFileValidator')

        return stack

    def _analyze_parameter_interactions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parameter interactions for expert diagnostics"""
        interactions = {}

        # CCO parameter interactions
        if 'cco_params' in context:
            cco = context['cco_params']

            if 'lambda_' in cco and 'weak_coverage_threshold' in cco:
                lambda_val = cco['lambda_']
                threshold = cco['weak_coverage_threshold']

                interactions['cco_coverage_balance'] = {
                    'lambda_': lambda_val,
                    'threshold': threshold,
                    'analysis': 'High lambda with lenient threshold may cause over-optimization for weak areas'
                }

        return interactions

    def _suggest_optimizations(self, context: Dict[str, Any]) -> List[str]:
        """Suggest configuration optimizations"""
        suggestions = []

        app_type = context.get('app_type', '')

        # App-specific optimization suggestions
        if app_type == 'coverage_capacity_optimization':
            suggestions.extend(self._suggest_cco_optimizations(context))
        elif app_type == 'energy_savings':
            suggestions.extend(self._suggest_energy_optimizations(context))

        return suggestions

    def _suggest_cco_optimizations(self, context: Dict[str, Any]) -> List[str]:
        """Generate CCO-specific optimization suggestions"""
        suggestions = []
        cco_params = context.get('cco_params', {})

        # Lambda optimization
        lambda_val = cco_params.get('lambda_', 0.5)
        if abs(lambda_val - 0.6) > 0.1:
            suggestions.append(f"Consider lambda_ = 0.6 (current: {lambda_val}) for optimal urban performance")

        # Threshold optimization
        weak_threshold = cco_params.get('weak_coverage_threshold', -100)
        if weak_threshold > -110:
            suggestions.append(f"Consider lowering weak coverage threshold to -108 dBm for better edge detection")

        return suggestions

    def _suggest_energy_optimizations(self, context: Dict[str, Any]) -> List[str]:
        """Generate energy savings optimization suggestions"""
        suggestions = []
        energy_params = context.get('energy_params', {})

        # Power reduction optimization
        power_target = energy_params.get('power_reduction_target', 0)
        if power_target > 0.3:
            suggestions.append("Consider gradual power reduction: start with 15%, increase weekly")

        return suggestions


# Global instances
_global_health_checker = ValidationHealthChecker()
_global_error_reporter = SmartErrorReporter()


def get_validation_health_checker() -> ValidationHealthChecker:
    """Get global validation health checker"""
    return _global_health_checker


def get_smart_error_reporter() -> SmartErrorReporter:
    """Get global smart error reporter"""
    return _global_error_reporter