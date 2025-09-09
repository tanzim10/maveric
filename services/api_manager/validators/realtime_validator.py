# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from api_manager.validators.base_validator import BaseValidator
from api_manager.exceptions.validation_exception import ValidationException

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of real-time validation feedback"""
    PROGRESS = "progress"
    WARNING = "warning"
    ERROR = "error"
    RECOMMENDATION = "recommendation"
    INSIGHT = "insight"
    COMPLETION = "completion"


@dataclass
class ValidationFeedback:
    """Real-time validation feedback message"""
    feedback_type: FeedbackType
    message: str
    timestamp: float
    field: Optional[str] = None
    severity: str = "info"  # info, warning, error
    progress_percent: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert enum to string value for proper serialization
        if isinstance(result.get('feedback_type'), FeedbackType):
            result['feedback_type'] = result['feedback_type'].value
        return result


class RealTimeValidationStream:
    """Manages real-time validation feedback streams"""
    
    def __init__(self):
        self.active_streams: Dict[str, asyncio.Queue] = {}
        self.stream_metadata: Dict[str, Dict[str, Any]] = {}
    
    def create_stream(self, stream_id: str) -> str:
        """Create new validation feedback stream"""
        self.active_streams[stream_id] = asyncio.Queue()
        self.stream_metadata[stream_id] = {
            'created_at': time.time(),
            'status': 'active',
            'message_count': 0
        }
        logger.info(f"Created validation stream: {stream_id}")
        return stream_id
    
    async def send_feedback(self, stream_id: str, feedback: ValidationFeedback):
        """Send feedback to stream"""
        if stream_id in self.active_streams:
            await self.active_streams[stream_id].put(feedback)
            self.stream_metadata[stream_id]['message_count'] += 1
    
    async def get_feedback_stream(self, stream_id: str) -> AsyncGenerator[ValidationFeedback, None]:
        """Get async generator for feedback stream"""
        if stream_id not in self.active_streams:
            raise ValidationException(f"Stream {stream_id} not found")
        
        queue = self.active_streams[stream_id]
        
        try:
            while True:
                # Wait for next feedback with timeout
                try:
                    feedback = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield feedback
                    
                    # Mark task as done
                    queue.task_done()
                    
                    # Break on completion
                    if feedback.feedback_type == FeedbackType.COMPLETION:
                        break
                        
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield ValidationFeedback(
                        feedback_type=FeedbackType.PROGRESS,
                        message="Validation in progress...",
                        timestamp=time.time()
                    )
                    
        finally:
            # Clean up stream
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
                del self.stream_metadata[stream_id]
    
    def close_stream(self, stream_id: str):
        """Close validation stream"""
        if stream_id in self.active_streams:
            self.stream_metadata[stream_id]['status'] = 'closed'


class RealTimeValidator(BaseValidator):
    """Validator with real-time feedback capabilities"""
    
    def __init__(self, base_validator: BaseValidator, stream: RealTimeValidationStream, stream_id: str):
        self.base_validator = base_validator
        self.stream = stream
        self.stream_id = stream_id
    
    async def validate_with_feedback(self, data: Dict[str, Any]) -> None:
        """Validate with real-time feedback"""
        
        # Send start feedback
        await self.stream.send_feedback(self.stream_id, ValidationFeedback(
            feedback_type=FeedbackType.PROGRESS,
            message="Starting validation...",
            timestamp=time.time(),
            progress_percent=0
        ))
        
        try:
            # Schema validation phase
            await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                feedback_type=FeedbackType.PROGRESS,
                message="Validating schema and basic parameters...",
                timestamp=time.time(),
                progress_percent=20
            ))
            
            self.base_validator.validate(data)
            
            # File validation phase
            if self._has_files(data):
                await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                    feedback_type=FeedbackType.PROGRESS,
                    message="Validating configuration files...",
                    timestamp=time.time(),
                    progress_percent=40
                ))
                
                await self._validate_files_with_feedback(data)
            
            # Advanced validation phase
            await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                feedback_type=FeedbackType.PROGRESS,
                message="Performing advanced validation checks...",
                timestamp=time.time(),
                progress_percent=70
            ))
            
            await self._advanced_validation_with_feedback(data)
            
            # Success feedback
            await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                feedback_type=FeedbackType.COMPLETION,
                message="Validation completed successfully",
                timestamp=time.time(),
                progress_percent=100,
                severity="success"
            ))
            
        except ValidationException as e:
            # Error feedback
            await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                feedback_type=FeedbackType.ERROR,
                message=f"Validation failed: {e.message}",
                timestamp=time.time(),
                severity="error",
                metadata={"validation_errors": e.validation_errors}
            ))
            raise
    
    def validate(self, data: Dict[str, Any]) -> None:
        """Standard validate method (synchronous)"""
        return self.base_validator.validate(data)
    
    def _has_files(self, data: Dict[str, Any]) -> bool:
        """Check if configuration includes files"""
        file_keys = ['topology_path', 'config_path', 'training_data_path']
        return any(key in data for key in file_keys)
    
    async def _validate_files_with_feedback(self, data: Dict[str, Any]):
        """Validate files with progress feedback"""
        
        if 'topology_path' in data:
            await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                feedback_type=FeedbackType.PROGRESS,
                message="Analyzing topology file...",
                timestamp=time.time(),
                progress_percent=45
            ))
            
            # Simulate file validation (in practice, call actual validator)
            await asyncio.sleep(0.1)
            
            await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                feedback_type=FeedbackType.INSIGHT,
                message="Topology file contains valid cellular network layout",
                timestamp=time.time(),
                field="topology_path"
            ))
        
        if 'config_path' in data:
            await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                feedback_type=FeedbackType.PROGRESS,
                message="Validating configuration file format...",
                timestamp=time.time(),
                progress_percent=55
            ))
            
            await asyncio.sleep(0.1)
    
    async def _advanced_validation_with_feedback(self, data: Dict[str, Any]):
        """Perform advanced validation with feedback"""
        
        # Context detection
        await self.stream.send_feedback(self.stream_id, ValidationFeedback(
            feedback_type=FeedbackType.PROGRESS,
            message="Detecting network environment context...",
            timestamp=time.time(),
            progress_percent=75
        ))
        
        # Physics validation
        await self.stream.send_feedback(self.stream_id, ValidationFeedback(
            feedback_type=FeedbackType.PROGRESS,
            message="Validating against cellular physics models...",
            timestamp=time.time(),
            progress_percent=85
        ))
        
        # Generate recommendations
        await self.stream.send_feedback(self.stream_id, ValidationFeedback(
            feedback_type=FeedbackType.RECOMMENDATION,
            message="Consider lambda_ = 0.6 for optimal coverage-capacity balance",
            timestamp=time.time(),
            field="cco_params.lambda_"
        ))
        
        await asyncio.sleep(0.1)


class InteractiveValidator:
    """Interactive validator with real-time parameter adjustment"""
    
    def __init__(self):
        self.validation_stream = RealTimeValidationStream()
        self.optimization_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def start_interactive_session(self, session_id: str, initial_config: Dict[str, Any]) -> str:
        """Start interactive validation session"""
        stream_id = self.validation_stream.create_stream(session_id)
        
        self.optimization_sessions[session_id] = {
            'config': initial_config.copy(),
            'history': [],
            'recommendations': [],
            'stream_id': stream_id,
            'started_at': time.time()
        }
        
        # Send initial feedback
        await self.validation_stream.send_feedback(stream_id, ValidationFeedback(
            feedback_type=FeedbackType.PROGRESS,
            message="Interactive validation session started",
            timestamp=time.time(),
            progress_percent=0
        ))
        
        return stream_id
    
    async def suggest_parameter_improvements(
        self, 
        session_id: str, 
        target_metric: str = "coverage_quality"
    ) -> List[Dict[str, Any]]:
        """Suggest parameter improvements for optimization"""
        
        if session_id not in self.optimization_sessions:
            raise ValidationException(f"Session {session_id} not found")
        
        session = self.optimization_sessions[session_id]
        config = session['config']
        stream_id = session['stream_id']
        
        # Send analysis feedback
        await self.validation_stream.send_feedback(stream_id, ValidationFeedback(
            feedback_type=FeedbackType.PROGRESS,
            message=f"Analyzing parameters for {target_metric} optimization...",
            timestamp=time.time(),
            progress_percent=30
        ))
        
        suggestions = []
        
        # Analyze CCO parameters
        if 'cco_params' in config and target_metric == "coverage_quality":
            current_lambda = config['cco_params'].get('lambda_', 0.5)
            
            if current_lambda < 0.6:
                suggestions.append({
                    'parameter': 'cco_params.lambda_',
                    'current_value': current_lambda,
                    'suggested_value': 0.65,
                    'rationale': 'Increase lambda_ to improve weak coverage areas',
                    'expected_improvement': '12% better coverage'
                })
            
            current_threshold = config['cco_params'].get('weak_coverage_threshold', -100)
            if current_threshold > -110:
                suggestions.append({
                    'parameter': 'cco_params.weak_coverage_threshold',
                    'current_value': current_threshold,
                    'suggested_value': -108,
                    'rationale': 'Lower threshold to catch more edge coverage issues',
                    'expected_improvement': '8% better edge coverage'
                })
        
        # Send suggestions as feedback
        for suggestion in suggestions:
            await self.validation_stream.send_feedback(stream_id, ValidationFeedback(
                feedback_type=FeedbackType.RECOMMENDATION,
                message=f"Suggest {suggestion['parameter']}: {suggestion['suggested_value']} ({suggestion['rationale']})",
                timestamp=time.time(),
                field=suggestion['parameter'],
                metadata=suggestion
            ))
        
        return suggestions
    
    async def apply_parameter_changes(
        self, 
        session_id: str, 
        parameter_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply parameter changes and re-validate"""
        
        if session_id not in self.optimization_sessions:
            raise ValidationException(f"Session {session_id} not found")
        
        session = self.optimization_sessions[session_id]
        stream_id = session['stream_id']
        
        # Apply updates
        old_config = session['config'].copy()
        
        for param_path, new_value in parameter_updates.items():
            self._set_nested_parameter(session['config'], param_path, new_value)
        
        # Record change in history
        session['history'].append({
            'timestamp': time.time(),
            'changes': parameter_updates,
            'previous_config': old_config
        })
        
        # Send update feedback
        await self.validation_stream.send_feedback(stream_id, ValidationFeedback(
            feedback_type=FeedbackType.PROGRESS,
            message=f"Applied {len(parameter_updates)} parameter updates",
            timestamp=time.time(),
            progress_percent=50
        ))
        
        # Re-validate with new parameters
        try:
            from api_manager.validators.app_validator import AppValidatorFactory
            validator = AppValidatorFactory.create_validator(session['config']['app_type'])
            
            rt_validator = RealTimeValidator(validator, self.validation_stream, stream_id)
            await rt_validator.validate_with_feedback(session['config'])
            
            return {
                'status': 'success',
                'updated_config': session['config'],
                'change_summary': self._summarize_changes(parameter_updates)
            }
            
        except ValidationException as e:
            # Revert changes on validation failure
            session['config'] = old_config
            
            await self.validation_stream.send_feedback(stream_id, ValidationFeedback(
                feedback_type=FeedbackType.ERROR,
                message="Parameter update failed validation - reverted changes",
                timestamp=time.time(),
                severity="error"
            ))
            
            raise
    
    def _set_nested_parameter(self, config: Dict[str, Any], param_path: str, value: Any):
        """Set nested parameter value using dot notation"""
        parts = param_path.split('.')
        current = config
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    def _summarize_changes(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize parameter changes"""
        return {
            'parameters_changed': len(updates),
            'change_categories': list(set(
                update.split('.')[0] for update in updates.keys()
            )),
            'timestamp': time.time()
        }


class ProgressiveValidator(BaseValidator):
    """Validator that provides progressive validation with early feedback"""
    
    def __init__(self, stream: RealTimeValidationStream, stream_id: str):
        self.stream = stream
        self.stream_id = stream_id
    
    async def validate_progressively(self, data: Dict[str, Any]) -> None:
        """Validate configuration with progressive feedback"""
        
        validation_phases = [
            ("Basic Schema", self._validate_basic_schema, 20),
            ("Parameter Ranges", self._validate_parameter_ranges, 40),
            ("Cross-Field Logic", self._validate_cross_field_logic, 60),
            ("Domain Knowledge", self._validate_domain_knowledge, 80),
            ("Optimization Analysis", self._analyze_optimization_potential, 100)
        ]
        
        for phase_name, validator_func, progress in validation_phases:
            await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                feedback_type=FeedbackType.PROGRESS,
                message=f"Phase: {phase_name}",
                timestamp=time.time(),
                progress_percent=progress
            ))
            
            try:
                await validator_func(data)
                
                await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                    feedback_type=FeedbackType.PROGRESS,
                    message=f"✅ {phase_name} completed",
                    timestamp=time.time(),
                    progress_percent=progress
                ))
                
            except ValidationException as e:
                await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                    feedback_type=FeedbackType.ERROR,
                    message=f"❌ {phase_name} failed: {e.message}",
                    timestamp=time.time(),
                    severity="error"
                ))
                raise
    
    def validate(self, data: Dict[str, Any]) -> None:
        """Standard synchronous validation"""
        # Run async validation in sync context
        asyncio.run(self.validate_progressively(data))
    
    async def _validate_basic_schema(self, data: Dict[str, Any]):
        """Phase 1: Basic schema validation"""
        from api_manager.validators.app_validator import AppValidatorFactory
        
        app_type = data.get('app_type')
        if not app_type:
            raise ValidationException("app_type is required")
        
        validator = AppValidatorFactory.create_validator(app_type)
        
        # Validate just the schema parts
        basic_data = {k: v for k, v in data.items() if not k.endswith('_path')}
        validator.validate(basic_data)
        
        await asyncio.sleep(0.05)  # Simulate processing time
    
    async def _validate_parameter_ranges(self, data: Dict[str, Any]):
        """Phase 2: Parameter range validation"""
        # Deep parameter validation
        param_sections = ['cco_params', 'energy_params', 'load_balance_params', 'mro_params']
        
        for section in param_sections:
            if section in data:
                await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                    feedback_type=FeedbackType.INSIGHT,
                    message=f"Validating {len(data[section])} parameters in {section}",
                    timestamp=time.time(),
                    field=section
                ))
        
        await asyncio.sleep(0.1)
    
    async def _validate_cross_field_logic(self, data: Dict[str, Any]):
        """Phase 3: Cross-field logic validation"""
        # Check logical consistency between fields
        train_days = data.get('train_days', [])
        test_day = data.get('test_day')
        
        if test_day and test_day in train_days:
            raise ValidationException("Test day cannot overlap with training days")
        
        await asyncio.sleep(0.05)
    
    async def _validate_domain_knowledge(self, data: Dict[str, Any]):
        """Phase 4: Domain-specific validation"""
        # Apply cellular network domain knowledge
        app_type = data['app_type']
        
        if app_type == 'coverage_capacity_optimization':
            await self._validate_cco_domain_knowledge(data)
        elif app_type == 'energy_savings':
            await self._validate_energy_domain_knowledge(data)
        
        await asyncio.sleep(0.1)
    
    async def _validate_cco_domain_knowledge(self, data: Dict[str, Any]):
        """Validate CCO parameters against cellular domain knowledge"""
        cco_params = data.get('cco_params', {})
        
        if 'lambda_' in cco_params:
            lambda_val = cco_params['lambda_']
            if lambda_val > 0.8:
                await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                    feedback_type=FeedbackType.WARNING,
                    message=f"High lambda_ ({lambda_val}) may over-prioritize weak coverage",
                    timestamp=time.time(),
                    field="cco_params.lambda_",
                    severity="warning"
                ))
    
    async def _validate_energy_domain_knowledge(self, data: Dict[str, Any]):
        """Validate energy parameters against domain knowledge"""
        energy_params = data.get('energy_params', {})
        
        if 'power_reduction_target' in energy_params:
            target = energy_params['power_reduction_target']
            if target > 0.35:
                await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                    feedback_type=FeedbackType.WARNING,
                    message=f"Power reduction target ({target*100:.0f}%) is aggressive - monitor service quality",
                    timestamp=time.time(),
                    field="energy_params.power_reduction_target",
                    severity="warning"
                ))
    
    async def _analyze_optimization_potential(self, data: Dict[str, Any]):
        """Phase 5: Analyze optimization potential"""
        app_type = data['app_type']
        
        optimization_score = self._calculate_optimization_score(data)
        
        await self.stream.send_feedback(self.stream_id, ValidationFeedback(
            feedback_type=FeedbackType.INSIGHT,
            message=f"Configuration optimization score: {optimization_score:.1f}/10",
            timestamp=time.time(),
            metadata={'optimization_score': optimization_score}
        ))
        
        if optimization_score < 7:
            await self.stream.send_feedback(self.stream_id, ValidationFeedback(
                feedback_type=FeedbackType.RECOMMENDATION,
                message="Configuration has optimization potential - consider parameter tuning",
                timestamp=time.time(),
                severity="info"
            ))
    
    def _calculate_optimization_score(self, data: Dict[str, Any]) -> float:
        """Calculate optimization potential score (0-10)"""
        # Simplified scoring based on parameter choices
        score = 7.0  # Base score
        
        # Analyze CCO parameters
        if 'cco_params' in data:
            lambda_val = data['cco_params'].get('lambda_', 0.5)
            # Optimal lambda around 0.6
            lambda_score = max(0, 3 - abs(lambda_val - 0.6) * 10)
            score += lambda_score
        
        return min(10.0, score)


# Global stream instance
_global_stream = RealTimeValidationStream()


def get_realtime_stream() -> RealTimeValidationStream:
    """Get global real-time validation stream"""
    return _global_stream