# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from api_manager.validators.base_validator import BaseValidator
from api_manager.validators.app_validator import AppValidatorFactory
from api_manager.validators.context_aware_validator import (
    ContextAwareValidator, SmartParameterValidator, PhysicsBasedValidator, MLPoweredValidator
)
from api_manager.validators.validation_cache import create_cached_validator, get_validation_cache
from api_manager.validators.async_validator import AsyncValidationPipeline
from api_manager.exceptions.validation_exception import ValidationException

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation thoroughness levels"""
    BASIC = "basic"           # Schema validation only
    STANDARD = "standard"     # Schema + domain validation  
    COMPREHENSIVE = "comprehensive"  # Standard + context-aware + physics
    EXPERT = "expert"         # All validations + ML insights


@dataclass
class ValidationPlan:
    """Plan for validation execution"""
    level: ValidationLevel
    validators: List[str]
    estimated_time_ms: float
    requires_async: bool
    cache_enabled: bool
    
    
@dataclass 
class ValidationResult:
    """Complete validation result with insights"""
    success: bool
    validation_time: float
    level: ValidationLevel
    errors: List[Dict[str, Any]]
    warnings: List[str]
    recommendations: List[str]
    insights: Optional[Dict[str, Any]]
    cache_hit: bool
    context_detected: Optional[str]


class ValidationOrchestrator:
    """Orchestrates validation workflow across multiple validators"""
    
    def __init__(self):
        self.async_pipeline = None
        self.validation_cache = get_validation_cache()
        
    async def validate_configuration(
        self,
        data: Dict[str, Any],
        level: ValidationLevel = ValidationLevel.STANDARD,
        enable_cache: bool = True,
        enable_async: bool = False
    ) -> ValidationResult:
        """Orchestrate complete validation workflow"""
        
        start_time = time.time()
        app_type = data.get('app_type', 'unknown')
        
        logger.info(f"Starting {level.value} validation for {app_type}")
        
        # Create validation plan
        plan = self._create_validation_plan(data, level, enable_async)
        
        try:
            if enable_async and plan.requires_async:
                result = await self._async_validation_workflow(data, plan, enable_cache)
            else:
                result = await self._sync_validation_workflow(data, plan, enable_cache)
            
            result.validation_time = time.time() - start_time
            result.level = level
            
            logger.info(f"Validation completed in {result.validation_time:.3f}s (cache_hit: {result.cache_hit})")
            return result
            
        except Exception as e:
            logger.error(f"Validation orchestration failed: {e}")
            return ValidationResult(
                success=False,
                validation_time=time.time() - start_time,
                level=level,
                errors=[{"field": "orchestrator", "error": str(e)}],
                warnings=[],
                recommendations=[],
                insights=None,
                cache_hit=False,
                context_detected=None
            )
    
    def _create_validation_plan(
        self, 
        data: Dict[str, Any], 
        level: ValidationLevel,
        enable_async: bool
    ) -> ValidationPlan:
        """Create execution plan for validation"""
        
        validators = ["schema"]  # Always include schema validation
        estimated_time = 5  # Base time in ms
        requires_async = False
        
        if level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE, ValidationLevel.EXPERT]:
            validators.append("domain_specific")
            estimated_time += 10
        
        if level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.EXPERT]:
            validators.extend(["context_aware", "physics_based"])
            estimated_time += 50
            
            # Check if we need async processing
            if self._has_large_datasets(data):
                requires_async = True
                estimated_time += 200
        
        if level == ValidationLevel.EXPERT:
            validators.extend(["smart_parameter", "ml_powered"])
            estimated_time += 100
        
        return ValidationPlan(
            level=level,
            validators=validators,
            estimated_time_ms=estimated_time,
            requires_async=requires_async,
            cache_enabled=True
        )
    
    async def _sync_validation_workflow(
        self, 
        data: Dict[str, Any], 
        plan: ValidationPlan,
        enable_cache: bool
    ) -> ValidationResult:
        """Execute synchronous validation workflow"""
        
        app_type = data['app_type']
        errors = []
        warnings = []
        recommendations = []
        insights = {}
        cache_hit = False
        context_detected = None
        
        # 1. Schema validation (always required)
        try:
            base_validator = AppValidatorFactory.create_validator(app_type)
            
            if enable_cache:
                cached_validator = create_cached_validator(base_validator)
                cached_validator.validate(data)
                # Check if it was a cache hit (simplified)
                cache_hit = len(self.validation_cache.cache) > 0
            else:
                base_validator.validate(data)
                
        except ValidationException as e:
            errors.extend(e.validation_errors or [{"field": "schema", "error": e.message}])
        
        # 2. Context-aware validation
        if "context_aware" in plan.validators:
            try:
                context_validator = ContextAwareValidator()
                context_validator.validate(data)
                context_detected = getattr(context_validator, 'context', None)
                if context_detected:
                    context_detected = context_detected.value
            except ValidationException as e:
                errors.extend(e.validation_errors or [{"field": "context", "error": e.message}])
        
        # 3. Physics-based validation
        if "physics_based" in plan.validators:
            try:
                physics_validator = PhysicsBasedValidator()
                physics_validator.validate(data)
            except ValidationException as e:
                errors.extend(e.validation_errors or [{"field": "physics", "error": e.message}])
        
        # 4. Smart parameter analysis
        if "smart_parameter" in plan.validators:
            try:
                smart_validator = SmartParameterValidator()
                smart_validator.validate(data)
                
                # Extract insights
                validation_insights = data.get('_validation_insights', {})
                recommendations.extend(validation_insights.get('recommendations', []))
                warnings.extend(validation_insights.get('warnings', []))
                
            except ValidationException as e:
                errors.extend(e.validation_errors or [{"field": "smart_analysis", "error": e.message}])
        
        # 5. ML-powered validation
        if "ml_powered" in plan.validators:
            try:
                ml_validator = MLPoweredValidator()
                ml_validator.validate(data)
                
                # Extract ML insights
                ml_insights = data.get('_ml_insights', {})
                if ml_insights:
                    insights['ml_performance'] = ml_insights.get('predicted_performance')
                    recommendations.extend(ml_insights.get('recommendations', []))
                    
            except ValidationException as e:
                errors.extend(e.validation_errors or [{"field": "ml_analysis", "error": e.message}])
        
        return ValidationResult(
            success=len(errors) == 0,
            validation_time=0,  # Will be set by caller
            level=plan.level,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            insights=insights if insights else None,
            cache_hit=cache_hit,
            context_detected=context_detected
        )
    
    async def _async_validation_workflow(
        self, 
        data: Dict[str, Any], 
        plan: ValidationPlan,
        enable_cache: bool
    ) -> ValidationResult:
        """Execute asynchronous validation workflow"""
        
        if not self.async_pipeline:
            self.async_pipeline = AsyncValidationPipeline()
            await self.async_pipeline.start_workers()
        
        # Submit validation task
        task_id = f"validation_{int(time.time() * 1000)}"
        await self.async_pipeline.submit_validation(
            task_id=task_id,
            validator_type=data['app_type'],
            data=data
        )
        
        # Wait for completion
        task = await self.async_pipeline.wait_for_completion(task_id)
        
        if task.status.value == "completed":
            return ValidationResult(
                success=True,
                validation_time=task.completed_at - task.started_at,
                level=plan.level,
                errors=[],
                warnings=[],
                recommendations=[],
                insights=task.result,
                cache_hit=False,
                context_detected=None
            )
        else:
            return ValidationResult(
                success=False,
                validation_time=task.completed_at - task.started_at if task.completed_at else 0,
                level=plan.level,
                errors=[{"field": "async_task", "error": task.error}],
                warnings=[],
                recommendations=[],
                insights=None,
                cache_hit=False,
                context_detected=None
            )
    
    def _has_large_datasets(self, data: Dict[str, Any]) -> bool:
        """Check if validation involves large datasets requiring async processing"""
        # Check for large file indicators
        file_indicators = [
            'topology_path', 'config_path', 'training_data_path'
        ]
        
        for indicator in file_indicators:
            if indicator in data:
                return True
        
        # Check for large parameter sets
        param_sections = ['cco_params', 'energy_params', 'mro_params']
        total_params = sum(
            len(data.get(section, {})) for section in param_sections
        )
        
        return total_params > 20  # Large configuration
    
    async def validate_batch_configurations(
        self,
        configs: List[Dict[str, Any]],
        level: ValidationLevel = ValidationLevel.STANDARD
    ) -> List[ValidationResult]:
        """Validate multiple configurations in parallel"""
        
        logger.info(f"Starting batch validation of {len(configs)} configurations")
        
        # Create tasks for all configurations
        tasks = []
        for i, config in enumerate(configs):
            task = asyncio.create_task(
                self.validate_configuration(
                    config, 
                    level=level,
                    enable_async=True
                )
            )
            tasks.append(task)
        
        # Wait for all validations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        validation_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                validation_results.append(ValidationResult(
                    success=False,
                    validation_time=0,
                    level=level,
                    errors=[{"field": "batch_error", "error": str(result)}],
                    warnings=[],
                    recommendations=[],
                    insights=None,
                    cache_hit=False,
                    context_detected=None
                ))
            else:
                validation_results.append(result)
        
        # Summary statistics
        successful = sum(1 for r in validation_results if r.success)
        logger.info(f"Batch validation completed: {successful}/{len(configs)} successful")
        
        return validation_results
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation system performance metrics"""
        cache_stats = self.validation_cache.get_stats()
        
        return {
            'cache_stats': cache_stats,
            'async_pipeline_active': self.async_pipeline is not None,
            'supported_validation_levels': [level.value for level in ValidationLevel],
            'supported_app_types': AppValidatorFactory.get_supported_app_types()
        }


# Global orchestrator instance
_global_orchestrator = None


async def get_validation_orchestrator() -> ValidationOrchestrator:
    """Get global validation orchestrator"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = ValidationOrchestrator()
    return _global_orchestrator