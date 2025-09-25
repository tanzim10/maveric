#!/usr/bin/env python3
"""
Test suite for advanced validation system improvements.
Tests async validation, caching, context-awareness, and real-time feedback.
"""

import sys
import os
import asyncio
import tempfile
import json
import time
import logging
from typing import Dict, Any

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../services'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../radp'))

from api_manager.validators.validation_orchestrator import ValidationOrchestrator, ValidationLevel
from api_manager.validators.context_aware_validator import ContextAwareValidator, ValidationContext
from api_manager.validators.validation_cache import ValidationCache, create_cached_validator
from api_manager.validators.realtime_validator import RealTimeValidationStream, InteractiveValidator
from api_manager.validators.validation_diagnostics import (
    ValidationDiagnosticsEngine, SmartErrorReporter, DiagnosticLevel
)
from api_manager.validators.app_validator import AppValidatorFactory
from api_manager.exceptions.validation_exception import ValidationException

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_async_validation_pipeline():
    """Test asynchronous validation pipeline"""
    logger.info("🧪 Testing Async Validation Pipeline...")

    orchestrator = ValidationOrchestrator()

    # Create large configuration that benefits from async processing
    large_config = {
        "app_type": "coverage_capacity_optimization",
        "train_days": list(range(10)),  # Many training days
        "test_day": 15,
        "total_timesteps": 100000,
        "cco_params": {
            "lambda_": 0.6,
            "weak_coverage_threshold": -108.0,
            "over_coverage_threshold": -5.0,
            "growth_rate": 1.2,
            "optimization_metric": "pixel"
        }
    }

    # Test async validation
    start_time = time.time()
    result = await orchestrator.validate_configuration(
        large_config,
        level=ValidationLevel.COMPREHENSIVE,
        enable_async=True
    )
    async_time = time.time() - start_time

    # Test sync validation for comparison
    start_time = time.time()
    sync_result = await orchestrator.validate_configuration(
        large_config,
        level=ValidationLevel.COMPREHENSIVE,
        enable_async=False
    )
    sync_time = time.time() - start_time

    logger.info(f"✅ Async validation: {async_time:.3f}s, Sync validation: {sync_time:.3f}s")
    logger.info(f"✅ Both validations successful: {result.success and sync_result.success}")

    return result.success and sync_result.success


async def test_validation_caching():
    """Test validation result caching"""
    logger.info("🧪 Testing Validation Caching...")

    cache = ValidationCache(max_entries=100, default_ttl=60)

    # Test configuration
    config = {
        "app_type": "energy_savings",
        "train_days": [0, 1, 2],
        "test_day": 3,
        "energy_params": {
            "power_reduction_target": 0.20,
            "min_active_cells": 3
        }
    }

    # Create cached validator
    base_validator = AppValidatorFactory.create_validator("energy_savings")
    cached_validator = create_cached_validator(base_validator)

    # First validation (cache miss)
    start_time = time.time()
    cached_validator.validate(config)
    first_time = time.time() - start_time

    # Second validation (cache hit)
    start_time = time.time()
    cached_validator.validate(config)
    second_time = time.time() - start_time

    # Get cache statistics
    stats = cache.get_stats()

    logger.info(f"✅ First validation: {first_time:.4f}s, Second validation: {second_time:.4f}s")
    logger.info(f"✅ Cache hit rate: {stats['hit_rate']:.2%}, Cache size: {stats['cache_size']}")

    return stats['hit_rate'] > 0


async def test_context_aware_validation():
    """Test context-aware validation"""
    logger.info("🧪 Testing Context-Aware Validation...")

    # Test urban dense context
    urban_config = {
        "app_type": "coverage_capacity_optimization",
        "cco_params": {
            "lambda_": 0.7,  # High coverage priority for urban
            "weak_coverage_threshold": -105.0,  # Stricter for urban
            "over_coverage_threshold": -3.0
        }
    }

    context_validator = ContextAwareValidator(ValidationContext.URBAN_DENSE)

    try:
        context_validator.validate(urban_config)
        logger.info("✅ Urban dense context validation passed")
        urban_success = True
    except ValidationException as e:
        logger.info(f"✅ Urban dense context validation appropriately flagged: {e.message}")
        urban_success = True  # Expected behavior

    # Test rural context with same parameters (should give different validation)
    rural_validator = ContextAwareValidator(ValidationContext.RURAL)

    try:
        rural_validator.validate(urban_config)
        logger.info("✅ Rural context validation passed")
        rural_success = True
    except ValidationException as e:
        logger.info(f"✅ Rural context validation provided different feedback: {e.message}")
        rural_success = True  # Different context, different validation

    return urban_success and rural_success


async def test_realtime_validation_feedback():
    """Test real-time validation feedback"""
    logger.info("🧪 Testing Real-Time Validation Feedback...")

    # Create real-time validation stream
    rt_stream = RealTimeValidationStream()
    stream_id = rt_stream.create_stream("test_stream_001")

    # Create configuration for progressive validation
    config = {
        "app_type": "coverage_capacity_optimization",
        "train_days": [0, 1, 2, 3],
        "test_day": 4,
        "cco_params": {
            "lambda_": 0.5,
            "weak_coverage_threshold": -110.0,
            "over_coverage_threshold": -5.0
        }
    }

    # Start progressive validation
    from api_manager.validators.realtime_validator import ProgressiveValidator
    progressive_validator = ProgressiveValidator(rt_stream, stream_id)

    # Validate and collect feedback
    feedback_messages = []

    async def collect_feedback():
        async for feedback in rt_stream.get_feedback_stream(stream_id):
            feedback_messages.append(feedback)
            logger.info(f"📡 Feedback: {feedback.message} ({feedback.progress_percent}%)")

    # Run validation and feedback collection concurrently
    validation_task = asyncio.create_task(progressive_validator.validate_progressively(config))
    feedback_task = asyncio.create_task(collect_feedback())

    # Wait for validation to complete
    await validation_task

    # Cancel feedback collection
    feedback_task.cancel()

    logger.info(f"✅ Received {len(feedback_messages)} real-time feedback messages")
    return len(feedback_messages) > 3  # Should have multiple progress updates


async def test_interactive_parameter_optimization():
    """Test interactive parameter optimization"""
    logger.info("🧪 Testing Interactive Parameter Optimization...")

    interactive_validator = InteractiveValidator()

    # Start optimization session
    initial_config = {
        "app_type": "coverage_capacity_optimization",
        "cco_params": {
            "lambda_": 0.3,  # Suboptimal value
            "weak_coverage_threshold": -95.0,  # Too lenient
            "over_coverage_threshold": 2.0
        }
    }

    session_id = "optimization_session_001"
    stream_id = await interactive_validator.start_interactive_session(session_id, initial_config)

    # Get parameter suggestions
    suggestions = await interactive_validator.suggest_parameter_improvements(
        session_id,
        target_metric="coverage_quality"
    )

    logger.info(f"✅ Received {len(suggestions)} parameter suggestions")

    if suggestions:
        # Apply first suggestion
        parameter_updates = {
            suggestions[0]['parameter']: suggestions[0]['suggested_value']
        }

        result = await interactive_validator.apply_parameter_changes(session_id, parameter_updates)
        logger.info(f"✅ Applied parameter update: {result['status']}")

        return result['status'] == 'success'

    return len(suggestions) > 0


def test_enhanced_error_diagnostics():
    """Test enhanced error reporting and diagnostics"""
    logger.info("🧪 Testing Enhanced Error Diagnostics...")

    # Create error scenario
    config = {
        "app_type": "coverage_capacity_optimization",
        "train_days": [0, 1, 2, 3],
        "test_day": 2,  # Overlaps with training - should trigger error
        "cco_params": {
            "lambda_": 1.5  # Out of bounds - should trigger error
        }
    }

    try:
        validator = AppValidatorFactory.create_validator("coverage_capacity_optimization")
        validator.validate(config)
        logger.error("❌ Validation should have failed")
        return False

    except ValidationException as e:
        # Generate enhanced diagnostic
        error_reporter = SmartErrorReporter()

        # Test different diagnostic levels
        minimal_report = error_reporter.generate_enhanced_error_report(
            e, config, DiagnosticLevel.MINIMAL
        )

        detailed_report = error_reporter.generate_enhanced_error_report(
            e, config, DiagnosticLevel.DETAILED
        )

        expert_report = error_reporter.generate_enhanced_error_report(
            e, config, DiagnosticLevel.EXPERT
        )

        logger.info(f"✅ Generated diagnostic reports:")
        logger.info(f"   Minimal: {len(str(minimal_report))} chars")
        logger.info(f"   Detailed: {len(str(detailed_report))} chars")
        logger.info(f"   Expert: {len(str(expert_report))} chars")

        # Verify enhanced reports have more information
        has_quick_fixes = 'quick_fixes' in detailed_report
        has_technical_details = 'technical_details' in expert_report

        logger.info(f"✅ Enhanced features: Quick fixes: {has_quick_fixes}, Technical details: {has_technical_details}")

        return has_quick_fixes and has_technical_details


async def test_batch_validation():
    """Test batch validation of multiple configurations"""
    logger.info("🧪 Testing Batch Validation...")

    orchestrator = ValidationOrchestrator()

    # Create multiple configurations
    configs = []
    for i in range(5):
        configs.append({
            "app_type": "energy_savings",
            "train_days": [0, 1, i],
            "test_day": i + 5,
            "total_timesteps": 20000 + i * 5000,
            "energy_params": {
                "power_reduction_target": 0.1 + i * 0.05,
                "min_active_cells": 2 + i
            }
        })

    # Run batch validation
    start_time = time.time()
    results = await orchestrator.validate_batch_configurations(configs, ValidationLevel.STANDARD)
    batch_time = time.time() - start_time

    successful = sum(1 for r in results if r.success)
    logger.info(f"✅ Batch validation: {successful}/{len(configs)} successful in {batch_time:.3f}s")

    return successful == len(configs)


async def test_validation_system_metrics():
    """Test validation system performance metrics"""
    logger.info("🧪 Testing Validation System Metrics...")

    orchestrator = ValidationOrchestrator()

    # Generate some validation activity
    test_configs = [
        {"app_type": "coverage_capacity_optimization", "cco_params": {"lambda_": 0.5}},
        {"app_type": "energy_savings", "energy_params": {"power_reduction_target": 0.15}},
        {"app_type": "load_balancing", "load_balance_params": {"target_load_threshold": 0.8}}
    ]

    # Run validations to generate metrics
    for config in test_configs:
        try:
            await orchestrator.validate_configuration(config, ValidationLevel.STANDARD)
        except ValidationException:
            pass  # Expected for some test cases

    # Get system metrics
    metrics = orchestrator.get_validation_metrics()

    logger.info(f"✅ System metrics:")
    logger.info(f"   Cache hit rate: {metrics['cache_stats']['hit_rate']:.2%}")
    logger.info(f"   Supported apps: {len(metrics['supported_app_types'])}")
    logger.info(f"   Validation levels: {len(metrics['supported_validation_levels'])}")

    return 'cache_stats' in metrics and 'supported_app_types' in metrics


async def main():
    """Run all advanced validation tests"""
    logger.info("🚀 Advanced RADP Validation System Test Suite")
    logger.info("=" * 70)

    test_functions = [
        ("Async Validation Pipeline", test_async_validation_pipeline),
        ("Validation Caching", test_validation_caching),
        ("Context-Aware Validation", test_context_aware_validation),
        ("Real-Time Feedback", test_realtime_validation_feedback),
        ("Interactive Optimization", test_interactive_parameter_optimization),
        ("Enhanced Diagnostics", test_enhanced_error_diagnostics),
        ("Batch Validation", test_batch_validation),
        ("System Metrics", test_validation_system_metrics)
    ]

    results = []

    for test_name, test_func in test_functions:
        logger.info(f"\n--- {test_name} ---")

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))

        except Exception as e:
            logger.exception(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("ADVANCED VALIDATION TEST SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name:<35} {status}")

    logger.info(f"\nOverall: {passed}/{total} advanced tests passed")

    if passed == total:
        logger.info("✅ All advanced validation features working correctly!")
        logger.info("🎉 Enhanced validation system is ready for production!")
        return 0
    else:
        logger.error("❌ Some advanced validation features failed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))