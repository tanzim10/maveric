# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import logging

from api_manager.validators.base_validator import BaseValidator
from api_manager.exceptions.validation_exception import ValidationException, FileValidationException

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ValidationTask:
    """Represents a validation task"""
    task_id: str
    validator_type: str
    data: Dict[str, Any]
    status: ValidationStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0


class AsyncValidationPipeline:
    """Async validation pipeline for large-scale validation tasks"""

    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.tasks: Dict[str, ValidationTask] = {}
        self.task_queue = asyncio.Queue()
        self.worker_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._workers_started = False

    async def start_workers(self):
        """Start background worker tasks"""
        if self._workers_started:
            return

        self._workers_started = True
        workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_concurrent_tasks)
        ]
        logger.info(f"Started {len(workers)} async validation workers")

    async def submit_validation(
        self,
        task_id: str,
        validator_type: str,
        data: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Submit validation task to async pipeline"""
        task = ValidationTask(
            task_id=task_id,
            validator_type=validator_type,
            data=data,
            status=ValidationStatus.PENDING,
            created_at=time.time()
        )

        self.tasks[task_id] = task
        await self.task_queue.put((priority, task))

        logger.info(f"Submitted validation task: {task_id} ({validator_type})")
        return task_id

    async def get_task_status(self, task_id: str) -> Optional[ValidationTask]:
        """Get status of validation task"""
        return self.tasks.get(task_id)

    async def wait_for_completion(self, task_id: str, timeout: float = 300.0) -> ValidationTask:
        """Wait for task completion with timeout"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            task = self.tasks.get(task_id)
            if not task:
                raise ValidationException(f"Task {task_id} not found")

            if task.status in [ValidationStatus.COMPLETED, ValidationStatus.FAILED]:
                return task

            await asyncio.sleep(0.1)  # Check every 100ms

        raise ValidationException(f"Task {task_id} timed out after {timeout}s")

    async def _worker(self, worker_name: str):
        """Background worker for processing validation tasks"""
        logger.info(f"Async validation worker {worker_name} started")

        while True:
            try:
                # Get next task from queue
                priority, task = await self.task_queue.get()

                async with self.worker_semaphore:
                    await self._process_task(task, worker_name)

                self.task_queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.exception(f"Worker {worker_name} error: {e}")
                continue

    async def _process_task(self, task: ValidationTask, worker_name: str):
        """Process a single validation task"""
        task.status = ValidationStatus.RUNNING
        task.started_at = time.time()

        logger.info(f"Worker {worker_name} processing task {task.task_id}")

        try:
            # Import and create validator
            from api_manager.validators.app_validator import AppValidatorFactory
            validator = AppValidatorFactory.create_validator(task.validator_type)

            # For file-heavy validation, use async file processing
            if self._has_large_files(task.data):
                await self._async_file_validation(task, validator)
            else:
                # Standard synchronous validation for small configs
                validator.validate(task.data)

            task.status = ValidationStatus.COMPLETED
            task.completed_at = time.time()
            task.progress = 1.0
            task.result = {
                "status": "success",
                "validation_time": task.completed_at - task.started_at,
                "worker": worker_name
            }

            logger.info(f"Task {task.task_id} completed successfully")

        except Exception as e:
            task.status = ValidationStatus.FAILED
            task.completed_at = time.time()
            task.error = str(e)

            logger.error(f"Task {task.task_id} failed: {e}")

    def _has_large_files(self, data: Dict[str, Any]) -> bool:
        """Check if validation involves large files that need async processing"""
        file_paths = [
            data.get('topology_path'),
            data.get('config_path'),
            data.get('training_data_path')
        ]

        for path in file_paths:
            if path and isinstance(path, str):
                try:
                    import os
                    if os.path.exists(path):
                        size_mb = os.path.getsize(path) / (1024 * 1024)
                        if size_mb > 10:  # Files larger than 10MB
                            return True
                except OSError:
                    pass

        return False

    async def _async_file_validation(self, task: ValidationTask, validator: BaseValidator):
        """Perform async file validation for large files"""
        # Update progress as we validate different components
        task.progress = 0.1

        # Validate app parameters first (fast)
        validator.validate(task.data)
        task.progress = 0.3

        # Async file validation
        if task.data.get('topology_path'):
            await self._validate_topology_async(task.data['topology_path'])
            task.progress = 0.6

        if task.data.get('config_path'):
            await self._validate_config_async(task.data['config_path'])
            task.progress = 0.8

        # Complete any remaining validation
        task.progress = 1.0

    async def _validate_topology_async(self, file_path: str):
        """Async topology file validation"""
        # Simplified async file validation without external dependencies
        await asyncio.sleep(0.1)  # Simulate async file processing

        # Use standard file reading for now
        try:
            import pandas as pd
            df = pd.read_csv(file_path)

            # Validate in chunks for large files
            chunk_size = 10000
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                await self._process_topology_chunk_df(chunk)
                await asyncio.sleep(0.001)  # Yield control

        except Exception as e:
            from api_manager.exceptions.validation_exception import FileValidationException
            raise FileValidationException(f"Async topology validation failed: {str(e)}")

    async def _process_topology_chunk_df(self, chunk_df: pd.DataFrame):
        """Process a chunk of topology data as DataFrame"""
        # Validate chunk data ranges
        errors = []

        # Validate coordinates
        if 'cell_lat' in chunk_df.columns:
            invalid_lat = chunk_df[(chunk_df['cell_lat'] < -90) | (chunk_df['cell_lat'] > 90)]
            if not invalid_lat.empty:
                errors.append(f"Invalid latitude in {len(invalid_lat)} rows")

        if 'cell_lon' in chunk_df.columns:
            invalid_lon = chunk_df[(chunk_df['cell_lon'] < -180) | (chunk_df['cell_lon'] > 180)]
            if not invalid_lon.empty:
                errors.append(f"Invalid longitude in {len(invalid_lon)} rows")

        if errors:
            from api_manager.exceptions.validation_exception import FileValidationException
            raise FileValidationException(f"Topology chunk validation failed: {'; '.join(errors)}")

        # Yield control to allow other tasks to run
        await asyncio.sleep(0.001)

    async def _process_topology_chunk(self, chunk_data: List[str], header: str):
        """Process a chunk of topology data"""
        # Create temporary dataframe from chunk
        import io
        chunk_content = header + ''.join(chunk_data)
        df = pd.read_csv(io.StringIO(chunk_content))

        # Validate chunk data ranges
        errors = []

        # Validate coordinates
        invalid_lat = df[(df['cell_lat'] < -90) | (df['cell_lat'] > 90)]
        if not invalid_lat.empty:
            errors.append(f"Invalid latitude in {len(invalid_lat)} rows")

        invalid_lon = df[(df['cell_lon'] < -180) | (df['cell_lon'] > 180)]
        if not invalid_lon.empty:
            errors.append(f"Invalid longitude in {len(invalid_lon)} rows")

        if errors:
            raise FileValidationException(f"Topology chunk validation failed: {'; '.join(errors)}")

        # Yield control to allow other tasks to run
        await asyncio.sleep(0.001)

    async def _validate_config_async(self, file_path: str):
        """Async configuration file validation"""
        # Simulate async file reading
        await asyncio.sleep(0.05)

        # Use standard file reading for now
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Parse and validate config content
            import json
            config_data = json.loads(content)
            # Validate config structure
            await self._validate_config_content_async(config_data)
        except json.JSONDecodeError as e:
            from api_manager.exceptions.validation_exception import FileValidationException
            raise FileValidationException(f"Invalid JSON config: {str(e)}")
        except Exception as e:
            from api_manager.exceptions.validation_exception import FileValidationException
            raise FileValidationException(f"Config validation failed: {str(e)}")

    async def _validate_config_content_async(self, config: Dict[str, Any]):
        """Async validation of configuration content"""
        # Validate RF config if present
        if 'rf_config' in config:
            await self._validate_rf_config_async(config['rf_config'])

        # Validate other config sections
        await asyncio.sleep(0.001)  # Yield control


# Global async pipeline instance
_async_pipeline = None


async def get_async_pipeline() -> AsyncValidationPipeline:
    """Get or create global async validation pipeline"""
    global _async_pipeline
    if _async_pipeline is None:
        _async_pipeline = AsyncValidationPipeline()
        await _async_pipeline.start_workers()
    return _async_pipeline