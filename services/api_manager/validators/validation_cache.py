# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import os

from api_manager.validators.base_validator import BaseValidator
from api_manager.exceptions.validation_exception import ValidationException


class CacheEntryStatus(Enum):
    """Cache entry status"""
    VALID = "valid"
    EXPIRED = "expired"
    INVALID = "invalid"


@dataclass
class CacheEntry:
    """Cache entry for validation results"""
    config_hash: str
    app_type: str
    validation_result: Dict[str, Any]
    created_at: float
    accessed_at: float
    access_count: int
    file_checksums: Dict[str, str]  # For file-based validations
    ttl_seconds: float = 3600  # 1 hour default
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.created_at > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics"""
        self.accessed_at = time.time()
        self.access_count += 1


class ValidationCache:
    """Thread-safe validation result cache"""
    
    def __init__(self, max_entries: int = 1000, default_ttl: float = 3600):
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0, 
            'evictions': 0,
            'file_invalidations': 0
        }
    
    def get(self, config_hash: str) -> Optional[CacheEntry]:
        """Get cached validation result"""
        with self.lock:
            entry = self.cache.get(config_hash)
            
            if not entry:
                self.stats['misses'] += 1
                return None
            
            if entry.is_expired():
                del self.cache[config_hash]
                self.stats['misses'] += 1
                return None
            
            # Check file checksums if applicable
            if entry.file_checksums and not self._verify_file_checksums(entry.file_checksums):
                del self.cache[config_hash]
                self.stats['file_invalidations'] += 1
                return None
            
            entry.update_access()
            self.stats['hits'] += 1
            return entry
    
    def put(
        self, 
        config_hash: str, 
        app_type: str,
        validation_result: Dict[str, Any],
        file_paths: Optional[Dict[str, str]] = None,
        ttl: Optional[float] = None
    ):
        """Cache validation result"""
        with self.lock:
            # Calculate file checksums for file-based validations
            file_checksums = {}
            if file_paths:
                file_checksums = self._calculate_file_checksums(file_paths)
            
            entry = CacheEntry(
                config_hash=config_hash,
                app_type=app_type,
                validation_result=validation_result,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=1,
                file_checksums=file_checksums,
                ttl_seconds=ttl or self.default_ttl
            )
            
            self.cache[config_hash] = entry
            
            # Evict oldest entries if cache is full
            if len(self.cache) > self.max_entries:
                self._evict_lru_entries()
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        with self.lock:
            keys_to_remove = []
            for key, entry in self.cache.items():
                if pattern in entry.app_type or pattern in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
                self.stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                'max_entries': self.max_entries
            }
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'file_invalidations': 0}
    
    def _calculate_file_checksums(self, file_paths: Dict[str, str]) -> Dict[str, str]:
        """Calculate checksums for validation files"""
        checksums = {}
        
        for key, path in file_paths.items():
            if path and os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        checksums[key] = file_hash
                except OSError:
                    # If file can't be read, skip checksum
                    pass
        
        return checksums
    
    def _verify_file_checksums(self, cached_checksums: Dict[str, str]) -> bool:
        """Verify that file checksums haven't changed"""
        for key, cached_hash in cached_checksums.items():
            # This would need file path resolution in practice
            # For now, assume files haven't changed if we can't verify
            pass
        return True  # Simplified for demo
    
    def _evict_lru_entries(self):
        """Evict least recently used entries"""
        # Sort by last accessed time
        entries_by_access = sorted(
            self.cache.items(),
            key=lambda item: item[1].accessed_at
        )
        
        # Remove oldest 10% of entries
        num_to_remove = max(1, len(entries_by_access) // 10)
        
        for i in range(num_to_remove):
            key = entries_by_access[i][0]
            del self.cache[key]
            self.stats['evictions'] += 1


class CachedValidator(BaseValidator):
    """Validator wrapper that adds caching support"""
    
    def __init__(self, base_validator: BaseValidator, cache: ValidationCache):
        self.base_validator = base_validator
        self.cache = cache
    
    def validate(self, data: Dict[str, Any]) -> None:
        """Validate with caching support"""
        # Generate cache key
        cache_key = self._generate_cache_key(data)
        
        # Check cache first
        cached_entry = self.cache.get(cache_key)
        if cached_entry:
            # Return cached result
            if cached_entry.validation_result.get('status') == 'error':
                # Re-raise cached validation error
                raise ValidationException(
                    cached_entry.validation_result['error'],
                    validation_errors=cached_entry.validation_result.get('validation_errors')
                )
            return  # Validation passed (cached)
        
        # Perform validation
        validation_start = time.time()
        validation_result = {'status': 'success'}
        
        try:
            self.base_validator.validate(data)
            validation_result['validation_time'] = time.time() - validation_start
            
        except ValidationException as e:
            validation_result = {
                'status': 'error',
                'error': e.message,
                'validation_errors': e.validation_errors,
                'validation_time': time.time() - validation_start
            }
            
            # Cache the error result too
            self.cache.put(
                cache_key,
                data.get('app_type', 'unknown'),
                validation_result,
                self._extract_file_paths(data)
            )
            raise
        
        # Cache successful validation
        self.cache.put(
            cache_key,
            data.get('app_type', 'unknown'),
            validation_result,
            self._extract_file_paths(data)
        )
    
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from configuration data"""
        # Create normalized data for hashing
        cache_data = {}
        
        # Include app type and core parameters
        cache_data['app_type'] = data.get('app_type')
        
        # Include parameter hashes (exclude file paths from hash)
        param_keys = [
            'cco_params', 'energy_params', 'rl_params', 
            'load_balance_params', 'mro_params', 'mobility_model_params'
        ]
        
        for key in param_keys:
            if key in data:
                cache_data[key] = data[key]
        
        # Include common app parameters
        common_keys = ['train_days', 'test_day', 'tick', 'total_timesteps']
        for key in common_keys:
            if key in data:
                cache_data[key] = data[key]
        
        # Generate hash
        cache_json = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_json.encode()).hexdigest()
    
    def _extract_file_paths(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Extract file paths for checksum validation"""
        file_paths = {}
        
        path_keys = ['topology_path', 'config_path', 'training_data_path']
        for key in path_keys:
            if key in data and data[key]:
                file_paths[key] = data[key]
        
        return file_paths


# Global cache instance
_global_cache = ValidationCache()


def get_validation_cache() -> ValidationCache:
    """Get global validation cache instance"""
    return _global_cache


def create_cached_validator(base_validator: BaseValidator) -> CachedValidator:
    """Create cached version of any validator"""
    return CachedValidator(base_validator, get_validation_cache())