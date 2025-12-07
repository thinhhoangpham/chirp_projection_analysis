import numpy as np
import hashlib
from typing import Dict, Tuple, Any, Optional

class ComputationCache:
    """Cache manager for expensive computations to avoid redundant calculations

    OPTIMIZATION: Uses identity-based cache keys instead of MD5 hashing for O(1) lookup
    """

    def __init__(self):
        # Cache for transformed feature values
        self.transform_cache: Dict[tuple, np.ndarray] = {}

        # Cache for projection bounds
        self.bounds_cache: Dict[tuple, np.ndarray] = {}

        # Cache for projection arrays
        self.projection_cache: Dict[tuple, Any] = {}

        # Cache for class statistics
        self.stats_cache: Dict[str, Any] = {}

        # OPTIMIZATION: Global data hash cache - compute once, reuse everywhere
        # Key: (id(data_array), shape, dtype) -> Value: hash string
        self.data_hash_cache: Dict[tuple, str] = {}

        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0

    def get_data_hash(self, data: np.ndarray) -> str:
        """Get or compute hash for data array

        OPTIMIZATION: Caches data hash using identity-based key (O(1) lookup)
        instead of recomputing MD5 hash every time (O(N) operation)
        """
        # Create identity-based key
        data_key = (id(data), data.shape, str(data.dtype))

        if data_key in self.data_hash_cache:
            self.cache_hits += 1
            return self.data_hash_cache[data_key]

        # Cache miss - compute hash once
        self.cache_misses += 1
        data_hash = hashlib.md5(data.tobytes()).hexdigest()
        self.data_hash_cache[data_key] = data_hash
        return data_hash

    def _create_hash_key(self, *args) -> str:
        """Create a hash key from arguments"""
        # Convert arguments to strings and create hash
        key_str = str(args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_transformed_features(self, data: np.ndarray, feature_idx: int,
                               transform_type: str) -> Optional[np.ndarray]:
        """Get cached transformed features or None if not cached

        OPTIMIZATION: Uses identity-based key (O(1)) instead of MD5 hash (O(N))
        """
        # Create unique key based on data identity, feature index, and transform
        key = (id(data), data.shape, str(data.dtype), feature_idx, transform_type)

        if key in self.transform_cache:
            self.cache_hits += 1
            return self.transform_cache[key]

        self.cache_misses += 1
        return None

    def cache_transformed_features(self, data: np.ndarray, feature_idx: int,
                                 transform_type: str, transformed: np.ndarray):
        """Cache transformed features"""
        # Create unique key based on data identity, feature index, and transform
        key = (id(data), data.shape, str(data.dtype), feature_idx, transform_type)
        # OPTIMIZATION: Removed .copy() to save memory. Caller must not modify cached array.
        self.transform_cache[key] = transformed
    
    def get_projection_bounds(self, wi: np.ndarray, transforms: list, 
                            data_hash: str) -> Optional[np.ndarray]:
        """Get cached projection bounds"""
        # OPTIMIZATION: Use tuple key instead of string formatting
        key = (data_hash, tuple(wi), tuple(transforms))
        
        if key in self.bounds_cache:
            self.cache_hits += 1
            return self.bounds_cache[key]
        
        self.cache_misses += 1
        return None
    
    def cache_projection_bounds(self, wi: np.ndarray, transforms: list, 
                              data_hash: str, bounds: np.ndarray):
        """Cache projection bounds"""
        # OPTIMIZATION: Use tuple key instead of string formatting
        key = (data_hash, tuple(wi), tuple(transforms))
        # OPTIMIZATION: Removed .copy() to save memory. Caller must not modify cached array.
        self.bounds_cache[key] = bounds
    
    def get_projection_array(self, wi: np.ndarray, transforms: list, 
                           bounds: np.ndarray, data_hash: str) -> Optional[np.ndarray]:
        """Get cached projection array"""
        bounds_hash = hashlib.md5(bounds.tobytes()).hexdigest()
        # OPTIMIZATION: Use tuple key instead of string formatting
        key = (data_hash, tuple(wi), tuple(transforms), bounds_hash)
        
        if key in self.projection_cache:
            self.cache_hits += 1
            return self.projection_cache[key]
        
        self.cache_misses += 1
        return None
    
    def cache_projection_array(self, wi: np.ndarray, transforms: list, 
                             bounds: np.ndarray, data_hash: str, projection: np.ndarray):
        """Cache projection array"""
        bounds_hash = hashlib.md5(bounds.tobytes()).hexdigest()
        # OPTIMIZATION: Use tuple key instead of string formatting
        key = (data_hash, tuple(wi), tuple(transforms), bounds_hash)
        # OPTIMIZATION: Removed .copy() to save memory. Caller must not modify cached array.
        self.projection_cache[key] = projection
    
    def clear_cache(self):
        """Clear all caches"""
        self.transform_cache.clear()
        self.bounds_cache.clear()
        self.projection_cache.clear()
        self.stats_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'transform_cache_size': len(self.transform_cache),
            'bounds_cache_size': len(self.bounds_cache),
            'projection_cache_size': len(self.projection_cache),
            'stats_cache_size': len(self.stats_cache)
        }

# Global cache instance
_computation_cache = ComputationCache()

def get_computation_cache() -> ComputationCache:
    """Get the global computation cache instance"""
    return _computation_cache
