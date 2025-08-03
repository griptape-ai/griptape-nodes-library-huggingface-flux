"""
Memory management utilities for FLUX CUDA backend.

Handles GPU memory safety checks, pipeline cache management, and memory allocation.
"""

import os
import gc
import time
import asyncio
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Union, Callable, Tuple, List
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that require working memory."""
    INFERENCE = "inference"
    VAE_DECODE = "vae_decode"
    LONG_CONTEXT = "long_context"
    BATCH_INFERENCE = "batch_inference"


@dataclass
class ComponentInfo:
    """Information about cached components."""
    component: Any
    device: str
    memory_usage_mb: float
    last_used: float
    usage_count: int
    

@dataclass
class WorkingMemoryReservation:
    """Working memory reservation for operations."""
    operation_type: OperationType
    reserved_gb: float
    active_until: float


class FluxMemoryManager:
    """Manages GPU memory allocation, cache, and cleanup for FLUX models."""
    
    def __init__(self, shared_backend: Dict[str, Any], config: "FluxConfig" = None):
        """Initialize with advanced caching, async operations, and working memory management."""
        self._shared_backend = shared_backend
        self._config = config
        
        # Advanced pipeline cache
        self._pipeline_cache = {}
        
        # Enhanced component cache with metadata
        self._component_cache: Dict[str, ComponentInfo] = {}
        
        # Working memory management
        self._working_memory_reservations: List[WorkingMemoryReservation] = []
        # Use string keys to avoid enum comparison issues
        self._working_memory_config = {
            "inference": 3.0,
            "vae_decode": 4.5,
            "long_context": 6.0,
            "batch_inference": 8.0
        }
        
        # Async operations
        self._preload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="flux_preload")
        self._preload_futures = {}
        self._ping_pong_buffers = {}
        
        # Usage pattern tracking for intelligent prefetching
        self._usage_history = deque(maxlen=10)  # Track last 10 model switches
        self._model_affinity = defaultdict(list)  # Which models are used together
        
        # Performance tracking
        self._cache_hits = 0
        self._cache_misses = 0
        self._component_reuse_count = 0
        
        # Legacy fields
        self._last_cleanup_failed = False
        self._cleanup_failed_memory = 0.0
        self._cuda_allocator_logged = False
    
    @staticmethod
    def normalize_memory_dict(mem: dict, gpu_device: int | str = 0) -> dict:
        """Return a copy of *mem* with all GPU keys as 'cuda:N' strings.
        Accepts variants: int, numeric str, 'auto'. Raises on unknown keys."""
        if not mem:
            return {}
        norm = {}
        for key, val in mem.items():
            if key == 'cpu':
                norm['cpu'] = val
            elif key == 'auto':
                norm[f'cuda:{gpu_device}'] = val
            elif isinstance(key, int):
                norm[key] = val  # keep integers as-is (accelerate prefers ints)
            elif isinstance(key, str) and key.isdigit():
                # numeric string → int
                norm[int(key)] = val
            elif isinstance(key, str) and key.startswith('cuda:'):
                norm[key] = val
            else:
                raise ValueError(
                    f"Invalid device key in max_memory dict: {key!r}. Use 'cpu', 'auto', or GPU indices like 0 / 'cuda:0'."
                )
        return norm
    
    def check_memory_safety(self, quantization: str, system_constraints: dict = None):
        """Check if system has enough memory to safely load FLUX model."""
        torch = self._shared_backend['torch']
        
        # Get available memory info
        if torch.cuda.is_available():
            gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
            
            # Handle 'auto' device selection - convert to actual device number
            if gpu_device == 'auto':
                gpu_device = 0  # Default to first GPU when auto is selected
            
            gpu_props = torch.cuda.get_device_properties(gpu_device)
            total_gpu_memory_gb = gpu_props.total_memory / 1024**3
            allocated_gpu_memory_gb = torch.cuda.memory_allocated(gpu_device) / 1024**3
            available_gpu_memory_gb = total_gpu_memory_gb - allocated_gpu_memory_gb
            
            print(f"[FLUX MEMORY] GPU {gpu_device}: {available_gpu_memory_gb:.1f}GB available / {total_gpu_memory_gb:.1f}GB total")
            
            # Get memory limits from system constraints or use sensible defaults
            default_memory_requirements = {
                "4-bit": 8.0,    # Default minimum for 4-bit quantization
                "8-bit": 12.0,   # Default minimum for 8-bit quantization  
                "none": 24.0     # Default minimum for full precision
            }
            
            # Use system constraints for memory limits if provided
            if system_constraints:
                required_memory = system_constraints.get(f'{quantization}_memory_gb', 
                                                        system_constraints.get('gpu_memory_gb',
                                                                             default_memory_requirements.get(quantization, 24.0)))
                print(f"[FLUX MEMORY] Using system constraint memory limit: {required_memory}GB for {quantization}")
            else:
                required_memory = default_memory_requirements.get(quantization, 24.0)
                print(f"[FLUX MEMORY] Using default memory requirement: {required_memory}GB for {quantization}")
            
            # Check if constraint is too restrictive
            min_viable_memory = 6.0  # Absolute minimum to attempt loading
            if required_memory < min_viable_memory:
                print(f"[FLUX MEMORY] ⚠️ Memory constraint ({required_memory}GB) is very low - may cause failures")
            
            if available_gpu_memory_gb < required_memory:
                error_msg = (
                    f"⚠️ MEMORY WARNING: Only {available_gpu_memory_gb:.1f}GB available, "
                    f"but {quantization} quantization limit is {required_memory}GB. "
                    f"This may cause system hangs or OOM errors!"
                )
                print(f"[FLUX MEMORY] {error_msg}")
                
                # Only raise error if user hasn't explicitly set a constraint allowing this
                user_override = system_constraints and (
                    'allow_low_memory' in system_constraints or 
                    'gpu_memory_gb' in system_constraints or
                    f'{quantization}_memory_gb' in system_constraints
                )
                
                if not user_override and available_gpu_memory_gb < min_viable_memory:
                    # Check if this is likely a quantized model cleanup failure and we should wait for cycles
                    if (len(self._pipeline_cache) == 0 and 
                        available_gpu_memory_gb < 6.0):
                        
                        # Return a special marker indicating async memory check needed
                        return "ASYNC_MEMORY_CHECK_NEEDED"
                    
                    # Prepare error message with helpful hints
                    cleanup_failed_hint = ""
                    auto_fallback_suggestion = ""
                    
                    if len(self._pipeline_cache) == 0 and available_gpu_memory_gb < 6.0:
                        cleanup_failed_hint = (
                            "\n💡 This appears to be a quantized model cleanup failure. "
                            "Quantized models can be 'sticky' in GPU memory. "
                            "\n💡 SOLUTIONS: "
                            "\n   1. Set 'allow_low_memory': true to override this safety check"
                            "\n   2. Use 'none' (full precision) instead of quantization for this model"
                            "\n   3. Restart the workflow to fully clear GPU memory"
                        )
                        
                        # Smart fallback suggestions based on available memory
                        if quantization == "4-bit":
                            if available_gpu_memory_gb >= 10.0:
                                auto_fallback_suggestion = "\n🔄 SMART SUGGESTION: Try 8-bit quantization instead (you have enough memory)"
                            elif available_gpu_memory_gb >= 20.0:
                                auto_fallback_suggestion = "\n🔄 SMART SUGGESTION: Try 'none' (full precision) instead"
                        elif quantization == "8-bit":
                            if available_gpu_memory_gb >= 20.0:
                                auto_fallback_suggestion = "\n🔄 SMART SUGGESTION: Try 'none' (full precision) instead"
                        
                        if not auto_fallback_suggestion and available_gpu_memory_gb < 6.0:
                            auto_fallback_suggestion = "\n🔄 RECOMMENDATION: Restart workflow to fully clear GPU memory (current memory too low)"
                    
                    raise RuntimeError(
                        f"Insufficient GPU memory: {available_gpu_memory_gb:.1f}GB available, "
                        f"below minimum viable ({min_viable_memory}GB). Set 'allow_low_memory': true "
                        f"in system constraints to override this safety check.{cleanup_failed_hint}{auto_fallback_suggestion}"
                    )
                elif user_override:
                    print(f"[FLUX MEMORY] ⚠️ User override detected - proceeding despite low memory warning")
        else:
            print(f"[FLUX MEMORY] CUDA not available - using CPU (will be very slow)")
            
        # Check system RAM for CPU fallback
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / 1024**3
            print(f"[FLUX MEMORY] System RAM: {available_ram_gb:.1f}GB available")
            
            if available_ram_gb < 16.0:
                print(f"[FLUX MEMORY] ⚠️ Low system RAM ({available_ram_gb:.1f}GB), may cause swapping")
        except ImportError:
            print(f"[FLUX MEMORY] psutil not available, cannot check system RAM")

    def manage_pipeline_cache(self, new_cache_key: str, quantization: str, system_constraints: dict = None):
        """Intelligently manage pipeline cache with selective component reuse
        
        For FLUX models, we can reuse shared components (VAE, CLIP, T5) and only swap
        the model-specific transformer component for much faster switching.
        
        Cache key format: '{model_id}_{quantization}' (e.g., 'FLUX.1-dev_8-bit')
        """
        print(f"[FLUX CACHE] 🔧 manage_pipeline_cache called with key: '{new_cache_key}'")
        print(f"[FLUX CACHE] Current cache contains: {list(self._pipeline_cache.keys())}")
        
        if new_cache_key in self._pipeline_cache:
            print(f"[FLUX CACHE] ✅ Cache hit for {new_cache_key}")
            return
            
        # Check if we're switching between FLUX models with same quantization
        # Cache key format: "model_id_quantization" e.g. "black-forest-labs/FLUX.1-dev_8-bit"
        # Need to split on the last underscore to handle model IDs with slashes
        parts = new_cache_key.rsplit('_', 1)
        if len(parts) == 2:
            new_model_id, new_quant = parts
        else:
            new_model_id = new_cache_key
            new_quant = "none"
        
        print(f"[FLUX CACHE] 🔍 Checking for selective swap opportunity...")
        print(f"[FLUX CACHE] New model: '{new_model_id}', quantization: '{new_quant}'")
        
        # If we have cached pipelines, check if we can do selective swapping
        if self._pipeline_cache:
            print(f"[FLUX CACHE] Found {len(self._pipeline_cache)} cached pipelines: {list(self._pipeline_cache.keys())}")
            
            # Check if we're switching between compatible FLUX models
            old_cache_keys = list(self._pipeline_cache.keys())
            for old_key in old_cache_keys:
                parts = old_key.rsplit('_', 1)
                if len(parts) == 2:
                    old_model_id, old_quant = parts
                else:
                    old_model_id = old_key
                    old_quant = "none"
                
                print(f"[FLUX CACHE] Comparing with cached: '{old_model_id}', quantization: '{old_quant}'")
                
                # If same quantization and both are FLUX models, we can do selective swap
                if (new_quant == old_quant and 
                    "FLUX.1" in new_model_id and "FLUX.1" in old_model_id and
                    new_model_id != old_model_id):
                    
                    print(f"[FLUX CACHE] 🎯 SELECTIVE SWAP TRIGGERED:")
                    print(f"[FLUX CACHE]   From: {old_model_id} ({old_quant})")
                    print(f"[FLUX CACHE]   To: {new_model_id} ({new_quant})")
                    print(f"[FLUX CACHE]   Preserving: VAE, CLIP, T5 (9GB+ saved)")
                    
                    # Store the cached components before attempting selective swap
                    preserved_components = self._component_cache.copy()
                    
                    success = self._selective_transformer_swap(old_key, new_cache_key)
                    
                    if success:
                        print(f"[FLUX CACHE] ✅ Selective swap successful - cached components preserved")
                        return
                    else:
                        # Restore cached components after failed selective swap
                        print(f"[FLUX CACHE] 🔄 Restoring cached components after selective swap failure")
                        self._component_cache = preserved_components
                        print(f"[FLUX CACHE] 💡 Will still try to use cached components in new pipeline load")
                else:
                    print(f"[FLUX CACHE] No selective swap: quant_match={new_quant == old_quant}, new_flux={'FLUX.1' in new_model_id}, old_flux={'FLUX.1' in old_model_id}, different_models={new_model_id != old_model_id}")
        else:
            print(f"[FLUX CACHE] No cached pipelines found for selective swap")
            
        # Fall back to normal cache clearing if we can't do selective swap
        # But preserve cached components for reuse in new pipeline load
        if not self._pipeline_cache:
            return
            
        print(f"[FLUX CACHE] 🧹 Clearing pipeline cache (preserving components for reuse)...")
        
        # Get torch from shared backend - this is where torch is actually loaded
        torch = self._shared_backend['torch']
        
        # Log CUDA allocator configuration now that torch is available
        if not self._cuda_allocator_logged:
            try:
                if torch.cuda.is_available():
                    print(f"[FLUX CACHE] ✅ PyTorch CUDA allocator active with config: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")
                self._cuda_allocator_logged = True
            except Exception as e:
                print(f"[FLUX CACHE] ⚠️ Could not verify CUDA allocator config: {e}")
                self._cuda_allocator_logged = True
        
        # Check available GPU memory before deciding to clear
        gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
        if isinstance(gpu_device, str) and gpu_device == 'auto':
            gpu_device = 0
        
        available_gpu_memory_gb = (torch.cuda.get_device_properties(gpu_device).total_memory - 
                                   torch.cuda.memory_allocated(gpu_device)) / 1024**3
        total_gpu_memory_gb = torch.cuda.get_device_properties(gpu_device).total_memory / 1024**3
        
        # Estimate memory needed for different quantization modes  
        if quantization == "4-bit":
            needed_memory = 8.0  # Conservative estimate for 4-bit + overhead
        elif quantization == "8-bit":
            needed_memory = 12.0  # Conservative estimate for 8-bit + overhead  
        else:  # none/full precision
            needed_memory = 20.0  # Conservative estimate for full precision
        
        print(f"[FLUX CACHE] Available GPU memory: {available_gpu_memory_gb:.1f}GB")
        print(f"[FLUX CACHE] Estimated need for {quantization}: {needed_memory}GB")
        
        if available_gpu_memory_gb >= needed_memory:
            print(f"[FLUX CACHE] ✅ Sufficient memory available for {quantization} quantization")
            return
        
        # Advanced cleanup for quantized models
        if available_gpu_memory_gb < needed_memory:
            print(f"[FLUX CACHE] 🧹 Insufficient memory for {quantization} quantization")
            
            # Step 1: Enable PyTorch reference cycle detection for debugging  
            try:
                from torch.utils.viz._cycles import observe_tensor_cycles
                print(f"[FLUX CACHE] 🔍 Enabling reference cycle detection for cleanup debugging")
                
                def cycle_callback(html):
                    print(f"[FLUX CACHE] ⚠️ Reference cycle detected during cleanup - this may explain sticky memory")
                
                observe_tensor_cycles(cycle_callback)
            except ImportError:
                print(f"[FLUX CACHE] ⚠️ Reference cycle detection not available in this PyTorch version")
            
            # Step 2: Advanced pipeline cleanup
            old_pipelines = list(self._pipeline_cache.values())
            old_cache_keys = list(self._pipeline_cache.keys())
            print(f"[FLUX CACHE] 🧹 Clearing cached models: {old_cache_keys}")
            
            for old_pipeline in old_pipelines:
                try:
                    # Reset device mapping before any operations
                    if hasattr(old_pipeline, 'reset_device_map'):
                        print(f"[FLUX CACHE] 🔧 Resetting device map for pipeline...")
                        old_pipeline.reset_device_map()
                    
                    # For quantized models, don't try to move to CPU (bitsandbytes limitation)
                    # Instead, explicitly delete components
                    print(f"[FLUX CACHE] 🗑️ Explicitly clearing pipeline components...")
                    components_cleared = 0
                    for attr_name in ['transformer', 'vae', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2']:
                        if hasattr(old_pipeline, attr_name):
                            try:
                                component = getattr(old_pipeline, attr_name)
                                # Clear component parameters if it has them
                                if hasattr(component, 'parameters'):
                                    for param in component.parameters():
                                        if param.data is not None:
                                            del param.data
                                        if param.grad is not None:
                                            del param.grad
                                del component
                                delattr(old_pipeline, attr_name)
                                components_cleared += 1
                            except Exception as e:
                                print(f"[FLUX CACHE] Warning clearing {attr_name}: {e}")
                    print(f"[FLUX CACHE] 🗑️ Cleared {components_cleared} pipeline components")
                    
                    # Clear any remaining attributes
                    for attr in dir(old_pipeline):
                        if not attr.startswith('_') and hasattr(old_pipeline, attr):
                            try:
                                attr_val = getattr(old_pipeline, attr)
                                if hasattr(attr_val, 'cuda') or str(type(attr_val)).find('torch') != -1:
                                    delattr(old_pipeline, attr)
                            except:
                                pass
                    
                except Exception as e:
                    print(f"[FLUX CACHE] Warning during pipeline cleanup: {e}")
            
            # Step 3: Clear pipeline cache but preserve component cache for reuse
            print(f"[FLUX CACHE] 🧹 Clearing pipeline cache (preserving component cache)")
            self._pipeline_cache.clear()
            # NOTE: NOT clearing self._component_cache - we want to reuse VAE/CLIP/T5
            del old_pipelines
            
            # Step 4: Force synchronization before cleanup
            print(f"[FLUX CACHE] 🔄 Forcing CUDA synchronization...")
            torch.cuda.synchronize()
            
            # Step 5: Multi-round aggressive GPU memory cleanup for quantized models
            print(f"[FLUX CACHE] 🧹 Starting aggressive GPU memory cleanup for quantized models...")
            
            for cleanup_round in range(7):  # Increased rounds
                # Multiple different cleanup approaches
                if cleanup_round == 0:
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: Standard cleanup")
                    torch.cuda.empty_cache()
                elif cleanup_round == 1:
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: Python garbage collection")
                    gc.collect()
                elif cleanup_round == 2:
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: CUDA + Python GC")
                    torch.cuda.empty_cache()
                    gc.collect()
                elif cleanup_round == 3:
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: CUDA synchronize + cleanup")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                elif cleanup_round == 4:
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: IPC collection")
                    try:
                        torch.cuda.ipc_collect()
                    except:
                        pass
                    torch.cuda.empty_cache()
                elif cleanup_round == 5:
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: Reset memory stats")
                    try:
                        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                            torch.cuda.reset_peak_memory_stats()
                        if hasattr(torch.cuda, 'reset_max_memory_allocated'):
                            torch.cuda.reset_max_memory_allocated()
                    except:
                        pass
                    torch.cuda.empty_cache()
                else:  # Round 6 - final aggressive round
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: Combined aggressive cleanup")
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except:
                        pass
                
                current_allocated = torch.cuda.memory_allocated(gpu_device) / 1024**3
                print(f"[FLUX CACHE] Cleanup round {cleanup_round + 1}: {current_allocated:.1f}GB still allocated")
            
            # Step 6: Check memory improvement
            new_available = (torch.cuda.get_device_properties(gpu_device).total_memory - 
                           torch.cuda.memory_allocated(gpu_device)) / 1024**3
            freed_memory = new_available - available_gpu_memory_gb
            print(f"[FLUX CACHE] ✅ Freed {freed_memory:.1f}GB, now have {new_available:.1f}GB available")
            
            # Step 7: Nuclear option if memory is still insufficient
            final_available = new_available  # Initialize in broader scope
            if new_available < needed_memory:
                print(f"[FLUX CACHE] ⚠️ Still insufficient memory after cleanup ({new_available:.1f}GB < {needed_memory}GB)")
                print(f"[FLUX CACHE] 🚨 Quantized models can be 'sticky' in GPU memory - trying nuclear cleanup...")
                
                try:
                    if torch.cuda.is_available():
                        print(f"[FLUX CACHE] 💣 Attempting CUDA context reset to force memory cleanup...")
                        
                        # Advanced CUDA memory reset techniques
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                        
                        # Reset all memory statistics if available
                        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                            torch.cuda.reset_accumulated_memory_stats()
                        if hasattr(torch.cuda, 'reset_max_memory_allocated'):
                            torch.cuda.reset_max_memory_allocated()
                        if hasattr(torch.cuda, 'reset_max_memory_cached'):
                            torch.cuda.reset_max_memory_cached()
                        
                        # One final aggressive cleanup
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        final_available = total_gpu_memory_gb - torch.cuda.memory_allocated(gpu_device) / 1024**3
                        nuclear_freed = final_available - new_available
                        print(f"[FLUX CACHE] 💣 Nuclear cleanup freed additional {nuclear_freed:.1f}GB")
                        print(f"[FLUX CACHE] 💣 Final available: {final_available:.1f}GB")
                        
                        # Wait for reference cycle detection to complete (it runs asynchronously)
                        print(f"[FLUX CACHE] ⏱️ Waiting for reference cycle cleanup to complete...")
                        
                        # Check memory again after waiting for cycles to clear
                        torch.cuda.synchronize()
                        post_cycle_available = total_gpu_memory_gb - torch.cuda.memory_allocated(gpu_device) / 1024**3
                        cycle_freed = post_cycle_available - final_available
                        
                        if cycle_freed > 0.1:  # If significant memory was freed by cycle detection
                            print(f"[FLUX CACHE] ✅ Reference cycle cleanup freed additional {cycle_freed:.1f}GB!")
                            print(f"[FLUX CACHE] ✅ Final memory after cycle cleanup: {post_cycle_available:.1f}GB")
                            final_available = post_cycle_available
                        
                        if final_available >= needed_memory:
                            print(f"[FLUX CACHE] ✅ Cleanup succeeded! Enough memory now available")
                            return  # Exit early if we now have enough memory
                        else:
                            print(f"[FLUX CACHE] ⚠️ Even with cycle cleanup insufficient - this is expected behavior")
                            print(f"[FLUX CACHE] 💡 Quantized models can require restart to fully clear GPU memory")
                    
                except Exception as e:
                    print(f"[FLUX CACHE] ⚠️ Nuclear cleanup failed: {e}")
                    # If nuclear cleanup failed, keep using the best available memory value
                    final_available = new_available
                    
                print(f"[FLUX CACHE] ⚠️ Memory safety check will still trigger if insufficient memory remains")
                print(f"[FLUX CACHE] 💡 WORKAROUND: Set 'allow_low_memory': true in system constraints to force load")
                print(f"[FLUX CACHE] 💡 ALTERNATIVE: Restart the workflow to fully clear quantized model memory")
                
                # Smart fallback: Suggest automatic quantization adjustment (use final_available which includes cycle cleanup)
                if quantization == "4-bit" and final_available >= 10.0:
                    print(f"[FLUX CACHE] 🔄 AUTO-FALLBACK OPTION: You have {final_available:.1f}GB - try 8-bit instead")
                elif quantization in ["4-bit", "8-bit"] and final_available >= 20.0:
                    print(f"[FLUX CACHE] 🔄 AUTO-FALLBACK OPTION: You have {final_available:.1f}GB - try 'none' (full precision)")
                elif final_available < 6.0:
                    print(f"[FLUX CACHE] 🔄 RECOMMENDED: Restart workflow to fully clear GPU memory ({final_available:.1f}GB too low for any model)")
                
                # Store the failed cleanup info for smarter error handling
                self._last_cleanup_failed = True
                self._cleanup_failed_memory = final_available
                        
        else:
            print(f"[FLUX CACHE] ✅ Sufficient memory available for {quantization} quantization")
    
    def get_pipeline_cache(self):
        """Get reference to pipeline cache."""
        return self._pipeline_cache
    
    def cache_pipeline(self, cache_key: str, pipeline):
        """Cache the pipeline and its reusable components with advanced tracking."""
        # Cache the pipeline
        self._pipeline_cache[cache_key] = pipeline
        self._cache_hits += 1
        
        # Extract model_id for usage tracking  
        model_id = cache_key.split('_')[0] if '_' in cache_key else cache_key
        self.track_model_usage(model_id, "cache")
        
        # Cache shared components using advanced caching
        if hasattr(pipeline, 'vae'):
            self.cache_component_advanced('vae', pipeline.vae)
            
        if hasattr(pipeline, 'text_encoder'):
            self.cache_component_advanced('clip_encoder', pipeline.text_encoder)
            if hasattr(pipeline, 'tokenizer'):
                self.cache_component_advanced('clip_tokenizer', pipeline.tokenizer)
                
        if hasattr(pipeline, 'text_encoder_2'):
            self.cache_component_advanced('t5_encoder', pipeline.text_encoder_2)
            if hasattr(pipeline, 'tokenizer_2'):
                self.cache_component_advanced('t5_tokenizer', pipeline.tokenizer_2)
        
        # Start intelligent prefetching for next likely models
        candidates = self.suggest_preload_candidates(model_id)
        for candidate in candidates:
            if candidate not in [k.split('_')[0] for k in self._pipeline_cache.keys()]:
                print(f"[FLUX SMART] 🧠 Suggested preload candidate: {candidate}")
                # Could start preload here, but keep it optional for now
    
    def get_cached_pipeline(self, cache_key: str):
        """Get cached pipeline if exists."""
        pipeline = self._pipeline_cache.get(cache_key)
        if pipeline:
            self._cache_hits += 1
            # Track usage for this model
            model_id = cache_key.split('_')[0] if '_' in cache_key else cache_key
            self.track_model_usage(model_id, "load")
        else:
            self._cache_misses += 1
        return pipeline
        
    def get_cached_components(self):
        """Get cached shared components for reuse with new advanced format."""
        # Return components in the old format for compatibility
        old_format = {}
        for comp_type, comp_info in self._component_cache.items():
            if comp_info is not None:
                old_format[comp_type] = comp_info.component
            else:
                old_format[comp_type] = None
        return old_format
        
    def _clear_all_caches(self):
        """Clear all caches including shared components"""
        self._pipeline_cache.clear()
        # Clear advanced component cache 
        self._component_cache.clear()
        # Clean up any pending async operations
        self.cleanup_async_resources()
        print(f"[FLUX CACHE] 🧹 Cleared all pipeline and component caches")
        
    def _selective_transformer_swap(self, old_cache_key: str, new_cache_key: str) -> bool:
        """Perform selective transformer swap while keeping shared components. Returns success status."""
        try:
            # Get the old pipeline
            old_pipeline = self._pipeline_cache.get(old_cache_key)
            if not old_pipeline:
                print(f"[FLUX CACHE] ⚠️ No old pipeline found for key: {old_cache_key}")
                return False
                
            torch = self._shared_backend['torch']
            
            # For quantized models, we need a different approach
            print(f"[FLUX CACHE] 🗑️ Selective clear: removing old pipeline, keeping cached components...")
            
            # Simply remove the pipeline from cache without trying to manipulate its components
            # The cached components (VAE, CLIP, T5) are stored separately in _component_cache
            del self._pipeline_cache[old_cache_key]
            
            # Don't try to manipulate the transformer directly for quantized models
            # Let Python's garbage collector handle it
            del old_pipeline
            
            # Light memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            print(f"[FLUX CACHE] ✅ Old pipeline cleared, cached components preserved")
            print(f"[FLUX CACHE] 🚀 Ready for fast component-reuse loading")
            
            return True
            
        except Exception as e:
            print(f"[FLUX CACHE] ⚠️ Selective swap failed: {e}")
            return False
    
    def cleanup_failed_info(self):
        """Get info about last cleanup failure."""
        return self._last_cleanup_failed, self._cleanup_failed_memory
    
    # =========================================================================
    # ADVANCED MEMORY MANAGEMENT FEATURES
    # =========================================================================
    
    def reserve_working_memory(self, operation_type: OperationType, duration_seconds: float = 300.0) -> bool:
        """Reserve working memory for specific operations."""
        torch = self._shared_backend['torch']
        
        if not torch.cuda.is_available():
            return True
            
        # Clean up expired reservations
        current_time = time.time()
        self._working_memory_reservations = [
            r for r in self._working_memory_reservations 
            if r.active_until > current_time
        ]
        
        # Calculate required memory
        required_gb = self._working_memory_config[operation_type.value]
        
        # Check if we have enough free VRAM
        device = 0  # Default GPU
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / 1024**3
        allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
        
        # Account for existing reservations
        reserved_gb = sum(r.reserved_gb for r in self._working_memory_reservations)
        
        available_gb = total_gb - allocated_gb - reserved_gb
        
        if available_gb >= required_gb:
            reservation = WorkingMemoryReservation(
                operation_type=operation_type,
                reserved_gb=required_gb,
                active_until=current_time + duration_seconds
            )
            self._working_memory_reservations.append(reservation)
            print(f"[FLUX MEMORY] ✅ Reserved {required_gb:.1f}GB for {operation_type.value} (available: {available_gb:.1f}GB)")
            return True
        else:
            print(f"[FLUX MEMORY] ❌ Cannot reserve {required_gb:.1f}GB for {operation_type.value} (only {available_gb:.1f}GB available)")
            return False
    
    def release_working_memory(self, operation_type: OperationType):
        """Release working memory reservation for operation."""
        original_count = len(self._working_memory_reservations)
        self._working_memory_reservations = [
            r for r in self._working_memory_reservations 
            if r.operation_type != operation_type
        ]
        freed_count = original_count - len(self._working_memory_reservations)
        if freed_count > 0:
            print(f"[FLUX MEMORY] 🔓 Released {freed_count} working memory reservations for {operation_type.value}")
    
    def get_available_memory_gb(self) -> float:
        """Get available GPU memory accounting for reservations."""
        torch = self._shared_backend['torch']
        
        if not torch.cuda.is_available():
            return float('inf')
        
        device = 0
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / 1024**3
        allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
        
        # Clean up expired reservations
        current_time = time.time()
        self._working_memory_reservations = [
            r for r in self._working_memory_reservations 
            if r.active_until > current_time
        ]
        
        reserved_gb = sum(r.reserved_gb for r in self._working_memory_reservations)
        return total_gb - allocated_gb - reserved_gb
    
    def cache_component_advanced(self, component_type: str, component: Any, 
                                device: str = "cuda", force_cache: bool = False) -> bool:
        """Cache component with advanced metadata tracking."""
        torch = self._shared_backend['torch']
        
        # Calculate memory usage
        memory_mb = 0.0
        if hasattr(component, 'parameters'):
            try:
                memory_mb = sum(p.numel() * p.element_size() for p in component.parameters()) / 1024**2
            except:
                memory_mb = 100.0  # Fallback estimate
        
        # Check if we should cache this component
        if not force_cache and memory_mb > 2000:  # Don't cache components > 2GB by default
            print(f"[FLUX CACHE] ⚠️ Skipping cache for large {component_type} ({memory_mb:.1f}MB)")
            return False
            
        # Check if we already have this component type cached
        if component_type in self._component_cache:
            old_info = self._component_cache[component_type]
            print(f"[FLUX CACHE] 🔄 Replacing cached {component_type} (old: {old_info.memory_usage_mb:.1f}MB, new: {memory_mb:.1f}MB)")
        
        # Cache the component with metadata
        info = ComponentInfo(
            component=component,
            device=device,
            memory_usage_mb=memory_mb,
            last_used=time.time(),
            usage_count=0
        )
        
        self._component_cache[component_type] = info
        print(f"[FLUX CACHE] 💾 Cached {component_type} ({memory_mb:.1f}MB) on {device}")
        return True
    
    def get_cached_component(self, component_type: str) -> Optional[Any]:
        """Get cached component and update usage stats."""
        if component_type in self._component_cache:
            info = self._component_cache[component_type]
            info.last_used = time.time()
            info.usage_count += 1
            self._component_reuse_count += 1
            print(f"[FLUX CACHE] ✅ Reusing cached {component_type} (used {info.usage_count} times)")
            return info.component
        return None
    
    def preload_model_async(self, model_id: str, quantization: str = "none") -> str:
        """Start preloading a model in the background. Returns future_id."""
        future_id = f"{model_id}_{quantization}_{time.time()}"
        
        def _preload():
            try:
                print(f"[FLUX ASYNC] 🚀 Starting background preload: {model_id} ({quantization})")
                # Import here to avoid circular imports
                from .flux_pipeline_loader import FluxPipelineLoader
                loader = FluxPipelineLoader(self._shared_backend, self)
                
                # Perform the loading
                result = loader._load_full_pipeline_sync(model_id, quantization)
                print(f"[FLUX ASYNC] ✅ Background preload completed: {model_id}")
                return result
            except Exception as e:
                print(f"[FLUX ASYNC] ❌ Background preload failed: {model_id} - {e}")
                return None
        
        future = self._preload_executor.submit(_preload)
        self._preload_futures[future_id] = future
        
        print(f"[FLUX ASYNC] 📥 Queued background preload: {model_id} (id: {future_id})")
        return future_id
    
    def get_preloaded_model(self, future_id: str) -> Optional[Any]:
        """Get result from background preload."""
        if future_id in self._preload_futures:
            future = self._preload_futures[future_id]
            if future.done():
                try:
                    result = future.result()
                    del self._preload_futures[future_id]
                    print(f"[FLUX ASYNC] ✅ Retrieved preloaded model (id: {future_id})")
                    return result
                except Exception as e:
                    print(f"[FLUX ASYNC] ❌ Preload failed (id: {future_id}): {e}")
                    del self._preload_futures[future_id]
            else:
                print(f"[FLUX ASYNC] ⏳ Preload still in progress (id: {future_id})")
        return None
    
    def track_model_usage(self, model_id: str, operation: str = "load"):
        """Track model usage patterns for intelligent prefetching."""
        entry = {
            'model_id': model_id,
            'operation': operation,
            'timestamp': time.time()
        }
        self._usage_history.append(entry)
        
        # Update affinity patterns
        if len(self._usage_history) >= 2:
            prev_model = self._usage_history[-2]['model_id']
            if prev_model != model_id:
                self._model_affinity[prev_model].append(model_id)
                # Keep only recent affinities (last 5)
                self._model_affinity[prev_model] = self._model_affinity[prev_model][-5:]
    
    def suggest_preload_candidates(self, current_model: str) -> List[str]:
        """Suggest models to preload based on usage patterns."""
        candidates = []
        
        # Check affinity patterns
        if current_model in self._model_affinity:
            # Get most frequently used models after current_model
            affinity_models = self._model_affinity[current_model]
            # Count frequency and suggest top 2
            from collections import Counter
            freq = Counter(affinity_models)
            candidates.extend([model for model, _ in freq.most_common(2)])
        
        return candidates
    
    def print_performance_stats(self):
        """Print cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        print(f"\n[FLUX PERF] 📊 Cache Performance Stats:")
        print(f"[FLUX PERF]   Cache Hit Rate: {hit_rate:.1f}% ({self._cache_hits}/{total_requests})")
        print(f"[FLUX PERF]   Component Reuse: {self._component_reuse_count} times")
        print(f"[FLUX PERF]   Active Reservations: {len(self._working_memory_reservations)}")
        print(f"[FLUX PERF]   Cached Components: {len(self._component_cache)}")
        print(f"[FLUX PERF]   Background Tasks: {len(self._preload_futures)}")
        
        if self._component_cache:
            print(f"[FLUX PERF] 💾 Cached Components:")
            for comp_type, info in self._component_cache.items():
                print(f"[FLUX PERF]   {comp_type}: {info.memory_usage_mb:.1f}MB, used {info.usage_count}x")
    
    def cleanup_async_resources(self):
        """Clean up async resources on shutdown."""
        # Cancel pending preloads
        for future_id, future in self._preload_futures.items():
            if not future.done():
                future.cancel()
                print(f"[FLUX ASYNC] ❌ Cancelled pending preload: {future_id}")
        
        self._preload_futures.clear()
        self._preload_executor.shutdown(wait=False)