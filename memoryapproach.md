# Advanced Memory Management for Quantized FLUX Models

## Problem Statement

Quantized models using `bitsandbytes` exhibit "sticky" memory behavior where GPU memory is not properly released after model unloading. This is a known limitation of the `bitsandbytes` library where quantized parameters cannot be moved between devices (`to('cpu')` is not supported) and maintain references that prevent garbage collection.

### Observed Behavior
- 8-bit quantized FLUX.1-dev retains ~11.2GB GPU memory after cache clearing
- 4-bit quantized models similarly retain ~8-10GB 
- Standard `torch.cuda.empty_cache()` and `gc.collect()` insufficient
- Memory only fully released on process restart

## Multi-Layered Memory Management Solution

### 1. PyTorch CUDA Allocator Configuration

**Implementation**: Automatic configuration via environment variables in `__init__()`:
```python
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 
    'garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True')
```

**Technical Details**:
- `garbage_collection_threshold:0.8`: Triggers allocator cleanup at 80% GPU memory usage
- `max_split_size_mb:128`: Prevents memory fragmentation by limiting block splitting to 128MB
- `expandable_segments:True`: Enables dynamic memory segment growth

**Performance Impact**: The 80% threshold is set high enough to avoid triggering during normal inference (which uses 70-90% memory) but will activate during memory pressure situations like model switching.

### 2. Intelligent Cache Management

**Cache Key Strategy**: `{model_id}_{quantization}` format ensures different quantization modes are treated as separate models requiring cache invalidation.

**Memory Estimation**:
- 4-bit quantization: 8.0GB conservative estimate
- 8-bit quantization: 12.0GB conservative estimate  
- Full precision: 20.0GB conservative estimate

**Decision Logic**: Cache clearing only triggered when:
1. Requested model/quantization differs from cached
2. Available GPU memory < estimated requirement
3. Cache is not empty

### 3. Advanced Pipeline Cleanup

**Component-Level Destruction**:
```python
for attr_name in ['transformer', 'vae', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2']:
    component = getattr(old_pipeline, attr_name)
    if hasattr(component, 'parameters'):
        for param in component.parameters():
            if param.data is not None:
                del param.data
            if param.grad is not None:
                del param.grad
    del component
    delattr(old_pipeline, attr_name)
```

**Device Map Reset**: Calls `reset_device_map()` before cleanup to remove accelerate device mappings that prevent proper memory release.

**Attribute Scanning**: Iterates through pipeline attributes to identify and delete torch-related objects that may hold GPU references.

### 4. Multi-Round Memory Cleanup

**7-Round Cleanup Strategy**:
1. **Standard**: `torch.cuda.empty_cache()`
2. **Python GC**: `gc.collect()`
3. **Combined**: Both CUDA and Python cleanup
4. **Synchronized**: `torch.cuda.synchronize()` + cleanup
5. **IPC Collection**: `torch.cuda.ipc_collect()` for inter-process memory
6. **Statistics Reset**: Reset PyTorch memory tracking
7. **Nuclear**: Combined aggressive cleanup with multiple passes

**Timing**: 300ms delays between rounds to allow asynchronous cleanup operations to complete.

### 5. PyTorch Reference Cycle Detection

**Implementation**: Optional integration with `torch.utils.viz._cycles.observe_tensor_cycles` when available.

**Purpose**: Identifies circular references in quantized models that prevent garbage collection. Provides diagnostic information when cleanup fails.

**Patience Mechanism**: 
- **Cache cleanup**: Waits 5 seconds after aggressive cleanup for reference cycle detection
- **Memory safety check**: **Non-blocking 7-second wait** when suspected quantized model cleanup failure is detected
- **Griptape Integration**: Uses `yield from` pattern to return control to Griptape engine during waits
- Rechecks available memory after waiting and proceeds if sufficient memory becomes available

**Non-Blocking Implementation**: The memory safety check now yields control back to Griptape in 500ms chunks during the 7-second wait, preventing engine freezing while still allowing time for reference cycle detection.

**Observed Behavior**: Reference cycle detection can free significant memory (11+ GB observed) but runs asynchronously taking 5-10 seconds. The dual patience mechanism captures this delayed cleanup at both the cache level and safety check level without blocking the Griptape engine.

**Fallback**: Gracefully handles PyTorch versions without cycle detection support.

### 6. Dual-Layer Patience Strategy

**Implementation**: Two separate wait mechanisms to handle asynchronous reference cycle detection:

1. **Cache-Level Patience** (`_manage_pipeline_cache`):
   - Waits 5 seconds after aggressive cleanup
   - Rechecks memory and continues if sufficient
   - Early return prevents unnecessary safety check failures

2. **Safety-Level Patience** (`_check_memory_safety`):
   - Triggered when low memory detected with empty cache (indicating sticky models)
   - Waits 7 seconds specifically for reference cycle detection
   - Updates memory readings and proceeds if threshold met

**Rationale**: Reference cycle detection timing varies (5-10 seconds) and runs asynchronously. Single-point waiting was insufficient due to detection completing after initial safety checks.

**Technical Implementation**:
```python
# Non-blocking async wait pattern
def _async_wait(self, seconds: float, message: str = ""):
    chunk_size = 0.5  # 500ms chunks
    for i in range(int(seconds / chunk_size)):
        yield lambda: time.sleep(chunk_size)  # Yield control to Griptape

# Memory safety with async handling  
if memory_insufficient:
    return "ASYNC_MEMORY_CHECK_NEEDED"  # Signal async check needed

# In generate_image (generator function):
if pipeline == "ASYNC_MEMORY_CHECK_NEEDED":
    yield from self._async_memory_check(...)  # Delegate to async checker
```

### 7. Nuclear Memory Recovery

**CUDA Context Reset**:
```python
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.cuda.synchronize()
torch.cuda.reset_accumulated_memory_stats()
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()
```

**Limitations**: Even nuclear cleanup cannot overcome fundamental `bitsandbytes` limitations where quantized parameters remain "sticky" in GPU memory.

## Memory Safety Checks

### Enhanced Error Detection
- Detects when cache is empty but memory remains high (indicates quantized model cleanup failure)
- Provides context-aware fallback suggestions based on available memory
- Estimates memory requirements per quantization mode

### User Override Mechanism
- `allow_low_memory: true` in system constraints bypasses safety checks
- Enables forcing model loading when cleanup fails
- Risk: Potential system hangs or OOM errors

### Fallback Recommendations
**Automatic Suggestions**:
- 4-bit → 8-bit when 10GB+ available
- 4-bit/8-bit → full precision when 20GB+ available  
- Workflow restart when <6GB available

## Performance Considerations

### Memory Management During Inference

**Garbage Collection Threshold Impact**: The 80% threshold is tuned to avoid triggering during normal inference:
- FLUX inference typically uses 70-90% of allocated GPU memory
- Threshold set above normal usage to prevent mid-inference cleanup
- Only triggers during genuine memory pressure (model switching, memory leaks)

**Tuning for Performance**:
1. **Increase Threshold**: Set to 0.9 for pure inference workloads
2. **Disable During Inference**: Set `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (disables all caching)
3. **Custom Configuration**: Override environment variable before process start
4. **Workflow Design**: Load all models at start, avoid quantization switching during active inference

### Cache Invalidation Cost

**Pipeline Loading**: 30-60 seconds for complete model reload when cache invalidated
**Component Cleanup**: 2-5 seconds for aggressive cleanup rounds
**Memory Detection**: <100ms for memory checks and decisions

## Known Limitations

### Fundamental Constraints
1. **bitsandbytes Limitation**: Quantized models cannot be moved to CPU (`to('cpu')` unsupported)
2. **Memory Stickiness**: Some GPU memory remains allocated until process restart
3. **Reference Cycles**: Quantized parameters may create circular references preventing GC

### Workarounds
1. **Process Restart**: Most reliable method for complete memory recovery
2. **Quantization Mode Selection**: Use full precision to avoid sticky memory
3. **Memory Override**: Force loading with `allow_low_memory` flag
4. **Workflow Design**: Plan quantization switches to minimize memory pressure

## Technical Architecture

### Memory Monitoring
- Real-time GPU memory tracking via `torch.cuda.memory_allocated()`
- Total memory detection via `torch.cuda.get_device_properties()`
- Available memory calculation with device-specific handling

### Error Propagation
- Detailed error messages with cleanup failure context
- Automatic fallback suggestion generation
- User guidance for memory constraint resolution

### Integration Points
- GPU Configuration node provides `allow_low_memory` UI control
- System constraints dictionary carries memory configuration
- Pipeline cache maintains quantization-aware key structure

## Environment Variables

### PYTORCH_CUDA_ALLOC_CONF
**Default Configuration**: `garbage_collection_threshold:0.8,max_split_size_mb:128,expandable_segments:True`

**Override Method**: Set environment variable before Python process start:
```bash
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:256"
```

**Parameter Documentation**:
- `garbage_collection_threshold`: 0.0-1.0, memory usage fraction triggering cleanup
- `max_split_size_mb`: Integer MB, maximum memory block split size  
- `expandable_segments`: Boolean, enable dynamic memory segment expansion

### Alternative Configurations

**Inference-Optimized**: `garbage_collection_threshold:0.9,max_split_size_mb:512`
- Minimal cleanup during inference, maximum performance
- Only cleans up at very high memory usage (90%)
- Risk: Less effective cleanup of quantized model memory

**Memory-Constrained**: `garbage_collection_threshold:0.7,max_split_size_mb:64`
- More aggressive cleanup for systems with limited VRAM
- May cause brief interruptions during high memory usage
- Better for <12GB VRAM systems

**Quantization-Heavy**: `garbage_collection_threshold:0.75,max_split_size_mb:128`
- Balanced approach for workflows switching quantization modes frequently
- Moderate cleanup frequency
- Good compromise between performance and memory management

## Implementation Files

### Core Logic
- `huggingface_cuda/flux/flux_inference.py`: Main implementation
- `_manage_pipeline_cache()`: Cache management and cleanup orchestration
- `_check_memory_safety()`: Memory validation and error handling

### Configuration
- `huggingface_cuda/gpu_configuration.py`: UI controls for memory overrides
- `huggingface_cuda/griptape-nodes-library.json`: Dependencies including `psutil`

### Monitoring
- `psutil`: System RAM monitoring
- `torch.cuda`: GPU memory tracking and management
- PyTorch allocator: Internal memory pool management 