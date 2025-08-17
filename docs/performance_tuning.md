# Performance Tuning Guide

## Overview

This guide helps you optimize Dream-CAD for maximum performance on your hardware, with specific focus on NVIDIA RTX 3090 optimization across all supported models.

## Quick Performance Wins

### 1. Enable FP16 Precision (40-50% VRAM reduction)
```python
model = ModelFactory.create_model("model_name", precision="fp16")
```

### 2. Clear GPU Cache Between Generations
```python
import torch
torch.cuda.empty_cache()
```

### 3. Use Optimal Batch Size (Always 1 for single GPU)
```python
config = {"batch_size": 1}
```

### 4. Enable GPU Persistence Mode
```bash
sudo nvidia-smi -pm 1
```

### 5. Set GPU to Maximum Performance
```bash
sudo nvidia-smi -ac 10001,1875  # RTX 3090 specific
```

## Model-Specific Optimizations

### MVDream Optimization

#### Memory-Efficient Configuration
```python
config = {
    "rescale_factor": 0.5,      # Reduce from 1.0
    "n_views": 2,                # Reduce from 4
    "resolution": 256,           # Keep at 256
    "gradient_checkpointing": True,
    "attention_slicing": True,
    "vae_tiling": True,
}
```

#### Speed vs Quality Trade-offs
| Setting | Fast | Balanced | Quality |
|---------|------|----------|---------|
| rescale_factor | 0.3 | 0.5 | 0.7 |
| num_inference_steps | 30 | 50 | 100 |
| guidance_scale | 7.0 | 7.5 | 10.0 |
| n_views | 2 | 3 | 4 |
| Generation Time | 60 min | 90 min | 150 min |

### TripoSR Optimization

#### Maximum Speed Configuration
```python
config = {
    "resolution": 256,           # Minimum viable
    "marching_cubes_resolution": 128,
    "batch_size": 1,
    "precision": "fp16",
    "compile_model": True,       # PyTorch 2.0 compile
}
```

#### Batch Processing Optimization
```python
# Process multiple prompts efficiently
def batch_process_triposr(prompts, model):
    results = []
    model.warmup()  # Warmup for consistent timing
    
    for prompt in prompts:
        torch.cuda.empty_cache()  # Clear between runs
        result = model.generate_from_text(prompt)
        results.append(result)
    
    model.cooldown()  # Release resources
    return results
```

### Stable-Fast-3D Optimization

#### Game Asset Pipeline
```python
config = {
    # Speed optimizations
    "target_polycount": 10000,   # Lower = faster
    "texture_resolution": 1024,   # Balance quality/speed
    "simplification_ratio": 0.5,  # Aggressive reduction
    
    # Memory optimizations
    "sequential_processing": True,
    "cpu_offload_textures": True,
    
    # Quality settings
    "generate_pbr": True,
    "delight": True,
}
```

### TRELLIS Optimization

#### Quality Mode Selection
```python
# Fast mode (8GB VRAM, 30s)
fast_config = {
    "quality_mode": "fast",
    "slat_resolution": 64,
    "num_inference_steps": 20,
}

# Balanced mode (12GB VRAM, 60s)
balanced_config = {
    "quality_mode": "balanced",
    "slat_resolution": 96,
    "num_inference_steps": 40,
}

# HQ mode (16GB VRAM, 90s)
hq_config = {
    "quality_mode": "hq",
    "slat_resolution": 128,
    "num_inference_steps": 60,
}
```

### Hunyuan3D-Mini Optimization

#### Production Pipeline
```python
config = {
    # Memory optimization
    "sequential_uv_unwrap": True,
    "texture_compression": "dxt5",
    "mipmap_generation": False,
    
    # Speed optimization
    "fast_marching_cubes": True,
    "parallel_normal_calculation": True,
    
    # Quality balance
    "target_polycount": 20000,
    "texture_size": 1024,
}
```

## RTX 3090 Specific Optimizations

### Optimal Settings for RTX 3090

```python
RTX_3090_OPTIMAL = {
    "triposr": {
        "resolution": 512,
        "precision": "fp16",
        "batch_size": 1,
    },
    "stable-fast-3d": {
        "target_polycount": 30000,
        "texture_resolution": 2048,
        "precision": "fp16",
    },
    "trellis": {
        "quality_mode": "balanced",
        "precision": "fp16",
        "gradient_checkpointing": True,
    },
    "hunyuan3d-mini": {
        "target_polycount": 40000,
        "texture_size": 2048,
        "precision": "fp16",
    },
    "mvdream": {
        "rescale_factor": 0.5,
        "resolution": 256,
        "gradient_checkpointing": True,
        "attention_slicing": True,
    },
}
```

### Memory Management for 24GB VRAM

```python
class VRAMManager:
    def __init__(self, max_vram_gb=24):
        self.max_vram = max_vram_gb * 1024**3
        
    def get_optimal_config(self, model_name):
        """Get config that fits in VRAM budget."""
        available = torch.cuda.mem_get_info()[0]
        
        if available < 8 * 1024**3:  # Less than 8GB
            return self._get_minimal_config(model_name)
        elif available < 16 * 1024**3:  # Less than 16GB
            return self._get_balanced_config(model_name)
        else:  # 16GB or more
            return self._get_quality_config(model_name)
```

## System-Level Optimizations

### 1. CUDA Settings

```bash
# Optimize CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable TF32 for Ampere GPUs (RTX 3090)
export NVIDIA_TF32_OVERRIDE=1

# Reduce memory fragmentation
export CUDA_LAUNCH_BLOCKING=0
```

### 2. PyTorch Optimizations

```python
# Enable cudNN autotuner
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Use torch.compile (PyTorch 2.0+)
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Compile model for faster inference
compiled_model = torch.compile(model, mode="reduce-overhead")
```

### 3. CPU Optimizations

```bash
# Set CPU governor to performance
sudo cpupower frequency-set -g performance

# Disable CPU frequency scaling
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set process priority
nice -n -10 python generate.py
```

### 4. Disk I/O Optimizations

```python
# Use NVMe for cache
os.environ['HF_HOME'] = '/mnt/nvme/.huggingface'
os.environ['TORCH_HOME'] = '/mnt/nvme/.torch'

# Preallocate output buffer
output_buffer = io.BytesIO()
output_buffer.seek(100 * 1024 * 1024)  # Pre-allocate 100MB
```

## Multi-Model Pipeline Optimization

### Sequential Pipeline
```python
def optimized_pipeline(prompt):
    """Multi-stage pipeline with resource management."""
    
    # Stage 1: Quick preview (2 seconds)
    with torch.cuda.amp.autocast():
        triposr = ModelFactory.create_model("triposr", precision="fp16")
        preview = triposr.generate_from_text(prompt, resolution=256)
        triposr.unload_model()
        torch.cuda.empty_cache()
    
    # Stage 2: Add materials (5 seconds)
    with torch.cuda.amp.autocast():
        stable = ModelFactory.create_model("stable-fast-3d", precision="fp16")
        refined = stable.refine_mesh(preview, generate_pbr=True)
        stable.unload_model()
        torch.cuda.empty_cache()
    
    # Stage 3: Professional UV (8 seconds)
    with torch.cuda.amp.autocast():
        hunyuan = ModelFactory.create_model("hunyuan3d-mini", precision="fp16")
        final = hunyuan.optimize_uv(refined)
        hunyuan.unload_model()
    
    return final
```

### Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor

def parallel_generation(prompts, model_name):
    """Process multiple prompts in parallel."""
    
    def process_single(prompt):
        model = ModelFactory.create_model(model_name)
        result = model.generate_from_text(prompt)
        model.unload_model()
        return result
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(process_single, prompts))
    
    return results
```

## Monitoring and Profiling

### GPU Utilization Monitor
```python
import subprocess
import time

def monitor_gpu(duration=60):
    """Monitor GPU metrics during generation."""
    metrics = []
    start = time.time()
    
    while time.time() - start < duration:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            metrics.append({
                'utilization': float(values[0]),
                'memory_mb': float(values[1]),
                'temperature': float(values[2]),
                'timestamp': time.time() - start
            })
        
        time.sleep(1)
    
    return metrics
```

### Performance Profiler
```python
import torch.profiler as profiler

def profile_generation(model, prompt):
    """Profile model generation."""
    
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        with profiler.record_function("model_inference"):
            result = model.generate_from_text(prompt)
    
    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Save trace for visualization
    prof.export_chrome_trace("trace.json")
    
    return result
```

## Performance Benchmarks by Model

### Generation Time Comparison (RTX 3090)

| Model | Fast Mode | Balanced | Quality Mode |
|-------|-----------|----------|--------------|
| TripoSR | 0.5-1s | 1-2s | 2-3s |
| Stable-Fast-3D | 2-3s | 3-5s | 5-7s |
| Hunyuan3D-Mini | 3-5s | 5-10s | 10-15s |
| TRELLIS | 20-30s | 40-60s | 60-90s |
| MVDream | 60 min | 90 min | 150 min |

### VRAM Usage by Model

| Model | Minimum | Typical | Maximum |
|-------|---------|---------|---------|
| TripoSR | 4GB | 6GB | 8GB |
| Stable-Fast-3D | 6GB | 8GB | 10GB |
| Hunyuan3D-Mini | 10GB | 14GB | 18GB |
| TRELLIS | 8GB | 12GB | 16GB |
| MVDream | 16GB | 18GB | 22GB |

## Troubleshooting Performance Issues

### Issue: Slow Generation Despite Good GPU

**Diagnosis:**
```python
# Check if using GPU
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show your GPU

# Check GPU utilization
nvidia-smi  # Should show >80% utilization during generation
```

**Solutions:**
1. Ensure CUDA PyTorch: `pip install torch==2.7.1+cu118`
2. Check power mode: `nvidia-smi -q -d PERFORMANCE`
3. Disable CPU throttling
4. Check thermal throttling (keep <83°C)

### Issue: Frequent OOM Errors

**Solutions:**
```python
# 1. Reduce memory fragmentation
torch.cuda.empty_cache()
gc.collect()

# 2. Use gradient checkpointing
config["gradient_checkpointing"] = True

# 3. Reduce batch size
config["batch_size"] = 1

# 4. Enable CPU offloading
config["cpu_offload"] = True

# 5. Use memory-efficient attention
config["use_memory_efficient_attention"] = True
```

### Issue: Inconsistent Generation Times

**Solutions:**
```python
# 1. Warmup model
for _ in range(3):
    model.generate_from_text("warmup", num_inference_steps=1)

# 2. Lock GPU clocks
os.system("sudo nvidia-smi -lgc 1875")  # RTX 3090

# 3. Disable dynamic boosting
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

## Optimization Profiles

### Development Profile (Speed Priority)
```python
DEV_CONFIG = {
    "triposr": {"resolution": 256, "precision": "fp16"},
    "stable-fast-3d": {"target_polycount": 5000, "texture_resolution": 512},
    "trellis": {"quality_mode": "fast"},
    "hunyuan3d-mini": {"target_polycount": 10000, "texture_size": 512},
    "mvdream": {"rescale_factor": 0.3, "num_inference_steps": 20},
}
```

### Production Profile (Balanced)
```python
PROD_CONFIG = {
    "triposr": {"resolution": 512, "precision": "fp16"},
    "stable-fast-3d": {"target_polycount": 20000, "texture_resolution": 1024},
    "trellis": {"quality_mode": "balanced"},
    "hunyuan3d-mini": {"target_polycount": 30000, "texture_size": 2048},
    "mvdream": {"rescale_factor": 0.5, "num_inference_steps": 50},
}
```

### Quality Profile (Maximum Quality)
```python
QUALITY_CONFIG = {
    "triposr": {"resolution": 512, "precision": "fp32"},
    "stable-fast-3d": {"target_polycount": 50000, "texture_resolution": 2048},
    "trellis": {"quality_mode": "hq"},
    "hunyuan3d-mini": {"target_polycount": 50000, "texture_size": 4096},
    "mvdream": {"rescale_factor": 0.7, "num_inference_steps": 100},
}
```

## Optimization Checklist

### Before Production Deployment

- [ ] Profile baseline performance
- [ ] Enable FP16/BF16 precision
- [ ] Configure optimal batch size
- [ ] Set GPU to performance mode
- [ ] Enable torch.compile if using PyTorch 2.0+
- [ ] Configure memory-efficient settings
- [ ] Test with production workload
- [ ] Monitor thermal performance
- [ ] Set up performance logging
- [ ] Create fallback configurations

### Daily Optimization Tasks

- [ ] Clear GPU cache at start
- [ ] Check GPU temperature logs
- [ ] Review performance metrics
- [ ] Verify GPU utilization >80%
- [ ] Check for memory leaks

### Weekly Optimization Tasks

- [ ] Update GPU drivers if available
- [ ] Review and optimize slow queries
- [ ] Clean temporary files
- [ ] Analyze performance trends
- [ ] Test new optimization settings

## Advanced Techniques

### Dynamic Batching
```python
class DynamicBatcher:
    def __init__(self, max_batch_size=4, max_wait_time=5.0):
        self.queue = []
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
    
    def add_request(self, prompt):
        self.queue.append((prompt, time.time()))
        
        if len(self.queue) >= self.max_batch_size:
            return self.process_batch()
        
        # Check if oldest request has waited too long
        if self.queue and time.time() - self.queue[0][1] > self.max_wait_time:
            return self.process_batch()
        
        return None
```

### Memory Pool Management
```python
class MemoryPool:
    def __init__(self, pool_size_gb=20):
        self.pool_size = pool_size_gb * 1024**3
        self.allocated = {}
        
    def allocate(self, model_name, size_gb):
        if self.can_allocate(size_gb):
            self.allocated[model_name] = size_gb * 1024**3
            return True
        return False
    
    def release(self, model_name):
        if model_name in self.allocated:
            del self.allocated[model_name]
            torch.cuda.empty_cache()
```

### Model Caching Strategy
```python
class ModelCache:
    def __init__(self, max_models=2):
        self.cache = {}
        self.max_models = max_models
        self.access_times = {}
    
    def get_model(self, model_name):
        if model_name in self.cache:
            self.access_times[model_name] = time.time()
            return self.cache[model_name]
        
        # Evict LRU if cache full
        if len(self.cache) >= self.max_models:
            lru = min(self.access_times, key=self.access_times.get)
            self.cache[lru].unload_model()
            del self.cache[lru]
            del self.access_times[lru]
        
        # Load new model
        model = ModelFactory.create_model(model_name)
        self.cache[model_name] = model
        self.access_times[model_name] = time.time()
        return model
```

## Hardware-Specific Tuning

### RTX 3090 Power Settings
```bash
# Optimal power limit
sudo nvidia-smi -pl 350

# Memory clock optimization
sudo nvidia-smi -ac 10001,1875

# Enable persistence mode
sudo nvidia-smi -pm 1
```

### RTX 4090 Settings
```bash
# Higher power limit for 4090
sudo nvidia-smi -pl 450

# Optimized clocks
sudo nvidia-smi -ac 10501,2520
```

### Multi-GPU Systems
```python
# Distribute models across GPUs
gpu_assignments = {
    "triposr": 0,
    "stable-fast-3d": 0,
    "trellis": 1,
    "hunyuan3d-mini": 1,
    "mvdream": 1,
}

def get_model_with_gpu(model_name):
    gpu_id = gpu_assignments.get(model_name, 0)
    device = f"cuda:{gpu_id}"
    return ModelFactory.create_model(model_name, device=device)
```

## Conclusion

Key optimization strategies:
1. **Use FP16 precision** - 40-50% VRAM savings
2. **Clear cache regularly** - Prevent fragmentation
3. **Choose right quality mode** - Balance speed vs quality
4. **Monitor GPU metrics** - Ensure optimal utilization
5. **Profile before optimizing** - Measure, don't guess

Remember: The fastest generation is the one that doesn't fail. Prioritize stability over marginal speed gains.