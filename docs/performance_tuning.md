# MVDream Performance Tuning Guide for RTX 3090

This guide provides comprehensive performance optimization strategies for MVDream 3D generation on NVIDIA RTX 3090 (24GB VRAM).

## Quick Start: Optimal Settings

Based on extensive benchmarking, here are the recommended settings for balanced performance:

```yaml
# Optimal balanced configuration
model:
  rescale_factor: 0.5
  enable_xformers: true
  enable_memory_efficient_attention: true
  gradient_checkpointing: false
  compile_model: true

inference:
  batch_size: 2
  num_inference_steps: 30
  guidance_scale: 7.5
  n_views: 4
  resolution: 512
  fp16: true
  timestep_annealing: true
  annealing_eta: 0.8
```

**Expected Performance:**
- Generation time: ~45-90 minutes
- Peak VRAM usage: ~16GB
- Max GPU temperature: ~75°C

## Performance Factors

### 1. Rescale Factor (Impact: HIGH)

The rescale factor controls the internal processing resolution. Lower values = faster generation.

- **0.3**: Fastest, lowest quality (~30% faster)
- **0.5**: Balanced (recommended)
- **0.7**: High quality (~50% slower)

```python
# Example usage
config["model"]["rescale_factor"] = 0.5
```

### 2. Batch Size (Impact: HIGH)

Batch size affects both speed and memory usage.

| Batch Size | Speed | VRAM Usage | Recommendation |
|-----------|-------|------------|----------------|
| 1 | Slowest | ~8GB | Memory-constrained |
| 2 | Balanced | ~12GB | **Recommended** |
| 4 | Fastest | ~20GB | Quality focus |

```python
# Adjust based on available VRAM
if torch.cuda.get_device_properties(0).total_memory < 20e9:
    config["inference"]["batch_size"] = 1
```

### 3. Memory-Efficient Attention (Impact: MEDIUM)

Reduces memory usage with minimal quality impact.

```yaml
model:
  enable_xformers: true  # If available
  enable_memory_efficient_attention: true
```

**Installation** (if not available):
```bash
poetry run pip install xformers
```

### 4. Mixed Precision (FP16) (Impact: HIGH)

Using half-precision significantly reduces memory and increases speed.

```python
# Enable FP16
pipe = pipe.to(torch.float16)
# or in config
config["inference"]["fp16"] = true
```

**Benefits:**
- 40-50% memory reduction
- 20-30% speed improvement
- Negligible quality impact

### 5. Number of Inference Steps (Impact: HIGH)

Fewer steps = faster generation, but lower quality.

| Steps | Time | Quality | Use Case |
|-------|------|---------|----------|
| 20 | ~30 min | Draft | Testing |
| 30 | ~45 min | Good | **Balanced** |
| 50 | ~75 min | High | Production |
| 100 | ~150 min | Maximum | Final renders |

### 6. Model Compilation (Impact: MEDIUM)

PyTorch 2.0+ model compilation can improve speed.

```python
import torch
model = torch.compile(model, mode="reduce-overhead")
```

**Benefits:**
- 10-20% speed improvement
- One-time compilation overhead
- Requires PyTorch 2.0+

### 7. Gradient Checkpointing (Impact: MEDIUM)

Trade compute for memory - useful for large batches.

```yaml
model:
  gradient_checkpointing: true  # Enable for batch_size > 2
```

**Trade-offs:**
- Reduces memory by ~30%
- Increases computation by ~20%
- Best for memory-constrained scenarios

### 8. Resolution (Impact: HIGH)

Output resolution significantly affects both time and memory.

| Resolution | Time Multiplier | VRAM | Quality |
|-----------|----------------|------|---------|
| 256 | 1x | ~6GB | Low |
| 512 | 4x | ~16GB | **Standard** |
| 1024 | 16x | ~40GB | High (requires tiling) |

### 9. Number of Views (Impact: MEDIUM)

Reducing views speeds generation but affects 3D consistency.

```yaml
inference:
  n_views: 4  # Standard (front, back, left, right)
  # n_views: 2  # Faster but less consistent
```

### 10. CPU Offloading (Impact: LOW)

Move model layers to CPU when not in use.

```python
pipe.enable_sequential_cpu_offload()
```

**Trade-offs:**
- Enables larger batch sizes
- Significantly slower (2-3x)
- Use only when hitting VRAM limits

## Optimization Strategies

### Speed-Optimized Configuration

For fastest generation with acceptable quality:

```yaml
model:
  rescale_factor: 0.3
  enable_xformers: true
  enable_memory_efficient_attention: true
  compile_model: true

inference:
  batch_size: 1
  num_inference_steps: 20
  guidance_scale: 5.0
  n_views: 2
  resolution: 256
  fp16: true
  timestep_annealing: true
  annealing_eta: 0.5
```

**Performance:** ~20-30 minutes per generation

### Quality-Optimized Configuration

For highest quality output:

```yaml
model:
  rescale_factor: 0.7
  enable_xformers: true
  enable_memory_efficient_attention: true
  compile_model: true

inference:
  batch_size: 4
  num_inference_steps: 100
  guidance_scale: 10.0
  n_views: 4
  resolution: 512
  fp16: false
  timestep_annealing: false
  annealing_eta: 1.0
```

**Performance:** ~2-3 hours per generation

### Memory-Optimized Configuration

For systems with limited VRAM or running multiple processes:

```yaml
model:
  rescale_factor: 0.5
  enable_xformers: true
  enable_memory_efficient_attention: true
  gradient_checkpointing: true

inference:
  batch_size: 1
  num_inference_steps: 30
  guidance_scale: 7.5
  n_views: 4
  resolution: 256
  fp16: true
  cpu_offload: true
```

**Performance:** ~60-90 minutes, uses ~10GB VRAM

## Advanced Techniques

### 1. Dynamic Batching

Adjust batch size based on prompt complexity:

```python
def get_optimal_batch_size(prompt: str) -> int:
    """Determine batch size based on prompt complexity."""
    complexity_keywords = ["detailed", "intricate", "complex", "realistic"]
    if any(keyword in prompt.lower() for keyword in complexity_keywords):
        return 1  # Complex prompts need more memory
    return 2  # Standard prompts
```

### 2. Progressive Resolution

Start with low resolution and upscale:

```python
# Generate at low res
low_res_output = generate(resolution=256)
# Upscale with AI
high_res_output = upscale(low_res_output, target=512)
```

### 3. Timestep Annealing

Reduce computation in later timesteps:

```python
scheduler.set_timesteps(num_inference_steps, eta=0.8)
```

### 4. CUDA Graphs

For repeated generations with same config:

```python
# Capture CUDA graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(input)

# Replay for faster execution
g.replay()
```

### 5. Multi-GPU Support

Distribute computation across multiple GPUs:

```python
model = torch.nn.DataParallel(model, device_ids=[0, 1])
```

## Monitoring and Profiling

### GPU Metrics

Monitor during generation:

```bash
watch -n 1 nvidia-smi
```

### PyTorch Profiler

Profile to find bottlenecks:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.GPU]) as prof:
    output = generate(prompt)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Memory Profiling

Track memory usage:

```python
torch.cuda.memory_summary(device=0)
```

## Troubleshooting Performance Issues

### Issue: Generation takes >3 hours

**Solutions:**
1. Reduce `num_inference_steps` to 30
2. Enable FP16: `fp16: true`
3. Lower `rescale_factor` to 0.3
4. Reduce `resolution` to 256

### Issue: CUDA Out of Memory

**Solutions:**
1. Reduce `batch_size` to 1
2. Enable `gradient_checkpointing`
3. Lower `resolution`
4. Enable `cpu_offload`
5. Clear cache: `torch.cuda.empty_cache()`

### Issue: GPU Temperature >83°C

**Solutions:**
1. Reduce `batch_size`
2. Add delays between generations
3. Improve case cooling
4. Reduce GPU power limit:
   ```bash
   sudo nvidia-smi -pl 300  # Limit to 300W
   ```

### Issue: Inconsistent Generation Times

**Solutions:**
1. Disable CPU frequency scaling
2. Set GPU to performance mode:
   ```bash
   sudo nvidia-smi -pm 1
   ```
3. Close other GPU applications
4. Use consistent `torch.manual_seed()`

## Benchmarking Your Setup

Run the benchmark suite:

```bash
# Quick benchmark (5 minutes)
poetry run poe benchmark --quick

# Full benchmark (30+ minutes)
poetry run poe benchmark

# Update config with optimal values
poetry run poe benchmark --update-config
```

View results:
```bash
cat benchmarks/rtx3090_results.json
cat benchmarks/benchmark_report.md
```

## RTX 3090 Specific Optimizations

### Power Management

Optimal power settings for sustained generation:

```bash
# Set power limit to 350W (default)
sudo nvidia-smi -pl 350

# Enable persistence mode
sudo nvidia-smi -pm 1

# Set application clock
sudo nvidia-smi -ac 9501,1695
```

### Memory Clock Optimization

```bash
# Check current clocks
nvidia-smi -q -d CLOCK

# Set memory transfer rate
sudo nvidia-smi -ac 9751,1695
```

### Thermal Management

Keep GPU temperature optimal:

1. **Target Temperature:** 70-75°C
2. **Fan Curve:** Aggressive (70% at 70°C)
3. **Case Airflow:** Ensure positive pressure
4. **Thermal Pads:** Replace if >2 years old

### Driver Settings

NVIDIA Control Panel optimizations:
- Power Management: Prefer Maximum Performance
- Texture Filtering: High Performance
- Threaded Optimization: On
- CUDA - Sysmem Fallback: Prefer No Sysmem Fallback

## Configuration Templates

### Development/Testing

```yaml
# Fast iteration for development
profile: development
model:
  rescale_factor: 0.3
inference:
  num_inference_steps: 20
  resolution: 256
  fp16: true
```

### Production

```yaml
# Balanced for production use
profile: production
model:
  rescale_factor: 0.5
inference:
  num_inference_steps: 50
  resolution: 512
  fp16: true
```

### Maximum Quality

```yaml
# Highest quality, longest time
profile: quality
model:
  rescale_factor: 0.7
inference:
  num_inference_steps: 100
  resolution: 512
  fp16: false
```

## Performance Expectations

Based on RTX 3090 benchmarks:

| Configuration | Time | VRAM | Quality |
|--------------|------|------|---------|
| Speed | 20-30 min | 8GB | Draft |
| Balanced | 45-90 min | 16GB | Good |
| Quality | 2-3 hours | 20GB | Excellent |

## Additional Resources

- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

## Summary

For most use cases on RTX 3090, the optimal configuration balances:
- **Rescale factor:** 0.5
- **Batch size:** 2
- **FP16:** Enabled
- **Steps:** 30-50
- **Resolution:** 512

This achieves generation in 45-90 minutes with good quality and reliable performance.