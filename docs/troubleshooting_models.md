# Troubleshooting Guide for 3D Generation Models

## Quick Diagnostic

Run this command to diagnose common issues:
```bash
poetry run python scripts/diagnose.py
```

## Common Issues Across All Models

### 1. CUDA Out of Memory Error

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
1. **Enable FP16 precision:**
   ```python
   model = ModelFactory.create_model("model_name", precision="fp16")
   ```

2. **Clear GPU cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Reduce batch size:**
   ```python
   config = {"batch_size": 1}
   ```

4. **Use CPU offloading:**
   ```python
   model = ModelFactory.create_model("model_name", cpu_offload=True)
   ```

5. **Switch to lower VRAM model:**
   - MVDream (20GB) → Hunyuan3D (12GB) → TRELLIS (8GB) → Stable-Fast-3D (6GB) → TripoSR (4GB)

### 2. Model Download Failures

**Symptoms:**
```
HTTPError: 403 Client Error: Forbidden
ConnectionError: Failed to reach huggingface.co
```

**Solutions:**
1. **Check internet connection:**
   ```bash
   ping huggingface.co
   ```

2. **Set proxy if behind firewall:**
   ```bash
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

3. **Use alternative download method:**
   ```bash
   # Manual download with wget
   wget https://huggingface.co/model/resolve/main/model.safetensors
   ```

4. **Configure HuggingFace cache:**
   ```bash
   export HF_HOME=/mnt/datadrive_m2/.huggingface
   ```

### 3. Slow Generation Speed

**Symptoms:**
- Generation takes much longer than expected
- GPU utilization below 50%

**Solutions:**
1. **Check GPU power mode:**
   ```bash
   nvidia-smi -q -d PERFORMANCE
   sudo nvidia-smi -pm 1  # Enable persistence mode
   ```

2. **Verify CUDA is being used:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Should show your GPU
   ```

3. **Check thermal throttling:**
   ```bash
   nvidia-smi -q -d TEMPERATURE
   # If > 83°C, improve cooling
   ```

4. **Optimize settings:**
   ```python
   config = {
       "num_inference_steps": 30,  # Reduce from 50
       "guidance_scale": 7.0,       # Reduce from 7.5
   }
   ```

### 4. Poor Quality Output

**Symptoms:**
- Blurry or low-detail models
- Artifacts or noise in output
- Geometry doesn't match prompt

**Solutions:**
1. **Increase quality settings:**
   ```python
   config = {
       "num_inference_steps": 75,  # Increase steps
       "guidance_scale": 8.0,       # Increase guidance
       "quality_mode": "hq",        # For TRELLIS
   }
   ```

2. **Try different models:**
   - For quality: MVDream > Hunyuan3D > TRELLIS > Stable-Fast-3D > TripoSR

3. **Improve prompts:**
   ```python
   # Bad: "chair"
   # Good: "a modern wooden chair with leather seat, high quality, detailed"
   ```

## Model-Specific Issues

### TripoSR Issues

#### Issue: "No module named 'triposr'"
**Solution:**
```bash
pip install git+https://github.com/VAST-AI-Research/TripoSR.git
```

#### Issue: Low quality output
**Solution:**
```python
config = {
    "resolution": 512,  # Increase from 256
    "remove_background": True,  # Clean input
}
```

#### Issue: Black or empty output
**Solution:**
- Check input image format (PNG/JPG only)
- Ensure image has clear subject
- Try with white background

### Stable-Fast-3D Issues

#### Issue: Missing PBR textures
**Solution:**
```python
config = {
    "generate_pbr": True,
    "texture_resolution": 1024,
    "delight": True,  # Remove baked lighting
}
```

#### Issue: UV unwrapping failures
**Solution:**
```python
config = {
    "uv_padding": 4,  # Increase padding
    "uv_method": "angle_based",  # Try different method
}
```

#### Issue: High polycount warning
**Solution:**
```python
config = {
    "target_polycount": 10000,  # Reduce from default
    "simplification_ratio": 0.5,  # More aggressive reduction
}
```

### TRELLIS Issues

#### Issue: "Windows optimized fork not found"
**Solution:**
```bash
# Use official version
pip install git+https://github.com/microsoft/TRELLIS.git

# Or install Windows fork
pip install git+https://github.com/IgorAherne/TRELLIS-Windows.git
```

#### Issue: NeRF output not viewable
**Solution:**
```python
# Convert to mesh
result = model.convert_representation(
    nerf_data,
    from_type="nerf",
    to_type="mesh"
)
```

#### Issue: Quality mode OOM
**Solution:**
```python
config = {
    "quality_mode": "fast",  # Instead of "hq"
    "slat_resolution": 64,   # Reduce from 128
}
```

### Hunyuan3D-Mini Issues

#### Issue: Commercial license warning
**Solution:**
```python
# Suppress for development
model = ModelFactory.create_model(
    "hunyuan3d-mini",
    suppress_license_warning=True
)
# Note: Still requires license for commercial use
```

#### Issue: UV islands overlapping
**Solution:**
```python
config = {
    "uv_method": "smart_projection",
    "uv_island_margin": 0.01,  # Increase margin
    "uv_angle_limit": 66,       # Adjust angle threshold
}
```

#### Issue: Multi-view fusion failures
**Solution:**
```python
# Ensure consistent views
config = {
    "normalize_views": True,
    "fusion_threshold": 0.7,  # Lower for more lenient fusion
}
```

### MVDream Issues

#### Issue: Extremely slow generation
**Solution:**
```python
config = {
    "rescale_factor": 0.3,  # Reduce quality for speed
    "n_views": 2,           # Reduce from 4 views
    "resolution": 128,      # Reduce from 256
}
```

#### Issue: Multi-view inconsistency
**Solution:**
```python
config = {
    "guidance_scale": 10.0,  # Increase for consistency
    "view_batch_size": 1,    # Process views sequentially
}
```

#### Issue: Memory fragmentation
**Solution:**
```python
# Enable gradient checkpointing
config = {
    "gradient_checkpointing": True,
    "sequential_offload": True,
}
```

## Error Messages Reference

### PyTorch/CUDA Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: CUDA out of memory` | Insufficient VRAM | Reduce batch size, enable FP16 |
| `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` | CUDA/cuDNN mismatch | Reinstall PyTorch with correct CUDA |
| `AssertionError: Torch not compiled with CUDA enabled` | CPU-only PyTorch | Install CUDA version of PyTorch |
| `RuntimeError: NCCL error` | Multi-GPU issue | Set `NCCL_P2P_DISABLE=1` |

### Model Loading Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: model.safetensors` | Model not downloaded | Run model download script |
| `KeyError: 'model.layers.0.weight'` | Incompatible checkpoint | Download correct model version |
| `RuntimeError: Error(s) in loading state_dict` | Version mismatch | Update model or use compatibility mode |

### Generation Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Expected 4D tensor` | Wrong input format | Ensure proper image preprocessing |
| `RuntimeError: shape mismatch` | Resolution mismatch | Use model's expected resolution |
| `IndexError: list index out of range` | Empty generation | Check prompt and increase steps |

## Performance Optimization Tips

### 1. Memory Optimization Checklist
- [ ] Enable FP16/BF16 precision
- [ ] Use gradient checkpointing
- [ ] Enable CPU offloading
- [ ] Clear cache between generations
- [ ] Reduce batch size to 1
- [ ] Lower resolution/quality settings
- [ ] Close other GPU applications

### 2. Speed Optimization Checklist
- [ ] Use NVMe SSD for cache
- [ ] Enable GPU persistence mode
- [ ] Disable CPU power saving
- [ ] Use PyTorch 2.0+ with compile
- [ ] Enable xformers optimizations
- [ ] Warm up model before batch
- [ ] Use async processing

### 3. Quality Optimization Checklist
- [ ] Increase inference steps
- [ ] Raise guidance scale
- [ ] Use higher resolution
- [ ] Enable all quality features
- [ ] Try multiple seeds
- [ ] Use better prompts
- [ ] Post-process outputs

## Debugging Tools

### 1. GPU Memory Monitor
```python
def monitor_gpu():
    import torch
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    print(f"Cached: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
    print(f"Free: {torch.cuda.mem_get_info()[0]/1024**3:.2f}GB")
```

### 2. Generation Profiler
```python
import time
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    result = model.generate(prompt)
    
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 3. Model Validator
```python
def validate_model(model_name):
    try:
        model = ModelFactory.create_model(model_name)
        result = model.generate_from_text("test cube", num_inference_steps=1)
        print(f"✓ {model_name} working")
    except Exception as e:
        print(f"✗ {model_name} failed: {e}")
```

## Getting Help

### 1. Diagnostic Information to Provide
When reporting issues, include:
```bash
# System info
poetry run python scripts/diagnose.py > diagnostic.txt

# GPU info
nvidia-smi >> diagnostic.txt

# Python environment
poetry show >> diagnostic.txt

# Error traceback
# Copy full error message
```

### 2. Community Resources
- **GitHub Issues**: [Report bugs](https://github.com/Ladvien/dream-cad/issues)
- **Discussions**: [Ask questions](https://github.com/Ladvien/dream-cad/discussions)
- **Discord**: Community support channel

### 3. Log Files
Check these logs for detailed error information:
```bash
# Application logs
tail -f logs/dream_cad.log

# GPU logs
nvidia-smi -l 1 > gpu_monitor.log

# System logs
journalctl -u dream-cad -f
```

## Preventive Maintenance

### Daily Checks
1. Clear GPU cache
2. Check disk space
3. Monitor GPU temperature

### Weekly Tasks
1. Update GPU drivers
2. Clean temporary files
3. Restart services

### Monthly Tasks
1. Update dependencies
2. Run full benchmark suite
3. Review error logs

## FAQ

**Q: Which model should I use if I keep getting OOM errors?**
A: Start with TripoSR (4GB), then try Stable-Fast-3D (6GB) if quality is insufficient.

**Q: Why is generation much slower than benchmarks?**
A: Check GPU power mode, thermal throttling, and ensure CUDA is properly configured.

**Q: Can I run models on CPU only?**
A: Only TripoSR supports CPU fallback, but it's 10-50x slower than GPU.

**Q: How do I know if a model is using GPU?**
A: Monitor with `nvidia-smi` during generation - GPU utilization should be >50%.

**Q: Why do some models produce black/empty outputs?**
A: Usually due to failed preprocessing, wrong input format, or insufficient inference steps.