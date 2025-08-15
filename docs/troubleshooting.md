# MVDream Troubleshooting Guide

This comprehensive guide covers common issues and their solutions for MVDream 3D generation setup.

## Quick Diagnostics

Run the diagnostic tool first to identify issues:
```bash
poe diagnose
```

This will check:
- System configuration
- CUDA and GPU setup
- Python dependencies
- Model files
- Directory structure
- Available disk space

## Top 5 Common Errors and Solutions

### 1. PyTorch NCCL Library Error

**Error:**
```
ImportError: /path/to/libtorch_cuda.so: undefined symbol: ncclGroupSimulateEnd
```

**Cause:** Incompatibility between PyTorch version and NCCL library.

**Solutions:**
1. Reinstall PyTorch with matching CUDA version:
   ```bash
   poetry run pip uninstall torch torchvision -y
   poetry run pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. If error persists, set environment variable:
   ```bash
   export LD_PRELOAD=/opt/cuda/lib64/libnccl.so.2
   ```

3. As a last resort, disable NCCL:
   ```bash
   export NCCL_P2P_DISABLE=1
   ```

### 2. CUDA Out of Memory (OOM) Error

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Cause:** Model requires more VRAM than available.

**Solutions:**
1. Reduce batch size in `configs/mvdream-sd21.yaml`:
   ```yaml
   inference:
     batch_size: 1  # Reduce from 4 to 1
   ```

2. Enable memory-efficient attention:
   ```yaml
   model:
     enable_xformers: true
     enable_memory_efficient_attention: true
   ```

3. Clear GPU cache before generation:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. Monitor GPU memory usage:
   ```bash
   watch -n 1 nvidia-smi
   ```

5. Kill other GPU processes:
   ```bash
   sudo fuser -v /dev/nvidia*
   sudo kill -9 <PID>
   ```

### 3. Model File Not Found

**Error:**
```
FileNotFoundError: Model file 'sd-v2.1-base-4view.pt' not found
```

**Cause:** Model files haven't been downloaded or are in wrong location.

**Solutions:**
1. Download models:
   ```bash
   poe download-models
   ```

2. Verify model integrity:
   ```bash
   poe verify-models
   ```

3. Check model location:
   ```bash
   ls -la /mnt/datadrive_m2/dream-cad/models/
   ```

4. Update model path in code if needed:
   ```python
   model_path = Path("/mnt/datadrive_m2/dream-cad/models/sd-v2.1-base-4view.pt")
   ```

### 4. ImportError: Missing Dependencies

**Error:**
```
ImportError: No module named 'diffusers'
```

**Cause:** Required Python packages not installed.

**Solutions:**
1. Install all dependencies:
   ```bash
   poetry install --with dev
   ```

2. For specific missing packages:
   ```bash
   poetry add <package_name>
   ```

3. If Poetry fails, use pip directly:
   ```bash
   poetry run pip install diffusers transformers accelerate
   ```

4. Verify installation:
   ```bash
   poetry show
   ```

### 5. GPU Not Detected

**Error:**
```
torch.cuda.is_available() returns False
```

**Cause:** CUDA toolkit, drivers, or PyTorch CUDA support issues.

**Solutions:**
1. Check NVIDIA driver:
   ```bash
   nvidia-smi
   ```
   
   If not working, reinstall driver:
   ```bash
   sudo mhwd -i pci video-nvidia
   ```

2. Verify CUDA installation:
   ```bash
   nvcc --version
   ```

3. Set CUDA environment variables in `~/.zshrc`:
   ```bash
   export CUDA_HOME=/opt/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

4. Reinstall PyTorch with CUDA:
   ```bash
   poetry run pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

## Memory Optimization Tips

### Reduce Memory Usage

1. **Lower resolution generation:**
   ```yaml
   resolution: 256  # Instead of 512
   ```

2. **Use gradient checkpointing:**
   ```yaml
   model:
     gradient_checkpointing: true
   ```

3. **Reduce number of views:**
   ```yaml
   n_views: 2  # Instead of 4
   ```

4. **Enable CPU offloading:**
   ```python
   pipe = pipe.enable_sequential_cpu_offload()
   ```

### Monitor Memory Usage

```python
import torch
import psutil

# GPU memory
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"GPU cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")

# System RAM
print(f"RAM used: {psutil.virtual_memory().percent}%")
```

## Performance Tuning

### Optimal Settings for RTX 3090

```yaml
# configs/mvdream-sd21.yaml
inference:
  batch_size: 4
  num_inference_steps: 30
  guidance_scale: 7.5
  
model:
  enable_xformers: true
  enable_memory_efficient_attention: true
  
hardware:
  max_vram_gb: 20  # Leave 4GB buffer
```

### Speed Optimizations

1. **Use half precision (fp16):**
   ```python
   pipe = pipe.to(torch.float16)
   ```

2. **Compile model with torch.compile:**
   ```python
   model = torch.compile(model, mode="reduce-overhead")
   ```

3. **Use CUDA graphs:**
   ```python
   with torch.cuda.graph():
       output = model(input)
   ```

## Directory Structure Issues

### Missing Directories

Create all required directories:
```bash
mkdir -p /mnt/datadrive_m2/dream-cad/{models,outputs/{2d_test,3d_test},configs,logs,tests/results}
```

### Permission Issues

Fix permissions:
```bash
chmod -R 755 /mnt/datadrive_m2/dream-cad
chown -R $USER:$USER /mnt/datadrive_m2/dream-cad
```

## Disk Space Management

### Check Space

```bash
df -h /mnt/datadrive_m2
du -sh /mnt/datadrive_m2/dream-cad/*
```

### Clean Up

1. Remove old outputs:
   ```bash
   rm -rf /mnt/datadrive_m2/dream-cad/outputs/old_*
   ```

2. Clear HuggingFace cache:
   ```bash
   rm -rf /mnt/datadrive_m2/.huggingface/hub/models--*/blobs
   ```

3. Clear PyTorch cache:
   ```bash
   rm -rf ~/.cache/torch
   ```

## Environment Issues

### Virtual Environment Not Activated

```bash
cd /mnt/datadrive_m2/dream-cad
poetry shell
# or
source .venv/bin/activate
```

### Poetry Issues

Reset Poetry environment:
```bash
poetry env remove python
poetry install --with dev
```

## Network and Download Issues

### Slow Model Downloads

Use mirror or different source:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Connection Timeouts

Increase timeout:
```python
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
```

## Generation Quality Issues

### Poor Quality Output

1. Increase inference steps:
   ```yaml
   num_inference_steps: 50  # Instead of 30
   ```

2. Adjust guidance scale:
   ```yaml
   guidance_scale: 10.0  # Higher for more prompt adherence
   ```

3. Try different seeds:
   ```python
   generator = torch.Generator().manual_seed(42)
   ```

### Inconsistent Multi-View

1. Increase view consistency weight:
   ```yaml
   view_consistency_weight: 1.5
   ```

2. Use deterministic generation:
   ```python
   torch.use_deterministic_algorithms(True)
   ```

## Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check CUDA Errors

```python
import torch
torch.cuda.synchronize()
print(torch.cuda.get_last_error())
```

### Profile Performance

```python
with torch.profiler.profile() as prof:
    output = model(input)
print(prof.key_averages())
```

## Quick Fix Scripts

### Reset GPU

```bash
#!/bin/bash
# scripts/reset_gpu.sh
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
nvidia-smi -r
```

### Clear All Caches

```bash
#!/bin/bash
# scripts/clear_caches.sh
rm -rf ~/.cache/torch
rm -rf ~/.cache/huggingface
poetry run python -c "import torch; torch.cuda.empty_cache()"
```

### Test Basic Functionality

```python
#!/usr/bin/env python3
# scripts/test_basic.py
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
```

## Getting Help

If issues persist after trying these solutions:

1. Run diagnostics and save output:
   ```bash
   poe diagnose > diagnostic_output.txt
   ```

2. Check logs:
   ```bash
   tail -n 100 logs/mvdream.log
   ```

3. Create minimal reproducible example

4. Check project documentation:
   - [System Requirements](system-specs.md)
   - [CUDA Setup](cuda-setup.md)
   - [Model Management](models.md)
   - [Project Structure](project-structure.md)

5. Search existing issues or create new one with:
   - Diagnostic output
   - Error messages
   - System configuration
   - Steps to reproduce

## Common Warning Messages (Safe to Ignore)

- `UserWarning: xformers not available` - Performance will be slightly slower
- `FutureWarning: Deprecated loader` - Will be fixed in future updates
- `Setting pad_token_id to eos_token_id` - Normal behavior for text models
- `Some weights not initialized from pretrained` - Expected for fine-tuned models