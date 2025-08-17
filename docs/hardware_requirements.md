# Hardware Requirements Guide

## System Requirements Overview

Dream-CAD supports a range of hardware configurations, from entry-level GPUs to high-end workstations. This guide helps you understand the requirements for each model and optimize your setup.

## Minimum System Requirements

### Absolute Minimum (TripoSR only)
- **GPU**: NVIDIA GTX 1060 6GB or better
- **VRAM**: 4GB minimum
- **RAM**: 8GB system memory
- **CPU**: 4-core processor (Intel i5-8400 or AMD Ryzen 5 2600)
- **Storage**: 20GB free space
- **OS**: Ubuntu 20.04+ or Windows 10/11 with WSL2
- **CUDA**: 11.8 or higher

### Recommended Minimum (All models except MVDream)
- **GPU**: NVIDIA RTX 2070 Super or better
- **VRAM**: 8GB
- **RAM**: 16GB system memory
- **CPU**: 6-core processor (Intel i7-9700K or AMD Ryzen 5 3600)
- **Storage**: 50GB SSD space
- **OS**: Ubuntu 20.04+ or Manjaro
- **CUDA**: 11.8 or higher

## Per-Model Requirements

### TripoSR
| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU** | GTX 1060 6GB | RTX 3060 | RTX 4070 |
| **VRAM** | 4GB | 6GB | 8GB+ |
| **RAM** | 8GB | 16GB | 32GB |
| **Storage** | 10GB | 20GB SSD | 50GB NVMe |
| **Generation Time** | 2-3s | 1-2s | 0.5-1s |

**Notes:**
- Can run on most modern GPUs
- FP16 mode reduces VRAM by 40%
- CPU fallback available but 10x slower

### Stable-Fast-3D
| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU** | GTX 1660 Ti | RTX 3060 Ti | RTX 4070 Ti |
| **VRAM** | 6GB | 8GB | 12GB+ |
| **RAM** | 12GB | 16GB | 32GB |
| **Storage** | 15GB | 30GB SSD | 50GB NVMe |
| **Generation Time** | 5-7s | 3-5s | 2-3s |

**Notes:**
- PBR material generation requires additional 1GB VRAM
- Delighting process benefits from faster GPU
- Texture resolution scales with available VRAM

### TRELLIS
| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU** | RTX 2070 | RTX 3070 | RTX 4080 |
| **VRAM** | 8GB | 12GB | 16GB+ |
| **RAM** | 16GB | 24GB | 32GB |
| **Storage** | 20GB | 40GB SSD | 100GB NVMe |
| **Generation Time** | 60-90s | 40-60s | 20-30s |

**Quality Mode Requirements:**
- **Fast Mode**: 8GB VRAM, 30s generation
- **Balanced Mode**: 12GB VRAM, 60s generation
- **HQ Mode**: 16GB VRAM, 90s generation

### Hunyuan3D-Mini
| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU** | RTX 2080 | RTX 3080 | RTX 4090 |
| **VRAM** | 12GB | 16GB | 24GB |
| **RAM** | 16GB | 32GB | 64GB |
| **Storage** | 25GB | 50GB SSD | 100GB NVMe |
| **Generation Time** | 10-15s | 7-10s | 3-5s |

**Notes:**
- Professional UV unwrapping requires additional 2GB VRAM
- Higher polycount targets need more memory
- Multi-view fusion doubles VRAM requirement

### MVDream
| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU** | RTX 3080 | RTX 3090 | RTX 4090 |
| **VRAM** | 16GB | 20GB | 24GB |
| **RAM** | 24GB | 32GB | 64GB |
| **Storage** | 30GB | 60GB SSD | 200GB NVMe |
| **Generation Time** | 150-180min | 90-120min | 60-90min |

**Notes:**
- Cannot run on consumer GPUs with <16GB VRAM
- Benefits significantly from NVMe storage
- Memory-efficient mode available but 30% slower

## GPU Compatibility Matrix

### Fully Supported GPUs (All Models)
- **NVIDIA RTX 4090** (24GB) - Best performance
- **NVIDIA RTX 4080** (16GB) - Excellent for most models
- **NVIDIA RTX 3090/3090 Ti** (24GB) - Best value for all models
- **NVIDIA A100** (40/80GB) - Data center grade
- **NVIDIA A40** (48GB) - Professional workstation

### Partially Supported (Some Models)
- **NVIDIA RTX 4070 Ti** (12GB) - All except MVDream
- **NVIDIA RTX 3080/3080 Ti** (10/12GB) - TripoSR, Stable-Fast-3D, TRELLIS
- **NVIDIA RTX 3070/3070 Ti** (8GB) - TripoSR, Stable-Fast-3D
- **NVIDIA RTX 3060** (12GB) - Good for most except MVDream
- **NVIDIA RTX 2080 Ti** (11GB) - Legacy support

### Entry Level (Limited Models)
- **NVIDIA RTX 3060 Ti** (8GB) - TripoSR, Stable-Fast-3D
- **NVIDIA RTX 3050** (8GB) - TripoSR only
- **NVIDIA GTX 1660 Ti** (6GB) - TripoSR only
- **NVIDIA GTX 1070/1080** (8GB) - TripoSR with limitations

## Memory Optimization Techniques

### VRAM Reduction Strategies
1. **Enable FP16 Precision**
   ```python
   config = {"precision": "fp16"}  # Reduces VRAM by 40-50%
   ```

2. **CPU Offloading**
   ```python
   config = {"cpu_offload": True}  # Slower but saves 30% VRAM
   ```

3. **Gradient Checkpointing**
   ```python
   config = {"gradient_checkpointing": True}  # Saves 20% VRAM
   ```

4. **Reduce Batch Size**
   ```python
   config = {"batch_size": 1}  # Minimal VRAM usage
   ```

5. **Lower Resolution**
   ```python
   config = {"resolution": 256}  # Reduces VRAM quadratically
   ```

### System RAM Optimization
1. **Clear Cache Regularly**
   ```python
   import gc
   import torch
   torch.cuda.empty_cache()
   gc.collect()
   ```

2. **Limit Model Loading**
   - Load one model at a time
   - Unload models after use
   - Use model warmup/cooldown

3. **Optimize Swap Space**
   ```bash
   # Increase swap to 2x RAM
   sudo fallocate -l 64G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

## Storage Requirements

### Model Weights Storage
| Model | Weights Size | Cache Size | Total |
|-------|-------------|------------|-------|
| TripoSR | 1.5GB | 2GB | 3.5GB |
| Stable-Fast-3D | 2.5GB | 3GB | 5.5GB |
| TRELLIS | 4GB | 5GB | 9GB |
| Hunyuan3D-Mini | 3GB | 4GB | 7GB |
| MVDream | 5.2GB | 10GB | 15.2GB |
| **Total All Models** | **16.2GB** | **24GB** | **40.2GB** |

### Working Space Requirements
- **Temporary files**: 10-20GB per generation
- **Output storage**: 10-100MB per model
- **Benchmark data**: 5GB for full suite
- **Logs and metrics**: 1GB

### Recommended Storage Setup
```
/mnt/datadrive_m2/dream-cad/
├── models/          # 20GB - Model weights (SSD recommended)
├── cache/           # 30GB - Hugging Face cache (SSD preferred)
├── outputs/         # 50GB+ - Generated models (HDD acceptable)
├── temp/            # 20GB - Working directory (NVMe optimal)
└── benchmarks/      # 10GB - Performance data (Any storage)
```

## CPU Requirements

### Minimum CPU Specs
- **Cores**: 4 physical cores
- **Threads**: 8 threads
- **Frequency**: 3.0 GHz base
- **Architecture**: x86_64 with AVX2

### Recommended CPU Specs
- **Cores**: 8+ physical cores
- **Threads**: 16+ threads
- **Frequency**: 3.5+ GHz base
- **Architecture**: x86_64 with AVX512
- **Cache**: 16MB+ L3 cache

### CPU Impact on Performance
- **Model Loading**: 20-30% faster with better CPU
- **Preprocessing**: 40% improvement with AVX512
- **Postprocessing**: 50% faster mesh operations
- **Multi-model**: Enables parallel preprocessing

## Network Requirements

### Download Speeds Needed
- **Model Downloads**: 50 Mbps minimum (100+ Mbps recommended)
- **HuggingFace Access**: Stable connection required
- **Total Download Size**: ~20GB for all models

### Firewall Configuration
```bash
# Required ports
- 443 (HTTPS) - Model downloads
- 7860 (HTTP) - Gradio web interface
- 8080 (HTTP) - API server (optional)
```

## Operating System Support

### Linux (Recommended)
- **Ubuntu**: 20.04 LTS, 22.04 LTS, 24.04 LTS
- **Manjaro**: Latest stable
- **Debian**: 11 or 12
- **Fedora**: 38+
- **RHEL/CentOS**: 8.x or 9.x

### Windows
- **Windows 11**: With WSL2
- **Windows 10**: Version 2004+ with WSL2
- **Native Windows**: Experimental support

### macOS
- **Status**: Not supported (NVIDIA CUDA required)
- **Alternative**: Cloud deployment or remote Linux machine

## Cloud and Container Deployment

### AWS EC2 Instances
| Model | Instance Type | Cost/Hour |
|-------|--------------|-----------|
| TripoSR | g4dn.xlarge | $0.526 |
| Stable-Fast-3D | g4dn.2xlarge | $0.752 |
| TRELLIS | g4dn.4xlarge | $1.204 |
| Hunyuan3D | g5.2xlarge | $1.212 |
| MVDream | g5.4xlarge | $1.696 |

### Google Cloud Platform
| Model | Machine Type | Cost/Hour |
|-------|-------------|-----------|
| TripoSR | n1-standard-4 + T4 | $0.45 |
| Stable-Fast-3D | n1-standard-8 + T4 | $0.65 |
| TRELLIS | n1-highmem-4 + V100 | $1.50 |
| Hunyuan3D | a2-highgpu-1g | $1.75 |
| MVDream | a2-highgpu-2g | $3.50 |

### Docker Requirements
```dockerfile
# Minimum Docker resources
DOCKER_MEMORY=16GB
DOCKER_CPUS=4
DOCKER_GPU=all
```

## Performance Scaling

### Multi-GPU Support
- **Data Parallel**: Not supported (single model inference)
- **Model Parallel**: Experimental for MVDream
- **Multi-Instance**: Supported via queue system

### Expected Performance by GPU

#### Generation Time (seconds)
| Model | RTX 3060 | RTX 3070 | RTX 3080 | RTX 3090 | RTX 4090 |
|-------|----------|----------|----------|----------|----------|
| TripoSR | 2.0 | 1.5 | 1.0 | 0.8 | 0.5 |
| Stable-Fast-3D | 6.0 | 5.0 | 4.0 | 3.5 | 2.5 |
| TRELLIS | N/A | 70 | 50 | 40 | 25 |
| Hunyuan3D | N/A | N/A | 12 | 8 | 5 |
| MVDream | N/A | N/A | N/A | 5400 | 3600 |

## Troubleshooting Hardware Issues

### CUDA Out of Memory
```python
# Check available VRAM
import torch
print(f"Available: {torch.cuda.mem_get_info()[0]/1024**3:.1f}GB")
print(f"Total: {torch.cuda.mem_get_info()[1]/1024**3:.1f}GB")

# Clear cache
torch.cuda.empty_cache()
```

### Slow Generation
1. Check GPU utilization: `nvidia-smi`
2. Verify CUDA version: `nvcc --version`
3. Check thermal throttling: `nvidia-smi -q -d TEMPERATURE`
4. Monitor power limits: `nvidia-smi -q -d POWER`

### System Monitoring Commands
```bash
# GPU monitoring
watch -n 1 nvidia-smi

# System resources
htop

# Disk I/O
iotop

# Network usage
nethogs
```

## Recommended Builds

### Budget Build (~$1,500)
- **GPU**: RTX 3060 12GB ($400)
- **CPU**: AMD Ryzen 5 5600 ($150)
- **RAM**: 32GB DDR4 ($100)
- **Storage**: 1TB NVMe SSD ($80)
- **PSU**: 650W 80+ Gold ($80)
- **Case + Mobo**: ($200)
- **Capabilities**: All models except MVDream

### Performance Build (~$3,000)
- **GPU**: RTX 4070 Ti 12GB ($800)
- **CPU**: AMD Ryzen 7 7700X ($300)
- **RAM**: 64GB DDR5 ($300)
- **Storage**: 2TB NVMe Gen4 ($150)
- **PSU**: 850W 80+ Platinum ($150)
- **Case + Mobo**: ($400)
- **Capabilities**: All models with good performance

### Professional Build (~$6,000)
- **GPU**: RTX 4090 24GB ($1,800)
- **CPU**: AMD Ryzen 9 7950X ($550)
- **RAM**: 128GB DDR5 ($800)
- **Storage**: 4TB NVMe Gen5 ($400)
- **PSU**: 1000W 80+ Titanium ($300)
- **Case + Mobo**: ($600)
- **Capabilities**: All models at maximum performance

## Conclusion

Hardware requirements vary significantly between models:
- **Entry-level** (4-6GB VRAM): TripoSR only
- **Mid-range** (8-12GB VRAM): Most models except MVDream
- **High-end** (16-24GB VRAM): All models with optimal performance

Choose hardware based on:
1. Models you plan to use most frequently
2. Required generation speed
3. Quality expectations
4. Budget constraints
5. Future scalability needs