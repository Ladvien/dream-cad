# CUDA Setup Documentation

## Overview
This document describes the CUDA configuration for the MVDream project on Manjaro Linux with NVIDIA RTX 3090.

## System Configuration

### CUDA Version
- **Installed Version**: CUDA 12.9
- **Location**: `/opt/cuda`
- **Compiler**: nvcc 12.9.86

### GPU Information
- **Device**: NVIDIA GeForce RTX 3090
- **Compute Capability**: 8.6
- **Total Memory**: 24,124 MB (24GB)
- **Memory Bus Width**: 384-bit
- **L2 Cache**: 6MB

## Environment Variables

The following environment variables have been configured in `~/.zshrc`:

```bash
export CUDA_HOME=/opt/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

To apply changes:
```bash
source ~/.zshrc
```

## Verification Tests

### 1. Device Query Test
Location: `~/mvdream/tests/cuda_device_query.cu`

Compile and run:
```bash
nvcc -o cuda_device_query cuda_device_query.cu
./cuda_device_query
```

This test verifies:
- CUDA device detection
- GPU compute capability
- Memory configuration
- Thread and block limits

### 2. Bandwidth Test
Location: `~/mvdream/tests/cuda_bandwidth_test.cu`

Compile and run:
```bash
nvcc -o cuda_bandwidth_test cuda_bandwidth_test.cu
./cuda_bandwidth_test
```

This test measures:
- Device-to-device memory bandwidth
- Host-to-device transfer speeds (pageable and pinned)
- Device-to-host transfer speeds (pageable and pinned)

## Test Results

### Device Query
- ✅ RTX 3090 detected successfully
- ✅ Compute capability 8.6 confirmed
- ✅ 24GB VRAM available

### Bandwidth Test
- Device-to-Device: ~392 GB/s
- Host-to-Device (Pinned): ~19 GB/s
- Device-to-Host (Pinned): ~26 GB/s

Note: Device-to-device bandwidth is lower than theoretical maximum (936 GB/s) but acceptable for MVDream operations.

## Compatibility Notes

### CUDA 12.9 vs CUDA 11.8
- MVDream originally requires CUDA 11.8
- System has CUDA 12.9 installed (newer version)
- CUDA 12.9 is backward compatible with CUDA 11.8 applications
- PyTorch with CUDA 11.8 support should work with CUDA 12.9 runtime

### Python Integration
When installing PyTorch, use:
```bash
poetry add torch torchvision --source pytorch-cuda118
```

This will install PyTorch compiled for CUDA 11.8, which is compatible with our CUDA 12.9 installation.

## Troubleshooting

### Common Issues

1. **CUDA not found**
   - Verify environment variables are set
   - Run `source ~/.zshrc` to reload configuration
   - Check `nvcc --version` output

2. **Low bandwidth warnings**
   - Expected behavior under system load
   - Ensure no other GPU-intensive applications running
   - Check GPU temperature and throttling

3. **Compilation warnings**
   - "deprecated-gpu-targets" warnings are normal for CUDA 12.9
   - Can be suppressed with `-Wno-deprecated-gpu-targets` flag

## Quick Verification Commands

```bash
# Check CUDA version
nvcc --version

# Check GPU status
nvidia-smi

# Verify environment variables
echo $CUDA_HOME
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda

# Test CUDA availability in Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Next Steps

1. Install PyTorch with CUDA support
2. Configure MVDream dependencies
3. Test GPU memory allocation for model loading