# System Specifications - MVDream Setup

## Hardware Specifications

### GPU
- **Model**: NVIDIA GeForce RTX 3090
- **VRAM**: 24GB (24576 MiB)
- **Driver Version**: 575.64.03 (exceeds requirement of 470.x)
- **Compute Capability**: 8.6 (supported)

### System Memory
- **Total RAM**: 31Gi (32GB requirement met)
- **Available RAM**: ~24Gi free

### Storage
- **Root Partition**: 195G total
- **Available Space**: 3.1G on root partition
- **Warning**: Limited disk space available - may need to use external storage or clean up existing files

### Operating System
- **Platform**: Linux (Manjaro)
- **Kernel**: 6.15.7-1-MANJARO
- **System Update Status**: Up to date

### Power Supply
- **Note**: Power supply wattage needs to be verified physically (cannot be checked via software)
- **Requirement**: ≥750W for RTX 3090

## Project Directory Structure
```
~/mvdream/
├── benchmarks/   # Performance benchmark results
├── docs/         # Documentation
├── logs/         # Application logs
├── outputs/      # Generated outputs
├── scripts/      # Utility scripts
└── tests/        # Test files and results
```

## System Verification Summary
✅ Manjaro system is up to date
✅ NVIDIA RTX 3090 detected with 24GB VRAM
✅ NVIDIA driver version 575.64.03 (exceeds 470.x requirement)
✅ System has 32GB RAM
⚠️  Disk space is limited (only 3.1GB free) - consider cleanup or external storage
❓ Power supply wattage needs physical verification
✅ Project directory created at ~/mvdream with proper permissions

## Recommendations
1. **Critical**: Free up disk space or use external storage (minimum 50GB required)
2. Verify power supply meets 750W requirement
3. Consider mounting additional storage for model downloads and outputs

## Verification Date
- **Date**: 2025-08-15
- **Verified By**: System automated checks