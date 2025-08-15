# Dream-CAD MVDream Project Structure

## Overview
This document describes the complete directory structure of the Dream-CAD project with integrated MVDream support.

## Directory Tree

```
/mnt/datadrive_m2/dream-cad/
├── .venv/                          # Poetry virtual environment (Python 3.13)
├── docs/                           # Documentation
│   ├── system-specs.md            # System specifications
│   ├── cuda-setup.md              # CUDA configuration guide
│   ├── troubleshooting-diskspace.md # Disk space solutions
│   └── project-structure.md      # This file
├── tests/                          # Test suite
│   ├── test_story1_requirements.py # System requirements tests
│   ├── test_story2_cuda_setup.py  # CUDA installation tests
│   ├── test_story3_python_env.py  # Python environment tests
│   ├── test_story4_pytorch.py     # PyTorch installation tests
│   ├── test_story5_mvdream.py     # MVDream integration tests
│   ├── test_cuda.py               # GPU validation tests
│   ├── test_pytorch_cuda_compat.py # PyTorch-CUDA compatibility
│   ├── cuda_device_query.cu      # CUDA device query sample
│   └── cuda_bandwidth_test.cu    # CUDA bandwidth test
├── scripts/                        # Utility scripts
│   └── verify_requirements.py     # System verification script
├── extern/                         # External repositories
│   └── MVDream/                   # ByteDance MVDream (cloned)
│       ├── mvdream/               # MVDream Python package
│       │   ├── configs/           # Model configurations
│       │   ├── models/            # Model implementations
│       │   └── utils/             # Utility functions
│       ├── scripts/               # MVDream scripts
│       ├── setup.py               # MVDream setup
│       └── requirements.txt       # MVDream requirements
├── MVDream-threestudio/            # MVDream-threestudio (cloned)
│   ├── configs/                   # 3D generation configs
│   ├── systems/                   # 3D rendering systems
│   ├── models/                    # 3D model implementations
│   └── requirements.txt           # Threestudio requirements
├── mvdream/                        # Main project package
│   └── __init__.py               # Package initialization
├── benchmarks/                     # Performance benchmarks
├── logs/                          # Application logs
├── outputs/                       # Generated outputs
├── pyproject.toml                 # Poetry configuration
├── poetry.lock                    # Locked dependencies
├── README.md                      # Project documentation
├── CLAUDE.md                      # Project memory/learnings
├── STORIES.md                     # Agile stories tracking
└── .gitignore                     # Git ignore patterns
```

## Key Directories

### `/mnt/datadrive_m2/dream-cad/`
Main project root, stored on data drive to avoid disk space issues on the system drive.

### `.venv/`
Poetry-managed virtual environment containing all dependencies. Located on data drive to save space.

### `extern/MVDream/`
Cloned MVDream repository from ByteDance. Installed as editable package via pip.

### `MVDream-threestudio/`
Cloned MVDream-threestudio repository for 3D generation capabilities.

### `mvdream/`
Main project package for custom implementations and configurations.

## Storage Locations

All large files are stored on `/mnt/datadrive_m2/`:
- Virtual environment: `/mnt/datadrive_m2/dream-cad/.venv`
- Pip cache: `/mnt/datadrive_m2/.pip-cache`
- PyTorch models: `/mnt/datadrive_m2/.torch`
- Hugging Face cache: `/mnt/datadrive_m2/.huggingface`
- Poetry cache: `/mnt/datadrive_m2/.poetry-cache`

## Installed Packages

### Core ML Frameworks
- PyTorch 2.7.1+cu118
- pytorch-lightning 2.0.9
- transformers 4.27.1
- diffusers 0.34.0

### MVDream Components
- mvdream (from extern/MVDream) - installed as editable
- open-clip-torch 2.7.0
- omegaconf 2.3.0

### Image/Video Processing
- opencv-python
- imageio + imageio-ffmpeg
- matplotlib
- PIL (Pillow)

### 3D Processing
- trimesh
- PyMCubes
- pysdf
- networkx

### UI and Monitoring
- gradio
- tensorboard
- wandb

## Configuration Files

### `pyproject.toml`
Poetry project configuration with all dependencies specified. PyTorch installed separately via pip due to Python 3.13 compatibility.

### `poetry.lock`
Locked versions of all Poetry-managed dependencies.

### `.gitignore`
Configured to ignore:
- Python bytecode and caches
- Virtual environments
- Model files (*.pth, *.ckpt, *.safetensors)
- CUDA build artifacts
- IDE configurations

## Environment Variables

Required environment variables (set in ~/.zshrc):
```bash
export PIP_CACHE_DIR=/mnt/datadrive_m2/.pip-cache
export TORCH_HOME=/mnt/datadrive_m2/.torch
export HF_HOME=/mnt/datadrive_m2/.huggingface
export CUDA_HOME=/opt/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Usage

### Activate Environment
```bash
cd /mnt/datadrive_m2/dream-cad
poetry shell
```

### Import MVDream
```python
from mvdream import *
from mvdream.models import MVDreamModel
```

### Run Tests
```bash
poetry run poe test
poetry run poe test-gpu
```