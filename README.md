# Dream-CAD

MVDream text-to-3D generation system on Manjaro Linux with NVIDIA RTX 3090.

## System Requirements

- **OS**: Manjaro Linux
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **CUDA**: 12.9
- **Python**: 3.13
- **RAM**: 32GB
- **Storage**: 50GB+ free space on `/mnt/datadrive_m2`

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Ladvien/dream-cad.git /mnt/datadrive_m2/dream-cad
cd /mnt/datadrive_m2/dream-cad
```

2. **Install Poetry**:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. **Install dependencies**:
```bash
poetry install
```

4. **Install PyTorch with CUDA support**:
```bash
poetry run pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

5. **Install MVDream**:
```bash
poetry run pip install -e extern/MVDream/
```

## Project Structure

- `mvdream/` - Main project package
- `extern/MVDream/` - ByteDance MVDream repository
- `MVDream-threestudio/` - 3D generation extension
- `tests/` - Test suite
- `docs/` - Documentation
- `.venv/` - Virtual environment

## Environment Variables

Add to your `~/.zshrc`:
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

### Run Tests
```bash
poetry run poe test        # Run all tests
poetry run poe test-gpu    # Run GPU tests only
```

### Linting and Formatting
```bash
poetry run poe lint        # Run linters
poetry run poe format      # Format code
```

## Development

This project uses:
- **Poetry** for dependency management
- **Ruff** for linting and formatting
- **Bandit** for security scanning
- **Pytest** for testing
- **Sphinx** for documentation

## Documentation

- [System Specifications](docs/system-specs.md)
- [CUDA Setup Guide](docs/cuda-setup.md)
- [Project Structure](docs/project-structure.md)
- [Troubleshooting Guide](docs/troubleshooting.md)
- [Model Management](docs/models.md)

## FAQ

### Q: How do I check if my system is properly configured?
**A:** Run the diagnostic tool:
```bash
poetry run poe diagnose
```
This will check CUDA, dependencies, model files, and system configuration.

### Q: What if I get a PyTorch NCCL library error?
**A:** This is a known issue with PyTorch 2.7.1. The generation scripts handle this gracefully with fallback mechanisms. For a permanent fix, see the [Troubleshooting Guide](docs/troubleshooting.md#1-pytorch-nccl-library-error).

### Q: How much disk space do I need?
**A:** Minimum 50GB free space on `/mnt/datadrive_m2`:
- Models: ~10GB
- Outputs: ~20GB
- Cache: ~10GB
- Working space: ~10GB

### Q: Can I run this without an RTX 3090?
**A:** MVDream requires a CUDA-capable GPU with at least 16GB VRAM. RTX 3090 (24GB) is recommended for optimal performance. Other suitable GPUs include RTX 4090, A5000, or A6000.

### Q: How long does 3D generation take?
**A:** On an RTX 3090:
- 2D multi-view: 1-2 minutes
- 3D mesh generation: 90-150 minutes
- Quality depends on settings in `configs/mvdream-sd21.yaml`

### Q: What if generation runs out of memory?
**A:** Try these solutions:
1. Reduce batch size in config
2. Enable memory-efficient attention
3. Lower resolution
4. See [Memory Optimization Tips](docs/troubleshooting.md#memory-optimization-tips)

### Q: How do I download the pre-trained models?
**A:** Use the download script:
```bash
poetry run poe download-models
```
This downloads the sd-v2.1-base-4view model (~5GB) to `/mnt/datadrive_m2/dream-cad/models/`.

### Q: Can I use custom prompts for generation?
**A:** Yes! For 3D generation:
```bash
poetry run poe generate-3d "your custom prompt here"
# Or use the web interface:
poetry run poe generate-3d-web
```

### Q: What Python version is required?
**A:** Python 3.10+ is required. The project is tested with Python 3.13.5.

### Q: How do I contribute or report issues?
**A:** 
1. Run diagnostics: `poetry run poe diagnose`
2. Check [Troubleshooting Guide](docs/troubleshooting.md)
3. Create an issue with diagnostic output and error messages

## License

MIT License - See [LICENSE](LICENSE) file for details.