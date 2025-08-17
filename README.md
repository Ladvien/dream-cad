# 🎨 DreamCAD - Multi-Model 3D Generation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform text prompts into 3D models using state-of-the-art AI models. DreamCAD integrates 5 powerful 3D generation models in a unified, easy-to-use system.

## Overview

DreamCAD is a comprehensive 3D generation system that brings together multiple state-of-the-art AI models for creating 3D content from text prompts and images. Whether you need quick prototypes, game-ready assets, or high-quality production models, DreamCAD provides the right tool for the job.

## ✨ Features

- **5 Integrated Models**: TripoSR, Stable-Fast-3D, TRELLIS, Hunyuan3D, MVDream
- **Multiple Output Formats**: OBJ, PLY, STL, GLB, NeRF
- **Production Ready**: Queue management, monitoring, and analytics
- **Beautiful CLI**: Rich terminal interface with animations
- **Hardware Aware**: Automatic model selection based on available resources

## 🚀 Quick Start

## Installation

```bash
# Clone the repository
git clone https://github.com/Ladvien/dream-cad.git
cd dream-cad

# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate environment
poetry shell
```

## Usage

### Generate Your First 3D Model

```bash
# Launch Web UI (Recommended)
python scripts/launch_ui.py
# Open browser to http://localhost:7860

# Interactive CLI
python -m dream_cad.cli.main

# Direct generation
python -m dream_cad.examples.quick_generation
```

## 🎮 Available Models

### Supported Models

| Model | Speed | Quality | VRAM | Best For |
|-------|-------|---------|------|----------|
| ⚡ TripoSR | 0.5s | ★★★☆☆ | 4-6GB | Quick prototypes |
| 🎮 Stable-Fast-3D | 3s | ★★★★☆ | 6-8GB | Game assets with PBR |
| 💎 TRELLIS | 30s | ★★★★★ | 16-24GB | High quality |
| 🏭 Hunyuan3D | 5s | ★★★★★ | 12-16GB | Production assets |
| 👁️ MVDream | 60s | ★★★★☆ | 8-12GB | Multi-view consistency |

## 📚 Documentation

- [API Reference](docs/api_reference.md)
- [Model Comparison](docs/model_comparison.md)
- [Hardware Requirements](docs/hardware_requirements.md)
- [Configuration Guide](docs/configuration_examples.md)
- [Troubleshooting](docs/troubleshooting_models.md)

## 🛠️ CLI Tools

### DreamCAD CLI
Beautiful command-line interface with Rich styling:

```bash
# List available models
python cli/dream_cli.py models

# Show gallery of generated models
python cli/dream_cli.py gallery

# Interactive demo
python cli/dream_cli.py demo
```

### City Builder
Generate low-poly buildings for games:

```bash
# Generate a single building
python cli/dreamcad generate

# Batch generate multiple buildings
python cli/dreamcad batch

# Preview generated buildings
python cli/dreamcad preview
```

## 📁 Project Structure

```
dream-cad/
├── dream_cad/          # Core package
│   ├── models/         # Model implementations
│   ├── monitoring/     # System monitoring
│   ├── queue/          # Job queue management
│   └── benchmark/      # Performance testing
├── cli/                # Command-line tools
├── examples/           # Example scripts
├── docs/              # Documentation
└── configs/           # Configuration files
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run specific test
poetry run pytest tests/test_triposr.py

# Run with coverage
poetry run pytest --cov=dream_cad
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) by VAST AI Research
- [Stable-Fast-3D](https://github.com/Stability-AI/stable-fast-3d) by Stability AI
- [TRELLIS](https://github.com/Microsoft/TRELLIS) by Microsoft
- [Hunyuan3D](https://github.com/Tencent/Hunyuan3D) by Tencent
- [MVDream](https://github.com/bytedance/MVDream) by ByteDance

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Ladvien/dream-cad/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ladvien/dream-cad/discussions)

---

Made with ❤️ by the DreamCAD team