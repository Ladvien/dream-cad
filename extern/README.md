# External Dependencies (Git Submodules)

This directory contains external repositories managed as git submodules.

## Submodules

### MVDream
- **Path**: `extern/MVDream/`
- **Repository**: https://github.com/bytedance/MVDream
- **Purpose**: Multi-view diffusion model for 2D image generation
- **Description**: Generates 4 consistent views of an object from text prompts

### MVDream-threestudio
- **Path**: `extern/MVDream-threestudio/`
- **Repository**: https://github.com/bytedance/MVDream-threestudio
- **Purpose**: 3D generation pipeline using MVDream
- **Description**: Converts text prompts to 3D meshes using Score Distillation Sampling (SDS)

## Managing Submodules

### Initial Setup
After cloning the main repository, initialize submodules:
```bash
git submodule update --init --recursive
```

### Update Submodules
To update all submodules to their latest commits:
```bash
git submodule update --remote --merge
```

### Add New Submodule
```bash
git submodule add https://github.com/user/repo.git extern/repo-name
```

## Directory Structure
```
extern/
├── README.md (this file)
├── MVDream/                    # 2D multi-view generation
│   ├── mvdream/               # Python package
│   ├── scripts/               # Example scripts
│   └── requirements.txt       # Dependencies
└── MVDream-threestudio/       # 3D generation pipeline
    ├── configs/               # Configuration files
    ├── threestudio/           # Main package
    └── launch.py             # Training script
```

## Important Notes

1. **Don't modify submodule contents directly** - Changes should be made in the upstream repositories
2. **Use Poetry environment** - All Python commands should use `poetry run` to ensure dependencies are available
3. **CUDA dependencies** - MVDream-threestudio requires compiled CUDA extensions (tinycudann, nerfacc, nvdiffrast)
4. **Model weights** - Large model files are stored in `/mnt/datadrive_m2/models/` to save space