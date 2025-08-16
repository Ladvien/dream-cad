# Repository Structure

## Overview
This repository uses git submodules to manage external dependencies cleanly. All external repositories are located in the `extern/` directory as submodules.

## Directory Structure

```
dream-cad/
├── docs/                       # Documentation
├── extern/                     # External repositories (git submodules)
│   ├── MVDream/               # 2D multi-view generation
│   └── MVDream-threestudio/   # 3D generation pipeline
├── models/                    # Downloaded model weights (gitignored)
├── outputs/                   # Generated outputs (gitignored)
├── scripts/                   # Our custom scripts
│   ├── generate_3d.py        # 3D generation wrapper
│   ├── generate_3d_real.py   # Real 3D using threestudio
│   ├── generate_mvdream.py   # 2D multi-view generation
│   └── ...
├── tests/                     # Test suites
├── pyproject.toml            # Poetry configuration
└── .gitmodules               # Submodule definitions
```

## Submodules

### Why Submodules?
- **Clean separation**: External code is clearly separated from our code
- **Version control**: Each submodule tracks a specific commit
- **Easy updates**: Can update to latest versions with git commands
- **No duplication**: Avoids copying large repositories

### Current Submodules

1. **MVDream** (`extern/MVDream/`)
   - Multi-view diffusion model
   - Generates 4 consistent views from text
   - Source: https://github.com/bytedance/MVDream

2. **MVDream-threestudio** (`extern/MVDream-threestudio/`)
   - 3D generation using Score Distillation Sampling
   - Converts text to 3D meshes
   - Source: https://github.com/bytedance/MVDream-threestudio

## Working with Submodules

### First Time Setup
```bash
# Clone the repository with submodules
git clone --recursive https://github.com/yourusername/dream-cad.git

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

### Update Submodules
```bash
# Update all submodules to latest
git submodule update --remote --merge

# Update specific submodule
git submodule update --remote extern/MVDream
```

### Check Status
```bash
# See submodule status
git submodule status

# See what changed in submodules
git diff --submodule
```

## Key Paths

### Scripts Using Submodules
- `scripts/generate_3d_real.py` → Uses `extern/MVDream-threestudio/`
- `scripts/generate_mvdream.py` → Uses `extern/MVDream/`
- `run_3d_generation.sh` → Runs from `extern/MVDream-threestudio/`

### Configuration Files
- `extern/MVDream-threestudio/configs/` - 3D generation configs
- `extern/MVDream/mvdream/configs/` - 2D generation configs

### Output Locations
- `outputs/` - Our script outputs
- `extern/MVDream-threestudio/outputs/` - Threestudio training outputs

## Best Practices

1. **Don't modify submodule code directly** - Make changes in upstream repos
2. **Use Poetry environment** - Run with `poetry run python`
3. **Check submodule status** - Before committing, ensure submodules are at correct commits
4. **Document dependencies** - Keep this file updated when adding submodules

## Troubleshooting

### Submodule not found
```bash
git submodule update --init --recursive
```

### Submodule at wrong commit
```bash
git submodule update
```

### Need to change submodule URL
```bash
git submodule set-url extern/MVDream https://new-url.git
```