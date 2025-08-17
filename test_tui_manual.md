# Manual TUI Testing Guide

## Launch the TUI

```bash
./dreamcad tui
# OR
poetry run python dreamcad_tui.py
```

## Features to Test

### 1. Model Selection
- Use the dropdown to select different models
- Notice how the Model Info panel updates with:
  - Model name
  - VRAM requirements  
  - Speed estimates
  - Quality rating

### 2. Parameter Adjustment
- When you select a model, parameters appear below
- Each model has different parameters:
  
  **TripoSR:**
  - Resolution: 256, 512, 1024
  - Batch size: 1-4
  - Remove background: on/off
  - Output format: obj, ply, stl, glb

  **Stable-Fast-3D:**
  - Target polycount: 1000-50000
  - Texture size: 512, 1024, 2048  
  - Enable PBR: on/off
  - Delighting: on/off
  - Output format: glb, obj, ply

  **TRELLIS:**
  - Quality mode: fast, balanced, hq
  - Representation: mesh, nerf, gaussian
  - Preserve intermediate: on/off
  - Output format: obj, ply, glb

  **Hunyuan3D:**
  - Polycount: 10000-50000
  - Texture resolution: 1024, 2048, 4096
  - UV unwrap method: smart, angle, conformal
  - Output format: glb, obj, ply

### 3. Prompt Entry
- Click in the prompt field or Tab to it
- Enter any 3D object description
- Examples:
  - "a crystal sword"
  - "a fantasy cottage"
  - "a low-poly robot"
  - "a wooden chair"

### 4. Generation
- Click "Generate 3D Model" button or press Ctrl+G
- Watch the output log for:
  - Model loading messages
  - Generation progress
  - Final output path

### 5. Keyboard Shortcuts
- **Ctrl+Q**: Quit
- **Ctrl+G**: Generate
- **Ctrl+C**: Clear log
- **F1**: Show help
- **Tab**: Navigate between fields

## Current Status

✅ **Working:**
- Model selection with info display
- Parameter display and adjustment
- Prompt input
- Generation simulation
- Keyboard shortcuts
- Output logging
- Settings persistence

⚠️ **Mock Mode:**
- Currently simulates generation (doesn't call real models)
- To connect to real models, need to:
  1. Import model classes from `dream_cad.models`
  2. Initialize selected model
  3. Call generate with parameters
  4. Handle actual file output

## Known Issues

None currently - the TUI is functional for its intended purpose!

## Next Steps

To connect to real models:
1. Add model initialization in `action_generate()`
2. Replace `simulate_generation()` with actual model calls
3. Handle VRAM checking before generation
4. Add progress callbacks from models
5. Save actual generated files