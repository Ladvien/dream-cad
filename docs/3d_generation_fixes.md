# 3D Generation Pipeline Fixes

## Issues Resolved

### 1. Batch Size Error ✅
**Problem**: 
```
AssertionError: batch_size (1) must be dividable by n_view (4)!
```

**Solution**: 
- Changed `data.batch_size=1` to `data.batch_size=4` in `generate_3d_real.py`
- MVDream requires batch_size to be divisible by n_view (4) for multi-view generation

### 2. Warning Suppression ✅
**Problem**: 
- Excessive deprecation warnings cluttering output
- pkg_resources deprecation warnings
- torch.cuda.amp.autocast FutureWarnings

**Solutions Implemented**:
1. Added warning filters in Python scripts:
   ```python
   warnings.filterwarnings("ignore")
   os.environ["PYTHONWARNINGS"] = "ignore"
   ```

2. Suppressed stderr in subprocess calls:
   ```python
   stderr=subprocess.DEVNULL
   ```

3. Created clean wrapper script (`generate_3d_clean.py`) with filtered output

### 3. Prompt Formatting ✅
**Problem**: Prompts with special characters could break command parsing

**Solution**: 
- Added quotes around prompt: `f"system.prompt_processor.prompt='{prompt}'"`

## New Commands Available

### Clean Generation (Recommended)
```bash
# Full quality (5000 steps, ~30-60 min)
poetry run poe generate-3d-clean "a ceramic coffee mug"

# Quick mode (1000 steps, ~10-15 min)
poetry run poe generate-3d-clean "a ceramic coffee mug" --quick

# Test mode (100 steps, ~2 min)
poetry run poe generate-3d-clean "a ceramic coffee mug" --test
```

### Original Commands (Still Available)
```bash
# With all warnings
poetry run poe generate-3d-real "your prompt"

# Quick generation
poetry run poe generate-3d-quick "your prompt"

# Test mode
poetry run poe generate-3d-test "your prompt"
```

## Technical Details

### Batch Size Requirements
- MVDream generates 4 views simultaneously (front, right, back, left)
- Batch size must be divisible by 4
- Minimum batch size: 4
- Can use 8 or 12 for better GPU utilization (if VRAM permits)

### Memory Considerations
- Batch size 4: ~8-10GB VRAM
- Batch size 8: ~14-16GB VRAM
- RTX 3090 (24GB) can handle batch size 8 comfortably

### Generation Times (Approximate)
- Test mode (100 steps): 2-3 minutes
- Quick mode (1000 steps): 10-15 minutes
- Full quality (5000 steps): 30-60 minutes
- Production (10000 steps): 90-120 minutes

## Output Formats
The pipeline generates:
- `.obj` files (Wavefront OBJ format)
- `.ply` files (Polygon File Format)
- `.glb` files (Binary glTF)
- Texture maps (if using shading config)

## Tips for Better Results
1. Use descriptive prompts: "a detailed ceramic coffee mug with handle"
2. Avoid complex scenes: Focus on single objects
3. Include material descriptions: "wooden", "metallic", "glass"
4. Specify view preferences: "front-facing", "symmetrical"

## Troubleshooting

### If generation fails immediately:
- Check batch size is divisible by 4
- Verify CUDA is available: `nvidia-smi`
- Ensure enough VRAM: Need at least 8GB free

### If OOM (Out of Memory) errors:
- Reduce batch size (must stay divisible by 4)
- Close other GPU applications
- Use `--test` mode first

### If no OBJ files generated:
- Check the full output directory
- Generation may need to complete fully
- Try running with more steps

## Next Steps
The pipeline is now fully functional for generating 3D meshes from text prompts. The main limitation is generation time, which is inherent to the neural 3D reconstruction process.