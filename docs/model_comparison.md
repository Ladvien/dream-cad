# Model Comparison Guide

## Overview

This guide provides a detailed comparison of all supported 3D generation models in Dream-CAD, helping you choose the right model for your specific use case.

## Quick Comparison Table

| Feature | MVDream | TripoSR | Stable-Fast-3D | TRELLIS | Hunyuan3D-Mini |
|---------|---------|---------|----------------|---------|----------------|
| **Generation Speed** | 90-150 min | 0.5-2 sec | 3-5 sec | 30-60 sec | 5-10 sec |
| **Quality Score** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Min VRAM** | 16GB | 4GB | 6GB | 8GB | 12GB |
| **Optimal VRAM** | 20GB | 6GB | 8GB | 16GB | 16GB |
| **Output Formats** | OBJ, PLY | OBJ, PLY, GLB | GLB, OBJ | Multiple | GLB, OBJ |
| **PBR Materials** | ❌ | ❌ | ✅ | ❌ | ✅ |
| **UV Unwrapping** | Basic | Basic | ✅ | ❌ | ✅ Professional |
| **Multi-View Input** | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Best For** | High quality | Prototyping | Game assets | Research | Production |

## Detailed Model Profiles

### MVDream
**Architecture:** Multi-view diffusion model  
**Developer:** ByteDance

#### Strengths
- Highest quality output for complex geometries
- Excellent multi-view consistency
- Strong understanding of text prompts
- Good at generating detailed objects

#### Weaknesses
- Very slow generation (90-150 minutes)
- High VRAM requirement (16-20GB)
- No built-in PBR material support
- Limited UV unwrapping capabilities

#### Best Use Cases
- Final production renders
- Complex architectural models
- Detailed character models
- When quality is paramount over speed

#### Optimal Settings
```python
config = {
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "n_views": 4,
    "resolution": 256,
    "rescale_factor": 0.5  # For RTX 3090
}
```

### TripoSR
**Architecture:** Feed-forward transformer  
**Developer:** Stability AI & VAST

#### Strengths
- Extremely fast (0.5-2 seconds)
- Low VRAM requirement (4-6GB)
- Good for simple to medium complexity objects
- Excellent for rapid iteration

#### Weaknesses
- Lower quality for complex geometries
- No PBR material support
- Limited detail preservation
- Basic UV unwrapping

#### Best Use Cases
- Rapid prototyping
- Real-time applications
- Mobile/web deployment
- Concept exploration
- Batch processing large datasets

#### Optimal Settings
```python
config = {
    "resolution": 512,
    "batch_size": 1,
    "precision": "fp16",
    "remove_background": True
}
```

### Stable-Fast-3D
**Architecture:** Optimized diffusion model  
**Developer:** Stability AI

#### Strengths
- Full PBR material generation
- Optimized topology for games
- Good UV unwrapping
- Reasonable speed (3-5 seconds)
- Polycount control

#### Weaknesses
- Medium quality for organic shapes
- Limited to single-view input
- Requires delighting post-processing

#### Best Use Cases
- Game asset creation
- Unity/Unreal Engine projects
- Mobile game development
- VR/AR applications
- Assets requiring PBR materials

#### Optimal Settings
```python
config = {
    "target_polycount": 20000,
    "generate_pbr": True,
    "delight": True,
    "uv_padding": 2,
    "texture_resolution": 1024
}
```

### TRELLIS
**Architecture:** Multi-representation model  
**Developer:** Microsoft

#### Strengths
- Multiple 3D representations (NeRF, Gaussian, Mesh)
- High quality outputs
- Multi-view consistency
- Novel representation support
- Good for research

#### Weaknesses
- Moderate speed (30-60 seconds)
- Higher VRAM usage for quality modes
- Complex pipeline
- Limited game engine compatibility

#### Best Use Cases
- Research projects
- Novel view synthesis
- NeRF/Gaussian splatting workflows
- Experimental 3D reconstruction
- When multiple representations needed

#### Optimal Settings
```python
config = {
    "quality_mode": "balanced",  # fast, balanced, hq
    "representation": "mesh",
    "num_views": 4,
    "slat_resolution": 128
}
```

### Hunyuan3D-Mini
**Architecture:** Optimized production model  
**Developer:** Tencent

#### Strengths
- Professional UV unwrapping
- Complete PBR material generation
- Production-ready outputs
- Good polycount control
- Multi-view fusion support

#### Weaknesses
- Commercial license restrictions
- Moderate VRAM requirement (12-16GB)
- Slower than TripoSR/Stable-Fast-3D

#### Best Use Cases
- Production asset creation
- Professional game development
- Marketing renders
- High-quality product visualization
- Assets requiring perfect UVs

#### Optimal Settings
```python
config = {
    "target_polycount": 30000,
    "uv_method": "smart_projection",
    "texture_size": 2048,
    "simplification_ratio": 0.95,
    "generate_all_maps": True
}
```

## Model Selection Decision Tree

```
Start: What is your primary need?
│
├─> Speed Critical (< 5 seconds)
│   ├─> Simple objects → TripoSR
│   └─> Game assets → Stable-Fast-3D
│
├─> Quality Critical
│   ├─> Highest quality → MVDream
│   ├─> Production ready → Hunyuan3D-Mini
│   └─> Multiple representations → TRELLIS
│
├─> Memory Constrained (< 8GB VRAM)
│   ├─> Fastest → TripoSR
│   └─> With PBR → Stable-Fast-3D
│
└─> Balanced Requirements
    ├─> Game development → Stable-Fast-3D
    ├─> Research → TRELLIS
    └─> Production → Hunyuan3D-Mini
```

## Performance Benchmarks

### Speed Comparison (RTX 3090)
```
TripoSR:         ████ 0.5-2s
Stable-Fast-3D:  ████████ 3-5s
Hunyuan3D-Mini:  ████████████████ 5-10s
TRELLIS:         ████████████████████████████████ 30-60s
MVDream:         ████████████████████████████████████████ 90-150min
```

### Quality Comparison (100-point scale)
```
MVDream:         ████████████████████ 95
Hunyuan3D-Mini:  ██████████████████ 90
TRELLIS:         █████████████████ 85
Stable-Fast-3D:  ███████████████ 75
TripoSR:         █████████████ 65
```

### VRAM Usage (GB)
```
TripoSR:         ████ 4-6GB
Stable-Fast-3D:  ██████ 6-8GB
TRELLIS:         ████████ 8-16GB
Hunyuan3D-Mini:  ████████████ 12-16GB
MVDream:         ████████████████ 16-20GB
```

## Workflow Recommendations

### For Game Development
1. **Prototype Phase**: Use TripoSR for rapid iteration
2. **Asset Creation**: Switch to Stable-Fast-3D for PBR materials
3. **Hero Assets**: Use Hunyuan3D-Mini for main characters/props
4. **Final Polish**: Consider MVDream for cinematic quality

### For Research
1. **Experimentation**: TRELLIS for multiple representations
2. **Dataset Creation**: TripoSR for batch processing
3. **Quality Analysis**: MVDream as quality baseline
4. **Novel Views**: TRELLIS with NeRF/Gaussian outputs

### For Production
1. **Concept Art**: TripoSR for quick explorations
2. **Asset Pipeline**: Hunyuan3D-Mini for UV-perfect models
3. **Material Creation**: Stable-Fast-3D for PBR textures
4. **Final Renders**: MVDream for marketing materials

## Multi-Model Pipeline

For optimal results, consider using multiple models in sequence:

```python
# Example: Concept to Production Pipeline
pipeline = [
    ("triposr", {"resolution": 256}),      # Quick preview
    ("stable-fast-3d", {"target_polycount": 10000}),  # Refine with PBR
    ("hunyuan3d-mini", {"uv_method": "smart_projection"})  # Final production
]
```

## Cost Analysis

### GPU Hours per 100 Models (RTX 3090)
- **TripoSR**: 0.05 hours ($0.025 at $0.50/hour)
- **Stable-Fast-3D**: 0.14 hours ($0.07)
- **Hunyuan3D-Mini**: 0.25 hours ($0.125)
- **TRELLIS**: 1.4 hours ($0.70)
- **MVDream**: 200 hours ($100)

### Recommended Batch Sizes
- **TripoSR**: 10-50 models per batch
- **Stable-Fast-3D**: 5-20 models per batch
- **Hunyuan3D-Mini**: 3-10 models per batch
- **TRELLIS**: 1-5 models per batch
- **MVDream**: Single model processing

## Troubleshooting Model Selection

### Common Issues and Solutions

#### "Out of Memory" Errors
- **Solution**: Switch to lower VRAM model
- TripoSR (4GB) → Stable-Fast-3D (6GB) → TRELLIS (8GB) → Hunyuan3D (12GB) → MVDream (16GB)

#### Poor Quality Output
- **Solution**: Move up quality hierarchy
- TripoSR → Stable-Fast-3D → TRELLIS → Hunyuan3D-Mini → MVDream

#### Slow Generation
- **Solution**: Use faster models or optimize settings
- Reduce inference steps, lower resolution, enable FP16

#### Missing PBR Materials
- **Solution**: Use Stable-Fast-3D or Hunyuan3D-Mini
- Post-process with external tools if needed

## Future Model Additions

Planned models for integration:
- **Wonder3D**: Single image to 3D with consistency
- **DreamGaussian**: Fast Gaussian splatting generation
- **One-2-3-45++**: Enhanced single-view reconstruction
- **Magic123**: High-quality zero-shot generation

## Conclusion

Choose your model based on:
1. **Time constraints** - How fast do you need results?
2. **Quality requirements** - What level of detail is needed?
3. **Hardware limitations** - How much VRAM is available?
4. **Output requirements** - Do you need PBR materials? UV maps?
5. **Budget constraints** - How much GPU time can you afford?

For most users, we recommend starting with **TripoSR** for exploration, then moving to **Stable-Fast-3D** or **Hunyuan3D-Mini** for production work.