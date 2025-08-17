# Configuration Examples

This guide provides configuration examples for different use cases and scenarios across all supported models.

## Basic Configuration Structure

All models accept configuration through the `ModelFactory`:

```python
from dream_cad import ModelFactory

# Basic usage
model = ModelFactory.create_model(
    model_name="triposr",
    **config_options
)

# Or pass config separately
config = {
    "resolution": 512,
    "num_inference_steps": 30,
}
result = model.generate_from_text(prompt, **config)
```

## Quick Start Configurations

### Fastest Generation (TripoSR)
```python
config = {
    "model_name": "triposr",
    "resolution": 256,
    "batch_size": 1,
    "precision": "fp16",
    "remove_background": True,
}
```

### Best Quality (MVDream)
```python
config = {
    "model_name": "mvdream",
    "num_inference_steps": 100,
    "guidance_scale": 10.0,
    "n_views": 4,
    "resolution": 512,
    "rescale_factor": 0.7,
}
```

### Balanced Performance (Hunyuan3D-Mini)
```python
config = {
    "model_name": "hunyuan3d-mini",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "target_polycount": 30000,
    "texture_size": 1024,
    "uv_method": "smart_projection",
}
```

## Use Case Specific Configurations

### Game Asset Development

#### Mobile Game Asset (Low Poly)
```python
config = {
    "model_name": "stable-fast-3d",
    "target_polycount": 5000,
    "texture_resolution": 512,
    "generate_pbr": True,
    "delight": True,
    "simplification_ratio": 0.3,
    "output_format": "glb",
}
```

#### PC/Console Game Asset (Medium Quality)
```python
config = {
    "model_name": "stable-fast-3d",
    "target_polycount": 20000,
    "texture_resolution": 2048,
    "generate_pbr": True,
    "generate_normal_map": True,
    "generate_roughness_map": True,
    "generate_metallic_map": True,
    "delight": True,
    "uv_padding": 4,
    "output_format": "glb",
}
```

#### Hero Asset (High Quality)
```python
config = {
    "model_name": "hunyuan3d-mini",
    "target_polycount": 50000,
    "texture_size": 4096,
    "uv_method": "smart_projection",
    "uv_island_margin": 0.01,
    "generate_all_maps": True,
    "simplification_ratio": 0.95,
    "multi_view_fusion": True,
    "output_format": "glb",
}
```

### 3D Printing

#### FDM Printing
```python
config = {
    "model_name": "triposr",
    "resolution": 512,
    "output_format": "stl",
    "ensure_manifold": True,
    "minimum_thickness": 2.0,  # mm
    "scale_to_size": 100,  # mm
}
```

#### Resin Printing (High Detail)
```python
config = {
    "model_name": "mvdream",
    "num_inference_steps": 75,
    "resolution": 512,
    "output_format": "stl",
    "ensure_watertight": True,
    "target_polycount": 100000,
    "smooth_normals": True,
}
```

### Architectural Visualization

#### Quick Concept
```python
config = {
    "model_name": "triposr",
    "resolution": 512,
    "remove_background": True,
    "output_format": "obj",
}
```

#### Detailed Model
```python
config = {
    "model_name": "mvdream",
    "num_inference_steps": 100,
    "guidance_scale": 8.0,
    "n_views": 4,
    "resolution": 512,
    "output_format": "obj",
}
```

### Research and Development

#### Multiple Representations
```python
config = {
    "model_name": "trellis",
    "quality_mode": "balanced",
    "representations": ["mesh", "nerf", "gaussian_splatting"],
    "slat_resolution": 128,
    "num_views": 4,
}
```

#### Quick Iteration
```python
config = {
    "model_name": "triposr",
    "resolution": 256,
    "batch_size": 1,
    "precision": "fp16",
    "save_intermediate": False,
}
```

## Memory Optimization Configurations

### Minimum VRAM Usage (4GB GPU)
```python
config = {
    "model_name": "triposr",
    "resolution": 256,
    "batch_size": 1,
    "precision": "fp16",
    "cpu_offload": True,
    "gradient_checkpointing": True,
}
```

### 8GB GPU Configuration
```python
config = {
    "model_name": "stable-fast-3d",
    "target_polycount": 10000,
    "texture_resolution": 1024,
    "precision": "fp16",
    "cpu_offload": False,
    "sequential_processing": True,
}
```

### 12GB GPU Configuration
```python
config = {
    "model_name": "hunyuan3d-mini",
    "target_polycount": 30000,
    "texture_size": 2048,
    "precision": "fp16",
    "cpu_offload": False,
}
```

### 24GB GPU Configuration
```python
config = {
    "model_name": "mvdream",
    "num_inference_steps": 100,
    "guidance_scale": 10.0,
    "n_views": 4,
    "resolution": 512,
    "precision": "fp32",
    "cpu_offload": False,
}
```

## Batch Processing Configurations

### High Throughput
```python
from dream_cad.queue import JobQueue, BatchProcessor

# Queue configuration
queue_config = {
    "max_concurrent_jobs": 3,
    "enable_warmup": True,
    "enable_cooldown": False,
    "priority_scheduling": True,
}

# Model configuration for batch
batch_config = {
    "model_name": "triposr",
    "resolution": 256,
    "batch_size": 1,
    "precision": "fp16",
    "save_intermediate": False,
}

# Create batch job
queue = JobQueue(**queue_config)
jobs = queue.create_batch(
    prompts=prompt_list,
    model_name="triposr",
    config=batch_config,
    priority="normal"
)
```

### Quality Focus
```python
batch_config = {
    "model_name": "hunyuan3d-mini",
    "target_polycount": 30000,
    "texture_size": 2048,
    "num_inference_steps": 75,
    "save_all_outputs": True,
    "generate_thumbnails": True,
}
```

## Environment-Specific Configurations

### Development Environment
```python
config = {
    "debug_mode": True,
    "save_intermediate": True,
    "verbose_logging": True,
    "mock_mode": False,  # Set True for testing without GPU
    "profile_performance": True,
    "validate_outputs": True,
}
```

### Production Environment
```python
config = {
    "debug_mode": False,
    "save_intermediate": False,
    "verbose_logging": False,
    "enable_monitoring": True,
    "alert_on_failure": True,
    "retry_on_oom": True,
    "max_retries": 3,
}
```

### CI/CD Pipeline
```python
config = {
    "mock_mode": True,  # Use mock models
    "quick_test": True,
    "num_inference_steps": 1,
    "resolution": 64,
    "skip_validation": False,
}
```

## Advanced Configurations

### Multi-Stage Pipeline
```python
# Stage 1: Quick preview
stage1_config = {
    "model_name": "triposr",
    "resolution": 256,
    "precision": "fp16",
}

# Stage 2: Refine with PBR
stage2_config = {
    "model_name": "stable-fast-3d",
    "input_mesh": "stage1_output.obj",
    "generate_pbr": True,
    "target_polycount": 20000,
}

# Stage 3: Professional UV
stage3_config = {
    "model_name": "hunyuan3d-mini",
    "input_mesh": "stage2_output.glb",
    "uv_method": "smart_projection",
    "texture_size": 2048,
}
```

### A/B Testing Configuration
```python
ab_test_config = {
    "model_a": "triposr",
    "model_b": "stable-fast-3d",
    "config_a": {
        "resolution": 512,
        "precision": "fp16",
    },
    "config_b": {
        "target_polycount": 20000,
        "generate_pbr": True,
    },
    "num_samples": 10,
    "metrics": ["speed", "quality", "memory"],
}
```

### Custom Preprocessing
```python
config = {
    "preprocessing": {
        "resize_input": True,
        "target_size": 512,
        "remove_background": True,
        "center_object": True,
        "normalize_lighting": True,
    },
    "model_config": {
        "model_name": "stable-fast-3d",
        "generate_pbr": True,
    },
    "postprocessing": {
        "smooth_normals": True,
        "optimize_mesh": True,
        "compress_textures": True,
    },
}
```

## Configuration Files

### YAML Configuration
```yaml
# config.yaml
model:
  name: hunyuan3d-mini
  precision: fp16
  
generation:
  num_inference_steps: 50
  guidance_scale: 7.5
  seed: 42
  
output:
  format: glb
  target_polycount: 30000
  texture_size: 2048
  
optimization:
  cpu_offload: false
  gradient_checkpointing: true
  
monitoring:
  track_memory: true
  log_performance: true
```

### JSON Configuration
```json
{
  "model": {
    "name": "stable-fast-3d",
    "version": "latest"
  },
  "generation": {
    "target_polycount": 20000,
    "texture_resolution": 2048,
    "generate_pbr": true,
    "delight": true
  },
  "output": {
    "format": "glb",
    "save_path": "./outputs",
    "create_thumbnail": true
  }
}
```

### Environment Variables
```bash
# .env file
DREAM_CAD_MODEL=triposr
DREAM_CAD_PRECISION=fp16
DREAM_CAD_RESOLUTION=512
DREAM_CAD_CACHE_DIR=/mnt/datadrive_m2/.cache
DREAM_CAD_OUTPUT_DIR=./outputs
DREAM_CAD_GPU_ID=0
DREAM_CAD_MAX_RETRIES=3
```

## Model-Specific Advanced Options

### MVDream Advanced
```python
config = {
    "model_name": "mvdream",
    "diffusion_scheduler": "DDIM",
    "scheduler_steps": 50,
    "eta": 0.0,
    "clip_skip": 1,
    "vae_tiling": True,
    "attention_slicing": True,
    "channel_last_format": True,
}
```

### TripoSR Advanced
```python
config = {
    "model_name": "triposr",
    "marching_cubes_resolution": 256,
    "threshold": 0.0,
    "bounding_box_scale": 1.1,
    "vertex_color_source": "texture",
}
```

### Stable-Fast-3D Advanced
```python
config = {
    "model_name": "stable-fast-3d",
    "pbr_workflow": "metallic",  # or "specular"
    "texture_filtering": "trilinear",
    "mipmap_levels": 4,
    "anisotropic_filtering": 8,
    "shadow_softness": 0.5,
}
```

### TRELLIS Advanced
```python
config = {
    "model_name": "trellis",
    "nerf_config": {
        "grid_resolution": 128,
        "num_samples": 64,
        "learning_rate": 1e-3,
    },
    "gaussian_config": {
        "num_gaussians": 10000,
        "init_scale": 0.1,
        "pruning_threshold": 0.01,
    },
}
```

### Hunyuan3D-Mini Advanced
```python
config = {
    "model_name": "hunyuan3d-mini",
    "uv_packing_algorithm": "optimal",
    "texture_bleed_pixels": 8,
    "normal_map_strength": 1.0,
    "ao_map_quality": "high",
    "displacement_strength": 0.1,
}
```

## Performance Profiling Configuration

```python
config = {
    "profiling": {
        "enabled": True,
        "profile_memory": True,
        "profile_time": True,
        "profile_operations": True,
        "save_trace": True,
        "trace_path": "./profiles/",
    },
    "benchmarking": {
        "warmup_runs": 3,
        "test_runs": 10,
        "save_results": True,
        "compare_baseline": True,
    },
}
```

## Debugging Configuration

```python
config = {
    "debug": {
        "verbose": True,
        "save_all_intermediates": True,
        "validate_inputs": True,
        "validate_outputs": True,
        "check_gradients": True,
        "log_level": "DEBUG",
        "breakpoint_on_error": False,
    },
}
```

## Configuration Best Practices

1. **Start with presets**: Use provided configurations as starting points
2. **Test incrementally**: Change one parameter at a time
3. **Monitor resources**: Always check VRAM usage with new configs
4. **Document settings**: Keep notes on what works for your use case
5. **Version control**: Track configuration files in git
6. **Validate outputs**: Always check quality after config changes
7. **Profile performance**: Measure impact of configuration changes
8. **Use appropriate precision**: FP16 for speed, FP32 for quality
9. **Consider pipeline**: Some settings work better in combination
10. **Hardware specific**: Tune configurations for your specific GPU