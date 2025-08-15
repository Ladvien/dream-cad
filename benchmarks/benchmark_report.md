# MVDream RTX 3090 Benchmark Report

Generated: 2025-08-15T12:00:02.024699

## Summary

- Total benchmarks run: 6
- Successful benchmarks: 6
- Average generation time: 26.4 seconds
- Average peak VRAM: 9.3 GB
- Average max GPU temp: 54.7°C
- **Meets <2hr requirement: ✅ Yes**

## Optimal Configurations

### Fastest Generation
- Time: 7.3 seconds
- Config: `{
  "rescale_factor": 0.3,
  "batch_size": 1,
  "enable_xformers": true,
  "enable_memory_efficient_attention": true,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "n_views": 4,
  "resolution": 256,
  "fp16": true,
  "gradient_checkpointing": false,
  "cpu_offload": false,
  "compile_model": true,
  "timestep_annealing": true,
  "annealing_eta": 0.8
}`

### Most Memory Efficient
- Peak VRAM: 4.0 GB
- Config: `{
  "rescale_factor": 0.3,
  "batch_size": 1,
  "enable_xformers": true,
  "enable_memory_efficient_attention": true,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "n_views": 4,
  "resolution": 256,
  "fp16": true,
  "gradient_checkpointing": false,
  "cpu_offload": false,
  "compile_model": true,
  "timestep_annealing": true,
  "annealing_eta": 0.8
}`

### Best Quality
- Config: `{
  "rescale_factor": 0.3,
  "batch_size": 1,
  "enable_xformers": false,
  "enable_memory_efficient_attention": false,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "n_views": 4,
  "resolution": 256,
  "fp16": false,
  "gradient_checkpointing": false,
  "cpu_offload": false,
  "compile_model": false,
  "timestep_annealing": false,
  "annealing_eta": 1.0
}`

### Optimal Balanced (Recommended)
- Time: 7.3 seconds
- VRAM: 4.0 GB
- Max Temp: 52.0°C
- Config: `{
  "rescale_factor": 0.3,
  "batch_size": 1,
  "enable_xformers": true,
  "enable_memory_efficient_attention": true,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "n_views": 4,
  "resolution": 256,
  "fp16": true,
  "gradient_checkpointing": false,
  "cpu_offload": false,
  "compile_model": true,
  "timestep_annealing": true,
  "annealing_eta": 0.8
}`

## Test Prompts Used

- a simple wooden cube
- a ceramic coffee mug with handle
- an ornate golden goblet with gemstones
- a low-poly cartoon character
- a realistic human head sculpture
