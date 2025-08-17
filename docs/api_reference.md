# API Reference

## Core Classes

### ModelFactory

Factory class for creating 3D generation models.

```python
from dream_cad import ModelFactory
```

#### Methods

##### `create_model(model_name: str, **kwargs) -> Model3D`

Creates and returns a model instance.

**Parameters:**
- `model_name` (str): Name of the model to create. Options: "mvdream", "triposr", "stable-fast-3d", "trellis", "hunyuan3d-mini"
- `**kwargs`: Model-specific configuration options

**Returns:**
- `Model3D`: Model instance ready for generation

**Example:**
```python
model = ModelFactory.create_model("triposr", precision="fp16")
```

##### `list_available_models() -> List[str]`

Returns list of all available model names.

**Returns:**
- `List[str]`: List of model names

**Example:**
```python
models = ModelFactory.list_available_models()
# ['mvdream', 'triposr', 'stable-fast-3d', 'trellis', 'hunyuan3d-mini']
```

### Model3D (Abstract Base Class)

Base class for all 3D generation models.

```python
from dream_cad.models.base import Model3D
```

#### Properties

- `name` (str): Model identifier
- `capabilities` (ModelCapabilities): Model capabilities and requirements
- `is_loaded` (bool): Whether model is currently loaded in memory

#### Methods

##### `generate_from_text(prompt: str, **kwargs) -> GenerationResult`

Generate 3D model from text prompt.

**Parameters:**
- `prompt` (str): Text description of desired 3D model
- `seed` (Optional[int]): Random seed for reproducibility
- `num_inference_steps` (int): Number of denoising steps
- `guidance_scale` (float): Classifier-free guidance scale
- `negative_prompt` (Optional[str]): Negative prompt for guidance
- `**kwargs`: Model-specific parameters

**Returns:**
- `GenerationResult`: Generated 3D model data and metadata

**Example:**
```python
result = model.generate_from_text(
    "a wooden chair",
    num_inference_steps=50,
    guidance_scale=7.5
)
```

##### `generate_from_image(image: Union[PIL.Image, np.ndarray, str], **kwargs) -> GenerationResult`

Generate 3D model from input image.

**Parameters:**
- `image`: Input image as PIL Image, numpy array, or file path
- `remove_background` (bool): Whether to remove background
- `**kwargs`: Model-specific parameters

**Returns:**
- `GenerationResult`: Generated 3D model data and metadata

**Example:**
```python
result = model.generate_from_image(
    "input.png",
    remove_background=True
)
```

##### `save(result: GenerationResult, path: str, format: Optional[str] = None) -> str`

Save generation result to file.

**Parameters:**
- `result` (GenerationResult): Generation result to save
- `path` (str): Output file path
- `format` (Optional[str]): Output format (obj, ply, stl, glb). If None, inferred from path

**Returns:**
- `str`: Path to saved file

**Example:**
```python
output_path = model.save(result, "output.glb")
```

##### `load_model() -> None`

Load model into memory (called automatically).

##### `unload_model() -> None`

Unload model from memory to free resources.

**Example:**
```python
model.unload_model()  # Free VRAM
```

### GenerationResult

Result container for 3D generation.

```python
from dream_cad.models.base import GenerationResult
```

#### Attributes

- `vertices` (np.ndarray): Mesh vertices (N, 3)
- `faces` (np.ndarray): Mesh faces (M, 3)
- `vertex_colors` (Optional[np.ndarray]): Vertex colors (N, 3)
- `uv_coords` (Optional[np.ndarray]): UV coordinates (N, 2)
- `texture` (Optional[np.ndarray]): Texture image
- `materials` (Optional[Dict]): Material properties
- `metadata` (Dict[str, Any]): Generation metadata
- `generation_time` (float): Time taken in seconds
- `model_name` (str): Model used for generation

#### Methods

##### `save(path: str, format: Optional[str] = None) -> str`

Save result to file.

**Example:**
```python
result.save("output.obj")
```

### ModelCapabilities

Model capability descriptor.

```python
from dream_cad.models.base import ModelCapabilities
```

#### Attributes

- `min_vram_gb` (float): Minimum VRAM required
- `recommended_vram_gb` (float): Recommended VRAM
- `supports_text_to_3d` (bool): Supports text input
- `supports_image_to_3d` (bool): Supports image input
- `supports_multi_view` (bool): Supports multi-view input
- `output_formats` (List[str]): Supported output formats
- `has_pbr_materials` (bool): Generates PBR materials
- `has_uv_unwrapping` (bool): Performs UV unwrapping
- `typical_generation_time` (str): Expected generation time
- `max_resolution` (int): Maximum output resolution
- `license_type` (str): License information

### ModelRegistry

Registry for model capabilities and metadata.

```python
from dream_cad.models.registry import ModelRegistry
```

#### Methods

##### `register(name: str, capabilities: ModelCapabilities) -> None`

Register a model's capabilities.

##### `get_capabilities(name: str) -> Optional[ModelCapabilities]`

Get capabilities for a model.

##### `list_models() -> List[str]`

List all registered models.

##### `get_models_for_vram(vram_gb: float) -> List[str]`

Get models that can run with given VRAM.

**Example:**
```python
registry = ModelRegistry()
models = registry.get_models_for_vram(8.0)
# ['triposr', 'stable-fast-3d']
```

## Queue System

### JobQueue

Queue for managing generation jobs.

```python
from dream_cad.queue import JobQueue
```

#### Methods

##### `add_job(job: GenerationJob) -> str`

Add a job to the queue.

**Returns:**
- `str`: Job ID

##### `create_batch(prompts: List[str], model_name: str, **kwargs) -> List[str]`

Create batch of jobs from prompts.

**Parameters:**
- `prompts` (List[str]): List of prompts
- `model_name` (str): Model to use
- `priority` (JobPriority): Job priority level
- `**kwargs`: Generation parameters

**Returns:**
- `List[str]`: List of job IDs

**Example:**
```python
queue = JobQueue()
job_ids = queue.create_batch(
    ["chair", "table", "lamp"],
    model_name="triposr",
    priority=JobPriority.HIGH
)
```

##### `get_next_job() -> Optional[GenerationJob]`

Get next job from queue based on priority.

##### `update_job_status(job_id: str, status: JobStatus) -> None`

Update job status.

##### `get_job_status(job_id: str) -> Optional[JobStatus]`

Get current job status.

### BatchProcessor

Processor for batch job execution.

```python
from dream_cad.queue import BatchProcessor
```

#### Methods

##### `start_processing(num_workers: int = 1) -> None`

Start processing jobs from queue.

##### `stop_processing() -> None`

Stop processing jobs.

##### `process_job(job: GenerationJob) -> GenerationResult`

Process a single job.

**Example:**
```python
processor = BatchProcessor(queue, resource_manager)
processor.start_processing(num_workers=2)
```

### ResourceManager

GPU resource management.

```python
from dream_cad.queue import ResourceManager
```

#### Methods

##### `discover_gpus() -> List[GPUInfo]`

Discover available GPUs.

##### `assign_job_to_gpu(job_id: str, model_name: str) -> Optional[int]`

Assign job to available GPU.

##### `release_gpu(gpu_id: int, job_id: str) -> None`

Release GPU after job completion.

##### `can_model_run(model_name: str, gpu_id: int) -> bool`

Check if model can run on GPU.

## Benchmarking

### ModelBenchmark

Benchmark runner for models.

```python
from dream_cad.benchmark import ModelBenchmark
```

#### Methods

##### `run_single_benchmark(config: BenchmarkConfig) -> BenchmarkResult`

Run single benchmark test.

##### `run_benchmark_suite(configs: List[BenchmarkConfig]) -> List[BenchmarkResult]`

Run suite of benchmarks.

**Example:**
```python
benchmark = ModelBenchmark("triposr")
config = BenchmarkConfig(
    prompt="test object",
    test_runs=10,
    warmup_runs=2
)
results = benchmark.run_single_benchmark(config)
```

### QualityAssessor

Assess quality of generated meshes.

```python
from dream_cad.benchmark import QualityAssessor
```

#### Methods

##### `assess_mesh_file(file_path: Path) -> QualityMetrics`

Assess quality of mesh file.

**Returns:**
- `QualityMetrics`: Quality assessment results

**Example:**
```python
assessor = QualityAssessor()
metrics = assessor.assess_mesh_file(Path("model.obj"))
print(f"Quality score: {metrics.overall_mesh_quality}")
```

### PerformanceTracker

Track performance over time.

```python
from dream_cad.benchmark import PerformanceTracker
```

#### Methods

##### `track_performance(model_name: str, metrics: PerformanceMetrics) -> None`

Track performance metrics.

##### `get_performance_trend(model_name: str, days: int = 30) -> Dict`

Get performance trend data.

##### `generate_report() -> Dict`

Generate performance report.

## UI Components

### ModelSelectionUI

Gradio UI for model selection.

```python
from dream_cad.ui import ModelSelectionUI
```

#### Methods

##### `create_interface() -> gr.Interface`

Create Gradio interface.

**Example:**
```python
ui = ModelSelectionUI()
interface = ui.create_interface()
interface.launch(share=True)
```

## Data Classes

### BenchmarkConfig

Configuration for benchmarking.

```python
from dream_cad.benchmark import BenchmarkConfig
```

**Fields:**
- `model_name` (str): Model to benchmark
- `prompt` (str): Test prompt
- `test_runs` (int): Number of test runs
- `warmup_runs` (int): Number of warmup runs
- `save_outputs` (bool): Whether to save outputs
- `output_dir` (Optional[Path]): Output directory

### BenchmarkResult

Result from benchmark run.

```python
from dream_cad.benchmark import BenchmarkResult
```

**Fields:**
- `model_name` (str): Model tested
- `prompt` (str): Prompt used
- `success` (bool): Whether generation succeeded
- `generation_time_seconds` (float): Generation time
- `peak_vram_gb` (float): Peak VRAM usage
- `peak_ram_gb` (float): Peak RAM usage
- `output_path` (Optional[str]): Path to output
- `error_message` (Optional[str]): Error if failed
- `quality_metrics` (Optional[QualityMetrics]): Quality assessment

### QualityMetrics

Quality assessment metrics.

```python
from dream_cad.benchmark import QualityMetrics
```

**Fields:**
- `mesh_validity_score` (float): 0-100 score
- `mesh_manifold_score` (float): 0-100 score
- `mesh_watertight_score` (float): 0-100 score
- `vertex_count` (int): Number of vertices
- `face_count` (int): Number of faces
- `overall_mesh_quality` (float): Overall quality score
- `game_ready_score` (float): Game readiness score

### GenerationJob

Job for queue system.

```python
from dream_cad.queue import GenerationJob
```

**Fields:**
- `id` (str): Unique job ID
- `prompt` (str): Generation prompt
- `model_name` (str): Model to use
- `config` (Dict[str, Any]): Generation config
- `status` (JobStatus): Current status
- `priority` (JobPriority): Job priority
- `created_at` (datetime): Creation time
- `started_at` (Optional[datetime]): Start time
- `completed_at` (Optional[datetime]): Completion time
- `result` (Optional[GenerationResult]): Result if completed
- `error` (Optional[str]): Error if failed

## Enums

### JobStatus

```python
from dream_cad.queue import JobStatus

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### JobPriority

```python
from dream_cad.queue import JobPriority

class JobPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
```

### RepresentationType

```python
from dream_cad.models.trellis import RepresentationType

class RepresentationType(Enum):
    MESH = "mesh"
    NERF = "nerf"
    GAUSSIAN_SPLATTING = "gaussian_splatting"
    SLAT = "slat"
```

## Utility Functions

### clear_gpu_cache()

Clear GPU memory cache.

```python
from dream_cad.utils import clear_gpu_cache

clear_gpu_cache()
```

### get_available_vram()

Get available VRAM in GB.

```python
from dream_cad.utils import get_available_vram

vram_gb = get_available_vram()
```

### validate_prompt(prompt: str) -> bool

Validate generation prompt.

```python
from dream_cad.utils import validate_prompt

is_valid = validate_prompt("a wooden chair")
```

## Example Workflows

### Simple Generation

```python
from dream_cad import ModelFactory

# Create model
model = ModelFactory.create_model("triposr")

# Generate
result = model.generate_from_text("a ceramic vase")

# Save
result.save("vase.glb")
```

### Batch Processing

```python
from dream_cad.queue import JobQueue, BatchProcessor, ResourceManager

# Setup
queue = JobQueue()
resource_manager = ResourceManager()
processor = BatchProcessor(queue, resource_manager)

# Add jobs
prompts = ["chair", "table", "lamp", "vase", "sculpture"]
job_ids = queue.create_batch(prompts, "triposr")

# Process
processor.start_processing(num_workers=2)

# Monitor
for job_id in job_ids:
    status = queue.get_job_status(job_id)
    print(f"Job {job_id}: {status}")
```

### Quality Assessment

```python
from dream_cad import ModelFactory
from dream_cad.benchmark import QualityAssessor

# Generate
model = ModelFactory.create_model("stable-fast-3d")
result = model.generate_from_text("game asset sword")
path = result.save("sword.glb")

# Assess
assessor = QualityAssessor()
metrics = assessor.assess_mesh_file(Path(path))

print(f"Polycount: {metrics.face_count}")
print(f"Game ready: {metrics.game_ready_score}/100")
print(f"UV quality: {metrics.uv_coverage}%")
```

### A/B Testing

```python
from dream_cad.benchmark import ABTester, ABTestConfig, ModelBenchmark

# Setup
tester = ABTester()
config = ABTestConfig(
    model_a="triposr",
    model_b="stable-fast-3d",
    test_prompts=["chair", "table", "lamp"],
    num_samples_per_prompt=5
)

# Create benchmarks
benchmark_a = ModelBenchmark("triposr")
benchmark_b = ModelBenchmark("stable-fast-3d")

# Run test
result = tester.run_ab_test(config, benchmark_a, benchmark_b)
print(f"Winner: {result.winner}")
print(f"Statistical significance: {result.p_value}")
```

### Custom Model Integration

```python
from dream_cad.models.base import Model3D
from dream_cad.models.factory import register_model

@register_model("custom_model")
class CustomModel(Model3D):
    def __init__(self, **kwargs):
        super().__init__("custom_model", **kwargs)
    
    def generate_from_text(self, prompt, **kwargs):
        # Implementation
        pass
    
    def generate_from_image(self, image, **kwargs):
        # Implementation
        pass

# Use custom model
model = ModelFactory.create_model("custom_model")
```

## Error Handling

All methods may raise the following exceptions:

- `ModelNotFoundError`: Model not available
- `InsufficientVRAMError`: Not enough GPU memory
- `GenerationError`: Generation failed
- `InvalidConfigError`: Invalid configuration
- `FileFormatError`: Unsupported file format

Example:
```python
from dream_cad.exceptions import InsufficientVRAMError

try:
    model = ModelFactory.create_model("mvdream")
    result = model.generate_from_text("complex scene")
except InsufficientVRAMError as e:
    print(f"Not enough VRAM: {e}")
    # Fall back to smaller model
    model = ModelFactory.create_model("triposr")
```

## Environment Variables

Configure Dream-CAD through environment variables:

```bash
# Model defaults
export DREAM_CAD_DEFAULT_MODEL=triposr
export DREAM_CAD_PRECISION=fp16

# Paths
export DREAM_CAD_MODEL_PATH=/path/to/models
export DREAM_CAD_CACHE_DIR=/path/to/cache
export DREAM_CAD_OUTPUT_DIR=/path/to/outputs

# Performance
export DREAM_CAD_GPU_ID=0
export DREAM_CAD_MAX_WORKERS=2
export DREAM_CAD_BATCH_SIZE=1

# Monitoring
export DREAM_CAD_LOG_LEVEL=INFO
export DREAM_CAD_PROFILE=false
```

## Version Information

```python
import dream_cad
print(dream_cad.__version__)  # 1.0.0
```