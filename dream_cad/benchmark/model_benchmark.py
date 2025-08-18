import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import psutil
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
try:
    from ..models.factory import ModelFactory
    from ..models.base import Model3D
    MODEL_FACTORY_AVAILABLE = True
except ImportError:
    MODEL_FACTORY_AVAILABLE = False
    ModelFactory = None
    Model3D = None
logger = logging.getLogger(__name__)
@dataclass
class BenchmarkConfig:
    model_name: str
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    output_format: str = "glb"
    seed: Optional[int] = 42
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    fp16: bool = True
    enable_xformers: bool = True
    batch_size: int = 1
    warmup_runs: int = 1
    test_runs: int = 3
    timeout_seconds: int = 600
    save_outputs: bool = True
    output_dir: Optional[Path] = None
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.output_dir:
            data["output_dir"] = str(self.output_dir)
        return data
    def get_hash(self) -> str:
        config_str = f"{self.model_name}_{self.prompt}_{self.num_inference_steps}_{self.guidance_scale}"
        return hashlib.md5(config_str.encode(), usedforsecurity=False).hexdigest()[:8]
@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    success: bool
    generation_time_seconds: float
    warmup_time_seconds: float = 0.0
    model_load_time_seconds: float = 0.0
    peak_vram_gb: float = 0.0
    avg_vram_gb: float = 0.0
    peak_ram_gb: float = 0.0
    avg_ram_gb: float = 0.0
    peak_gpu_temp_c: float = 0.0
    avg_gpu_temp_c: float = 0.0
    peak_gpu_utilization: float = 0.0
    avg_gpu_utilization: float = 0.0
    output_path: Optional[str] = None
    output_size_mb: float = 0.0
    vertex_count: int = 0
    face_count: int = 0
    texture_resolution: Optional[Tuple[int, int]] = None
    has_pbr_materials: bool = False
    mesh_quality_score: float = 0.0
    texture_quality_score: float = 0.0
    prompt_adherence_score: float = 0.0
    overall_quality_score: float = 0.0
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    benchmark_version: str = "1.0.0"
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["config"] = self.config.to_dict()
        return data
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        data = data.copy()
        if "config" in data:
            data["config"] = BenchmarkConfig(**data["config"])
        return cls(**data)
class ModelBenchmark:
    def __init__(
        self,
        model_name: str,
        output_dir: Optional[Path] = None,
        enable_monitoring: bool = True,
    ):
        self.model_name = model_name
        self.output_dir = output_dir or Path("benchmarks") / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_monitoring = enable_monitoring
        self.model: Optional[Model3D] = None
        self.monitor_data: List[Dict[str, float]] = []
        self.monitoring = False
        self.hardware_info = self._get_hardware_info()
    def _get_hardware_info(self) -> Dict[str, Any]:
        info = {
            "cpu": psutil.cpu_count(),
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "platform": "linux",
        }
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = props.total_memory / (1024**3)
            info["cuda_version"] = torch.version.cuda
        return info
    def load_model(self) -> float:
        if not MODEL_FACTORY_AVAILABLE:
            logger.warning("Model factory not available, using mock model")
            return 0.0
        start_time = time.time()
        try:
            self.model = ModelFactory.create_model(self.model_name)
            load_time = time.time() - start_time
            logger.info(f"Loaded model {self.model_name} in {load_time:.2f}s")
            return load_time
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    def unload_model(self) -> None:
        if self.model:
            try:
                if hasattr(self.model, 'cleanup'):
                    self.model.cleanup()
            except Exception as e:
                logger.warning(f"Error during model cleanup: {e}")
            finally:
                self.model = None
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    def warmup(self, config: BenchmarkConfig) -> float:
        if not self.model:
            return 0.0
        logger.info(f"Running {config.warmup_runs} warmup iterations...")
        warmup_time = 0.0
        for i in range(config.warmup_runs):
            start_time = time.time()
            try:
                with self.model as model:
                    _ = model.generate_from_text(
                        prompt="a simple cube",
                        num_inference_steps=1,
                    )
                warmup_time += time.time() - start_time
            except Exception as e:
                logger.warning(f"Warmup run {i+1} failed: {e}")
        return warmup_time
    def _monitor_resources(self) -> Dict[str, float]:
        metrics = {
            "timestamp": time.time(),
            "ram_gb": psutil.virtual_memory().used / (1024**3),
            "ram_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
        }
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                metrics["vram_gb"] = torch.cuda.memory_allocated() / (1024**3)
                metrics["vram_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", 
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    temp, util = result.stdout.strip().split(", ")
                    metrics["gpu_temp_c"] = float(temp)
                    metrics["gpu_utilization"] = float(util)
            except Exception as e:
                logger.debug(f"Could not get GPU metrics: {e}")
        return metrics
    def run_single_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        result = BenchmarkResult(
            config=config,
            success=False,
            generation_time_seconds=0.0,
            hardware_info=self.hardware_info,
        )
        self.monitor_data.clear()
        try:
            if not self.model:
                result.model_load_time_seconds = self.load_model()
            if config.warmup_runs > 0:
                result.warmup_time_seconds = self.warmup(config)
            before_metrics = self._monitor_resources()
            logger.info(f"Running benchmark: {config.model_name} - {config.prompt[:50]}...")
            start_time = time.time()
            with self.model as model:
                generation_result = model.generate_from_text(
                    prompt=config.prompt,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    seed=config.seed,
                )
            generation_time = time.time() - start_time
            result.generation_time_seconds = generation_time
            result.success = True
            if config.save_outputs and config.output_dir:
                output_path = config.output_dir / f"{config.get_hash()}.{config.output_format}"
                if hasattr(generation_result, 'save'):
                    generation_result.save(output_path)
                    result.output_path = str(output_path)
                    if output_path.exists():
                        result.output_size_mb = output_path.stat().st_size / (1024**2)
            if hasattr(generation_result, 'vertices'):
                result.vertex_count = len(generation_result.vertices)
            if hasattr(generation_result, 'faces'):
                result.face_count = len(generation_result.faces)
            after_metrics = self._monitor_resources()
            if before_metrics and after_metrics:
                result.peak_vram_gb = after_metrics.get("vram_gb", 0)
                result.peak_ram_gb = after_metrics.get("ram_gb", 0)
                result.peak_gpu_temp_c = after_metrics.get("gpu_temp_c", 0)
                result.peak_gpu_utilization = after_metrics.get("gpu_utilization", 0)
            logger.info(f"Benchmark completed in {generation_time:.2f}s")
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.error_type = type(e).__name__
            logger.error(f"Benchmark failed: {e}")
        return result
    def run_benchmark_suite(
        self,
        configs: List[BenchmarkConfig],
        save_results: bool = True,
    ) -> List[BenchmarkResult]:
        results = []
        for i, config in enumerate(configs, 1):
            logger.info(f"Running benchmark {i}/{len(configs)}")
            run_results = []
            for run in range(config.test_runs):
                logger.info(f"  Run {run+1}/{config.test_runs}")
                result = self.run_single_benchmark(config)
                run_results.append(result)
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(2)
            if run_results:
                avg_result = self._average_results(run_results)
                results.append(avg_result)
        if save_results:
            self._save_results(results)
        self.unload_model()
        return results
    def _average_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        if not results:
            return None
        avg_result = BenchmarkResult(
            config=results[0].config,
            success=all(r.success for r in results),
            generation_time_seconds=0.0,
            hardware_info=results[0].hardware_info,
        )
        successful_results = [r for r in results if r.success]
        if successful_results:
            n = len(successful_results)
            avg_result.generation_time_seconds = sum(r.generation_time_seconds for r in successful_results) / n
            avg_result.warmup_time_seconds = sum(r.warmup_time_seconds for r in successful_results) / n
            avg_result.model_load_time_seconds = sum(r.model_load_time_seconds for r in successful_results) / n
            avg_result.peak_vram_gb = max(r.peak_vram_gb for r in successful_results)
            avg_result.peak_ram_gb = max(r.peak_ram_gb for r in successful_results)
            avg_result.avg_vram_gb = sum(r.avg_vram_gb for r in successful_results) / n
            avg_result.avg_ram_gb = sum(r.avg_ram_gb for r in successful_results) / n
            avg_result.peak_gpu_temp_c = max(r.peak_gpu_temp_c for r in successful_results)
            avg_result.avg_gpu_temp_c = sum(r.avg_gpu_temp_c for r in successful_results) / n
        if successful_results:
            last_result = successful_results[-1]
            avg_result.output_path = last_result.output_path
            avg_result.output_size_mb = last_result.output_size_mb
            avg_result.vertex_count = last_result.vertex_count
            avg_result.face_count = last_result.face_count
        return avg_result
    def _save_results(self, results: List[BenchmarkResult]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        try:
            with results_file.open("w") as f:
                json.dump(
                    [r.to_dict() for r in results],
                    f,
                    indent=2,
                )
            logger.info(f"Results saved to {results_file}")
        except IOError as e:
            logger.error(f"Failed to save results: {e}")