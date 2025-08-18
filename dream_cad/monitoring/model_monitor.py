import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque, defaultdict
import psutil
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
logger = logging.getLogger(__name__)
@dataclass
class ModelMetrics:
    model_name: str
    timestamp: datetime
    generation_time_seconds: float = 0.0
    preprocessing_time: float = 0.0
    inference_time: float = 0.0
    postprocessing_time: float = 0.0
    vram_used_gb: float = 0.0
    vram_peak_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_peak_gb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0
    output_polycount: int = 0
    output_texture_resolution: int = 0
    quality_score: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    prompt: str = ""
    output_path: Optional[str] = None
    model_config: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetrics":
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
class ModelMonitor:
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        history_size: int = 1000,
        sampling_interval: float = 1.0,
    ):
        self.storage_dir = storage_dir or Path("monitoring/metrics")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history_size = history_size
        self.sampling_interval = sampling_interval
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.current_metrics: Dict[str, ModelMetrics] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.resource_samples: List[Dict[str, float]] = []
        self._resource_lock = threading.Lock()
        self.active_models: Dict[str, datetime] = {}
        self.model_events: List[Dict[str, Any]] = []
        self._load_history()
    def start_generation(self, model_name: str, prompt: str, config: Dict[str, Any]) -> None:
        metrics = ModelMetrics(
            model_name=model_name,
            timestamp=datetime.now(),
            prompt=prompt,
            model_config=config,
        )
        self.current_metrics[model_name] = metrics
        self._start_resource_monitoring()
        logger.info(f"Started monitoring generation for {model_name}")
    def end_generation(
        self,
        model_name: str,
        success: bool = True,
        output_path: Optional[str] = None,
        error_message: Optional[str] = None,
        quality_metrics: Optional[Dict[str, Any]] = None,
    ) -> ModelMetrics:
        if model_name not in self.current_metrics:
            logger.warning(f"No active monitoring for {model_name}")
            return ModelMetrics(model_name=model_name, timestamp=datetime.now())
        metrics = self.current_metrics[model_name]
        self._stop_resource_monitoring()
        with self._resource_lock:
            if self.resource_samples:
                metrics.vram_peak_gb = max(s.get("vram_gb", 0) for s in self.resource_samples)
                metrics.ram_peak_gb = max(s.get("ram_gb", 0) for s in self.resource_samples)
                metrics.gpu_utilization = sum(s.get("gpu_util", 0) for s in self.resource_samples) / len(self.resource_samples)
                metrics.gpu_temperature = max(s.get("gpu_temp", 0) for s in self.resource_samples)
        metrics.success = success
        metrics.output_path = output_path
        metrics.error_message = error_message
        metrics.generation_time_seconds = (datetime.now() - metrics.timestamp).total_seconds()
        if quality_metrics:
            metrics.output_polycount = quality_metrics.get("polycount", 0)
            metrics.output_texture_resolution = quality_metrics.get("texture_resolution", 0)
            metrics.quality_score = quality_metrics.get("quality_score", 0.0)
        self.metrics_history[model_name].append(metrics)
        self._save_metrics(metrics)
        del self.current_metrics[model_name]
        self.resource_samples.clear()
        logger.info(f"Completed monitoring for {model_name}: success={success}")
        return metrics
    def record_model_event(
        self,
        model_name: str,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "event_type": event_type,
            "details": details or {},
        }
        self.model_events.append(event)
        if event_type == "load":
            self.active_models[model_name] = datetime.now()
        elif event_type == "unload" and model_name in self.active_models:
            del self.active_models[model_name]
        self._save_event(event)
        logger.info(f"Model event: {model_name} - {event_type}")
    def get_model_stats(self, model_name: str, hours: int = 24) -> Dict[str, Any]:
        cutoff = datetime.now() - timedelta(hours=hours)
        metrics = [
            m for m in self.metrics_history.get(model_name, [])
            if m.timestamp > cutoff
        ]
        if not metrics:
            return {
                "model_name": model_name,
                "total_generations": 0,
                "success_rate": 0.0,
                "avg_generation_time": 0.0,
                "avg_vram_gb": 0.0,
                "peak_vram_gb": 0.0,
                "avg_quality_score": 0.0,
            }
        successful = [m for m in metrics if m.success]
        return {
            "model_name": model_name,
            "total_generations": len(metrics),
            "successful_generations": len(successful),
            "failed_generations": len(metrics) - len(successful),
            "success_rate": len(successful) / len(metrics) if metrics else 0.0,
            "avg_generation_time": sum(m.generation_time_seconds for m in successful) / len(successful) if successful else 0.0,
            "min_generation_time": min(m.generation_time_seconds for m in successful) if successful else 0.0,
            "max_generation_time": max(m.generation_time_seconds for m in successful) if successful else 0.0,
            "avg_vram_gb": sum(m.vram_peak_gb for m in successful) / len(successful) if successful else 0.0,
            "peak_vram_gb": max(m.vram_peak_gb for m in successful) if successful else 0.0,
            "avg_quality_score": sum(m.quality_score for m in successful) / len(successful) if successful else 0.0,
            "total_time_hours": sum(m.generation_time_seconds for m in successful) / 3600,
            "errors": [m.error_message for m in metrics if not m.success and m.error_message],
        }
    def get_all_model_stats(self, hours: int = 24) -> Dict[str, Dict[str, Any]]:
        stats = {}
        for model_name in self.metrics_history.keys():
            stats[model_name] = self.get_model_stats(model_name, hours)
        return stats
    def get_resource_usage(self) -> Dict[str, Any]:
        usage = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "ram_gb": psutil.virtual_memory().used / (1024**3),
            "ram_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage("/").percent,
        }
        gpu_metrics = self._get_gpu_metrics()
        if gpu_metrics:
            usage.update(gpu_metrics)
        usage["active_models"] = list(self.active_models.keys())
        usage["models_loaded_time"] = {
            model: (datetime.now() - load_time).total_seconds()
            for model, load_time in self.active_models.items()
        }
        return usage
    def _start_resource_monitoring(self) -> None:
        if self.monitoring_active:
            return
        self.monitoring_active = True
        self.resource_samples.clear()
        def monitor_loop():
            while self.monitoring_active:
                sample = {
                    "timestamp": time.time(),
                    "ram_gb": psutil.virtual_memory().used / (1024**3),
                }
                gpu_metrics = self._get_gpu_metrics()
                if gpu_metrics:
                    sample["vram_gb"] = gpu_metrics.get("vram_gb", 0)
                    sample["gpu_util"] = gpu_metrics.get("gpu_utilization", 0)
                    sample["gpu_temp"] = gpu_metrics.get("gpu_temperature", 0)
                with self._resource_lock:
                    self.resource_samples.append(sample)
                time.sleep(self.sampling_interval)
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    def _stop_resource_monitoring(self) -> None:
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            self.monitor_thread = None
    def _get_gpu_metrics(self) -> Optional[Dict[str, float]]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                return {
                    "vram_gb": float(values[0]) / 1024,
                    "vram_total_gb": float(values[1]) / 1024,
                    "gpu_utilization": float(values[2]),
                    "gpu_temperature": float(values[3]),
                }
        except Exception as e:
            logger.debug(f"Could not get GPU metrics: {e}")
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                return {
                    "vram_gb": torch.cuda.memory_allocated() / (1024**3),
                    "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                }
            except Exception:
                pass
        return None
    def _save_metrics(self, metrics: ModelMetrics) -> None:
        try:
            date_str = metrics.timestamp.strftime("%Y%m%d")
            file_path = self.storage_dir / f"{metrics.model_name}_{date_str}.jsonl"
            with file_path.open("a") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    def _save_event(self, event: Dict[str, Any]) -> None:
        try:
            file_path = self.storage_dir / "model_events.jsonl"
            with file_path.open("a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to save event: {e}")
    def _load_history(self) -> None:
        try:
            for metrics_file in self.storage_dir.glob("*.jsonl"):
                if metrics_file.name == "model_events.jsonl":
                    continue
                model_name = metrics_file.stem.rsplit("_", 1)[0]
                with metrics_file.open() as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            metrics = ModelMetrics.from_dict(data)
                            self.metrics_history[model_name].append(metrics)
                        except Exception as e:
                            logger.debug(f"Could not load metric: {e}")
            logger.info(f"Loaded metrics for {len(self.metrics_history)} models")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")