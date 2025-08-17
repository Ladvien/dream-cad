"""Resource management for intelligent job scheduling."""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

try:
    import torch
    import psutil
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
    # Mock for testing
    class MockTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False
            @staticmethod
            def device_count():
                return 0
            @staticmethod
            def get_device_name(idx):
                return "Mock GPU"
            @staticmethod
            def get_device_properties(idx):
                class Props:
                    total_memory = 24 * 1024**3
                return Props()
            @staticmethod
            def memory_allocated(idx):
                return 0
            @staticmethod
            def memory_reserved(idx):
                return 0
            @staticmethod
            def empty_cache():
                pass
    
    torch = MockTorch()
    
    class MockProcess:
        def memory_info(self):
            class MemInfo:
                rss = 1024**3
            return MemInfo()
        def cpu_percent(self):
            return 10.0
    
    class MockPsutil:
        @staticmethod
        def virtual_memory():
            class VMem:
                total = 32 * 1024**3
                available = 16 * 1024**3
                percent = 50.0
            return VMem()
        @staticmethod
        def Process():
            return MockProcess()
    
    psutil = MockPsutil()

logger = logging.getLogger(__name__)


@dataclass
class GPUDevice:
    """GPU device information."""
    index: int
    name: str
    total_memory_gb: float
    allocated_memory_gb: float
    free_memory_gb: float
    utilization_percent: float = 0.0
    temperature_c: float = 0.0
    assigned_models: Set[str] = field(default_factory=set)
    current_job_id: Optional[str] = None
    
    @property
    def is_available(self) -> bool:
        """Check if GPU is available for new jobs."""
        return self.current_job_id is None and self.free_memory_gb > 1.0


@dataclass
class ModelProfile:
    """Model resource profile."""
    model_name: str
    min_vram_gb: float
    optimal_vram_gb: float
    supports_multi_gpu: bool = False
    supports_cpu_offload: bool = False
    load_time_seconds: float = 10.0
    unload_time_seconds: float = 5.0
    last_used: Optional[datetime] = None
    usage_count: int = 0
    total_generation_time: float = 0.0
    
    @property
    def average_generation_time(self) -> float:
        """Get average generation time."""
        if self.usage_count == 0:
            return 0.0
        return self.total_generation_time / self.usage_count


class ResourceManager:
    """Manage system resources for optimal job scheduling."""
    
    def __init__(
        self,
        check_interval: int = 5,
        enable_multi_gpu: bool = True,
        enable_cpu_offload: bool = False,
        model_cache_dir: Optional[Path] = None,
    ):
        """Initialize resource manager."""
        self.check_interval = check_interval
        self.enable_multi_gpu = enable_multi_gpu
        self.enable_cpu_offload = enable_cpu_offload
        self.model_cache_dir = model_cache_dir or Path.home() / ".cache" / "dream_cad"
        
        # GPU devices
        self.gpus: Dict[int, GPUDevice] = {}
        self.discover_gpus()
        
        # Model profiles
        self.model_profiles: Dict[str, ModelProfile] = self._load_default_profiles()
        
        # Loaded models
        self.loaded_models: Dict[str, Tuple[str, int]] = {}  # model_name -> (job_id, gpu_index)
        
        # Resource locks
        self.lock = threading.RLock()
        
        # Monitoring
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            "total_jobs_assigned": 0,
            "total_gpu_switches": 0,
            "total_memory_freed": 0,
            "peak_memory_usage": 0,
        }
    
    def _load_default_profiles(self) -> Dict[str, ModelProfile]:
        """Load default model profiles."""
        return {
            "mvdream": ModelProfile(
                model_name="mvdream",
                min_vram_gb=16.0,
                optimal_vram_gb=20.0,
                supports_multi_gpu=False,
                supports_cpu_offload=True,
                load_time_seconds=30.0,
            ),
            "triposr": ModelProfile(
                model_name="triposr",
                min_vram_gb=4.0,
                optimal_vram_gb=6.0,
                supports_multi_gpu=False,
                supports_cpu_offload=True,
                load_time_seconds=10.0,
            ),
            "stable-fast-3d": ModelProfile(
                model_name="stable-fast-3d",
                min_vram_gb=6.0,
                optimal_vram_gb=8.0,
                supports_multi_gpu=False,
                supports_cpu_offload=True,
                load_time_seconds=15.0,
            ),
            "trellis": ModelProfile(
                model_name="trellis",
                min_vram_gb=8.0,
                optimal_vram_gb=12.0,
                supports_multi_gpu=True,
                supports_cpu_offload=True,
                load_time_seconds=20.0,
            ),
            "hunyuan3d-mini": ModelProfile(
                model_name="hunyuan3d-mini",
                min_vram_gb=12.0,
                optimal_vram_gb=16.0,
                supports_multi_gpu=False,
                supports_cpu_offload=True,
                load_time_seconds=25.0,
            ),
        }
    
    def discover_gpus(self) -> None:
        """Discover available GPUs."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            logger.warning("No GPUs available")
            return
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"Discovered {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_memory_gb = props.total_memory / (1024**3)
            allocated_gb = torch.cuda.memory_allocated(i) / (1024**3)
            
            self.gpus[i] = GPUDevice(
                index=i,
                name=torch.cuda.get_device_name(i),
                total_memory_gb=total_memory_gb,
                allocated_memory_gb=allocated_gb,
                free_memory_gb=total_memory_gb - allocated_gb,
            )
            
            logger.info(
                f"GPU {i}: {self.gpus[i].name} "
                f"({total_memory_gb:.1f}GB total, {self.gpus[i].free_memory_gb:.1f}GB free)"
            )
    
    def update_gpu_stats(self) -> None:
        """Update GPU statistics."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        with self.lock:
            for gpu_idx, gpu in self.gpus.items():
                allocated_gb = torch.cuda.memory_allocated(gpu_idx) / (1024**3)
                gpu.allocated_memory_gb = allocated_gb
                gpu.free_memory_gb = gpu.total_memory_gb - allocated_gb
                
                # Track peak usage
                if allocated_gb > self.stats["peak_memory_usage"]:
                    self.stats["peak_memory_usage"] = allocated_gb
    
    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage."""
        mem = psutil.virtual_memory()
        
        return {
            "cpu_percent": psutil.Process().cpu_percent(),
            "ram_total_gb": mem.total / (1024**3),
            "ram_available_gb": mem.available / (1024**3),
            "ram_percent": mem.percent,
        }
    
    def can_run_model(
        self,
        model_name: str,
        required_vram_gb: Optional[float] = None,
    ) -> Tuple[bool, Optional[int], str]:
        """Check if a model can run with current resources."""
        with self.lock:
            profile = self.model_profiles.get(model_name)
            if not profile:
                return False, None, f"Unknown model: {model_name}"
            
            min_vram = required_vram_gb or profile.min_vram_gb
            
            # Check if model is already loaded
            if model_name in self.loaded_models:
                job_id, gpu_idx = self.loaded_models[model_name]
                if gpu_idx in self.gpus:
                    return True, gpu_idx, "Model already loaded"
            
            # Find available GPU
            best_gpu = None
            best_free_memory = 0
            
            for gpu_idx, gpu in self.gpus.items():
                if gpu.is_available and gpu.free_memory_gb >= min_vram:
                    if gpu.free_memory_gb > best_free_memory:
                        best_gpu = gpu_idx
                        best_free_memory = gpu.free_memory_gb
            
            if best_gpu is not None:
                return True, best_gpu, f"GPU {best_gpu} available"
            
            # Check if we can free memory
            if self._can_free_memory(min_vram):
                return True, None, "Memory can be freed"
            
            # Check CPU offload option
            if self.enable_cpu_offload and profile.supports_cpu_offload:
                sys_resources = self.get_system_resources()
                if sys_resources["ram_available_gb"] >= min_vram * 2:
                    return True, None, "CPU offload available"
            
            return False, None, f"Insufficient resources (need {min_vram:.1f}GB VRAM)"
    
    def _can_free_memory(self, required_gb: float) -> bool:
        """Check if we can free enough memory."""
        total_freeable = sum(
            gpu.allocated_memory_gb
            for gpu in self.gpus.values()
            if not gpu.current_job_id
        )
        return total_freeable >= required_gb
    
    def assign_job_to_gpu(
        self,
        job_id: str,
        model_name: str,
        preferred_gpu: Optional[int] = None,
    ) -> Optional[int]:
        """Assign a job to a GPU."""
        with self.lock:
            can_run, gpu_idx, message = self.can_run_model(model_name)
            
            if not can_run:
                logger.error(f"Cannot run model {model_name}: {message}")
                return None
            
            # Use preferred GPU if specified and available
            if preferred_gpu is not None and preferred_gpu in self.gpus:
                gpu = self.gpus[preferred_gpu]
                if gpu.is_available:
                    gpu_idx = preferred_gpu
            
            # If no specific GPU yet, find best one
            if gpu_idx is None:
                profile = self.model_profiles[model_name]
                for idx, gpu in self.gpus.items():
                    if gpu.is_available and gpu.free_memory_gb >= profile.min_vram_gb:
                        gpu_idx = idx
                        break
            
            if gpu_idx is not None:
                # Assign job to GPU
                gpu = self.gpus[gpu_idx]
                gpu.current_job_id = job_id
                gpu.assigned_models.add(model_name)
                
                # Track loaded model
                self.loaded_models[model_name] = (job_id, gpu_idx)
                
                # Update stats
                self.stats["total_jobs_assigned"] += 1
                
                logger.info(f"Assigned job {job_id} to GPU {gpu_idx}")
                return gpu_idx
            
            return None
    
    def release_job_from_gpu(self, job_id: str) -> None:
        """Release a job from GPU."""
        with self.lock:
            for gpu in self.gpus.values():
                if gpu.current_job_id == job_id:
                    gpu.current_job_id = None
                    logger.info(f"Released job {job_id} from GPU {gpu.index}")
                    break
    
    def unload_model(self, model_name: str) -> None:
        """Unload a model from GPU."""
        with self.lock:
            if model_name in self.loaded_models:
                job_id, gpu_idx = self.loaded_models[model_name]
                
                if gpu_idx in self.gpus:
                    gpu = self.gpus[gpu_idx]
                    gpu.assigned_models.discard(model_name)
                    
                    # Clear CUDA cache
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        self.stats["total_memory_freed"] += gpu.allocated_memory_gb
                
                del self.loaded_models[model_name]
                self.stats["total_gpu_switches"] += 1
                
                logger.info(f"Unloaded model {model_name} from GPU {gpu_idx}")
    
    def get_model_assignment_score(
        self,
        model_name: str,
        gpu_idx: int,
    ) -> float:
        """Calculate score for assigning a model to a specific GPU."""
        if gpu_idx not in self.gpus:
            return 0.0
        
        gpu = self.gpus[gpu_idx]
        profile = self.model_profiles.get(model_name)
        
        if not profile:
            return 0.0
        
        score = 100.0
        
        # Penalty for insufficient memory
        if gpu.free_memory_gb < profile.min_vram_gb:
            return 0.0
        
        # Bonus for optimal memory
        if gpu.free_memory_gb >= profile.optimal_vram_gb:
            score += 20.0
        
        # Penalty for fragmentation
        memory_ratio = gpu.free_memory_gb / gpu.total_memory_gb
        score += memory_ratio * 10.0
        
        # Bonus if model already loaded
        if model_name in gpu.assigned_models:
            score += 50.0
        
        # Penalty for switching models
        if gpu.assigned_models and model_name not in gpu.assigned_models:
            score -= 30.0
        
        return score
    
    def optimize_model_placement(
        self,
        pending_models: List[str],
    ) -> Dict[str, int]:
        """Optimize placement of models across GPUs."""
        with self.lock:
            placement = {}
            
            # Sort models by resource requirements (largest first)
            sorted_models = sorted(
                pending_models,
                key=lambda m: self.model_profiles.get(m, ModelProfile(m, 0, 0)).optimal_vram_gb,
                reverse=True,
            )
            
            for model_name in sorted_models:
                best_gpu = None
                best_score = 0.0
                
                for gpu_idx in self.gpus:
                    score = self.get_model_assignment_score(model_name, gpu_idx)
                    if score > best_score:
                        best_score = score
                        best_gpu = gpu_idx
                
                if best_gpu is not None and best_score > 0:
                    placement[model_name] = best_gpu
            
            return placement
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
            )
            self.monitor_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Resource monitoring loop."""
        while self.monitoring:
            try:
                self.update_gpu_stats()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.check_interval)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        with self.lock:
            gpu_summary = []
            for gpu in self.gpus.values():
                gpu_summary.append({
                    "index": gpu.index,
                    "name": gpu.name,
                    "total_memory_gb": gpu.total_memory_gb,
                    "free_memory_gb": gpu.free_memory_gb,
                    "utilization_percent": gpu.utilization_percent,
                    "current_job": gpu.current_job_id,
                    "assigned_models": list(gpu.assigned_models),
                })
            
            return {
                "gpus": gpu_summary,
                "system": self.get_system_resources(),
                "loaded_models": list(self.loaded_models.keys()),
                "stats": self.stats,
            }