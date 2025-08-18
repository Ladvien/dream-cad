import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from .job_queue import GenerationJob, JobQueue, JobStatus
from .resource_manager import ResourceManager
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
class ModelInstance:
    model_name: str
    model: Any
    gpu_index: Optional[int]
    loaded_at: datetime
    last_used: datetime
    usage_count: int = 0
    total_generation_time: float = 0.0
    is_warming_up: bool = False
    is_cooling_down: bool = False
    @property
    def idle_time(self) -> float:
        return (datetime.now() - self.last_used).total_seconds()
    @property
    def average_generation_time(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.total_generation_time / self.usage_count
class FailoverStrategy:
    def __init__(self):
        self.model_alternatives = {
            "mvdream": ["stable-fast-3d", "triposr"],
            "hunyuan3d-mini": ["trellis", "stable-fast-3d"],
            "trellis": ["hunyuan3d-mini", "stable-fast-3d"],
            "stable-fast-3d": ["triposr", "hunyuan3d-mini"],
            "triposr": ["stable-fast-3d", "trellis"],
        }
        self.failure_counts: Dict[str, int] = {}
        self.blacklisted_models: Set[str] = set()
    def get_alternative_model(self, failed_model: str) -> Optional[str]:
        if failed_model in self.model_alternatives:
            for alternative in self.model_alternatives[failed_model]:
                if alternative not in self.blacklisted_models:
                    return alternative
        return None
    def record_failure(self, model_name: str) -> None:
        self.failure_counts[model_name] = self.failure_counts.get(model_name, 0) + 1
        if self.failure_counts[model_name] >= 3:
            self.blacklisted_models.add(model_name)
            logger.warning(f"Model {model_name} blacklisted after 3 failures")
    def reset_model(self, model_name: str) -> None:
        self.failure_counts[model_name] = 0
        self.blacklisted_models.discard(model_name)
class BatchProcessor:
    def __init__(
        self,
        job_queue: JobQueue,
        resource_manager: ResourceManager,
        max_workers: int = 2,
        max_loaded_models: int = 3,
        model_idle_timeout: int = 300,
        enable_warm_up: bool = True,
        enable_failover: bool = True,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.job_queue = job_queue
        self.resource_manager = resource_manager
        self.max_workers = max_workers
        self.max_loaded_models = max_loaded_models
        self.model_idle_timeout = model_idle_timeout
        self.enable_warm_up = enable_warm_up
        self.enable_failover = enable_failover
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.loaded_models: Dict[str, ModelInstance] = {}
        self.model_lock = threading.RLock()
        self.failover_strategy = FailoverStrategy()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing = False
        self.processing_thread: Optional[threading.Thread] = None
        self.active_jobs: Dict[str, GenerationJob] = {}
        self.stats = {
            "total_processed": 0,
            "total_succeeded": 0,
            "total_failed": 0,
            "total_retried": 0,
            "model_switches": 0,
            "failovers": 0,
        }
        self.resource_manager.start_monitoring()
    def load_model(
        self,
        model_name: str,
        gpu_index: Optional[int] = None,
        warm_up: bool = True,
    ) -> Optional[ModelInstance]:
        with self.model_lock:
            if model_name in self.loaded_models:
                instance = self.loaded_models[model_name]
                instance.last_used = datetime.now()
                return instance
            if len(self.loaded_models) >= self.max_loaded_models:
                self._unload_least_recently_used()
            try:
                if not MODEL_FACTORY_AVAILABLE:
                    logger.error("Model factory not available")
                    return None
                logger.info(f"Loading model {model_name}...")
                model = ModelFactory.create_model(model_name)
                if gpu_index is None:
                    gpu_index = self.resource_manager.assign_job_to_gpu(
                        f"model_{model_name}",
                        model_name,
                    )
                instance = ModelInstance(
                    model_name=model_name,
                    model=model,
                    gpu_index=gpu_index,
                    loaded_at=datetime.now(),
                    last_used=datetime.now(),
                )
                if warm_up and self.enable_warm_up:
                    instance.is_warming_up = True
                    self._warm_up_model(instance)
                    instance.is_warming_up = False
                self.loaded_models[model_name] = instance
                logger.info(f"Model {model_name} loaded successfully")
                return instance
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                self.failover_strategy.record_failure(model_name)
                return None
    def _warm_up_model(self, instance: ModelInstance) -> None:
        logger.info(f"Warming up model {instance.model_name}...")
        try:
            test_prompt = "a simple cube"
            start_time = time.time()
            with instance.model as model:
                _ = model.generate_from_text(
                    prompt=test_prompt,
                    num_inference_steps=1,
                )
            warm_up_time = time.time() - start_time
            logger.info(f"Model {instance.model_name} warmed up in {warm_up_time:.2f}s")
        except Exception as e:
            logger.warning(f"Failed to warm up model {instance.model_name}: {e}")
    def _unload_least_recently_used(self) -> None:
        if not self.loaded_models:
            return
        lru_model = min(
            self.loaded_models.values(),
            key=lambda m: m.last_used,
        )
        self.unload_model(lru_model.model_name)
    def unload_model(self, model_name: str, cool_down: bool = True) -> None:
        with self.model_lock:
            if model_name not in self.loaded_models:
                return
            instance = self.loaded_models[model_name]
            if cool_down and not instance.is_cooling_down:
                instance.is_cooling_down = True
                self._cool_down_model(instance)
            try:
                if hasattr(instance.model, 'cleanup'):
                    instance.model.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up model {model_name}: {e}")
            if instance.gpu_index is not None:
                self.resource_manager.release_job_from_gpu(f"model_{model_name}")
                self.resource_manager.unload_model(model_name)
            del self.loaded_models[model_name]
            self.stats["model_switches"] += 1
            logger.info(f"Model {model_name} unloaded")
    def _cool_down_model(self, instance: ModelInstance) -> None:
        logger.info(f"Cooling down model {instance.model_name}...")
        if instance.usage_count > 0:
            avg_time = instance.average_generation_time
            logger.info(
                f"Model {instance.model_name} stats: "
                f"{instance.usage_count} uses, {avg_time:.2f}s avg generation time"
            )
        try:
            if hasattr(instance.model, 'clear_cache'):
                instance.model.clear_cache()
        except Exception as e:
            logger.warning(f"Failed to clear cache for {instance.model_name}: {e}")
    def process_job(self, job: GenerationJob) -> bool:
        logger.info(f"Processing job {job.id}: {job.prompt[:50]}...")
        try:
            instance = self.load_model(job.model_name)
            if not instance:
                if self.enable_failover:
                    alternative = self.failover_strategy.get_alternative_model(job.model_name)
                    if alternative:
                        logger.info(f"Failover: Using {alternative} instead of {job.model_name}")
                        job.model_name = alternative
                        instance = self.load_model(alternative)
                        self.stats["failovers"] += 1
                if not instance:
                    raise RuntimeError(f"Failed to load model {job.model_name}")
            self.job_queue.update_job(
                job.id,
                status=JobStatus.RUNNING,
                assigned_device=f"GPU-{instance.gpu_index}" if instance.gpu_index else "CPU",
            )
            start_time = time.time()
            output_path = self._generate_with_model(instance, job)
            generation_time = time.time() - start_time
            instance.usage_count += 1
            instance.total_generation_time += generation_time
            instance.last_used = datetime.now()
            self.job_queue.update_job(
                job.id,
                status=JobStatus.COMPLETED,
                output_path=str(output_path),
                generation_time_seconds=generation_time,
                progress_percent=100.0,
            )
            self.stats["total_succeeded"] += 1
            self.failover_strategy.reset_model(job.model_name)
            logger.info(f"Job {job.id} completed successfully in {generation_time:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            self.job_queue.update_job(
                job.id,
                status=JobStatus.FAILED,
                error_message=str(e),
            )
            self.stats["total_failed"] += 1
            self.failover_strategy.record_failure(job.model_name)
            return False
    def _generate_with_model(
        self,
        instance: ModelInstance,
        job: GenerationJob,
    ) -> Path:
        output_dir = Path("outputs") / job.model_name / job.id
        output_dir.mkdir(parents=True, exist_ok=True)
        def progress_callback(percent: float):
            self.job_queue.update_job(job.id, progress_percent=percent)
        try:
            with instance.model as model:
                result = model.generate_from_text(
                    prompt=job.prompt,
                    output_path=output_dir,
                    **job.config,
                )
            output_file = output_dir / f"{job.id}.{job.output_format}"
            if hasattr(result, 'save'):
                result.save(output_file)
            return output_file
        finally:
            if hasattr(instance.model, '__exit__'):
                pass
            else:
                if hasattr(instance.model, 'cleanup'):
                    try:
                        instance.model.cleanup()
                    except Exception as e:
                        logger.warning(f"Error during model cleanup: {e}")
    def process_batch(self, batch_size: int = 5) -> List[GenerationJob]:
        processed = []
        for _ in range(batch_size):
            job = self.job_queue.get_next_job()
            if not job:
                break
            self.active_jobs[job.id] = job
            success = self.process_job(job)
            del self.active_jobs[job.id]
            if success:
                processed.append(job)
            self.stats["total_processed"] += 1
        return processed
    def start_processing(self) -> None:
        if not self.processing:
            self.processing = True
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
            )
            self.processing_thread.start()
            logger.info("Batch processing started")
    def stop_processing(self) -> None:
        if self.processing:
            self.processing = False
            if self.processing_thread:
                self.processing_thread.join(timeout=10)
            for model_name in list(self.loaded_models.keys()):
                self.unload_model(model_name)
            self.executor.shutdown(wait=True)
            self.resource_manager.stop_monitoring()
            logger.info("Batch processing stopped")
    def _processing_loop(self) -> None:
        while self.processing:
            try:
                self._check_idle_models()
                job = self.job_queue.get_next_job()
                if job:
                    future = self.executor.submit(self.process_job, job)
                    try:
                        future.result(timeout=600)
                    except TimeoutError:
                        logger.error(f"Job {job.id} timed out")
                        self.job_queue.update_job(
                            job.id,
                            status=JobStatus.FAILED,
                            error_message="Generation timed out",
                        )
                else:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(5)
    def _check_idle_models(self) -> None:
        with self.model_lock:
            for model_name, instance in list(self.loaded_models.items()):
                if instance.idle_time > self.model_idle_timeout:
                    if not instance.is_warming_up and not instance.is_cooling_down:
                        logger.info(f"Unloading idle model {model_name}")
                        self.unload_model(model_name)
    def get_status(self) -> Dict[str, Any]:
        return {
            "processing": self.processing,
            "loaded_models": list(self.loaded_models.keys()),
            "active_jobs": list(self.active_jobs.keys()),
            "stats": self.stats,
            "resource_summary": self.resource_manager.get_resource_summary(),
            "failover_blacklist": list(self.failover_strategy.blacklisted_models),
        }