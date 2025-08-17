"""Batch processor with model warm-up, cool-down, and failover mechanisms."""

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

# Import model factory
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
    """Represents a loaded model instance."""
    model_name: str
    model: Any  # Model3D instance
    gpu_index: Optional[int]
    loaded_at: datetime
    last_used: datetime
    usage_count: int = 0
    total_generation_time: float = 0.0
    is_warming_up: bool = False
    is_cooling_down: bool = False
    
    @property
    def idle_time(self) -> float:
        """Get idle time in seconds."""
        return (datetime.now() - self.last_used).total_seconds()
    
    @property
    def average_generation_time(self) -> float:
        """Get average generation time."""
        if self.usage_count == 0:
            return 0.0
        return self.total_generation_time / self.usage_count


class FailoverStrategy:
    """Failover strategy for handling model failures."""
    
    def __init__(self):
        """Initialize failover strategy."""
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
        """Get alternative model for failover."""
        if failed_model in self.model_alternatives:
            for alternative in self.model_alternatives[failed_model]:
                if alternative not in self.blacklisted_models:
                    return alternative
        return None
    
    def record_failure(self, model_name: str) -> None:
        """Record a model failure."""
        self.failure_counts[model_name] = self.failure_counts.get(model_name, 0) + 1
        
        # Blacklist model after 3 failures
        if self.failure_counts[model_name] >= 3:
            self.blacklisted_models.add(model_name)
            logger.warning(f"Model {model_name} blacklisted after 3 failures")
    
    def reset_model(self, model_name: str) -> None:
        """Reset failure count for a model."""
        self.failure_counts[model_name] = 0
        self.blacklisted_models.discard(model_name)


class BatchProcessor:
    """Process generation jobs in batches with intelligent scheduling."""
    
    def __init__(
        self,
        job_queue: JobQueue,
        resource_manager: ResourceManager,
        max_workers: int = 2,
        max_loaded_models: int = 3,
        model_idle_timeout: int = 300,  # 5 minutes
        enable_warm_up: bool = True,
        enable_failover: bool = True,
        checkpoint_dir: Optional[Path] = None,
    ):
        """Initialize batch processor."""
        self.job_queue = job_queue
        self.resource_manager = resource_manager
        self.max_workers = max_workers
        self.max_loaded_models = max_loaded_models
        self.model_idle_timeout = model_idle_timeout
        self.enable_warm_up = enable_warm_up
        self.enable_failover = enable_failover
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        
        # Model instances
        self.loaded_models: Dict[str, ModelInstance] = {}
        self.model_lock = threading.RLock()
        
        # Failover strategy
        self.failover_strategy = FailoverStrategy()
        
        # Worker pool
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Processing state
        self.processing = False
        self.processing_thread: Optional[threading.Thread] = None
        self.active_jobs: Dict[str, GenerationJob] = {}
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "total_succeeded": 0,
            "total_failed": 0,
            "total_retried": 0,
            "model_switches": 0,
            "failovers": 0,
        }
        
        # Start resource monitoring
        self.resource_manager.start_monitoring()
    
    def load_model(
        self,
        model_name: str,
        gpu_index: Optional[int] = None,
        warm_up: bool = True,
    ) -> Optional[ModelInstance]:
        """Load a model with optional warm-up."""
        with self.model_lock:
            # Check if already loaded
            if model_name in self.loaded_models:
                instance = self.loaded_models[model_name]
                instance.last_used = datetime.now()
                return instance
            
            # Check if we need to unload models
            if len(self.loaded_models) >= self.max_loaded_models:
                self._unload_least_recently_used()
            
            # Load model
            try:
                if not MODEL_FACTORY_AVAILABLE:
                    logger.error("Model factory not available")
                    return None
                
                logger.info(f"Loading model {model_name}...")
                
                # Create model instance
                model = ModelFactory.create_model(model_name)
                
                # Assign to GPU
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
                
                # Warm up model
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
        """Warm up a model with a test generation."""
        logger.info(f"Warming up model {instance.model_name}...")
        
        try:
            # Simple warm-up generation
            test_prompt = "a simple cube"
            start_time = time.time()
            
            with instance.model as model:
                _ = model.generate_from_text(
                    prompt=test_prompt,
                    num_inference_steps=1,  # Minimal steps for warm-up
                )
            
            warm_up_time = time.time() - start_time
            logger.info(f"Model {instance.model_name} warmed up in {warm_up_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"Failed to warm up model {instance.model_name}: {e}")
    
    def _unload_least_recently_used(self) -> None:
        """Unload the least recently used model."""
        if not self.loaded_models:
            return
        
        # Find LRU model
        lru_model = min(
            self.loaded_models.values(),
            key=lambda m: m.last_used,
        )
        
        self.unload_model(lru_model.model_name)
    
    def unload_model(self, model_name: str, cool_down: bool = True) -> None:
        """Unload a model with optional cool-down."""
        with self.model_lock:
            if model_name not in self.loaded_models:
                return
            
            instance = self.loaded_models[model_name]
            
            # Cool down model
            if cool_down and not instance.is_cooling_down:
                instance.is_cooling_down = True
                self._cool_down_model(instance)
            
            # Clean up
            try:
                if hasattr(instance.model, 'cleanup'):
                    instance.model.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up model {model_name}: {e}")
            
            # Release GPU resources
            if instance.gpu_index is not None:
                self.resource_manager.release_job_from_gpu(f"model_{model_name}")
                self.resource_manager.unload_model(model_name)
            
            del self.loaded_models[model_name]
            self.stats["model_switches"] += 1
            
            logger.info(f"Model {model_name} unloaded")
    
    def _cool_down_model(self, instance: ModelInstance) -> None:
        """Cool down a model before unloading."""
        logger.info(f"Cooling down model {instance.model_name}...")
        
        # Save model statistics
        if instance.usage_count > 0:
            avg_time = instance.average_generation_time
            logger.info(
                f"Model {instance.model_name} stats: "
                f"{instance.usage_count} uses, {avg_time:.2f}s avg generation time"
            )
        
        # Clear any cached data
        try:
            if hasattr(instance.model, 'clear_cache'):
                instance.model.clear_cache()
        except Exception as e:
            logger.warning(f"Failed to clear cache for {instance.model_name}: {e}")
    
    def process_job(self, job: GenerationJob) -> bool:
        """Process a single job."""
        logger.info(f"Processing job {job.id}: {job.prompt[:50]}...")
        
        try:
            # Load model
            instance = self.load_model(job.model_name)
            if not instance:
                # Try failover
                if self.enable_failover:
                    alternative = self.failover_strategy.get_alternative_model(job.model_name)
                    if alternative:
                        logger.info(f"Failover: Using {alternative} instead of {job.model_name}")
                        job.model_name = alternative
                        instance = self.load_model(alternative)
                        self.stats["failovers"] += 1
                
                if not instance:
                    raise RuntimeError(f"Failed to load model {job.model_name}")
            
            # Update job status
            self.job_queue.update_job(
                job.id,
                status=JobStatus.RUNNING,
                assigned_device=f"GPU-{instance.gpu_index}" if instance.gpu_index else "CPU",
            )
            
            # Generate
            start_time = time.time()
            output_path = self._generate_with_model(instance, job)
            generation_time = time.time() - start_time
            
            # Update instance stats
            instance.usage_count += 1
            instance.total_generation_time += generation_time
            instance.last_used = datetime.now()
            
            # Update job
            self.job_queue.update_job(
                job.id,
                status=JobStatus.COMPLETED,
                output_path=str(output_path),
                generation_time_seconds=generation_time,
                progress_percent=100.0,
            )
            
            # Update stats
            self.stats["total_succeeded"] += 1
            
            # Reset failure count on success
            self.failover_strategy.reset_model(job.model_name)
            
            logger.info(f"Job {job.id} completed successfully in {generation_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            
            # Update job
            self.job_queue.update_job(
                job.id,
                status=JobStatus.FAILED,
                error_message=str(e),
            )
            
            # Update stats
            self.stats["total_failed"] += 1
            
            # Record failure
            self.failover_strategy.record_failure(job.model_name)
            
            return False
    
    def _generate_with_model(
        self,
        instance: ModelInstance,
        job: GenerationJob,
    ) -> Path:
        """Generate output using a model instance."""
        output_dir = Path("outputs") / job.model_name / job.id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress callback
        def progress_callback(percent: float):
            self.job_queue.update_job(job.id, progress_percent=percent)
        
        # Generate
        try:
            with instance.model as model:
                result = model.generate_from_text(
                    prompt=job.prompt,
                    output_path=output_dir,
                    **job.config,
                )
            
            # Save output
            output_file = output_dir / f"{job.id}.{job.output_format}"
            if hasattr(result, 'save'):
                result.save(output_file)
            
            return output_file
        finally:
            # Ensure cleanup even on error
            if hasattr(instance.model, '__exit__'):
                pass  # Already handled by context manager
            else:
                # Manual cleanup if not a context manager
                if hasattr(instance.model, 'cleanup'):
                    try:
                        instance.model.cleanup()
                    except Exception as e:
                        logger.warning(f"Error during model cleanup: {e}")
    
    def process_batch(self, batch_size: int = 5) -> List[GenerationJob]:
        """Process a batch of jobs."""
        processed = []
        
        for _ in range(batch_size):
            # Get next job
            job = self.job_queue.get_next_job()
            if not job:
                break
            
            # Process job
            self.active_jobs[job.id] = job
            success = self.process_job(job)
            del self.active_jobs[job.id]
            
            if success:
                processed.append(job)
            
            self.stats["total_processed"] += 1
        
        return processed
    
    def start_processing(self) -> None:
        """Start batch processing."""
        if not self.processing:
            self.processing = True
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
            )
            self.processing_thread.start()
            logger.info("Batch processing started")
    
    def stop_processing(self) -> None:
        """Stop batch processing."""
        if self.processing:
            self.processing = False
            if self.processing_thread:
                self.processing_thread.join(timeout=10)
            
            # Unload all models
            for model_name in list(self.loaded_models.keys()):
                self.unload_model(model_name)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Stop resource monitoring
            self.resource_manager.stop_monitoring()
            
            logger.info("Batch processing stopped")
    
    def _processing_loop(self) -> None:
        """Main processing loop."""
        while self.processing:
            try:
                # Check for idle models
                self._check_idle_models()
                
                # Get next job
                job = self.job_queue.get_next_job()
                
                if job:
                    # Submit job to executor
                    future = self.executor.submit(self.process_job, job)
                    
                    # Handle timeout
                    try:
                        future.result(timeout=600)  # 10 minute timeout
                    except TimeoutError:
                        logger.error(f"Job {job.id} timed out")
                        self.job_queue.update_job(
                            job.id,
                            status=JobStatus.FAILED,
                            error_message="Generation timed out",
                        )
                else:
                    # No jobs, sleep briefly
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(5)
    
    def _check_idle_models(self) -> None:
        """Check for and unload idle models."""
        with self.model_lock:
            for model_name, instance in list(self.loaded_models.items()):
                if instance.idle_time > self.model_idle_timeout:
                    if not instance.is_warming_up and not instance.is_cooling_down:
                        logger.info(f"Unloading idle model {model_name}")
                        self.unload_model(model_name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status."""
        return {
            "processing": self.processing,
            "loaded_models": list(self.loaded_models.keys()),
            "active_jobs": list(self.active_jobs.keys()),
            "stats": self.stats,
            "resource_summary": self.resource_manager.get_resource_summary(),
            "failover_blacklist": list(self.failover_strategy.blacklisted_models),
        }