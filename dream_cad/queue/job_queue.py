import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, List, Dict, Set
from queue import PriorityQueue
import logging
logger = logging.getLogger(__name__)
class JobStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"
class JobPriority(Enum):
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0
@dataclass
class ModelRequirements:
    model_name: str
    min_vram_gb: float
    estimated_time_seconds: float
    supports_batch: bool = False
    max_batch_size: int = 1
    requires_gpu: bool = True
    supports_fp16: bool = True
    supports_cpu_offload: bool = False
@dataclass
class GenerationJob:
    id: str
    prompt: str
    model_name: str
    config: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    queued_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    model_requirements: Optional[ModelRequirements] = None
    assigned_device: Optional[str] = None
    memory_allocated_gb: float = 0.0
    output_path: Optional[str] = None
    output_format: str = "glb"
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    depends_on: List[str] = field(default_factory=list)
    batch_id: Optional[str] = None
    batch_position: Optional[int] = None
    progress_percent: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    checkpoint_path: Optional[str] = None
    gpu_memory_used_gb: float = 0.0
    cpu_memory_used_gb: float = 0.0
    generation_time_seconds: float = 0.0
    user_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["priority"] = self.priority.value
        if self.model_requirements:
            data["model_requirements"] = asdict(self.model_requirements)
        return data
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationJob":
        data = data.copy()
        if "status" in data:
            data["status"] = JobStatus(data["status"])
        if "priority" in data:
            data["priority"] = JobPriority(data["priority"])
        if "model_requirements" in data and data["model_requirements"]:
            data["model_requirements"] = ModelRequirements(**data["model_requirements"])
        return cls(**data)
    def can_run(self, completed_jobs: Set[str]) -> bool:
        return all(dep_id in completed_jobs for dep_id in self.depends_on)
    def is_retriable(self) -> bool:
        return self.retry_count < self.max_retries
    def estimate_completion_time(self) -> Optional[datetime]:
        if not self.started_at or self.progress_percent == 0:
            return None
        start_time = datetime.fromisoformat(self.started_at)
        elapsed = (datetime.now() - start_time).total_seconds()
        if self.progress_percent > 0:
            total_estimated = elapsed / (self.progress_percent / 100)
            remaining = total_estimated - elapsed
            return datetime.now() + timedelta(seconds=remaining)
        return None
class JobQueue:
    def __init__(
        self,
        queue_file: Path = Path("queue.json"),
        max_queue_size: int = 1000,
        enable_persistence: bool = True,
    ):
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self.queue_file = queue_file
        self.max_queue_size = max_queue_size
        self.enable_persistence = enable_persistence
        self.pending_queue: PriorityQueue = PriorityQueue()
        self.jobs: Dict[str, GenerationJob] = {}
        self.completed_jobs: Set[str] = set()
        self.model_queues: Dict[str, List[str]] = {}
        self.lock = threading.RLock()
        self.stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_cancelled": 0,
            "model_counts": {},
        }
        if enable_persistence:
            self.load_queue()
    def add_job(
        self,
        prompt: str,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        priority: JobPriority = JobPriority.NORMAL,
        depends_on: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerationJob:
        with self.lock:
            if len(self.jobs) >= self.max_queue_size:
                raise ValueError(f"Queue is full (max {self.max_queue_size} jobs)")
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.stats['total_submitted']}"
            job = GenerationJob(
                id=job_id,
                prompt=prompt,
                model_name=model_name,
                config=config or {},
                priority=priority,
                depends_on=depends_on or [],
                user_id=user_id,
                tags=tags or [],
                **kwargs,
            )
            self.jobs[job_id] = job
            if model_name not in self.model_queues:
                self.model_queues[model_name] = []
            self.model_queues[model_name].append(job_id)
            if job.can_run(self.completed_jobs):
                self._enqueue_job(job)
            self.stats["total_submitted"] += 1
            if model_name not in self.stats["model_counts"]:
                self.stats["model_counts"][model_name] = 0
            self.stats["model_counts"][model_name] += 1
            if self.enable_persistence:
                self.save_queue()
            logger.info(f"Added job {job_id}: {prompt[:50]}... (model: {model_name})")
            return job
    def _enqueue_job(self, job: GenerationJob) -> None:
        priority_tuple = (
            job.priority.value,
            job.created_at,
            job.id,
        )
        self.pending_queue.put((priority_tuple, job.id))
        job.status = JobStatus.QUEUED
        job.queued_at = datetime.now().isoformat()
    def get_next_job(
        self,
        model_name: Optional[str] = None,
        required_vram_gb: Optional[float] = None,
    ) -> Optional[GenerationJob]:
        with self.lock:
            while not self.pending_queue.empty():
                _, job_id = self.pending_queue.get()
                if job_id not in self.jobs:
                    continue
                job = self.jobs[job_id]
                if model_name and job.model_name != model_name:
                    self._enqueue_job(job)
                    continue
                if required_vram_gb and job.model_requirements:
                    if job.model_requirements.min_vram_gb > required_vram_gb:
                        self._enqueue_job(job)
                        continue
                if not job.can_run(self.completed_jobs):
                    self._enqueue_job(job)
                    continue
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now().isoformat()
                if self.enable_persistence:
                    self.save_queue()
                return job
            self._check_dependencies()
            return None
    def _check_dependencies(self) -> None:
        jobs_to_check = list(self.jobs.items())
        for job_id, job in jobs_to_check:
            if job.status == JobStatus.PENDING and job.can_run(self.completed_jobs):
                self._enqueue_job(job)
    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress_percent: Optional[float] = None,
        error_message: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError(f"Job {job_id} not found")
            job = self.jobs[job_id]
            if status:
                job.status = status
                if status == JobStatus.COMPLETED:
                    job.completed_at = datetime.now().isoformat()
                    self.completed_jobs.add(job_id)
                    self.stats["total_completed"] += 1
                    self._check_dependencies()
                elif status == JobStatus.FAILED:
                    job.completed_at = datetime.now().isoformat()
                    self.stats["total_failed"] += 1
                    if job.is_retriable():
                        job.retry_count += 1
                        job.status = JobStatus.RETRYING
                        self._enqueue_job(job)
                        logger.info(f"Retrying job {job_id} (attempt {job.retry_count})")
                elif status == JobStatus.CANCELLED:
                    self.stats["total_cancelled"] += 1
            if progress_percent is not None:
                job.progress_percent = progress_percent
            if error_message is not None:
                job.error_message = error_message
            if output_path is not None:
                job.output_path = output_path
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            if self.enable_persistence:
                self.save_queue()
    def cancel_job(self, job_id: str) -> None:
        self.update_job(job_id, status=JobStatus.CANCELLED)
    def pause_job(self, job_id: str) -> None:
        self.update_job(job_id, status=JobStatus.PAUSED)
    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        with self.lock:
            return self.jobs.get(job_id)
    def get_jobs_by_status(self, status: JobStatus) -> List[GenerationJob]:
        with self.lock:
            return [job for job in self.jobs.values() if job.status == status]
    def get_jobs_by_model(self, model_name: str) -> List[GenerationJob]:
        with self.lock:
            job_ids = self.model_queues.get(model_name, [])
            return [self.jobs[job_id] for job_id in job_ids if job_id in self.jobs]
    def get_queue_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                **self.stats,
                "queue_size": self.pending_queue.qsize(),
                "total_jobs": len(self.jobs),
                "running_jobs": len(self.get_jobs_by_status(JobStatus.RUNNING)),
                "pending_jobs": len(self.get_jobs_by_status(JobStatus.PENDING)),
                "queued_jobs": len(self.get_jobs_by_status(JobStatus.QUEUED)),
            }
    def create_batch(
        self,
        prompts: List[str],
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[GenerationJob]:
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        jobs = []
        for i, prompt in enumerate(prompts):
            job = self.add_job(
                prompt=prompt,
                model_name=model_name,
                config=config,
                batch_id=batch_id,
                batch_position=i,
                **kwargs,
            )
            jobs.append(job)
        logger.info(f"Created batch {batch_id} with {len(jobs)} jobs")
        return jobs
    def save_queue(self) -> None:
        try:
            data = {
                "jobs": {
                    job_id: job.to_dict()
                    for job_id, job in self.jobs.items()
                },
                "completed_jobs": list(self.completed_jobs),
                "stats": self.stats,
                "model_queues": self.model_queues,
            }
            with self.queue_file.open("w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Queue saved to {self.queue_file}")
        except Exception as e:
            logger.error(f"Failed to save queue: {e}")
    def load_queue(self) -> None:
        if not self.queue_file.exists():
            return
        try:
            with self.queue_file.open() as f:
                data = json.load(f)
            for job_id, job_data in data.get("jobs", {}).items():
                job = GenerationJob.from_dict(job_data)
                self.jobs[job_id] = job
                if job.status in [JobStatus.QUEUED, JobStatus.RETRYING]:
                    self._enqueue_job(job)
                elif job.status == JobStatus.RUNNING:
                    job.status = JobStatus.QUEUED
                    self._enqueue_job(job)
            self.completed_jobs = set(data.get("completed_jobs", []))
            self.stats = data.get("stats", self.stats)
            self.model_queues = data.get("model_queues", {})
            self._check_dependencies()
            logger.info(f"Loaded {len(self.jobs)} jobs from queue")
        except Exception as e:
            logger.error(f"Failed to load queue: {e}")
    def clear_completed(self, older_than_hours: int = 24) -> int:
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            removed_count = 0
            jobs_to_remove = []
            for job_id, job in self.jobs.items():
                if job.status == JobStatus.COMPLETED and job.completed_at:
                    completed_time = datetime.fromisoformat(job.completed_at)
                    if completed_time < cutoff_time:
                        jobs_to_remove.append(job_id)
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                self.completed_jobs.discard(job_id)
                removed_count += 1
            if removed_count > 0 and self.enable_persistence:
                self.save_queue()
            logger.info(f"Cleared {removed_count} completed jobs")
            return removed_count