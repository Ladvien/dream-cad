import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from enum import Enum
import time
import uuid

from textual import work
from textual.message import Message

from dream_cad.models.factory import ModelFactory
from dream_cad.models.base import ModelConfig as BaseModelConfig, OutputFormat

class JobStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    GENERATING = "generating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class GenerationJob:
    id: str
    prompt: str
    model_name: str
    output_format: OutputFormat
    status: JobStatus
    progress: float = 0.0
    progress_message: str = ""
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_path: Optional[Path] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

class ProgressUpdate(Message):
    def __init__(self, job_id: str, progress: float, message: str):
        super().__init__()
        self.job_id = job_id
        self.progress = progress
        self.message = message

class JobComplete(Message):
    def __init__(self, job: GenerationJob):
        super().__init__()
        self.job = job

class GenerationWorker:
    def __init__(self, app):
        self.app = app
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.active_jobs: Dict[str, GenerationJob] = {}
        self.model_cache: Dict[str, Any] = {}
        self.cancel_flags: Dict[str, bool] = {}
        
    def create_job(
        self,
        prompt: str,
        model_name: str,
        output_format: OutputFormat = OutputFormat.OBJ
    ) -> GenerationJob:
        job = GenerationJob(
            id=str(uuid.uuid4()),
            prompt=prompt,
            model_name=model_name,
            output_format=output_format,
            status=JobStatus.PENDING
        )
        self.active_jobs[job.id] = job
        return job
        
    async def process_job(self, job: GenerationJob) -> None:
        try:
            self.cancel_flags[job.id] = False
            job.status = JobStatus.LOADING
            job.started_at = datetime.now()
            
            await self._update_progress(job, 0.1, "Initializing...")
            
            # Check if model is available
            if not self.app.is_model_available(job.model_name):
                # Try to download if auto_download is enabled
                if self.app.config.config.models.auto_download:
                    model_info = self.app.get_model_info(job.model_name)
                    if model_info and model_info.get('vram_compatible', False):
                        # Download the model
                        job.status = JobStatus.DOWNLOADING
                        await self._update_progress(job, 0.15, f"Downloading {job.model_name}...")
                        
                        success = await self._download_model(job)
                        if success:
                            # Model downloaded, proceed with generation
                            await self._update_progress(job, 0.3, "Model downloaded, loading...")
                        else:
                            # Download failed, fall back to demo only as last resort
                            if self.app.config.config.models.fallback_to_mock:
                                await self._update_progress(job, 0.2, "Download failed, using demo mode...")
                                await self._generate_mock(job)
                                return
                            else:
                                raise Exception(f"Failed to download model {job.model_name}")
                    else:
                        # Model not compatible, use demo if enabled
                        if self.app.config.config.models.fallback_to_mock:
                            await self._update_progress(job, 0.2, "Model incompatible, using demo mode...")
                            await self._generate_mock(job)
                            return
                        else:
                            raise Exception(f"Model {job.model_name} not compatible with system")
                else:
                    # Auto-download disabled, use demo if enabled
                    if self.app.config.config.models.fallback_to_mock:
                        await self._update_progress(job, 0.2, "Model not available, using demo mode...")
                        await self._generate_mock(job)
                        return
                    else:
                        raise Exception(f"Model {job.model_name} not available and auto-download disabled")
            
            # Model is available (or just downloaded), proceed with real generation
            model = await self._get_or_load_model(job)
            
            if self.cancel_flags.get(job.id):
                job.status = JobStatus.CANCELLED
                return
                
            await self._generate_real(job, model)
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            self.app.logger.error(f"Generation failed for job {job.id}: {e}")
        finally:
            job.completed_at = datetime.now()
            self.app.post_message(JobComplete(job))
            
    async def _get_or_load_model(self, job: GenerationJob):
        if job.model_name in self.model_cache:
            await self._update_progress(job, 0.2, "Using cached model")
            return self.model_cache[job.model_name]
            
        await self._update_progress(job, 0.15, f"Loading {job.model_name}...")
        
        config = BaseModelConfig(
            model_name=job.model_name,
            output_dir=Path(self.app.config.config.output_dir),
            cache_dir=Path(self.app.config.config.cache_dir),
            device="cuda" if self._has_cuda() else "cpu",
            progress_callback=lambda msg: asyncio.create_task(
                self._update_progress(job, job.progress, msg)
            )
        )
        
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            self.executor,
            ModelFactory.create_model,
            job.model_name,
            config
        )
        
        if self.app.config.config.models.cache_models:
            self._manage_cache(job.model_name, model)
            
        await self._update_progress(job, 0.3, "Model loaded")
        return model
        
    async def _generate_real(self, job: GenerationJob, model) -> None:
        job.status = JobStatus.GENERATING
        await self._update_progress(job, 0.4, "Starting generation...")
        
        def progress_callback(step: int, total: int, message: str):
            progress = 0.4 + (step / total) * 0.5
            asyncio.create_task(self._update_progress(job, progress, message))
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            model.generate_from_text,
            job.prompt,
            None,
            {"output_format": job.output_format}
        )
        
        if result.success:
            job.status = JobStatus.SAVING
            await self._update_progress(job, 0.95, "Saving output...")
            
            job.output_path = result.output_path
            job.metadata = result.metadata or {}
            job.metadata["generation_time"] = result.generation_time
            job.metadata["memory_used_gb"] = result.memory_used_gb
            
            job.status = JobStatus.COMPLETED
            await self._update_progress(job, 1.0, "Complete!")
        else:
            raise Exception(result.error_message or "Generation failed")
            
    async def _generate_mock(self, job: GenerationJob) -> None:
        job.status = JobStatus.GENERATING
        await self._update_progress(job, 0.2, "Demo mode - generating placeholder...")
        
        for i in range(5):
            if self.cancel_flags.get(job.id):
                job.status = JobStatus.CANCELLED
                return
            await self._update_progress(job, 0.2 + i * 0.15, f"Processing step {i+1}/5...")
            await asyncio.sleep(0.5)
            
        job.status = JobStatus.SAVING
        await self._update_progress(job, 0.95, "Saving mock output...")
        
        output_dir = Path(self.app.config.config.output_dir) / "demo"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        output_file = output_dir / f"demo_{timestamp}.obj"
        
        vertices, faces = self._generate_simple_mesh(job.prompt)
        self._save_obj(output_file, vertices, faces, job.prompt)
        
        job.output_path = output_file
        job.metadata = {
            "demo_mode": True,
            "generation_time": 2.5
        }
        
        job.status = JobStatus.COMPLETED
        await self._update_progress(job, 1.0, "Demo generation complete!")
        
    def _generate_simple_mesh(self, prompt: str):
        import numpy as np
        
        prompt_lower = prompt.lower()
        
        if "cube" in prompt_lower or "box" in prompt_lower:
            vertices = np.array([
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
            ])
            faces = np.array([
                [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]
            ])
        elif "sphere" in prompt_lower or "ball" in prompt_lower:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            
            vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            faces = []
            for i in range(len(u) - 1):
                for j in range(len(v) - 1):
                    a = i * len(v) + j
                    b = (i + 1) * len(v) + j
                    c = (i + 1) * len(v) + j + 1
                    d = i * len(v) + j + 1
                    faces.extend([[a, b, c], [a, c, d]])
            faces = np.array(faces)
        else:
            vertices = np.array([
                [0, 0, 0], [1, 0, 0], [0.5, 0, 1],
                [0.5, 1, 0.5]
            ])
            faces = np.array([
                [0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]
            ])
            
        return vertices, faces
        
    def _save_obj(self, path: Path, vertices, faces, comment: str = ""):
        with open(path, 'w') as f:
            f.write(f"# DreamCAD Demo Mode\n")
            if comment:
                f.write(f"# Prompt: {comment}\n")
            f.write(f"# Generated: {datetime.now()}\n\n")
            
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")
            
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                
    def _manage_cache(self, model_name: str, model):
        max_cached = self.app.config.config.models.max_cached_models
        
        if len(self.model_cache) >= max_cached:
            oldest = list(self.model_cache.keys())[0]
            old_model = self.model_cache.pop(oldest)
            if hasattr(old_model, 'cleanup'):
                old_model.cleanup()
                
        self.model_cache[model_name] = model
        
    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    async def _download_model(self, job: GenerationJob) -> bool:
        """Download a model with progress tracking"""
        try:
            from ..core.model_detector import ModelDetector, KNOWN_MODELS
            
            detector = ModelDetector(self.app.config)
            model_info = KNOWN_MODELS.get(job.model_name)
            
            if not model_info:
                return False
            
            # Create progress callback
            def download_progress(downloaded: float, total: float):
                if total > 0:
                    percent = downloaded / total
                    job.progress = 0.15 + (percent * 0.15)  # Map to 15-30% of total progress
                    size_mb = total / (1024 * 1024)
                    downloaded_mb = downloaded / (1024 * 1024)
                    job.progress_message = f"Downloading: {downloaded_mb:.1f}/{size_mb:.1f} MB ({percent:.0%})"
                    asyncio.create_task(self._update_progress(job, job.progress, job.progress_message))
            
            # Download the model
            success = await detector.download_model(job.model_name, progress_callback=download_progress)
            
            if success:
                # Update available models in app
                self.app.available_models = await detector.scan_models()
                
            return success
            
        except Exception as e:
            self.app.logger.error(f"Download failed for {job.model_name}: {e}")
            return False
    
    async def _update_progress(self, job: GenerationJob, progress: float, message: str):
        job.progress = progress
        job.progress_message = message
        self.app.post_message(ProgressUpdate(job.id, progress, message))
        
    def cancel_job(self, job_id: str) -> bool:
        if job_id in self.cancel_flags:
            self.cancel_flags[job_id] = True
            return True
        return False
        
    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        return self.active_jobs.get(job_id)
        
    def clear_completed_jobs(self):
        completed = [
            job_id for job_id, job in self.active_jobs.items()
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
        ]
        for job_id in completed:
            del self.active_jobs[job_id]
            self.cancel_flags.pop(job_id, None)