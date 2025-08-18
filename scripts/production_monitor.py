import argparse
import json
import logging
import logging.handlers
import os
import pickle
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import psutil
import yaml
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError as e:
    print(f"Warning: PyTorch import error: {e}")
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
LOG_DIR = Path("/mnt/datadrive_m2/dream-cad/logs")
LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = Path("/mnt/datadrive_m2/dream-cad/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
QUEUE_FILE = Path("/mnt/datadrive_m2/dream-cad/generation_queue.json")
def setup_logging(name: str = "mvdream") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_file = LOG_DIR / f"{name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
logger = setup_logging("mvdream.production")
@dataclass
class GPUMetrics:
    timestamp: str
    gpu_name: str
    temperature_c: float
    utilization_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    power_draw_w: float
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
@dataclass
class SystemMetrics:
    timestamp: str
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
@dataclass
class GenerationJob:
    id: str
    prompt: str
    config: dict[str, Any]
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
class GPUMonitor:
    def __init__(self, interval: int = 30, alert_threshold: float = 90.0):
        self.interval = interval
        self.alert_threshold = alert_threshold
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_log = LOG_DIR / "gpu_metrics.jsonl"
        self.alerts_log = LOG_DIR / "alerts.log"
        self.alert_logger = setup_logging("mvdream.alerts")
    def get_gpu_metrics(self) -> Optional[GPUMetrics]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 6:
                memory_used = float(parts[3]) / 1024
                memory_total = float(parts[4]) / 1024
                return GPUMetrics(
                    timestamp=datetime.now().isoformat(),
                    gpu_name=parts[0],
                    temperature_c=float(parts[1]),
                    utilization_percent=float(parts[2]),
                    memory_used_gb=memory_used,
                    memory_total_gb=memory_total,
                    memory_percent=(memory_used / memory_total) * 100,
                    power_draw_w=float(parts[5]) if parts[5] != "[N/A]" else 0.0,
                )
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {e}")
            return None
    def get_system_metrics(self) -> SystemMetrics:
        cpu_percent = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("/mnt/datadrive_m2")
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            ram_used_gb=ram.used / (1024**3),
            ram_total_gb=ram.total / (1024**3),
            ram_percent=ram.percent,
            disk_used_gb=disk.used / (1024**3),
            disk_total_gb=disk.total / (1024**3),
            disk_percent=disk.percent,
        )
    def check_alerts(self, gpu_metrics: GPUMetrics, sys_metrics: SystemMetrics) -> None:
        alerts = []
        if gpu_metrics.memory_percent > self.alert_threshold:
            alerts.append(
                f"HIGH GPU MEMORY: {gpu_metrics.memory_percent:.1f}% "
                f"({gpu_metrics.memory_used_gb:.1f}/{gpu_metrics.memory_total_gb:.1f} GB)"
            )
        if gpu_metrics.temperature_c > 83:
            alerts.append(f"HIGH GPU TEMPERATURE: {gpu_metrics.temperature_c}°C")
        if sys_metrics.ram_percent > self.alert_threshold:
            alerts.append(
                f"HIGH RAM USAGE: {sys_metrics.ram_percent:.1f}% "
                f"({sys_metrics.ram_used_gb:.1f}/{sys_metrics.ram_total_gb:.1f} GB)"
            )
        if sys_metrics.disk_percent > 90:
            alerts.append(
                f"LOW DISK SPACE: {sys_metrics.disk_percent:.1f}% used "
                f"({sys_metrics.disk_used_gb:.1f}/{sys_metrics.disk_total_gb:.1f} GB)"
            )
        for alert in alerts:
            self.alert_logger.warning(alert)
            logger.warning(f"ALERT: {alert}")
    def monitor_loop(self) -> None:
        logger.info(f"Starting GPU monitoring (interval: {self.interval}s)")
        while self.monitoring:
            try:
                gpu_metrics = self.get_gpu_metrics()
                sys_metrics = self.get_system_metrics()
                if gpu_metrics:
                    with self.metrics_log.open("a") as f:
                        f.write(json.dumps(gpu_metrics.to_dict()) + "\n")
                    self.check_alerts(gpu_metrics, sys_metrics)
                    logger.debug(
                        f"GPU: {gpu_metrics.temperature_c}°C, "
                        f"{gpu_metrics.memory_percent:.1f}% VRAM, "
                        f"{gpu_metrics.utilization_percent:.1f}% util | "
                        f"System: {sys_metrics.cpu_percent:.1f}% CPU, "
                        f"{sys_metrics.ram_percent:.1f}% RAM"
                    )
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.interval)
    def start(self) -> None:
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("GPU monitoring started")
    def stop(self) -> None:
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            logger.info("GPU monitoring stopped")
class CheckpointManager:
    def __init__(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
    def save_checkpoint(
        self,
        job_id: str,
        step: int,
        state_dict: dict[str, Any],
        metadata: dict[str, Any],
    ) -> Path:
        checkpoint_path = self.checkpoint_dir / f"{job_id}_step_{step}.ckpt"
        checkpoint = {
            "job_id": job_id,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "state_dict": state_dict,
            "metadata": metadata,
        }
        with checkpoint_path.open("wb") as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        self._cleanup_old_checkpoints(job_id, keep=3)
        return checkpoint_path
    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        with checkpoint_path.open("rb") as f:
            checkpoint = pickle.load(f)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint
    def get_latest_checkpoint(self, job_id: str) -> Optional[Path]:
        pattern = f"{job_id}_step_*.ckpt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
        return checkpoints[-1]
    def _cleanup_old_checkpoints(self, job_id: str, keep: int = 3) -> None:
        pattern = f"{job_id}_step_*.ckpt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        if len(checkpoints) > keep:
            checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
            for checkpoint in checkpoints[:-keep]:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")
class GenerationQueue:
    def __init__(self, queue_file: Path = QUEUE_FILE):
        self.queue_file = queue_file
        self.jobs: list[GenerationJob] = []
        self.processing = False
        self.current_job: Optional[GenerationJob] = None
        self.load_queue()
    def load_queue(self) -> None:
        if self.queue_file.exists():
            try:
                with self.queue_file.open() as f:
                    data = json.load(f)
                    self.jobs = [GenerationJob(**job) for job in data.get("jobs", [])]
                logger.info(f"Loaded {len(self.jobs)} jobs from queue")
            except Exception as e:
                logger.error(f"Failed to load queue: {e}")
                self.jobs = []
    def save_queue(self) -> None:
        try:
            data = {"jobs": [job.to_dict() for job in self.jobs]}
            with self.queue_file.open("w") as f:
                json.dump(data, f, indent=2)
            logger.debug("Queue saved")
        except Exception as e:
            logger.error(f"Failed to save queue: {e}")
    def add_job(self, prompt: str, config: dict[str, Any]) -> GenerationJob:
        job = GenerationJob(
            id=f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.jobs)}",
            prompt=prompt,
            config=config,
            status="pending",
            created_at=datetime.now().isoformat(),
        )
        self.jobs.append(job)
        self.save_queue()
        logger.info(f"Added job {job.id}: {prompt[:50]}...")
        return job
    def get_next_job(self) -> Optional[GenerationJob]:
        for job in self.jobs:
            if job.status == "pending":
                return job
        return None
    def update_job_status(
        self,
        job_id: str,
        status: str,
        **kwargs: Any,
    ) -> None:
        for job in self.jobs:
            if job.id == job_id:
                job.status = status
                if status == "running":
                    job.started_at = datetime.now().isoformat()
                elif status in ["completed", "failed"]:
                    job.completed_at = datetime.now().isoformat()
                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                self.save_queue()
                logger.info(f"Updated job {job_id} status: {status}")
                break
    def get_queue_status(self) -> dict[str, Any]:
        status_counts = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
        }
        for job in self.jobs:
            status_counts[job.status] = status_counts.get(job.status, 0) + 1
        return {
            "total_jobs": len(self.jobs),
            "status_counts": status_counts,
            "current_job": self.current_job.id if self.current_job else None,
        }
class ProductionManager:
    def __init__(self):
        self.gpu_monitor = GPUMonitor()
        self.checkpoint_manager = CheckpointManager()
        self.queue = GenerationQueue()
        self.running = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    def _signal_handler(self, signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
    def process_job(self, job: GenerationJob) -> None:
        logger.info(f"Processing job {job.id}: {job.prompt}")
        try:
            self.queue.update_job_status(job.id, "running")
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint(job.id)
            start_step = 0
            if latest_checkpoint:
                logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
                checkpoint = self.checkpoint_manager.load_checkpoint(latest_checkpoint)
                start_step = checkpoint["step"]
            total_steps = job.config.get("num_inference_steps", 50)
            checkpoint_interval = 1000
            for step in range(start_step, total_steps):
                time.sleep(0.1)
                if step > 0 and step % checkpoint_interval == 0:
                    state_dict = {"step": step}
                    metadata = {"prompt": job.prompt, "config": job.config}
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        job.id, step, state_dict, metadata
                    )
                    self.queue.update_job_status(
                        job.id, "running", checkpoint_path=str(checkpoint_path)
                    )
                if step % 10 == 0:
                    logger.debug(f"Job {job.id}: Step {step}/{total_steps}")
            output_path = f"/mnt/datadrive_m2/dream-cad/outputs/{job.id}"
            self.queue.update_job_status(
                job.id, "completed", output_path=output_path
            )
            logger.info(f"Completed job {job.id}")
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            self.queue.update_job_status(
                job.id, "failed", error_message=str(e)
            )
    def run_queue_processor(self) -> None:
        logger.info("Starting queue processor")
        while self.running:
            job = self.queue.get_next_job()
            if job:
                self.queue.current_job = job
                self.process_job(job)
                self.queue.current_job = None
            else:
                time.sleep(5)
    def start(self) -> None:
        logger.info("Starting production manager")
        self.running = True
        self.gpu_monitor.start()
        queue_thread = threading.Thread(target=self.run_queue_processor, daemon=True)
        queue_thread.start()
        logger.info("Production manager started")
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()
    def shutdown(self) -> None:
        logger.info("Shutting down production manager")
        self.running = False
        self.gpu_monitor.stop()
        logger.info("Production manager stopped")
    def show_status(self) -> None:
        print("\n" + "=" * 60)
        print("MVDream Production Status")
        print("=" * 60)
        gpu_metrics = self.gpu_monitor.get_gpu_metrics()
        if gpu_metrics:
            print(f"\n📊 GPU Status ({gpu_metrics.gpu_name}):")
            print(f"  Temperature: {gpu_metrics.temperature_c}°C")
            print(f"  VRAM: {gpu_metrics.memory_used_gb:.1f}/{gpu_metrics.memory_total_gb:.1f} GB "
                  f"({gpu_metrics.memory_percent:.1f}%)")
            print(f"  Utilization: {gpu_metrics.utilization_percent:.1f}%")
            print(f"  Power: {gpu_metrics.power_draw_w:.1f}W")
        sys_metrics = self.gpu_monitor.get_system_metrics()
        print(f"\n💻 System Status:")
        print(f"  CPU: {sys_metrics.cpu_percent:.1f}%")
        print(f"  RAM: {sys_metrics.ram_used_gb:.1f}/{sys_metrics.ram_total_gb:.1f} GB "
              f"({sys_metrics.ram_percent:.1f}%)")
        print(f"  Disk: {sys_metrics.disk_used_gb:.1f}/{sys_metrics.disk_total_gb:.1f} GB "
              f"({sys_metrics.disk_percent:.1f}%)")
        queue_status = self.queue.get_queue_status()
        print(f"\n📋 Queue Status:")
        print(f"  Total Jobs: {queue_status['total_jobs']}")
        for status, count in queue_status["status_counts"].items():
            print(f"    {status.capitalize()}: {count}")
        if queue_status["current_job"]:
            print(f"  Currently Processing: {queue_status['current_job']}")
        print("=" * 60 + "\n")
def main():
    parser = argparse.ArgumentParser(description="MVDream Production Monitor")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    start_parser = subparsers.add_parser("start", help="Start production manager")
    monitor_parser = subparsers.add_parser("monitor", help="Start monitoring only")
    monitor_parser.add_argument(
        "--interval", type=int, default=30, help="Monitoring interval in seconds"
    )
    status_parser = subparsers.add_parser("status", help="Show system status")
    queue_parser = subparsers.add_parser("queue", help="Queue management")
    queue_subparsers = queue_parser.add_subparsers(dest="queue_command")
    add_parser = queue_subparsers.add_parser("add", help="Add job to queue")
    add_parser.add_argument("prompt", help="Generation prompt")
    add_parser.add_argument("--config", type=Path, help="Config file path")
    list_parser = queue_subparsers.add_parser("list", help="List queue jobs")
    recover_parser = subparsers.add_parser("recover", help="Recover from checkpoint")
    recover_parser.add_argument("job_id", help="Job ID to recover")
    args = parser.parse_args()
    if args.command == "start":
        manager = ProductionManager()
        manager.start()
    elif args.command == "monitor":
        monitor = GPUMonitor(interval=args.interval)
        monitor.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop()
    elif args.command == "status":
        manager = ProductionManager()
        manager.show_status()
    elif args.command == "queue":
        queue = GenerationQueue()
        if args.queue_command == "add":
            config = {}
            if args.config and args.config.exists():
                with args.config.open() as f:
                    config = yaml.safe_load(f)
            job = queue.add_job(args.prompt, config)
            print(f"Added job: {job.id}")
        elif args.queue_command == "list":
            status = queue.get_queue_status()
            print(f"Queue: {status['total_jobs']} jobs")
            for job in queue.jobs:
                print(f"  [{job.status}] {job.id}: {job.prompt[:50]}...")
    elif args.command == "recover":
        manager = CheckpointManager()
        checkpoint_path = manager.get_latest_checkpoint(args.job_id)
        if checkpoint_path:
            print(f"Found checkpoint: {checkpoint_path}")
            checkpoint = manager.load_checkpoint(checkpoint_path)
            print(f"Step: {checkpoint['step']}")
            print(f"Timestamp: {checkpoint['timestamp']}")
        else:
            print(f"No checkpoint found for job {args.job_id}")
    else:
        parser.print_help()
if __name__ == "__main__":
    main()