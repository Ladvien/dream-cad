import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
import numpy as np
from .job_queue import GenerationJob, JobStatus, JobQueue
logger = logging.getLogger(__name__)
@dataclass
class ModelPerformance:
    model_name: str
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    average_generation_time: float = 0.0
    median_generation_time: float = 0.0
    min_generation_time: float = float('inf')
    max_generation_time: float = 0.0
    total_gpu_hours: float = 0.0
    average_memory_usage_gb: float = 0.0
    success_rate: float = 0.0
    def update(self, job: GenerationJob) -> None:
        self.total_jobs += 1
        if job.status == JobStatus.COMPLETED:
            self.successful_jobs += 1
            if job.generation_time_seconds and job.generation_time_seconds > 0:
                if job.generation_time_seconds < self.min_generation_time:
                    self.min_generation_time = job.generation_time_seconds
                if job.generation_time_seconds > self.max_generation_time:
                    self.max_generation_time = job.generation_time_seconds
                self.total_gpu_hours += job.generation_time_seconds / 3600
        elif job.status == JobStatus.FAILED:
            self.failed_jobs += 1
        if self.total_jobs > 0:
            self.success_rate = self.successful_jobs / self.total_jobs
        else:
            self.success_rate = 0.0
@dataclass 
class QueueMetrics:
    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_cancelled: int = 0
    total_pending: int = 0
    total_running: int = 0
    average_wait_time: float = 0.0
    average_processing_time: float = 0.0
    throughput_per_hour: float = 0.0
    queue_efficiency: float = 0.0
    peak_queue_size: int = 0
    total_gpu_hours: float = 0.0
    estimated_cost_usd: float = 0.0
class QueueAnalytics:
    def __init__(
        self,
        job_queue: JobQueue,
        analytics_dir: Path = Path("analytics"),
        gpu_cost_per_hour: float = 0.5,
    ):
        self.job_queue = job_queue
        self.analytics_dir = analytics_dir
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.analytics_dir.mkdir(exist_ok=True)
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.hourly_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.user_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    def analyze_queue(self) -> QueueMetrics:
        metrics = QueueMetrics()
        all_jobs = list(self.job_queue.jobs.values())
        for job in all_jobs:
            if job.status == JobStatus.COMPLETED:
                metrics.total_completed += 1
            elif job.status == JobStatus.FAILED:
                metrics.total_failed += 1
            elif job.status == JobStatus.CANCELLED:
                metrics.total_cancelled += 1
            elif job.status == JobStatus.PENDING:
                metrics.total_pending += 1
            elif job.status == JobStatus.RUNNING:
                metrics.total_running += 1
        metrics.total_submitted = len(all_jobs)
        wait_times = []
        processing_times = []
        for job in all_jobs:
            if job.queued_at and job.started_at:
                wait_time = (
                    datetime.fromisoformat(job.started_at) -
                    datetime.fromisoformat(job.queued_at)
                ).total_seconds()
                wait_times.append(wait_time)
            if job.generation_time_seconds > 0:
                processing_times.append(job.generation_time_seconds)
                metrics.total_gpu_hours += job.generation_time_seconds / 3600
        if wait_times:
            metrics.average_wait_time = np.mean(wait_times)
        if processing_times:
            metrics.average_processing_time = np.mean(processing_times)
        if all_jobs:
            earliest = min(
                datetime.fromisoformat(job.created_at)
                for job in all_jobs
            )
            time_span_hours = (datetime.now() - earliest).total_seconds() / 3600
            if time_span_hours > 0:
                metrics.throughput_per_hour = metrics.total_completed / time_span_hours
        if metrics.total_submitted > 0:
            metrics.queue_efficiency = metrics.total_completed / metrics.total_submitted
        metrics.estimated_cost_usd = metrics.total_gpu_hours * self.gpu_cost_per_hour
        metrics.peak_queue_size = max(
            len(self.job_queue.get_jobs_by_status(JobStatus.QUEUED)),
            metrics.total_pending,
        )
        return metrics
    def analyze_models(self) -> Dict[str, ModelPerformance]:
        self.model_performance.clear()
        for job in self.job_queue.jobs.values():
            model_name = job.model_name
            if model_name not in self.model_performance:
                self.model_performance[model_name] = ModelPerformance(model_name)
            self.model_performance[model_name].update(job)
        for model_name, perf in self.model_performance.items():
            times = [
                job.generation_time_seconds
                for job in self.job_queue.jobs.values()
                if job.model_name == model_name and job.generation_time_seconds > 0
            ]
            if times:
                perf.average_generation_time = np.mean(times)
                perf.median_generation_time = np.median(times)
            memory_usages = [
                job.gpu_memory_used_gb
                for job in self.job_queue.jobs.values()
                if job.model_name == model_name and job.gpu_memory_used_gb > 0
            ]
            if memory_usages:
                perf.average_memory_usage_gb = np.mean(memory_usages)
        return self.model_performance
    def analyze_time_series(self) -> Dict[str, Dict[str, int]]:
        self.hourly_stats.clear()
        for job in self.job_queue.jobs.values():
            created_hour = datetime.fromisoformat(job.created_at).replace(
                minute=0, second=0, microsecond=0
            ).isoformat()
            self.hourly_stats[created_hour]["submitted"] += 1
            if job.status == JobStatus.COMPLETED:
                self.hourly_stats[created_hour]["completed"] += 1
            elif job.status == JobStatus.FAILED:
                self.hourly_stats[created_hour]["failed"] += 1
        return dict(self.hourly_stats)
    def analyze_users(self) -> Dict[str, Dict[str, int]]:
        self.user_stats.clear()
        for job in self.job_queue.jobs.values():
            if job.user_id:
                self.user_stats[job.user_id]["total_jobs"] += 1
                if job.status == JobStatus.COMPLETED:
                    self.user_stats[job.user_id]["successful_jobs"] += 1
                elif job.status == JobStatus.FAILED:
                    self.user_stats[job.user_id]["failed_jobs"] += 1
                if job.generation_time_seconds > 0:
                    self.user_stats[job.user_id]["total_gpu_seconds"] += int(
                        job.generation_time_seconds
                    )
        return dict(self.user_stats)
    def get_job_dependencies(self) -> Dict[str, List[str]]:
        dependencies = {}
        for job in self.job_queue.jobs.values():
            if job.depends_on:
                dependencies[job.id] = job.depends_on
        return dependencies
    def predict_completion_time(self, job_id: str) -> Optional[datetime]:
        job = self.job_queue.get_job(job_id)
        if not job:
            return None
        if job.status == JobStatus.COMPLETED:
            return datetime.fromisoformat(job.completed_at) if job.completed_at else None
        if job.status == JobStatus.RUNNING:
            return job.estimate_completion_time()
        metrics = self.analyze_queue()
        model_perf = self.model_performance.get(job.model_name)
        if not model_perf:
            return None
        jobs_ahead = sum(
            1 for j in self.job_queue.jobs.values()
            if j.status in [JobStatus.QUEUED, JobStatus.RUNNING]
            and j.created_at < job.created_at
        )
        estimated_wait = jobs_ahead * model_perf.average_generation_time
        return datetime.now() + timedelta(seconds=estimated_wait)
    def generate_report(self) -> Dict[str, Any]:
        metrics = self.analyze_queue()
        model_performance = self.analyze_models()
        time_series = self.analyze_time_series()
        user_stats = self.analyze_users()
        report = {
            "generated_at": datetime.now().isoformat(),
            "queue_metrics": asdict(metrics),
            "model_performance": {
                name: asdict(perf)
                for name, perf in model_performance.items()
            },
            "time_series": time_series,
            "user_statistics": user_stats,
            "top_users": self._get_top_users(user_stats, 10),
            "busiest_hours": self._get_busiest_hours(time_series, 5),
            "model_recommendations": self._get_model_recommendations(model_performance),
        }
        report_file = self.analytics_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with report_file.open("w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Analytics report saved to {report_file}")
        return report
    def _get_top_users(
        self,
        user_stats: Dict[str, Dict[str, int]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        sorted_users = sorted(
            user_stats.items(),
            key=lambda x: x[1]["total_jobs"],
            reverse=True,
        )[:limit]
        return [
            {
                "user_id": user_id,
                "stats": stats,
            }
            for user_id, stats in sorted_users
        ]
    def _get_busiest_hours(
        self,
        time_series: Dict[str, Dict[str, int]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        sorted_hours = sorted(
            time_series.items(),
            key=lambda x: x[1]["submitted"],
            reverse=True,
        )[:limit]
        return [
            {
                "hour": hour,
                "stats": stats,
            }
            for hour, stats in sorted_hours
        ]
    def _get_model_recommendations(
        self,
        model_performance: Dict[str, ModelPerformance],
    ) -> Dict[str, str]:
        recommendations = {}
        for model_name, perf in model_performance.items():
            if perf.success_rate < 0.5:
                recommendations[model_name] = (
                    f"Low success rate ({perf.success_rate:.1%}). "
                    "Consider using alternative models or adjusting parameters."
                )
            elif perf.average_generation_time > 300:
                recommendations[model_name] = (
                    f"High generation time ({perf.average_generation_time:.1f}s). "
                    "Consider using faster models for time-sensitive tasks."
                )
            elif perf.average_memory_usage_gb > 20:
                recommendations[model_name] = (
                    f"High memory usage ({perf.average_memory_usage_gb:.1f}GB). "
                    "May cause OOM errors on smaller GPUs."
                )
            else:
                recommendations[model_name] = "Performance within normal parameters."
        return recommendations
    def plot_metrics(self) -> None:
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot generation")
            return
        metrics = self.analyze_queue()
        model_performance = self.analyze_models()
        time_series = self.analyze_time_series()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax1 = axes[0, 0]
        statuses = ['Completed', 'Failed', 'Pending', 'Running']
        counts = [
            metrics.total_completed,
            metrics.total_failed,
            metrics.total_pending,
            metrics.total_running,
        ]
        ax1.pie(counts, labels=statuses, autopct='%1.1f%%')
        ax1.set_title('Job Status Distribution')
        ax2 = axes[0, 1]
        if model_performance:
            models = list(model_performance.keys())
            success_rates = [p.success_rate for p in model_performance.values()]
            ax2.bar(models, success_rates)
            ax2.set_title('Model Success Rates')
            ax2.set_ylabel('Success Rate')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
        ax3 = axes[1, 0]
        if time_series:
            hours = sorted(time_series.keys())
            submitted = [time_series[h]["submitted"] for h in hours]
            completed = [time_series[h]["completed"] for h in hours]
            ax3.plot(range(len(hours)), submitted, label='Submitted', marker='o')
            ax3.plot(range(len(hours)), completed, label='Completed', marker='s')
            ax3.set_title('Jobs Over Time')
            ax3.set_xlabel('Time (hours)')
            ax3.set_ylabel('Job Count')
            ax3.legend()
        ax4 = axes[1, 1]
        gen_times = [
            job.generation_time_seconds
            for job in self.job_queue.jobs.values()
            if job.generation_time_seconds > 0
        ]
        if gen_times:
            ax4.hist(gen_times, bins=20, edgecolor='black')
            ax4.set_title('Generation Time Distribution')
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Frequency')
        plt.tight_layout()
        plot_file = self.analytics_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Metrics plot saved to {plot_file}")
    def export_csv(self, output_file: Optional[Path] = None) -> Path:
        if output_file is None:
            output_file = self.analytics_dir / f"queue_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        rows = []
        for job in self.job_queue.jobs.values():
            rows.append({
                "id": job.id,
                "prompt": job.prompt[:100],
                "model": job.model_name,
                "status": job.status.value,
                "priority": job.priority.value,
                "created_at": job.created_at,
                "completed_at": job.completed_at or "",
                "generation_time": job.generation_time_seconds,
                "gpu_memory_gb": job.gpu_memory_used_gb,
                "output_path": job.output_path or "",
                "error": job.error_message or "",
            })
        import csv
        with output_file.open("w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        logger.info(f"Queue data exported to {output_file}")
        return output_file