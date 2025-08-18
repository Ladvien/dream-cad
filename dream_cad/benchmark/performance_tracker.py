import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import statistics
import numpy as np
logger = logging.getLogger(__name__)
@dataclass
class PerformanceMetrics:
    model_name: str
    timestamp: str
    avg_generation_time: float
    min_generation_time: float
    max_generation_time: float
    std_generation_time: float
    p50_generation_time: float
    p95_generation_time: float
    p99_generation_time: float
    avg_vram_gb: float
    peak_vram_gb: float
    avg_ram_gb: float
    peak_ram_gb: float
    avg_gpu_temp_c: float
    peak_gpu_temp_c: float
    avg_gpu_utilization: float
    peak_gpu_utilization: float
    successful_generations: int
    failed_generations: int
    success_rate: float
    generations_per_hour: float
    avg_cost_per_generation: float
    total_gpu_hours: float
    total_cost_usd: float
    avg_quality_score: float
    min_quality_score: float
    max_quality_score: float
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
class PerformanceTracker:
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        gpu_cost_per_hour: float = 0.5,
    ):
        self.storage_dir = storage_dir or Path("benchmarks/performance")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.history: Dict[str, List[PerformanceMetrics]] = {}
        self.load_history()
    def load_history(self) -> None:
        history_file = self.storage_dir / "performance_history.json"
        if history_file.exists():
            try:
                with history_file.open() as f:
                    data = json.load(f)
                for model_name, metrics_list in data.items():
                    self.history[model_name] = [
                        PerformanceMetrics(**m) for m in metrics_list
                    ]
                logger.info(f"Loaded performance history for {len(self.history)} models")
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
    def save_history(self) -> None:
        history_file = self.storage_dir / "performance_history.json"
        try:
            data = {}
            for model_name, metrics_list in self.history.items():
                data[model_name] = [m.to_dict() for m in metrics_list]
            with history_file.open("w") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved performance history")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    def calculate_metrics(
        self,
        model_name: str,
        results: List[Any],
    ) -> Optional[PerformanceMetrics]:
        if not results:
            logger.warning(f"No results provided for {model_name}")
            return None
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        if not successful:
            logger.warning(f"No successful results for {model_name}")
            return None
        gen_times = [r.generation_time_seconds for r in successful]
        metrics = PerformanceMetrics(
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            avg_generation_time=statistics.mean(gen_times),
            min_generation_time=min(gen_times),
            max_generation_time=max(gen_times),
            std_generation_time=statistics.stdev(gen_times) if len(gen_times) > 1 else 0,
            p50_generation_time=statistics.median(gen_times),
            p95_generation_time=np.percentile(gen_times, 95),
            p99_generation_time=np.percentile(gen_times, 99),
            successful_generations=len(successful),
            failed_generations=len(failed),
            success_rate=len(successful) / len(results),
            generations_per_hour=3600 / statistics.mean(gen_times) if gen_times else 0,
            avg_vram_gb=0,
            peak_vram_gb=0,
            avg_ram_gb=0,
            peak_ram_gb=0,
            avg_gpu_temp_c=0,
            peak_gpu_temp_c=0,
            avg_gpu_utilization=0,
            peak_gpu_utilization=0,
            avg_cost_per_generation=0,
            total_gpu_hours=0,
            total_cost_usd=0,
            avg_quality_score=0,
            min_quality_score=0,
            max_quality_score=0,
        )
        if any(hasattr(r, 'peak_vram_gb') for r in successful):
            vram_values = [r.peak_vram_gb for r in successful if hasattr(r, 'peak_vram_gb')]
            if vram_values:
                metrics.avg_vram_gb = statistics.mean(vram_values)
                metrics.peak_vram_gb = max(vram_values)
        if any(hasattr(r, 'peak_ram_gb') for r in successful):
            ram_values = [r.peak_ram_gb for r in successful if hasattr(r, 'peak_ram_gb')]
            if ram_values:
                metrics.avg_ram_gb = statistics.mean(ram_values)
                metrics.peak_ram_gb = max(ram_values)
        if any(hasattr(r, 'peak_gpu_temp_c') for r in successful):
            temp_values = [r.peak_gpu_temp_c for r in successful if hasattr(r, 'peak_gpu_temp_c')]
            if temp_values:
                metrics.avg_gpu_temp_c = statistics.mean(temp_values)
                metrics.peak_gpu_temp_c = max(temp_values)
        if any(hasattr(r, 'peak_gpu_utilization') for r in successful):
            util_values = [r.peak_gpu_utilization for r in successful if hasattr(r, 'peak_gpu_utilization')]
            if util_values:
                metrics.avg_gpu_utilization = statistics.mean(util_values)
                metrics.peak_gpu_utilization = max(util_values)
        metrics.total_gpu_hours = sum(gen_times) / 3600
        metrics.total_cost_usd = metrics.total_gpu_hours * self.gpu_cost_per_hour
        metrics.avg_cost_per_generation = metrics.total_cost_usd / len(successful)
        if any(hasattr(r, 'overall_quality_score') for r in successful):
            quality_scores = [r.overall_quality_score for r in successful if hasattr(r, 'overall_quality_score')]
            if quality_scores:
                metrics.avg_quality_score = statistics.mean(quality_scores)
                metrics.min_quality_score = min(quality_scores)
                metrics.max_quality_score = max(quality_scores)
        return metrics
    def track_performance(
        self,
        model_name: str,
        metrics: PerformanceMetrics,
    ) -> None:
        if model_name not in self.history:
            self.history[model_name] = []
        self.history[model_name].append(metrics)
        self.save_history()
        logger.info(f"Tracked performance for {model_name}")
    def get_performance_trend(
        self,
        model_name: str,
        days: int = 30,
    ) -> Dict[str, List[float]]:
        if model_name not in self.history:
            return {}
        cutoff = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m for m in self.history[model_name]
            if datetime.fromisoformat(m.timestamp) > cutoff
        ]
        if not recent_metrics:
            return {}
        trends = {
            "timestamps": [m.timestamp for m in recent_metrics],
            "generation_times": [m.avg_generation_time for m in recent_metrics],
            "success_rates": [m.success_rate for m in recent_metrics],
            "quality_scores": [m.avg_quality_score for m in recent_metrics],
            "vram_usage": [m.peak_vram_gb for m in recent_metrics],
            "costs": [m.avg_cost_per_generation for m in recent_metrics],
        }
        return trends
    def compare_models(
        self,
        model_names: List[str],
        metric: str = "avg_generation_time",
    ) -> Dict[str, float]:
        comparison = {}
        for model_name in model_names:
            if model_name in self.history and self.history[model_name]:
                latest = self.history[model_name][-1]
                if hasattr(latest, metric):
                    comparison[model_name] = getattr(latest, metric)
        return comparison
    def get_best_model(
        self,
        criteria: str = "balanced",
        min_quality: float = 60.0,
        max_time: float = 60.0,
        max_vram: float = 24.0,
    ) -> Optional[str]:
        candidates = []
        for model_name, metrics_list in self.history.items():
            if not metrics_list:
                continue
            latest = metrics_list[-1]
            if latest.avg_quality_score < min_quality:
                continue
            if latest.avg_generation_time > max_time:
                continue
            if latest.peak_vram_gb > max_vram:
                continue
            if criteria == "speed":
                score = 1 / latest.avg_generation_time
            elif criteria == "quality":
                score = latest.avg_quality_score
            elif criteria == "efficiency":
                score = latest.avg_quality_score / latest.avg_generation_time
            elif criteria == "cost":
                score = 1 / latest.avg_cost_per_generation
            else:
                speed_score = min(1, 10 / latest.avg_generation_time)
                quality_score = latest.avg_quality_score / 100
                cost_score = min(1, 0.1 / latest.avg_cost_per_generation)
                score = (speed_score + quality_score + cost_score) / 3
            candidates.append((model_name, score))
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        return None
    def generate_report(self) -> Dict[str, Any]:
        report = {
            "generated_at": datetime.now().isoformat(),
            "models": {},
            "rankings": {},
        }
        for model_name, metrics_list in self.history.items():
            if metrics_list:
                latest = metrics_list[-1]
                report["models"][model_name] = latest.to_dict()
        if report["models"]:
            speed_ranking = sorted(
                report["models"].items(),
                key=lambda x: x[1]["avg_generation_time"],
            )
            report["rankings"]["fastest"] = [name for name, _ in speed_ranking[:3]]
            quality_ranking = sorted(
                report["models"].items(),
                key=lambda x: x[1]["avg_quality_score"],
                reverse=True,
            )
            report["rankings"]["highest_quality"] = [name for name, _ in quality_ranking[:3]]
            efficiency_scores = []
            for name, metrics in report["models"].items():
                if metrics["avg_generation_time"] > 0:
                    efficiency = metrics["avg_quality_score"] / metrics["avg_generation_time"]
                    efficiency_scores.append((name, efficiency))
            efficiency_ranking = sorted(efficiency_scores, key=lambda x: x[1], reverse=True)
            report["rankings"]["most_efficient"] = [name for name, _ in efficiency_ranking[:3]]
        report_file = self.storage_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with report_file.open("w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Generated performance report: {report_file}")
        return report