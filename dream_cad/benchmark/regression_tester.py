import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
@dataclass
class RegressionThresholds:
    max_time_increase_percent: float = 20.0
    min_quality_decrease_percent: float = 10.0
    max_vram_increase_gb: float = 2.0
    max_failure_rate_increase_percent: float = 5.0
    min_samples: int = 3
@dataclass 
class RegressionResult:
    model_name: str
    tested_at: str
    baseline_version: str
    current_version: str
    time_regression: bool = False
    time_baseline: float = 0.0
    time_current: float = 0.0
    time_change_percent: float = 0.0
    quality_regression: bool = False
    quality_baseline: float = 0.0
    quality_current: float = 0.0
    quality_change_percent: float = 0.0
    memory_regression: bool = False
    memory_baseline: float = 0.0
    memory_current: float = 0.0
    memory_change_gb: float = 0.0
    failure_regression: bool = False
    failure_baseline: float = 0.0
    failure_current: float = 0.0
    failure_change_percent: float = 0.0
    has_regression: bool = False
    regression_summary: str = ""
    recommendations: List[str] = None
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        self.has_regression = any([
            self.time_regression,
            self.quality_regression,
            self.memory_regression,
            self.failure_regression,
        ])
        regressions = []
        if self.time_regression:
            regressions.append(f"Generation time increased by {self.time_change_percent:.1f}%")
            self.recommendations.append("Investigate recent changes to model architecture or processing pipeline")
        if self.quality_regression:
            regressions.append(f"Quality decreased by {abs(self.quality_change_percent):.1f}%")
            self.recommendations.append("Review model weights and generation parameters")
        if self.memory_regression:
            regressions.append(f"Memory usage increased by {self.memory_change_gb:.1f}GB")
            self.recommendations.append("Check for memory leaks or inefficient tensor operations")
        if self.failure_regression:
            regressions.append(f"Failure rate increased by {self.failure_change_percent:.1f}%")
            self.recommendations.append("Review error logs and add additional error handling")
        if regressions:
            self.regression_summary = "Regressions detected: " + "; ".join(regressions)
        else:
            self.regression_summary = "No regressions detected"
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
class RegressionTester:
    def __init__(
        self,
        baseline_dir: Optional[Path] = None,
        thresholds: Optional[RegressionThresholds] = None,
    ):
        self.baseline_dir = baseline_dir or Path("benchmarks/baselines")
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = thresholds or RegressionThresholds()
        self.baselines: Dict[str, Dict[str, Any]] = {}
        self.load_baselines()
    def load_baselines(self) -> None:
        for baseline_file in self.baseline_dir.glob("*.json"):
            try:
                with baseline_file.open() as f:
                    data = json.load(f)
                    model_name = baseline_file.stem.replace("baseline_", "")
                    self.baselines[model_name] = data
            except Exception as e:
                logger.error(f"Failed to load baseline {baseline_file}: {e}")
        if self.baselines:
            logger.info(f"Loaded baselines for {len(self.baselines)} models")
    def save_baseline(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        version: str = "latest",
    ) -> None:
        baseline_data = {
            "model_name": model_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
        }
        baseline_file = self.baseline_dir / f"baseline_{model_name}.json"
        with baseline_file.open("w") as f:
            json.dump(baseline_data, f, indent=2)
        self.baselines[model_name] = baseline_data
        logger.info(f"Saved baseline for {model_name}")
    def test_regression(
        self,
        model_name: str,
        current_metrics: Dict[str, Any],
        version: str = "current",
    ) -> RegressionResult:
        result = RegressionResult(
            model_name=model_name,
            tested_at=datetime.now().isoformat(),
            baseline_version="unknown",
            current_version=version,
        )
        if model_name not in self.baselines:
            logger.warning(f"No baseline found for {model_name}")
            result.regression_summary = "No baseline available for comparison"
            return result
        baseline = self.baselines[model_name]
        baseline_metrics = baseline["metrics"]
        result.baseline_version = baseline.get("version", "unknown")
        if "avg_generation_time" in baseline_metrics and "avg_generation_time" in current_metrics:
            result.time_baseline = baseline_metrics["avg_generation_time"]
            result.time_current = current_metrics["avg_generation_time"]
            if result.time_baseline > 0:
                result.time_change_percent = (
                    (result.time_current - result.time_baseline) / result.time_baseline * 100
                )
                result.time_regression = result.time_change_percent > self.thresholds.max_time_increase_percent
        if "avg_quality_score" in baseline_metrics and "avg_quality_score" in current_metrics:
            result.quality_baseline = baseline_metrics["avg_quality_score"]
            result.quality_current = current_metrics["avg_quality_score"]
            if result.quality_baseline > 0:
                result.quality_change_percent = (
                    (result.quality_current - result.quality_baseline) / result.quality_baseline * 100
                )
                result.quality_regression = (
                    result.quality_change_percent < -self.thresholds.min_quality_decrease_percent
                )
        if "peak_vram_gb" in baseline_metrics and "peak_vram_gb" in current_metrics:
            result.memory_baseline = baseline_metrics["peak_vram_gb"]
            result.memory_current = current_metrics["peak_vram_gb"]
            result.memory_change_gb = result.memory_current - result.memory_baseline
            result.memory_regression = result.memory_change_gb > self.thresholds.max_vram_increase_gb
        if "success_rate" in baseline_metrics and "success_rate" in current_metrics:
            result.failure_baseline = (1 - baseline_metrics["success_rate"]) * 100
            result.failure_current = (1 - current_metrics["success_rate"]) * 100
            result.failure_change_percent = result.failure_current - result.failure_baseline
            result.failure_regression = (
                result.failure_change_percent > self.thresholds.max_failure_rate_increase_percent
            )
        result.__post_init__()
        return result
    def test_all_models(
        self,
        current_metrics: Dict[str, Dict[str, Any]],
    ) -> List[RegressionResult]:
        results = []
        for model_name in self.baselines:
            if model_name in current_metrics:
                result = self.test_regression(model_name, current_metrics[model_name])
                results.append(result)
        return results
    def update_baseline_if_improved(
        self,
        model_name: str,
        current_metrics: Dict[str, Any],
        version: str = "latest",
    ) -> bool:
        if model_name not in self.baselines:
            self.save_baseline(model_name, current_metrics, version)
            return True
        baseline_metrics = self.baselines[model_name]["metrics"]
        better_time = current_metrics.get("avg_generation_time", float('inf')) < baseline_metrics.get("avg_generation_time", float('inf'))
        better_quality = current_metrics.get("avg_quality_score", 0) > baseline_metrics.get("avg_quality_score", 0)
        better_memory = current_metrics.get("peak_vram_gb", float('inf')) < baseline_metrics.get("peak_vram_gb", float('inf'))
        better_success = current_metrics.get("success_rate", 0) > baseline_metrics.get("success_rate", 0)
        improvements = sum([better_time, better_quality, better_memory, better_success])
        if improvements >= 2:
            result = self.test_regression(model_name, current_metrics, version)
            if not result.has_regression:
                logger.info(f"Updating baseline for {model_name} (improvements: {improvements}/4)")
                self.save_baseline(model_name, current_metrics, version)
                return True
        return False
    def generate_regression_report(
        self,
        results: List[RegressionResult],
    ) -> Dict[str, Any]:
        report = {
            "tested_at": datetime.now().isoformat(),
            "total_models": len(results),
            "models_with_regression": sum(1 for r in results if r.has_regression),
            "results": [r.to_dict() for r in results],
            "summary": {},
        }
        time_regressions = [r for r in results if r.time_regression]
        quality_regressions = [r for r in results if r.quality_regression]
        memory_regressions = [r for r in results if r.memory_regression]
        failure_regressions = [r for r in results if r.failure_regression]
        report["summary"] = {
            "time_regressions": [r.model_name for r in time_regressions],
            "quality_regressions": [r.model_name for r in quality_regressions],
            "memory_regressions": [r.model_name for r in memory_regressions],
            "failure_regressions": [r.model_name for r in failure_regressions],
        }
        if report["models_with_regression"] == 0:
            report["status"] = "PASS"
            report["message"] = "No regressions detected"
        else:
            report["status"] = "FAIL"
            report["message"] = f"Regressions detected in {report['models_with_regression']} models"
        report_file = self.baseline_dir.parent / f"regression_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with report_file.open("w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Generated regression report: {report_file}")
        return report