import json
import logging
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import statistics
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
logger = logging.getLogger(__name__)
@dataclass
class ABTestConfig:
    model_a: str
    model_b: str
    test_prompts: List[str]
    num_samples_per_prompt: int = 5
    confidence_level: float = 0.95
    min_difference_threshold: float = 0.05
    metrics_to_compare: List[str] = field(default_factory=lambda: [
        "generation_time",
        "quality_score", 
        "vram_usage",
        "success_rate",
    ])
    randomize_order: bool = True
    blind_evaluation: bool = True
@dataclass
class ABTestResult:
    config: ABTestConfig
    tested_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_samples: int = 0
    samples_model_a: int = 0
    samples_model_b: int = 0
    generation_time_a: List[float] = field(default_factory=list)
    generation_time_b: List[float] = field(default_factory=list)
    generation_time_winner: str = ""
    generation_time_p_value: float = 0.0
    generation_time_significant: bool = False
    quality_scores_a: List[float] = field(default_factory=list)
    quality_scores_b: List[float] = field(default_factory=list)
    quality_winner: str = ""
    quality_p_value: float = 0.0
    quality_significant: bool = False
    vram_usage_a: List[float] = field(default_factory=list)
    vram_usage_b: List[float] = field(default_factory=list)
    vram_winner: str = ""
    vram_p_value: float = 0.0
    vram_significant: bool = False
    success_count_a: int = 0
    success_count_b: int = 0
    success_rate_a: float = 0.0
    success_rate_b: float = 0.0
    success_winner: str = ""
    success_p_value: float = 0.0
    success_significant: bool = False
    overall_winner: str = ""
    confidence_level: float = 0.0
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    def calculate_statistics(self) -> None:
        alpha = 1 - self.config.confidence_level
        if self.generation_time_a and self.generation_time_b:
            if SCIPY_AVAILABLE:
                t_stat, p_value = stats.ttest_ind(self.generation_time_a, self.generation_time_b)
                self.generation_time_p_value = p_value
                self.generation_time_significant = p_value < alpha
            else:
                self.generation_time_p_value = 0.05
                mean_diff = abs(statistics.mean(self.generation_time_a) - statistics.mean(self.generation_time_b))
                avg_mean = (statistics.mean(self.generation_time_a) + statistics.mean(self.generation_time_b)) / 2
                self.generation_time_significant = (mean_diff / avg_mean) > 0.1
            mean_a = statistics.mean(self.generation_time_a)
            mean_b = statistics.mean(self.generation_time_b)
            if self.generation_time_significant:
                self.generation_time_winner = self.config.model_a if mean_a < mean_b else self.config.model_b
            else:
                self.generation_time_winner = "tie"
        if self.quality_scores_a and self.quality_scores_b:
            if SCIPY_AVAILABLE:
                t_stat, p_value = stats.ttest_ind(self.quality_scores_a, self.quality_scores_b)
                self.quality_p_value = p_value
                self.quality_significant = p_value < alpha
            else:
                self.quality_p_value = 0.05
                mean_diff = abs(statistics.mean(self.quality_scores_a) - statistics.mean(self.quality_scores_b))
                self.quality_significant = mean_diff > 5
            mean_a = statistics.mean(self.quality_scores_a)
            mean_b = statistics.mean(self.quality_scores_b)
            if self.quality_significant:
                self.quality_winner = self.config.model_a if mean_a > mean_b else self.config.model_b
            else:
                self.quality_winner = "tie"
        if self.vram_usage_a and self.vram_usage_b:
            if SCIPY_AVAILABLE:
                t_stat, p_value = stats.ttest_ind(self.vram_usage_a, self.vram_usage_b)
                self.vram_p_value = p_value
                self.vram_significant = p_value < alpha
            else:
                self.vram_p_value = 0.05
                mean_diff = abs(statistics.mean(self.vram_usage_a) - statistics.mean(self.vram_usage_b))
                self.vram_significant = mean_diff > 1
            mean_a = statistics.mean(self.vram_usage_a)
            mean_b = statistics.mean(self.vram_usage_b)
            if self.vram_significant:
                self.vram_winner = self.config.model_a if mean_a < mean_b else self.config.model_b
            else:
                self.vram_winner = "tie"
        if self.samples_model_a > 0 and self.samples_model_b > 0:
            self.success_rate_a = self.success_count_a / self.samples_model_a
            self.success_rate_b = self.success_count_b / self.samples_model_b
            successes = [self.success_count_a, self.success_count_b]
            totals = [self.samples_model_a, self.samples_model_b]
            if min(totals) >= 5:
                if SCIPY_AVAILABLE:
                    chi2, p_value = stats.chi2_contingency([
                        [self.success_count_a, self.samples_model_a - self.success_count_a],
                        [self.success_count_b, self.samples_model_b - self.success_count_b],
                    ])[:2]
                    self.success_p_value = p_value
                    self.success_significant = p_value < alpha
                else:
                    self.success_p_value = 0.05
                    rate_diff = abs(self.success_rate_a - self.success_rate_b)
                    self.success_significant = rate_diff > 0.1
                if self.success_significant:
                    self.success_winner = self.config.model_a if self.success_rate_a > self.success_rate_b else self.config.model_b
                else:
                    self.success_winner = "tie"
        wins = {self.config.model_a: 0, self.config.model_b: 0, "tie": 0}
        for winner in [self.generation_time_winner, self.quality_winner, 
                      self.vram_winner, self.success_winner]:
            if winner:
                wins[winner] += 1
        if wins[self.config.model_a] > wins[self.config.model_b]:
            self.overall_winner = self.config.model_a
        elif wins[self.config.model_b] > wins[self.config.model_a]:
            self.overall_winner = self.config.model_b
        else:
            self.overall_winner = "tie"
        self._generate_summary()
    def _generate_summary(self) -> None:
        summaries = []
        if self.generation_time_winner and self.generation_time_winner != "tie":
            mean_a = statistics.mean(self.generation_time_a) if self.generation_time_a else 0
            mean_b = statistics.mean(self.generation_time_b) if self.generation_time_b else 0
            diff = abs(mean_a - mean_b)
            summaries.append(f"{self.generation_time_winner} is {diff:.1f}s faster")
        if self.quality_winner and self.quality_winner != "tie":
            mean_a = statistics.mean(self.quality_scores_a) if self.quality_scores_a else 0
            mean_b = statistics.mean(self.quality_scores_b) if self.quality_scores_b else 0
            diff = abs(mean_a - mean_b)
            summaries.append(f"{self.quality_winner} has {diff:.1f} points higher quality")
        if self.vram_winner and self.vram_winner != "tie":
            mean_a = statistics.mean(self.vram_usage_a) if self.vram_usage_a else 0
            mean_b = statistics.mean(self.vram_usage_b) if self.vram_usage_b else 0
            diff = abs(mean_a - mean_b)
            summaries.append(f"{self.vram_winner} uses {diff:.1f}GB less VRAM")
        if self.success_winner and self.success_winner != "tie":
            diff = abs(self.success_rate_a - self.success_rate_b) * 100
            summaries.append(f"{self.success_winner} has {diff:.1f}% higher success rate")
        if summaries:
            self.summary = f"Overall winner: {self.overall_winner}. " + "; ".join(summaries)
        else:
            self.summary = "No significant differences found between models"
        self._generate_recommendations()
    def _generate_recommendations(self) -> None:
        self.recommendations = []
        if self.overall_winner != "tie":
            self.recommendations.append(f"Consider using {self.overall_winner} as the default model")
        if self.generation_time_winner != "tie":
            self.recommendations.append(f"For time-critical applications, use {self.generation_time_winner}")
        if self.quality_winner != "tie":
            self.recommendations.append(f"For quality-critical applications, use {self.quality_winner}")
        if self.vram_winner != "tie":
            self.recommendations.append(f"For memory-constrained environments, use {self.vram_winner}")
        if self.success_winner != "tie":
            self.recommendations.append(f"For reliability-critical applications, use {self.success_winner}")
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["config"] = asdict(self.config)
        return data
class ABTester:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
    ):
        self.output_dir = output_dir or Path("benchmarks/ab_tests")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_history: List[ABTestResult] = []
    def run_ab_test(
        self,
        config: ABTestConfig,
        model_a_runner: Any,
        model_b_runner: Any,
    ) -> ABTestResult:
        logger.info(f"Starting A/B test: {config.model_a} vs {config.model_b}")
        result = ABTestResult(config=config)
        test_samples = [
            {
                "prompt": prompt,
                "sample_id": f"{prompt[:20]}_{i}",
                "model": random.choice([config.model_a, config.model_b]) if config.randomize_order else (
                    config.model_a if i % 2 == 0 else config.model_b
                ),
            }
            for prompt in config.test_prompts
            for i in range(config.num_samples_per_prompt)
        ]
        if config.randomize_order:
            random.shuffle(test_samples)
        for sample in test_samples:
            model = sample["model"]
            prompt = sample["prompt"]
            logger.info(f"Testing {model}: {prompt[:50]}...")
            runner = model_a_runner if model == config.model_a else model_b_runner
            try:
                from .model_benchmark import BenchmarkConfig
                bench_config = BenchmarkConfig(
                    model_name=model,
                    prompt=prompt,
                    test_runs=1,
                    warmup_runs=0,
                )
                bench_result = runner.run_single_benchmark(bench_config)
                result.total_samples += 1
                if model == config.model_a:
                    result.samples_model_a += 1
                    if bench_result.success:
                        result.success_count_a += 1
                        result.generation_time_a.append(bench_result.generation_time_seconds)
                        result.quality_scores_a.append(bench_result.overall_quality_score)
                        result.vram_usage_a.append(bench_result.peak_vram_gb)
                else:
                    result.samples_model_b += 1
                    if bench_result.success:
                        result.success_count_b += 1
                        result.generation_time_b.append(bench_result.generation_time_seconds)
                        result.quality_scores_b.append(bench_result.overall_quality_score)
                        result.vram_usage_b.append(bench_result.peak_vram_gb)
            except Exception as e:
                logger.error(f"Failed to test {model} with '{prompt}': {e}")
        result.calculate_statistics()
        self.save_result(result)
        self.test_history.append(result)
        logger.info(f"A/B test complete. Winner: {result.overall_winner}")
        return result
    def save_result(self, result: ABTestResult) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ab_test_{result.config.model_a}_vs_{result.config.model_b}_{timestamp}.json"
        result_file = self.output_dir / filename
        with result_file.open("w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved A/B test result to {result_file}")
    def get_historical_comparison(
        self,
        model_a: str,
        model_b: str,
    ) -> List[ABTestResult]:
        results = []
        for result in self.test_history:
            if (result.config.model_a == model_a and result.config.model_b == model_b) or \
               (result.config.model_a == model_b and result.config.model_b == model_a):
                results.append(result)
        return results