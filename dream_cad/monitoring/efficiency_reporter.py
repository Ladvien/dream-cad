import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
logger = logging.getLogger(__name__)
@dataclass
class EfficiencyReport:
    report_date: datetime
    analysis_period_days: int
    model_rankings: Dict[str, float]
    most_efficient_model: str
    least_efficient_model: str
    vram_efficiency: Dict[str, float]
    gpu_efficiency: Dict[str, float]
    time_efficiency: Dict[str, float]
    quality_per_vram: Dict[str, float]
    quality_per_second: Dict[str, float]
    quality_per_dollar: Dict[str, float]
    optimization_opportunities: List[Dict[str, Any]]
    potential_savings: Dict[str, float]
    recommended_configs: Dict[str, Dict[str, Any]]
    batch_size_recommendations: Dict[str, int]
    model_comparisons: List[Dict[str, Any]]
    use_case_recommendations: Dict[str, str]
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["report_date"] = self.report_date.isoformat()
        return data
class EfficiencyReporter:
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
    ):
        self.storage_dir = storage_dir or Path("monitoring/reports")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.efficiency_cache: Dict[str, Dict[str, float]] = {}
    def generate_report(
        self,
        model_stats: Dict[str, Dict[str, Any]],
        usage_data: Dict[str, Any],
        analysis_period_days: int = 7,
    ) -> EfficiencyReport:
        efficiency_scores = self._calculate_efficiency_scores(model_stats)
        resource_efficiency = self._analyze_resource_efficiency(model_stats)
        quality_tradeoffs = self._calculate_quality_tradeoffs(model_stats)
        opportunities = self._find_optimization_opportunities(
            model_stats,
            efficiency_scores,
        )
        recommendations = self._generate_recommendations(
            model_stats,
            efficiency_scores,
            usage_data,
        )
        comparisons = self._compare_models(model_stats, efficiency_scores)
        if efficiency_scores:
            sorted_models = sorted(
                efficiency_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            most_efficient = sorted_models[0][0] if sorted_models else "N/A"
            least_efficient = sorted_models[-1][0] if sorted_models else "N/A"
        else:
            most_efficient = least_efficient = "N/A"
        report = EfficiencyReport(
            report_date=datetime.now(),
            analysis_period_days=analysis_period_days,
            model_rankings=efficiency_scores,
            most_efficient_model=most_efficient,
            least_efficient_model=least_efficient,
            vram_efficiency=resource_efficiency["vram"],
            gpu_efficiency=resource_efficiency["gpu"],
            time_efficiency=resource_efficiency["time"],
            quality_per_vram=quality_tradeoffs["per_vram"],
            quality_per_second=quality_tradeoffs["per_second"],
            quality_per_dollar=quality_tradeoffs["per_dollar"],
            optimization_opportunities=opportunities,
            potential_savings=self._calculate_potential_savings(opportunities),
            recommended_configs=recommendations["configs"],
            batch_size_recommendations=recommendations["batch_sizes"],
            model_comparisons=comparisons,
            use_case_recommendations=recommendations["use_cases"],
        )
        self._save_report(report)
        return report
    def _calculate_efficiency_scores(
        self,
        model_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        scores = {}
        for model_name, stats in model_stats.items():
            success_rate = stats.get("success_rate", 0)
            avg_time = stats.get("avg_generation_time", float('inf'))
            if avg_time > 0:
                speed_score = min(100, 60 / avg_time * 100)
            else:
                speed_score = 0
            avg_vram = stats.get("avg_vram_gb", float('inf'))
            if avg_vram > 0:
                resource_score = min(100, 8 / avg_vram * 100)
            else:
                resource_score = 0
            quality = stats.get("avg_quality_score", 50)
            efficiency = (
                success_rate * 100 * 0.3 +
                speed_score * 0.25 +
                resource_score * 0.25 +
                quality * 0.2
            )
            scores[model_name] = round(efficiency, 2)
        return scores
    def _analyze_resource_efficiency(
        self,
        model_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        vram_efficiency = {}
        gpu_efficiency = {}
        time_efficiency = {}
        min_vram = min(
            (stats.get("avg_vram_gb", float('inf')) for stats in model_stats.values()),
            default=1.0,
        )
        min_time = min(
            (stats.get("avg_generation_time", float('inf')) for stats in model_stats.values()),
            default=1.0,
        )
        for model_name, stats in model_stats.items():
            avg_vram = stats.get("avg_vram_gb", 0)
            peak_vram = stats.get("peak_vram_gb", avg_vram)
            if peak_vram > 0:
                vram_efficiency[model_name] = (avg_vram / peak_vram) * 100
            else:
                vram_efficiency[model_name] = 0
            gpu_util = stats.get("avg_gpu_utilization", 0)
            gen_time = stats.get("avg_generation_time", 1)
            if gen_time > 0:
                gpu_efficiency[model_name] = min(100, (gpu_util * 30) / gen_time)
            else:
                gpu_efficiency[model_name] = 0
            if gen_time > 0 and min_time > 0:
                time_efficiency[model_name] = (min_time / gen_time) * 100
            else:
                time_efficiency[model_name] = 0
        return {
            "vram": vram_efficiency,
            "gpu": gpu_efficiency,
            "time": time_efficiency,
        }
    def _calculate_quality_tradeoffs(
        self,
        model_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        per_vram = {}
        per_second = {}
        per_dollar = {}
        for model_name, stats in model_stats.items():
            quality = stats.get("avg_quality_score", 0)
            vram = stats.get("avg_vram_gb", 1)
            time = stats.get("avg_generation_time", 1)
            per_vram[model_name] = quality / vram if vram > 0 else 0
            per_second[model_name] = quality / time if time > 0 else 0
            cost = (vram * time / 3600) * 0.10
            per_dollar[model_name] = quality / cost if cost > 0 else 0
        return {
            "per_vram": per_vram,
            "per_second": per_second,
            "per_dollar": per_dollar,
        }
    def _find_optimization_opportunities(
        self,
        model_stats: Dict[str, Dict[str, Any]],
        efficiency_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        opportunities = []
        for model_name, stats in model_stats.items():
            score = efficiency_scores.get(model_name, 0)
            if stats.get("success_rate", 1) < 0.8:
                opportunities.append({
                    "model": model_name,
                    "type": "reliability",
                    "issue": f"Low success rate: {stats['success_rate']:.1%}",
                    "recommendation": "Review error logs and adjust configuration",
                    "potential_improvement": (0.95 - stats["success_rate"]) * 100,
                })
            if stats.get("avg_vram_gb", 0) > 16:
                opportunities.append({
                    "model": model_name,
                    "type": "memory",
                    "issue": f"High VRAM usage: {stats['avg_vram_gb']:.1f}GB",
                    "recommendation": "Enable memory optimization or reduce batch size",
                    "potential_improvement": 20,
                })
            if stats.get("avg_generation_time", 0) > 120:
                opportunities.append({
                    "model": model_name,
                    "type": "speed",
                    "issue": f"Slow generation: {stats['avg_generation_time']:.1f}s",
                    "recommendation": "Consider using faster model or optimize settings",
                    "potential_improvement": 30,
                })
            if score < 50:
                opportunities.append({
                    "model": model_name,
                    "type": "overall",
                    "issue": f"Low efficiency score: {score:.1f}",
                    "recommendation": "Consider replacing with more efficient model",
                    "potential_improvement": 50 - score,
                })
        opportunities.sort(
            key=lambda x: x["potential_improvement"],
            reverse=True,
        )
        return opportunities[:10]
    def _generate_recommendations(
        self,
        model_stats: Dict[str, Dict[str, Any]],
        efficiency_scores: Dict[str, float],
        usage_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        configs = {}
        batch_sizes = {}
        use_cases = {}
        for model_name, stats in model_stats.items():
            config = {}
            if stats.get("avg_vram_gb", 0) > 12:
                config["enable_memory_optimization"] = True
                config["precision"] = "fp16"
            else:
                config["enable_memory_optimization"] = False
                config["precision"] = "fp32"
            if efficiency_scores.get(model_name, 0) < 60:
                config["quality_mode"] = "fast"
            else:
                config["quality_mode"] = "balanced"
            configs[model_name] = config
            vram = stats.get("avg_vram_gb", 8)
            if vram < 6:
                batch_sizes[model_name] = 4
            elif vram < 10:
                batch_sizes[model_name] = 2
            else:
                batch_sizes[model_name] = 1
        if efficiency_scores:
            sorted_models = sorted(
                efficiency_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            if sorted_models:
                fastest_model = min(
                    model_stats.items(),
                    key=lambda x: x[1].get("avg_generation_time", float('inf')),
                    default=(None, None),
                )[0]
                if fastest_model:
                    use_cases["fast_prototyping"] = fastest_model
                highest_quality = max(
                    model_stats.items(),
                    key=lambda x: x[1].get("avg_quality_score", 0),
                    default=(None, None),
                )[0]
                if highest_quality:
                    use_cases["high_quality"] = highest_quality
                if sorted_models:
                    use_cases["balanced"] = sorted_models[0][0]
                lowest_vram = min(
                    model_stats.items(),
                    key=lambda x: x[1].get("avg_vram_gb", float('inf')),
                    default=(None, None),
                )[0]
                if lowest_vram:
                    use_cases["low_resource"] = lowest_vram
        return {
            "configs": configs,
            "batch_sizes": batch_sizes,
            "use_cases": use_cases,
        }
    def _compare_models(
        self,
        model_stats: Dict[str, Dict[str, Any]],
        efficiency_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        comparisons = []
        models = list(model_stats.keys())
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                stats1 = model_stats[model1]
                stats2 = model_stats[model2]
                comparison = {
                    "model1": model1,
                    "model2": model2,
                    "efficiency_diff": efficiency_scores.get(model1, 0) - efficiency_scores.get(model2, 0),
                    "speed_ratio": (
                        stats2.get("avg_generation_time", 1) /
                        stats1.get("avg_generation_time", 1)
                        if stats1.get("avg_generation_time", 0) > 0 else 1
                    ),
                    "vram_diff": stats1.get("avg_vram_gb", 0) - stats2.get("avg_vram_gb", 0),
                    "quality_diff": stats1.get("avg_quality_score", 0) - stats2.get("avg_quality_score", 0),
                    "recommendation": "",
                }
                if comparison["efficiency_diff"] > 10:
                    comparison["recommendation"] = f"Prefer {model1} for better efficiency"
                elif comparison["efficiency_diff"] < -10:
                    comparison["recommendation"] = f"Prefer {model2} for better efficiency"
                elif comparison["quality_diff"] > 10:
                    comparison["recommendation"] = f"Prefer {model1} for higher quality"
                elif comparison["quality_diff"] < -10:
                    comparison["recommendation"] = f"Prefer {model2} for higher quality"
                elif comparison["speed_ratio"] > 1.5:
                    comparison["recommendation"] = f"Prefer {model1} for faster generation"
                elif comparison["speed_ratio"] < 0.67:
                    comparison["recommendation"] = f"Prefer {model2} for faster generation"
                else:
                    comparison["recommendation"] = "Models are comparable"
                comparisons.append(comparison)
        comparisons.sort(
            key=lambda x: abs(x["efficiency_diff"]),
            reverse=True,
        )
        return comparisons[:20]
    def _calculate_potential_savings(
        self,
        opportunities: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        savings = {
            "vram_gb": 0,
            "time_hours": 0,
            "cost_dollars": 0,
        }
        for opp in opportunities:
            if opp["type"] == "memory":
                savings["vram_gb"] += opp.get("potential_improvement", 0) * 0.2
            elif opp["type"] == "speed":
                savings["time_hours"] += opp.get("potential_improvement", 0) * 0.01
            savings["cost_dollars"] += opp.get("potential_improvement", 0) * 0.1
        return savings
    def _save_report(self, report: EfficiencyReport) -> None:
        try:
            filename = f"efficiency_report_{report.report_date.strftime('%Y%m%d_%H%M%S')}.json"
            file_path = self.storage_dir / filename
            with file_path.open("w") as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"Saved efficiency report to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save efficiency report: {e}")