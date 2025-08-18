import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import statistics
logger = logging.getLogger(__name__)
@dataclass
class UsageReport:
    report_period: str
    start_date: datetime
    end_date: datetime
    model_usage_counts: Dict[str, int]
    model_usage_hours: Dict[str, float]
    model_success_rates: Dict[str, float]
    peak_usage_hours: List[int]
    average_daily_generations: float
    total_generations: int
    unique_prompts: int
    top_prompts: List[Tuple[str, int]]
    prompt_categories: Dict[str, int]
    avg_generation_times: Dict[str, float]
    quality_trends: Dict[str, List[float]]
    peak_concurrent_models: int
    avg_vram_usage: float
    peak_vram_usage: float
    model_preference_score: Dict[str, float]
    config_preferences: Dict[str, Dict[str, Any]]
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["start_date"] = self.start_date.isoformat()
        data["end_date"] = self.end_date.isoformat()
        return data
class UsageAnalytics:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("monitoring/analytics")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.generation_log: List[Dict[str, Any]] = []
        self.model_preferences: Dict[str, float] = defaultdict(float)
        self.prompt_history: List[str] = []
        self._load_history()
    def log_generation(
        self,
        model_name: str,
        prompt: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "prompt": prompt,
            "config": config,
            "success": metrics.get("success", True),
            "generation_time": metrics.get("generation_time_seconds", 0),
            "quality_score": metrics.get("quality_score", 0),
            "vram_used": metrics.get("vram_peak_gb", 0),
        }
        self.generation_log.append(entry)
        self.prompt_history.append(prompt)
        if entry["success"]:
            preference_score = 1.0
            if entry["quality_score"] > 0:
                preference_score *= (entry["quality_score"] / 100)
            self.model_preferences[model_name] += preference_score
        self._save_entry(entry)
    def generate_report(
        self,
        period: str = "daily",
        end_date: Optional[datetime] = None,
    ) -> UsageReport:
        if end_date is None:
            end_date = datetime.now()
        if period == "daily":
            start_date = end_date - timedelta(days=1)
        elif period == "weekly":
            start_date = end_date - timedelta(weeks=1)
        elif period == "monthly":
            start_date = end_date - timedelta(days=30)
        else:
            raise ValueError(f"Invalid period: {period}")
        period_logs = [
            log for log in self.generation_log
            if start_date <= datetime.fromisoformat(log["timestamp"]) <= end_date
        ]
        if not period_logs:
            return UsageReport(
                report_period=period,
                start_date=start_date,
                end_date=end_date,
                model_usage_counts={},
                model_usage_hours={},
                model_success_rates={},
                peak_usage_hours=[],
                average_daily_generations=0,
                total_generations=0,
                unique_prompts=0,
                top_prompts=[],
                prompt_categories={},
                avg_generation_times={},
                quality_trends={},
                peak_concurrent_models=0,
                avg_vram_usage=0,
                peak_vram_usage=0,
                model_preference_score={},
                config_preferences={},
            )
        report = UsageReport(
            report_period=period,
            start_date=start_date,
            end_date=end_date,
            model_usage_counts=self._calculate_model_usage(period_logs),
            model_usage_hours=self._calculate_usage_hours(period_logs),
            model_success_rates=self._calculate_success_rates(period_logs),
            peak_usage_hours=self._find_peak_hours(period_logs),
            average_daily_generations=len(period_logs) / max(1, (end_date - start_date).days),
            total_generations=len(period_logs),
            unique_prompts=len(set(log["prompt"] for log in period_logs)),
            top_prompts=self._get_top_prompts(period_logs),
            prompt_categories=self._categorize_prompts(period_logs),
            avg_generation_times=self._calculate_avg_times(period_logs),
            quality_trends=self._calculate_quality_trends(period_logs),
            peak_concurrent_models=self._estimate_peak_concurrent(period_logs),
            avg_vram_usage=self._calculate_avg_vram(period_logs),
            peak_vram_usage=self._calculate_peak_vram(period_logs),
            model_preference_score=self._calculate_preferences(period_logs),
            config_preferences=self._analyze_configs(period_logs),
        )
        self._save_report(report)
        return report
    def get_usage_trends(
        self,
        model_name: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, List[float]]:
        cutoff = datetime.now() - timedelta(days=days)
        logs = [
            log for log in self.generation_log
            if datetime.fromisoformat(log["timestamp"]) > cutoff
        ]
        if model_name:
            logs = [log for log in logs if log["model_name"] == model_name]
        daily_counts = defaultdict(int)
        daily_times = defaultdict(list)
        daily_quality = defaultdict(list)
        for log in logs:
            date = datetime.fromisoformat(log["timestamp"]).date()
            daily_counts[date] += 1
            daily_times[date].append(log["generation_time"])
            if log["quality_score"] > 0:
                daily_quality[date].append(log["quality_score"])
        dates = sorted(daily_counts.keys())
        return {
            "dates": [d.isoformat() for d in dates],
            "daily_counts": [daily_counts[d] for d in dates],
            "avg_generation_time": [
                statistics.mean(daily_times[d]) if daily_times[d] else 0
                for d in dates
            ],
            "avg_quality_score": [
                statistics.mean(daily_quality[d]) if daily_quality[d] else 0
                for d in dates
            ],
        }
    def get_model_comparison(self) -> Dict[str, Dict[str, Any]]:
        comparison = {}
        for model_name in set(log["model_name"] for log in self.generation_log):
            model_logs = [
                log for log in self.generation_log
                if log["model_name"] == model_name
            ]
            if not model_logs:
                continue
            successful = [log for log in model_logs if log["success"]]
            comparison[model_name] = {
                "total_uses": len(model_logs),
                "success_rate": len(successful) / len(model_logs) if model_logs else 0,
                "avg_generation_time": statistics.mean(
                    log["generation_time"] for log in successful
                ) if successful else 0,
                "avg_quality_score": statistics.mean(
                    log["quality_score"] for log in successful
                    if log["quality_score"] > 0
                ) if any(log["quality_score"] > 0 for log in successful) else 0,
                "avg_vram_gb": statistics.mean(
                    log["vram_used"] for log in successful
                    if log["vram_used"] > 0
                ) if any(log["vram_used"] > 0 for log in successful) else 0,
                "preference_score": self.model_preferences.get(model_name, 0),
            }
        return comparison
    def _calculate_model_usage(self, logs: List[Dict]) -> Dict[str, int]:
        counter = Counter(log["model_name"] for log in logs)
        return dict(counter)
    def _calculate_usage_hours(self, logs: List[Dict]) -> Dict[str, float]:
        usage_hours = defaultdict(float)
        for log in logs:
            usage_hours[log["model_name"]] += log["generation_time"] / 3600
        return dict(usage_hours)
    def _calculate_success_rates(self, logs: List[Dict]) -> Dict[str, float]:
        model_logs = defaultdict(list)
        for log in logs:
            model_logs[log["model_name"]].append(log["success"])
        return {
            model: sum(successes) / len(successes) if successes else 0
            for model, successes in model_logs.items()
        }
    def _find_peak_hours(self, logs: List[Dict]) -> List[int]:
        hour_counts = Counter(
            datetime.fromisoformat(log["timestamp"]).hour
            for log in logs
        )
        return [hour for hour, _ in hour_counts.most_common(3)]
    def _get_top_prompts(self, logs: List[Dict], n: int = 10) -> List[Tuple[str, int]]:
        prompt_counts = Counter(log["prompt"] for log in logs)
        return prompt_counts.most_common(n)
    def _categorize_prompts(self, logs: List[Dict]) -> Dict[str, int]:
        categories = defaultdict(int)
        for log in logs:
            prompt = log["prompt"].lower()
            if any(word in prompt for word in ["character", "person", "human", "face"]):
                categories["character"] += 1
            elif any(word in prompt for word in ["building", "house", "architecture"]):
                categories["architecture"] += 1
            elif any(word in prompt for word in ["car", "vehicle", "plane", "ship"]):
                categories["vehicle"] += 1
            elif any(word in prompt for word in ["animal", "creature", "dragon"]):
                categories["creature"] += 1
            elif any(word in prompt for word in ["furniture", "chair", "table", "sofa"]):
                categories["furniture"] += 1
            elif any(word in prompt for word in ["weapon", "sword", "gun"]):
                categories["weapon"] += 1
            else:
                categories["other"] += 1
        return dict(categories)
    def _calculate_avg_times(self, logs: List[Dict]) -> Dict[str, float]:
        model_times = defaultdict(list)
        for log in logs:
            if log["success"]:
                model_times[log["model_name"]].append(log["generation_time"])
        return {
            model: statistics.mean(times) if times else 0
            for model, times in model_times.items()
        }
    def _calculate_quality_trends(self, logs: List[Dict]) -> Dict[str, List[float]]:
        model_daily_quality = defaultdict(lambda: defaultdict(list))
        for log in logs:
            if log["quality_score"] > 0:
                date = datetime.fromisoformat(log["timestamp"]).date()
                model_daily_quality[log["model_name"]][date].append(log["quality_score"])
        trends = {}
        for model, daily_scores in model_daily_quality.items():
            dates = sorted(daily_scores.keys())
            trends[model] = [
                statistics.mean(daily_scores[date]) for date in dates
            ]
        return trends
    def _estimate_peak_concurrent(self, logs: List[Dict]) -> int:
        if not logs:
            return 0
        events = []
        for log in logs:
            start = datetime.fromisoformat(log["timestamp"])
            end = start + timedelta(seconds=log["generation_time"])
            events.append((start, 1))
            events.append((end, -1))
        events.sort()
        concurrent = 0
        peak = 0
        for _, delta in events:
            concurrent += delta
            peak = max(peak, concurrent)
        return peak
    def _calculate_avg_vram(self, logs: List[Dict]) -> float:
        vram_values = [log["vram_used"] for log in logs if log["vram_used"] > 0]
        return statistics.mean(vram_values) if vram_values else 0
    def _calculate_peak_vram(self, logs: List[Dict]) -> float:
        vram_values = [log["vram_used"] for log in logs if log["vram_used"] > 0]
        return max(vram_values) if vram_values else 0
    def _calculate_preferences(self, logs: List[Dict]) -> Dict[str, float]:
        model_scores = defaultdict(float)
        for model in set(log["model_name"] for log in logs):
            model_logs = [log for log in logs if log["model_name"] == model]
            usage_score = len(model_logs) / len(logs)
            success_rate = sum(1 for log in model_logs if log["success"]) / len(model_logs)
            quality_scores = [log["quality_score"] for log in model_logs if log["quality_score"] > 0]
            avg_quality = statistics.mean(quality_scores) / 100 if quality_scores else 0.5
            model_scores[model] = usage_score * 0.3 + success_rate * 0.3 + avg_quality * 0.4
        max_score = max(model_scores.values()) if model_scores else 1
        return {
            model: score / max_score
            for model, score in model_scores.items()
        }
    def _analyze_configs(self, logs: List[Dict]) -> Dict[str, Dict[str, Any]]:
        model_configs = defaultdict(list)
        for log in logs:
            if log["success"]:
                model_configs[log["model_name"]].append(log["config"])
        common_configs = {}
        for model, configs in model_configs.items():
            if not configs:
                continue
            param_values = defaultdict(list)
            for config in configs:
                for key, value in config.items():
                    param_values[key].append(value)
            common_configs[model] = {}
            for key, values in param_values.items():
                if all(isinstance(v, (int, float)) for v in values):
                    common_configs[model][key] = statistics.median(values)
                else:
                    common_configs[model][key] = Counter(values).most_common(1)[0][0]
        return common_configs
    def _save_entry(self, entry: Dict[str, Any]) -> None:
        try:
            file_path = self.storage_dir / "generation_log.jsonl"
            with file_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to save entry: {e}")
    def _save_report(self, report: UsageReport) -> None:
        try:
            filename = f"report_{report.report_period}_{report.end_date.strftime('%Y%m%d')}.json"
            file_path = self.storage_dir / filename
            with file_path.open("w") as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"Saved report to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    def _load_history(self) -> None:
        try:
            log_file = self.storage_dir / "generation_log.jsonl"
            if log_file.exists():
                with log_file.open() as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            self.generation_log.append(entry)
                            if entry.get("success"):
                                self.model_preferences[entry["model_name"]] += 1
                        except Exception as e:
                            logger.debug(f"Could not load entry: {e}")
            logger.info(f"Loaded {len(self.generation_log)} historical entries")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")