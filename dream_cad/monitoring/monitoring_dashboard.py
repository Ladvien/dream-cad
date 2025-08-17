"""Monitoring dashboard for multi-model 3D generation system."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .model_monitor import ModelMonitor
from .usage_analytics import UsageAnalytics
from .performance_alerts import PerformanceAlerts, AlertConfig
from .resource_forecaster import ResourceForecaster
from .efficiency_reporter import EfficiencyReporter
from .cost_analyzer import CostAnalyzer

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Central monitoring dashboard for multi-model system."""
    
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        alert_config: Optional[AlertConfig] = None,
    ):
        """Initialize monitoring dashboard.
        
        Args:
            storage_dir: Base directory for monitoring data
            alert_config: Alert configuration
        """
        self.storage_dir = storage_dir or Path("monitoring")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_monitor = ModelMonitor(self.storage_dir / "metrics")
        self.usage_analytics = UsageAnalytics(self.storage_dir / "analytics")
        self.performance_alerts = PerformanceAlerts(alert_config, self.storage_dir / "alerts")
        self.resource_forecaster = ResourceForecaster(self.storage_dir / "forecasts")
        self.efficiency_reporter = EfficiencyReporter(self.storage_dir / "reports")
        self.cost_analyzer = CostAnalyzer(self.storage_dir / "costs")
        
        # Dashboard state
        self.active_generation: Optional[str] = None
        self.dashboard_metrics: Dict[str, Any] = {}
        
        logger.info("Monitoring dashboard initialized")
    
    def start_generation(
        self,
        model_name: str,
        prompt: str,
        config: Dict[str, Any],
    ) -> None:
        """Start monitoring a generation.
        
        Args:
            model_name: Name of the model
            prompt: Generation prompt
            config: Model configuration
        """
        # Start model monitoring
        self.model_monitor.start_generation(model_name, prompt, config)
        
        # Log to analytics
        self.usage_analytics.log_generation(
            model_name,
            prompt,
            config,
            {"status": "started"},
        )
        
        # Record model event
        self.model_monitor.record_model_event(
            model_name,
            "generation_start",
            {"prompt": prompt[:100], "config": config},
        )
        
        self.active_generation = model_name
        logger.info(f"Started monitoring generation for {model_name}")
    
    def end_generation(
        self,
        model_name: str,
        success: bool = True,
        output_path: Optional[str] = None,
        error_message: Optional[str] = None,
        quality_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """End monitoring a generation.
        
        Args:
            model_name: Name of the model
            success: Whether generation succeeded
            output_path: Path to output file
            error_message: Error message if failed
            quality_metrics: Quality assessment results
            
        Returns:
            Generation metrics and alerts
        """
        # End model monitoring
        metrics = self.model_monitor.end_generation(
            model_name,
            success,
            output_path,
            error_message,
            quality_metrics,
        )
        
        # Log final metrics to analytics
        self.usage_analytics.log_generation(
            model_name,
            metrics.prompt,
            metrics.model_config,
            metrics.to_dict(),
        )
        
        # Update model profile for forecasting
        self.resource_forecaster.update_model_profile(
            model_name,
            metrics.to_dict(),
        )
        
        # Check for alerts
        alerts = self.performance_alerts.check_metrics(
            model_name,
            metrics.to_dict(),
        )
        
        # Record model event
        self.model_monitor.record_model_event(
            model_name,
            "generation_end",
            {"success": success, "duration": metrics.generation_time_seconds},
        )
        
        self.active_generation = None
        
        return {
            "metrics": metrics.to_dict(),
            "alerts": [alert.to_dict() for alert in alerts],
        }
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary.
        
        Returns:
            Dashboard summary with all key metrics
        """
        # Get current resource usage
        resources = self.model_monitor.get_resource_usage()
        
        # Get model statistics
        model_stats = self.model_monitor.get_all_model_stats(hours=24)
        
        # Get active alerts
        active_alerts = self.performance_alerts.get_active_alerts()
        
        # Get usage trends
        trends = {}
        for model in model_stats:
            trends[model] = self.usage_analytics.get_usage_trends(model, days=7)
        
        # Get model comparison
        comparison = self.usage_analytics.get_model_comparison()
        
        # Generate forecast if we have pending jobs (simplified)
        forecast = None
        if resources.get("active_models"):
            forecast = self.resource_forecaster.forecast(
                [],  # Would get from queue in production
                resources,
                time_horizon_minutes=60,
            )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": {
                "active_generation": self.active_generation,
                "active_models": resources.get("active_models", []),
                "resource_usage": resources,
            },
            "model_statistics": model_stats,
            "active_alerts": [alert.to_dict() for alert in active_alerts],
            "usage_trends": trends,
            "model_comparison": comparison,
            "forecast": forecast.to_dict() if forecast else None,
        }
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate comprehensive daily report.
        
        Returns:
            Daily report with all analyses
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        # Get model stats
        model_stats = self.model_monitor.get_all_model_stats(hours=24)
        
        # Generate usage report
        usage_report = self.usage_analytics.generate_report(
            period="daily",
            end_date=end_date,
        )
        
        # Generate efficiency report
        efficiency_report = self.efficiency_reporter.generate_report(
            model_stats,
            usage_report.to_dict(),
            analysis_period_days=1,
        )
        
        # Get metrics for cost analysis
        metrics_data = []
        for model, history in self.model_monitor.metrics_history.items():
            for metric in history:
                if metric.timestamp >= start_date:
                    metrics_data.append(metric.to_dict())
        
        # Generate cost report
        cost_report = None
        if metrics_data:
            cost_report = self.cost_analyzer.generate_report(
                metrics_data,
                start_date,
                end_date,
            )
        
        # Check model stats for alerts
        alerts_triggered = []
        for model, stats in model_stats.items():
            alerts = self.performance_alerts.check_model_stats(model, stats)
            alerts_triggered.extend(alerts)
        
        return {
            "report_date": end_date.isoformat(),
            "period": "daily",
            "model_statistics": model_stats,
            "usage_report": usage_report.to_dict(),
            "efficiency_report": efficiency_report.to_dict(),
            "cost_report": cost_report.to_dict() if cost_report else None,
            "alerts_triggered": [alert.to_dict() for alert in alerts_triggered],
            "summary": self._generate_summary(
                model_stats,
                usage_report,
                efficiency_report,
                cost_report,
            ),
        }
    
    def model_loaded(self, model_name: str) -> None:
        """Record model loading event.
        
        Args:
            model_name: Name of the model loaded
        """
        self.model_monitor.record_model_event(
            model_name,
            "load",
            {"timestamp": datetime.now().isoformat()},
        )
        logger.info(f"Model loaded: {model_name}")
    
    def model_unloaded(self, model_name: str) -> None:
        """Record model unloading event.
        
        Args:
            model_name: Name of the model unloaded
        """
        self.model_monitor.record_model_event(
            model_name,
            "unload",
            {"timestamp": datetime.now().isoformat()},
        )
        logger.info(f"Model unloaded: {model_name}")
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health.
        
        Returns:
            System health status
        """
        resources = self.model_monitor.get_resource_usage()
        
        # Check resource alerts
        resource_alerts = self.performance_alerts.check_system_resources(resources)
        
        # Determine health status
        if any(alert.severity == "critical" for alert in resource_alerts):
            health_status = "critical"
        elif any(alert.severity == "error" for alert in resource_alerts):
            health_status = "degraded"
        elif any(alert.severity == "warning" for alert in resource_alerts):
            health_status = "warning"
        else:
            health_status = "healthy"
        
        # Get recent error rate
        model_stats = self.model_monitor.get_all_model_stats(hours=1)
        total_generations = sum(s["total_generations"] for s in model_stats.values())
        failed_generations = sum(s["failed_generations"] for s in model_stats.values())
        error_rate = failed_generations / total_generations if total_generations > 0 else 0
        
        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "resource_usage": resources,
            "active_alerts": [alert.to_dict() for alert in resource_alerts],
            "error_rate": error_rate,
            "active_models": len(resources.get("active_models", [])),
            "recommendations": self._get_health_recommendations(
                health_status,
                resources,
                error_rate,
            ),
        }
    
    def forecast_capacity(
        self,
        queue_jobs: List[Dict[str, Any]],
        time_horizon_minutes: int = 60,
    ) -> Dict[str, Any]:
        """Forecast system capacity.
        
        Args:
            queue_jobs: List of pending jobs
            time_horizon_minutes: Forecast horizon
            
        Returns:
            Capacity forecast
        """
        resources = self.model_monitor.get_resource_usage()
        
        forecast = self.resource_forecaster.forecast(
            queue_jobs,
            resources,
            time_horizon_minutes,
        )
        
        return {
            "forecast": forecast.to_dict(),
            "current_resources": resources,
            "queue_size": len(queue_jobs),
        }
    
    def update_performance_baseline(self, model_name: str) -> None:
        """Update performance baseline for a model.
        
        Args:
            model_name: Name of the model
        """
        stats = self.model_monitor.get_model_stats(model_name, hours=168)  # 1 week
        
        if stats["total_generations"] > 0:
            self.performance_alerts.update_baseline(model_name, stats)
            logger.info(f"Updated performance baseline for {model_name}")
    
    def _generate_summary(
        self,
        model_stats: Dict[str, Any],
        usage_report: Any,
        efficiency_report: Any,
        cost_report: Optional[Any],
    ) -> Dict[str, Any]:
        """Generate executive summary.
        
        Args:
            model_stats: Model statistics
            usage_report: Usage report
            efficiency_report: Efficiency report
            cost_report: Cost report (optional)
            
        Returns:
            Executive summary
        """
        # Calculate totals
        total_generations = sum(s.get("total_generations", 0) for s in model_stats.values())
        total_successes = sum(s.get("successful_generations", 0) for s in model_stats.values())
        overall_success_rate = total_successes / total_generations if total_generations > 0 else 0
        
        # Find best performing model
        best_model = None
        if model_stats:
            best_model = max(
                model_stats.items(),
                key=lambda x: x[1]["success_rate"] * x[1].get("avg_quality_score", 0),
                default=(None, None),
            )[0]
        
        summary = {
            "total_generations": total_generations,
            "success_rate": overall_success_rate,
            "best_performing_model": best_model,
            "most_efficient_model": efficiency_report.most_efficient_model,
            "peak_usage_hours": usage_report.peak_usage_hours,
            "unique_prompts": usage_report.unique_prompts,
        }
        
        if cost_report:
            summary["total_cost"] = cost_report.total_cost
            summary["cost_per_generation"] = cost_report.avg_cost_per_generation
        
        return summary
    
    def _get_health_recommendations(
        self,
        health_status: str,
        resources: Dict[str, Any],
        error_rate: float,
    ) -> List[str]:
        """Get health-based recommendations.
        
        Args:
            health_status: Current health status
            resources: Resource usage
            error_rate: Recent error rate
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if health_status == "critical":
            recommendations.append("System critical - consider pausing new jobs")
        elif health_status == "degraded":
            recommendations.append("System degraded - monitor closely")
        
        # Check specific resources
        if resources.get("ram_percent", 0) > 80:
            recommendations.append("High RAM usage - consider closing other applications")
        
        if resources.get("vram_gb", 0) > 20:
            recommendations.append("High VRAM usage - consider unloading unused models")
        
        if error_rate > 0.2:
            recommendations.append("High error rate - review recent failures")
        
        if not recommendations:
            recommendations.append("System healthy - all metrics within normal range")
        
        return recommendations