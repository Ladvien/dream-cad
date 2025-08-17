"""Tests for production monitoring system."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

# Mock torch if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = Mock()


class TestModelMonitor(unittest.TestCase):
    """Test ModelMonitor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / "metrics"
        
        # Import after mocking
        from dream_cad.monitoring.model_monitor import ModelMonitor
        self.monitor = ModelMonitor(storage_dir=self.storage_dir)
    
    def test_start_generation(self):
        """Test starting generation monitoring."""
        self.monitor.start_generation(
            model_name="test_model",
            prompt="test prompt",
            config={"batch_size": 1},
        )
        
        self.assertIn("test_model", self.monitor.current_metrics)
        metrics = self.monitor.current_metrics["test_model"]
        self.assertEqual(metrics.model_name, "test_model")
        self.assertEqual(metrics.prompt, "test prompt")
    
    def test_end_generation_success(self):
        """Test ending successful generation."""
        self.monitor.start_generation(
            model_name="test_model",
            prompt="test prompt",
            config={"batch_size": 1},
        )
        
        metrics = self.monitor.end_generation(
            model_name="test_model",
            success=True,
            output_path="/tmp/output.obj",
            quality_metrics={"polycount": 10000, "quality_score": 85},
        )
        
        self.assertTrue(metrics.success)
        self.assertEqual(metrics.output_path, "/tmp/output.obj")
        self.assertEqual(metrics.output_polycount, 10000)
        self.assertEqual(metrics.quality_score, 85)
    
    def test_model_events(self):
        """Test model event tracking."""
        self.monitor.record_model_event(
            model_name="test_model",
            event_type="load",
            details={"memory_gb": 8},
        )
        
        self.assertEqual(len(self.monitor.model_events), 1)
        event = self.monitor.model_events[0]
        self.assertEqual(event["model_name"], "test_model")
        self.assertEqual(event["event_type"], "load")
        self.assertIn("test_model", self.monitor.active_models)
    
    def test_model_stats(self):
        """Test model statistics calculation."""
        # Add some metrics history
        from dream_cad.monitoring.model_monitor import ModelMetrics
        
        for i in range(5):
            metrics = ModelMetrics(
                model_name="test_model",
                timestamp=datetime.now() - timedelta(hours=i),
                generation_time_seconds=60 + i * 10,
                vram_peak_gb=8 + i * 0.5,
                success=i != 2,  # One failure
                quality_score=80 + i * 2,
            )
            self.monitor.metrics_history["test_model"].append(metrics)
        
        stats = self.monitor.get_model_stats("test_model", hours=24)
        
        self.assertEqual(stats["total_generations"], 5)
        self.assertEqual(stats["successful_generations"], 4)
        self.assertEqual(stats["failed_generations"], 1)
        self.assertEqual(stats["success_rate"], 0.8)
        self.assertTrue(stats["avg_generation_time"] > 0)


class TestUsageAnalytics(unittest.TestCase):
    """Test UsageAnalytics functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / "analytics"
        
        from dream_cad.monitoring.usage_analytics import UsageAnalytics
        self.analytics = UsageAnalytics(storage_dir=self.storage_dir)
    
    def test_log_generation(self):
        """Test logging generation data."""
        self.analytics.log_generation(
            model_name="test_model",
            prompt="create a cube",
            config={"quality": "high"},
            metrics={"success": True, "generation_time_seconds": 45},
        )
        
        self.assertEqual(len(self.analytics.generation_log), 1)
        entry = self.analytics.generation_log[0]
        self.assertEqual(entry["model_name"], "test_model")
        self.assertEqual(entry["prompt"], "create a cube")
        self.assertTrue(entry["success"])
    
    def test_generate_report(self):
        """Test report generation."""
        # Add test data
        for i in range(10):
            self.analytics.log_generation(
                model_name=f"model_{i % 3}",
                prompt=f"prompt {i}",
                config={},
                metrics={
                    "success": i != 5,
                    "generation_time_seconds": 30 + i * 5,
                    "quality_score": 70 + i * 2,
                    "vram_peak_gb": 6 + i * 0.5,
                },
            )
        
        report = self.analytics.generate_report(period="daily")
        
        self.assertEqual(report.total_generations, 10)
        self.assertEqual(len(report.model_usage_counts), 3)
        self.assertTrue(report.unique_prompts > 0)
        self.assertIsInstance(report.top_prompts, list)
    
    def test_model_comparison(self):
        """Test model comparison analysis."""
        # Add test data
        for model in ["model_a", "model_b"]:
            for i in range(5):
                self.analytics.log_generation(
                    model_name=model,
                    prompt="test",
                    config={},
                    metrics={
                        "success": True,
                        "generation_time_seconds": 30 if model == "model_a" else 60,
                        "quality_score": 80 if model == "model_a" else 90,
                        "vram_peak_gb": 8,
                    },
                )
        
        comparison = self.analytics.get_model_comparison()
        
        self.assertIn("model_a", comparison)
        self.assertIn("model_b", comparison)
        self.assertEqual(comparison["model_a"]["total_uses"], 5)
        self.assertEqual(comparison["model_b"]["total_uses"], 5)


class TestPerformanceAlerts(unittest.TestCase):
    """Test PerformanceAlerts functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / "alerts"
        
        from dream_cad.monitoring.performance_alerts import PerformanceAlerts, AlertConfig
        config = AlertConfig(
            max_generation_time=60,
            min_success_rate=0.8,
            max_vram_usage_gb=20,
        )
        self.alerts = PerformanceAlerts(config=config, storage_dir=self.storage_dir)
    
    def test_check_metrics_alerts(self):
        """Test metric-based alert generation."""
        metrics = {
            "generation_time_seconds": 120,  # Exceeds threshold
            "vram_peak_gb": 22,  # Exceeds threshold
            "gpu_temperature": 85,  # Exceeds threshold
            "success": True,
        }
        
        alerts = self.alerts.check_metrics("test_model", metrics)
        
        self.assertTrue(len(alerts) >= 2)  # At least 2 alerts should trigger
        alert_types = [alert.alert_type for alert in alerts]
        self.assertIn("performance", alert_types)
        self.assertIn("resource", alert_types)
    
    def test_check_model_stats_alerts(self):
        """Test statistics-based alert generation."""
        stats = {
            "success_rate": 0.6,  # Below threshold
            "avg_generation_time": 45,
            "avg_quality_score": 75,
        }
        
        alerts = self.alerts.check_model_stats("test_model", stats)
        
        self.assertTrue(len(alerts) >= 1)
        self.assertEqual(alerts[0].alert_type, "quality")
    
    def test_alert_cooldown(self):
        """Test alert cooldown mechanism."""
        metrics = {"generation_time_seconds": 120}
        
        # First alert should trigger
        alerts1 = self.alerts.check_metrics("test_model", metrics)
        self.assertTrue(len(alerts1) > 0)
        
        # Second alert should be suppressed (cooldown)
        alerts2 = self.alerts.check_metrics("test_model", metrics)
        # Should not create new alerts due to cooldown
        self.assertEqual(len(self.alerts.active_alerts), len(alerts1))


class TestResourceForecaster(unittest.TestCase):
    """Test ResourceForecaster functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / "forecasts"
        
        from dream_cad.monitoring.resource_forecaster import ResourceForecaster
        self.forecaster = ResourceForecaster(storage_dir=self.storage_dir)
    
    def test_forecast_empty_queue(self):
        """Test forecasting with empty queue."""
        forecast = self.forecaster.forecast(
            queue_jobs=[],
            current_resources={"vram_gb": 5, "gpu_utilization": 20},
            time_horizon_minutes=60,
        )
        
        self.assertEqual(forecast.predicted_vram_gb, 5)  # Should maintain current
        self.assertEqual(forecast.estimated_queue_completion_time, 0)
        self.assertTrue(forecast.can_accept_new_jobs)
    
    def test_forecast_with_queue(self):
        """Test forecasting with jobs in queue."""
        queue_jobs = [
            {"model_name": "model_a", "prompt": "test"},
            {"model_name": "model_b", "prompt": "test"},
        ]
        
        # Set up model profiles
        self.forecaster.model_profiles["model_a"] = {
            "avg_vram_gb": 8,
            "avg_generation_time": 60,
        }
        self.forecaster.model_profiles["model_b"] = {
            "avg_vram_gb": 10,
            "avg_generation_time": 90,
        }
        
        forecast = self.forecaster.forecast(
            queue_jobs=queue_jobs,
            current_resources={"vram_gb": 0, "gpu_utilization": 0},
            time_horizon_minutes=60,
        )
        
        self.assertTrue(forecast.predicted_vram_gb > 0)
        self.assertTrue(forecast.estimated_queue_completion_time > 0)
        self.assertIsInstance(forecast.risk_level, str)
    
    def test_update_model_profile(self):
        """Test updating model resource profile."""
        metrics = {
            "vram_peak_gb": 8.5,
            "generation_time_seconds": 75,
            "gpu_utilization": 65,
            "success": True,
        }
        
        self.forecaster.update_model_profile("test_model", metrics)
        
        self.assertIn("test_model", self.forecaster.model_profiles)
        profile = self.forecaster.model_profiles["test_model"]
        self.assertEqual(len(profile["vram_gb_samples"]), 1)
        self.assertEqual(profile["vram_gb_samples"][0], 8.5)


class TestEfficiencyReporter(unittest.TestCase):
    """Test EfficiencyReporter functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / "reports"
        
        from dream_cad.monitoring.efficiency_reporter import EfficiencyReporter
        self.reporter = EfficiencyReporter(storage_dir=self.storage_dir)
    
    def test_generate_report(self):
        """Test efficiency report generation."""
        model_stats = {
            "model_a": {
                "success_rate": 0.95,
                "avg_generation_time": 30,
                "avg_vram_gb": 6,
                "avg_quality_score": 85,
            },
            "model_b": {
                "success_rate": 0.80,
                "avg_generation_time": 60,
                "avg_vram_gb": 10,
                "avg_quality_score": 90,
            },
        }
        
        usage_data = {"total_generations": 100}
        
        report = self.reporter.generate_report(
            model_stats,
            usage_data,
            analysis_period_days=7,
        )
        
        self.assertIn("model_a", report.model_rankings)
        self.assertIn("model_b", report.model_rankings)
        self.assertTrue(report.model_rankings["model_a"] > 0)
        self.assertIsInstance(report.optimization_opportunities, list)
        self.assertIsInstance(report.recommended_configs, dict)


class TestCostAnalyzer(unittest.TestCase):
    """Test CostAnalyzer functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / "costs"
        
        from dream_cad.monitoring.cost_analyzer import CostAnalyzer
        self.analyzer = CostAnalyzer(storage_dir=self.storage_dir)
    
    def test_generate_report(self):
        """Test cost report generation."""
        metrics_data = [
            {
                "model_name": "model_a",
                "generation_time_seconds": 60,
                "vram_peak_gb": 8,
                "success": True,
                "quality_score": 85,
                "timestamp": datetime.now().isoformat(),
            },
            {
                "model_name": "model_b",
                "generation_time_seconds": 90,
                "vram_peak_gb": 12,
                "success": True,
                "quality_score": 90,
                "timestamp": datetime.now().isoformat(),
            },
        ]
        
        period_start = datetime.now() - timedelta(days=1)
        period_end = datetime.now()
        
        report = self.analyzer.generate_report(
            metrics_data,
            period_start,
            period_end,
        )
        
        self.assertTrue(report.total_cost > 0)
        self.assertTrue(report.compute_cost > 0)
        self.assertIn("model_a", report.model_costs)
        self.assertIn("model_b", report.model_costs)
        self.assertTrue(report.avg_cost_per_generation > 0)


class TestMonitoringDashboard(unittest.TestCase):
    """Test MonitoringDashboard functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / "monitoring"
        
        from dream_cad.monitoring.monitoring_dashboard import MonitoringDashboard
        self.dashboard = MonitoringDashboard(storage_dir=self.storage_dir)
    
    def test_start_end_generation(self):
        """Test generation lifecycle monitoring."""
        self.dashboard.start_generation(
            model_name="test_model",
            prompt="test prompt",
            config={"quality": "high"},
        )
        
        self.assertEqual(self.dashboard.active_generation, "test_model")
        
        result = self.dashboard.end_generation(
            model_name="test_model",
            success=True,
            output_path="/tmp/output.obj",
            quality_metrics={"quality_score": 85},
        )
        
        self.assertIsNone(self.dashboard.active_generation)
        self.assertIn("metrics", result)
        self.assertIn("alerts", result)
    
    def test_dashboard_summary(self):
        """Test dashboard summary generation."""
        summary = self.dashboard.get_dashboard_summary()
        
        self.assertIn("timestamp", summary)
        self.assertIn("system_status", summary)
        self.assertIn("model_statistics", summary)
        self.assertIn("active_alerts", summary)
        self.assertIn("usage_trends", summary)
    
    def test_system_health_check(self):
        """Test system health checking."""
        health = self.dashboard.check_system_health()
        
        self.assertIn("status", health)
        self.assertIn(health["status"], ["healthy", "warning", "degraded", "critical"])
        self.assertIn("recommendations", health)
        self.assertIsInstance(health["recommendations"], list)
    
    @patch('dream_cad.monitoring.monitoring_dashboard.datetime')
    def test_daily_report(self, mock_datetime):
        """Test daily report generation."""
        mock_datetime.now.return_value = datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        # Add some test data
        self.dashboard.model_monitor.metrics_history["test_model"] = []
        
        report = self.dashboard.generate_daily_report()
        
        self.assertIn("report_date", report)
        self.assertIn("model_statistics", report)
        self.assertIn("usage_report", report)
        self.assertIn("efficiency_report", report)
        self.assertIn("summary", report)


def run_tests():
    """Run all monitoring tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()