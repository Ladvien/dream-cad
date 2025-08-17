"""Tests for the benchmarking system."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dream_cad.benchmark import (
    ModelBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    QualityAssessor,
    QualityMetrics,
    PerformanceTracker,
    PerformanceMetrics,
    RegressionTester,
    ABTester,
    ABTestConfig,
    BenchmarkRunner,
)


class TestBenchmarkConfig(unittest.TestCase):
    """Test BenchmarkConfig class."""
    
    def test_config_creation(self):
        """Test creating benchmark configuration."""
        config = BenchmarkConfig(
            model_name="triposr",
            prompt="test prompt",
            num_inference_steps=50,
        )
        
        self.assertEqual(config.model_name, "triposr")
        self.assertEqual(config.prompt, "test prompt")
        self.assertEqual(config.num_inference_steps, 50)
        self.assertTrue(config.fp16)  # Default value
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = BenchmarkConfig(
            model_name="triposr",
            prompt="test",
            output_dir=Path("/tmp/test"),
        )
        
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["model_name"], "triposr")
        self.assertEqual(config_dict["output_dir"], "/tmp/test")
    
    def test_config_hash(self):
        """Test config hash generation."""
        config1 = BenchmarkConfig(model_name="model1", prompt="prompt1")
        config2 = BenchmarkConfig(model_name="model1", prompt="prompt1")
        config3 = BenchmarkConfig(model_name="model2", prompt="prompt1")
        
        # Same configs should have same hash
        self.assertEqual(config1.get_hash(), config2.get_hash())
        # Different configs should have different hash
        self.assertNotEqual(config1.get_hash(), config3.get_hash())


class TestBenchmarkResult(unittest.TestCase):
    """Test BenchmarkResult class."""
    
    def test_result_creation(self):
        """Test creating benchmark result."""
        config = BenchmarkConfig(model_name="test", prompt="test")
        result = BenchmarkResult(
            config=config,
            success=True,
            generation_time_seconds=10.5,
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.generation_time_seconds, 10.5)
        self.assertIsNotNone(result.timestamp)
    
    def test_result_serialization(self):
        """Test result serialization."""
        config = BenchmarkConfig(model_name="test", prompt="test")
        result = BenchmarkResult(
            config=config,
            success=True,
            generation_time_seconds=10.5,
            peak_vram_gb=4.2,
        )
        
        # To dict
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["success"], True)
        self.assertEqual(result_dict["peak_vram_gb"], 4.2)
        
        # From dict
        result2 = BenchmarkResult.from_dict(result_dict)
        self.assertEqual(result2.success, result.success)
        self.assertEqual(result2.peak_vram_gb, result.peak_vram_gb)


class TestModelBenchmark(unittest.TestCase):
    """Test ModelBenchmark class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "benchmarks"
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        benchmark = ModelBenchmark(
            model_name="triposr",
            output_dir=self.output_dir,
        )
        
        self.assertEqual(benchmark.model_name, "triposr")
        self.assertTrue(self.output_dir.exists())
    
    @patch('dream_cad.benchmark.model_benchmark.ModelFactory')
    def test_model_loading(self, mock_factory):
        """Test model loading."""
        mock_model = MagicMock()
        mock_factory.create_model.return_value = mock_model
        
        # Enable model factory
        import dream_cad.benchmark.model_benchmark as mb
        mb.MODEL_FACTORY_AVAILABLE = True
        mb.ModelFactory = mock_factory
        
        benchmark = ModelBenchmark("triposr", self.output_dir)
        load_time = benchmark.load_model()
        
        self.assertGreater(load_time, 0)
        self.assertEqual(benchmark.model, mock_model)
        mock_factory.create_model.assert_called_once_with("triposr")
    
    def test_resource_monitoring(self):
        """Test resource monitoring."""
        benchmark = ModelBenchmark("triposr", self.output_dir)
        metrics = benchmark._monitor_resources()
        
        self.assertIn("timestamp", metrics)
        self.assertIn("ram_gb", metrics)
        self.assertIn("cpu_percent", metrics)
    
    def test_averaging_results(self):
        """Test averaging multiple results."""
        config = BenchmarkConfig(model_name="test", prompt="test")
        
        results = [
            BenchmarkResult(config=config, success=True, generation_time_seconds=10.0),
            BenchmarkResult(config=config, success=True, generation_time_seconds=12.0),
            BenchmarkResult(config=config, success=True, generation_time_seconds=11.0),
        ]
        
        benchmark = ModelBenchmark("test", self.output_dir)
        avg_result = benchmark._average_results(results)
        
        self.assertIsNotNone(avg_result)
        self.assertEqual(avg_result.generation_time_seconds, 11.0)  # Average
        self.assertTrue(avg_result.success)


class TestQualityAssessor(unittest.TestCase):
    """Test QualityAssessor class."""
    
    def test_quality_metrics_creation(self):
        """Test creating quality metrics."""
        metrics = QualityMetrics(
            vertex_count=1000,
            face_count=500,
            mesh_validity_score=95.0,
        )
        
        self.assertEqual(metrics.vertex_count, 1000)
        self.assertEqual(metrics.face_count, 500)
        self.assertEqual(metrics.mesh_validity_score, 95.0)
    
    def test_overall_score_calculation(self):
        """Test calculating overall quality scores."""
        metrics = QualityMetrics(
            mesh_validity_score=80,
            mesh_manifold_score=90,
            mesh_watertight_score=100,
            mesh_smoothness_score=70,
            edge_quality_score=85,
            face_quality_score=75,
            face_count=15000,  # Good for games
            uv_coverage=60,
            uv_overlaps=0,
            material_count=2,
        )
        
        metrics.calculate_overall_scores()
        
        self.assertGreater(metrics.overall_mesh_quality, 0)
        self.assertGreater(metrics.game_ready_score, 0)
    
    def test_quality_comparison(self):
        """Test comparing quality between models."""
        assessor = QualityAssessor()
        
        metrics1 = QualityMetrics(
            overall_mesh_quality=80,
            overall_texture_quality=70,
            game_ready_score=75,
            face_count=10000,
        )
        
        metrics2 = QualityMetrics(
            overall_mesh_quality=70,
            overall_texture_quality=80,
            game_ready_score=70,
            face_count=20000,
        )
        
        comparison = assessor.compare_quality(metrics1, metrics2)
        
        self.assertIn("mesh_quality_diff", comparison)
        self.assertIn("winner", comparison)
        self.assertEqual(comparison["winner"], 1)  # metrics1 wins


class TestPerformanceTracker(unittest.TestCase):
    """Test PerformanceTracker class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / "performance"
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            model_name="triposr",
            timestamp=datetime.now().isoformat(),
            avg_generation_time=10.5,
            min_generation_time=8.0,
            max_generation_time=13.0,
            std_generation_time=2.0,
            p50_generation_time=10.0,
            p95_generation_time=12.5,
            p99_generation_time=12.9,
            avg_vram_gb=6.0,
            peak_vram_gb=8.0,
            avg_ram_gb=12.0,
            peak_ram_gb=16.0,
            avg_gpu_temp_c=65.0,
            peak_gpu_temp_c=75.0,
            avg_gpu_utilization=80.0,
            peak_gpu_utilization=95.0,
            successful_generations=10,
            failed_generations=1,
            success_rate=0.91,
            generations_per_hour=342.86,
            avg_cost_per_generation=0.001,
            total_gpu_hours=0.029,
            total_cost_usd=0.015,
            avg_quality_score=75.0,
            min_quality_score=60.0,
            max_quality_score=90.0,
        )
        
        self.assertEqual(metrics.model_name, "triposr")
        self.assertEqual(metrics.avg_generation_time, 10.5)
        self.assertEqual(metrics.success_rate, 0.91)
    
    def test_metrics_calculation(self):
        """Test calculating metrics from results."""
        tracker = PerformanceTracker(self.storage_dir)
        
        # Create mock results
        config = BenchmarkConfig(model_name="test", prompt="test")
        results = [
            BenchmarkResult(config=config, success=True, generation_time_seconds=10.0),
            BenchmarkResult(config=config, success=True, generation_time_seconds=12.0),
            BenchmarkResult(config=config, success=False, generation_time_seconds=0.0),
        ]
        
        metrics = tracker.calculate_metrics("test", results)
        
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.successful_generations, 2)
        self.assertEqual(metrics.failed_generations, 1)
        self.assertAlmostEqual(metrics.success_rate, 0.667, places=2)
    
    def test_best_model_selection(self):
        """Test selecting best model based on criteria."""
        tracker = PerformanceTracker(self.storage_dir)
        
        # Add mock history
        tracker.history["model1"] = [
            PerformanceMetrics(
                model_name="model1",
                timestamp=datetime.now().isoformat(),
                avg_generation_time=10.0,
                min_generation_time=8.0,
                max_generation_time=12.0,
                std_generation_time=1.0,
                p50_generation_time=10.0,
                p95_generation_time=11.5,
                p99_generation_time=11.9,
                avg_vram_gb=6.0,
                peak_vram_gb=8.0,
                avg_ram_gb=12.0,
                peak_ram_gb=16.0,
                avg_gpu_temp_c=65.0,
                peak_gpu_temp_c=75.0,
                avg_gpu_utilization=80.0,
                peak_gpu_utilization=95.0,
                successful_generations=10,
                failed_generations=0,
                success_rate=1.0,
                generations_per_hour=360,
                avg_cost_per_generation=0.01,
                total_gpu_hours=0.028,
                total_cost_usd=0.014,
                avg_quality_score=80.0,
                min_quality_score=75,
                max_quality_score=85,
            )
        ]
        
        tracker.history["model2"] = [
            PerformanceMetrics(
                model_name="model2",
                timestamp=datetime.now().isoformat(),
                avg_generation_time=5.0,
                min_generation_time=4.0,
                max_generation_time=6.0,
                std_generation_time=0.5,
                p50_generation_time=5.0,
                p95_generation_time=5.8,
                p99_generation_time=5.95,
                avg_vram_gb=10.0,
                peak_vram_gb=12.0,
                avg_ram_gb=14.0,
                peak_ram_gb=18.0,
                avg_gpu_temp_c=70.0,
                peak_gpu_temp_c=80.0,
                avg_gpu_utilization=85.0,
                peak_gpu_utilization=98.0,
                successful_generations=10,
                failed_generations=0,
                success_rate=1.0,
                generations_per_hour=720,
                avg_cost_per_generation=0.005,
                total_gpu_hours=0.014,
                total_cost_usd=0.007,
                avg_quality_score=70.0,
                min_quality_score=65,
                max_quality_score=75,
            )
        ]
        
        # Test different criteria
        best_speed = tracker.get_best_model(criteria="speed")
        self.assertEqual(best_speed, "model2")
        
        best_quality = tracker.get_best_model(criteria="quality")
        self.assertEqual(best_quality, "model1")


class TestRegressionTester(unittest.TestCase):
    """Test RegressionTester class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_dir = Path(self.temp_dir) / "baselines"
    
    def test_baseline_saving(self):
        """Test saving baseline metrics."""
        tester = RegressionTester(self.baseline_dir)
        
        metrics = {
            "avg_generation_time": 10.0,
            "avg_quality_score": 75.0,
            "peak_vram_gb": 8.0,
            "success_rate": 0.95,
        }
        
        tester.save_baseline("test_model", metrics, "v1.0")
        
        self.assertIn("test_model", tester.baselines)
        self.assertEqual(tester.baselines["test_model"]["metrics"], metrics)
    
    def test_regression_detection(self):
        """Test detecting performance regression."""
        tester = RegressionTester(self.baseline_dir)
        
        # Set baseline
        baseline_metrics = {
            "avg_generation_time": 10.0,
            "avg_quality_score": 75.0,
            "peak_vram_gb": 8.0,
            "success_rate": 0.95,
        }
        tester.save_baseline("test_model", baseline_metrics)
        
        # Test with regression (slower generation)
        current_metrics = {
            "avg_generation_time": 15.0,  # 50% slower
            "avg_quality_score": 75.0,
            "peak_vram_gb": 8.0,
            "success_rate": 0.95,
        }
        
        result = tester.test_regression("test_model", current_metrics)
        
        self.assertTrue(result.has_regression)
        self.assertTrue(result.time_regression)
        self.assertFalse(result.quality_regression)
    
    def test_baseline_update(self):
        """Test updating baseline when improved."""
        tester = RegressionTester(self.baseline_dir)
        
        # Set initial baseline
        baseline_metrics = {
            "avg_generation_time": 10.0,
            "avg_quality_score": 70.0,
            "peak_vram_gb": 8.0,
            "success_rate": 0.90,
        }
        tester.save_baseline("test_model", baseline_metrics)
        
        # Test with improved metrics
        improved_metrics = {
            "avg_generation_time": 8.0,  # Faster
            "avg_quality_score": 80.0,  # Better quality
            "peak_vram_gb": 7.0,  # Less memory
            "success_rate": 0.95,  # Higher success
        }
        
        updated = tester.update_baseline_if_improved("test_model", improved_metrics)
        
        self.assertTrue(updated)
        self.assertEqual(tester.baselines["test_model"]["metrics"]["avg_generation_time"], 8.0)


class TestABTester(unittest.TestCase):
    """Test ABTester class."""
    
    def test_ab_test_config(self):
        """Test A/B test configuration."""
        config = ABTestConfig(
            model_a="model1",
            model_b="model2",
            test_prompts=["prompt1", "prompt2"],
            num_samples_per_prompt=3,
        )
        
        self.assertEqual(config.model_a, "model1")
        self.assertEqual(config.model_b, "model2")
        self.assertEqual(len(config.test_prompts), 2)
    
    def test_ab_result_statistics(self):
        """Test A/B test statistical calculations."""
        config = ABTestConfig(model_a="A", model_b="B", test_prompts=["test"])
        result = config.__class__.__module__
        
        from dream_cad.benchmark.ab_tester import ABTestResult
        
        result = ABTestResult(config=config)
        
        # Add sample data
        result.generation_time_a = [10, 11, 12, 10, 11]
        result.generation_time_b = [8, 9, 8, 9, 8]
        result.quality_scores_a = [70, 75, 72, 73, 71]
        result.quality_scores_b = [80, 82, 81, 83, 79]
        result.samples_model_a = 5
        result.samples_model_b = 5
        result.success_count_a = 5
        result.success_count_b = 5
        
        # Calculate statistics
        result.calculate_statistics()
        
        # Model B should win on speed and quality
        self.assertEqual(result.generation_time_winner, "B")
        self.assertEqual(result.quality_winner, "B")
        self.assertEqual(result.overall_winner, "B")


class TestBenchmarkRunner(unittest.TestCase):
    """Test BenchmarkRunner class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "benchmarks"
    
    def test_runner_initialization(self):
        """Test benchmark runner initialization."""
        runner = BenchmarkRunner(
            output_dir=self.output_dir,
            models=["triposr", "stable-fast-3d"],
        )
        
        self.assertEqual(len(runner.models), 2)
        self.assertIsNotNone(runner.quality_assessor)
        self.assertIsNotNone(runner.performance_tracker)
    
    def test_report_generation(self):
        """Test generating comprehensive report."""
        runner = BenchmarkRunner(output_dir=self.output_dir)
        
        # Add mock results
        config = BenchmarkConfig(model_name="test", prompt="test")
        runner.all_results["model1"] = [
            BenchmarkResult(config=config, success=True, generation_time_seconds=10.0, overall_quality_score=75.0),
            BenchmarkResult(config=config, success=True, generation_time_seconds=11.0, overall_quality_score=80.0),
        ]
        
        report = runner.generate_comprehensive_report()
        
        self.assertIn("results", report)
        self.assertIn("model1", report["results"])
        self.assertIn("recommendations", report)
    
    def test_recommendations_generation(self):
        """Test generating model recommendations."""
        runner = BenchmarkRunner(output_dir=self.output_dir)
        
        report = {
            "results": {
                "fast_model": {
                    "avg_generation_time": 5.0,
                    "avg_quality_score": 70.0,
                    "avg_vram_gb": 6.0,
                },
                "quality_model": {
                    "avg_generation_time": 15.0,
                    "avg_quality_score": 90.0,
                    "avg_vram_gb": 12.0,
                },
                "balanced_model": {
                    "avg_generation_time": 10.0,
                    "avg_quality_score": 80.0,
                    "avg_vram_gb": 8.0,
                },
            }
        }
        
        recommendations = runner._generate_recommendations(report)
        
        self.assertIn("speed_critical", recommendations)
        self.assertIn("quality_critical", recommendations)
        self.assertIn("balanced", recommendations)
        
        # Check correct model selection
        self.assertIn("fast_model", recommendations["speed_critical"])
        self.assertIn("quality_model", recommendations["quality_critical"])


if __name__ == "__main__":
    unittest.main()