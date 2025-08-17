"""Main benchmark runner for comprehensive model evaluation."""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model_benchmark import ModelBenchmark, BenchmarkConfig, BenchmarkResult
from .quality_assessor import QualityAssessor
from .performance_tracker import PerformanceTracker
from .regression_tester import RegressionTester
from .ab_tester import ABTester, ABTestConfig

logger = logging.getLogger(__name__)


# Standard test prompts for cross-model comparison
STANDARD_PROMPTS = {
    "simple": [
        "a simple wooden cube",
        "a red sphere",
        "a blue cylinder",
    ],
    "medium": [
        "a ceramic coffee mug with handle",
        "a leather wallet",
        "a wooden chair",
    ],
    "complex": [
        "an ornate golden goblet with gemstones",
        "a detailed dragon sculpture",
        "a futuristic motorcycle",
    ],
    "stylized": [
        "a low-poly cartoon character",
        "an anime-style sword",
        "a pixelated game asset",
    ],
    "organic": [
        "a realistic human head sculpture",
        "a detailed tree with branches",
        "a coral reef formation",
    ],
    "architectural": [
        "a medieval castle tower",
        "a modern skyscraper",
        "an ancient temple",
    ],
}

# Model configurations optimized for each model
MODEL_CONFIGS = {
    "mvdream": {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "n_views": 4,
        "resolution": 256,
    },
    "triposr": {
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "resolution": 512,
    },
    "stable-fast-3d": {
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "target_polycount": 20000,
    },
    "trellis": {
        "num_inference_steps": 40,
        "guidance_scale": 7.5,
        "quality_mode": "balanced",
    },
    "hunyuan3d-mini": {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "target_polycount": 30000,
    },
}


class BenchmarkRunner:
    """Comprehensive benchmark runner for all models."""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        models: Optional[List[str]] = None,
        enable_quality_assessment: bool = True,
        enable_regression_testing: bool = True,
        enable_continuous: bool = False,
        gpu_cost_per_hour: float = 0.5,
    ):
        """Initialize benchmark runner."""
        self.output_dir = output_dir or Path("benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Models to benchmark
        self.models = models or list(MODEL_CONFIGS.keys())
        
        # Components
        self.quality_assessor = QualityAssessor() if enable_quality_assessment else None
        self.performance_tracker = PerformanceTracker(
            storage_dir=self.output_dir / "performance",
            gpu_cost_per_hour=gpu_cost_per_hour,
        )
        self.regression_tester = RegressionTester(
            baseline_dir=self.output_dir / "baselines",
        ) if enable_regression_testing else None
        self.ab_tester = ABTester(
            output_dir=self.output_dir / "ab_tests",
        )
        
        self.enable_continuous = enable_continuous
        
        # Results storage
        self.all_results: Dict[str, List[BenchmarkResult]] = {}
    
    def run_comprehensive_benchmark(
        self,
        prompt_category: str = "all",
        num_runs: int = 3,
        save_outputs: bool = True,
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark across all models."""
        logger.info(f"Starting comprehensive benchmark for {len(self.models)} models")
        
        # Select prompts
        if prompt_category == "all":
            prompts = []
            for category_prompts in STANDARD_PROMPTS.values():
                prompts.extend(category_prompts[:1])  # Take first from each category
        else:
            prompts = STANDARD_PROMPTS.get(prompt_category, STANDARD_PROMPTS["simple"])
        
        # Benchmark each model
        for model_name in self.models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking model: {model_name}")
            logger.info(f"{'='*60}")
            
            results = self.benchmark_model(
                model_name=model_name,
                prompts=prompts,
                num_runs=num_runs,
                save_outputs=save_outputs,
            )
            
            self.all_results[model_name] = results
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        # Save report
        report_file = self.output_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with report_file.open("w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive benchmark complete. Report saved to {report_file}")
        
        return report
    
    def benchmark_model(
        self,
        model_name: str,
        prompts: List[str],
        num_runs: int = 3,
        save_outputs: bool = True,
    ) -> List[BenchmarkResult]:
        """Benchmark a single model."""
        # Get model configuration
        model_config = MODEL_CONFIGS.get(model_name, {})
        
        # Create benchmark configs
        configs = []
        for prompt in prompts:
            config = BenchmarkConfig(
                model_name=model_name,
                prompt=prompt,
                test_runs=num_runs,
                warmup_runs=1,
                save_outputs=save_outputs,
                output_dir=self.output_dir / "outputs" / model_name,
                **model_config,
            )
            configs.append(config)
        
        # Create model benchmark
        model_benchmark = ModelBenchmark(
            model_name=model_name,
            output_dir=self.output_dir / model_name,
        )
        
        # Run benchmarks
        results = model_benchmark.run_benchmark_suite(configs)
        
        # Assess quality if enabled
        if self.quality_assessor and save_outputs:
            for result in results:
                if result.output_path and Path(result.output_path).exists():
                    quality_metrics = self.quality_assessor.assess_mesh_file(Path(result.output_path))
                    
                    # Update result with quality scores
                    result.mesh_quality_score = quality_metrics.overall_mesh_quality
                    result.texture_quality_score = quality_metrics.overall_texture_quality
                    result.overall_quality_score = (
                        quality_metrics.overall_mesh_quality + 
                        quality_metrics.overall_texture_quality +
                        quality_metrics.game_ready_score
                    ) / 3
        
        # Track performance
        if results:
            performance_metrics = self.performance_tracker.calculate_metrics(model_name, results)
            if performance_metrics:
                self.performance_tracker.track_performance(model_name, performance_metrics)
        
        # Test for regression if enabled
        if self.regression_tester and results:
            current_metrics = {
                "avg_generation_time": sum(r.generation_time_seconds for r in results if r.success) / len([r for r in results if r.success]),
                "avg_quality_score": sum(r.overall_quality_score for r in results if r.success) / len([r for r in results if r.success]),
                "peak_vram_gb": max(r.peak_vram_gb for r in results if r.peak_vram_gb > 0) if any(r.peak_vram_gb > 0 for r in results) else 0,
                "success_rate": len([r for r in results if r.success]) / len(results),
            }
            
            regression_result = self.regression_tester.test_regression(model_name, current_metrics)
            
            if regression_result.has_regression:
                logger.warning(f"Regression detected for {model_name}: {regression_result.regression_summary}")
            else:
                # Update baseline if improved
                self.regression_tester.update_baseline_if_improved(model_name, current_metrics)
        
        # Cleanup
        model_benchmark.unload_model()
        
        return results
    
    def run_ab_test(
        self,
        model_a: str,
        model_b: str,
        prompts: Optional[List[str]] = None,
        num_samples: int = 5,
    ) -> Any:
        """Run A/B test between two models."""
        if prompts is None:
            # Use a mix of prompts
            prompts = [
                STANDARD_PROMPTS["simple"][0],
                STANDARD_PROMPTS["medium"][0],
                STANDARD_PROMPTS["complex"][0],
            ]
        
        config = ABTestConfig(
            model_a=model_a,
            model_b=model_b,
            test_prompts=prompts,
            num_samples_per_prompt=num_samples,
        )
        
        # Create model benchmarks
        benchmark_a = ModelBenchmark(model_a, self.output_dir / model_a)
        benchmark_b = ModelBenchmark(model_b, self.output_dir / model_b)
        
        # Run A/B test
        result = self.ab_tester.run_ab_test(config, benchmark_a, benchmark_b)
        
        # Cleanup
        benchmark_a.unload_model()
        benchmark_b.unload_model()
        
        return result
    
    def run_continuous_benchmark(
        self,
        interval_hours: int = 24,
        max_iterations: int = 0,
    ) -> None:
        """Run continuous benchmarking."""
        logger.info("Starting continuous benchmarking...")
        
        iteration = 0
        while self.enable_continuous and (max_iterations == 0 or iteration < max_iterations):
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Continuous benchmark iteration {iteration}")
            logger.info(f"{'='*60}")
            
            # Run lightweight benchmark
            for model_name in self.models:
                # Use single prompt for continuous monitoring
                results = self.benchmark_model(
                    model_name=model_name,
                    prompts=[STANDARD_PROMPTS["simple"][0]],
                    num_runs=1,
                    save_outputs=False,
                )
                
                # Check for significant changes
                if self.regression_tester and results:
                    current_metrics = {
                        "avg_generation_time": results[0].generation_time_seconds if results else 0,
                        "success_rate": 1.0 if results[0].success else 0.0,
                    }
                    
                    regression_result = self.regression_tester.test_regression(model_name, current_metrics)
                    if regression_result.has_regression:
                        logger.error(f"ALERT: Regression detected for {model_name}")
                        # Could send notification here
            
            # Sleep until next iteration
            if self.enable_continuous and (max_iterations == 0 or iteration < max_iterations):
                logger.info(f"Sleeping for {interval_hours} hours...")
                time.sleep(interval_hours * 3600)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "models_tested": self.models,
            "results": {},
            "performance_comparison": {},
            "quality_comparison": {},
            "recommendations": {},
        }
        
        # Compile results for each model
        for model_name, results in self.all_results.items():
            if not results:
                continue
            
            successful = [r for r in results if r.success]
            if not successful:
                continue
            
            # Calculate aggregate metrics
            model_report = {
                "total_tests": len(results),
                "successful_tests": len(successful),
                "success_rate": len(successful) / len(results) if results else 0,
                "avg_generation_time": sum(r.generation_time_seconds for r in successful) / len(successful) if successful else 0,
                "avg_quality_score": sum(r.overall_quality_score for r in successful) / len(successful) if successful and any(r.overall_quality_score for r in successful) else 0,
                "avg_vram_gb": sum(r.peak_vram_gb for r in successful) / len(successful) if successful and any(r.peak_vram_gb for r in successful) else 0,
                "total_time_seconds": sum(r.generation_time_seconds for r in successful),
            }
            
            report["results"][model_name] = model_report
        
        # Performance comparison
        if report["results"]:
            # Speed ranking
            speed_ranking = sorted(
                report["results"].items(),
                key=lambda x: x[1]["avg_generation_time"],
            )
            report["performance_comparison"]["fastest_models"] = [name for name, _ in speed_ranking[:3]]
            
            # Quality ranking
            quality_ranking = sorted(
                report["results"].items(),
                key=lambda x: x[1]["avg_quality_score"],
                reverse=True,
            )
            report["quality_comparison"]["highest_quality_models"] = [name for name, _ in quality_ranking[:3]]
            
            # Efficiency ranking (quality per second)
            efficiency_scores = []
            for name, metrics in report["results"].items():
                if metrics["avg_generation_time"] > 0:
                    efficiency = metrics["avg_quality_score"] / metrics["avg_generation_time"]
                    efficiency_scores.append((name, efficiency))
            
            efficiency_ranking = sorted(efficiency_scores, key=lambda x: x[1], reverse=True)
            report["performance_comparison"]["most_efficient_models"] = [name for name, _ in efficiency_ranking[:3]]
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)
        
        # Add performance tracker report
        if self.performance_tracker:
            perf_report = self.performance_tracker.generate_report()
            report["performance_trends"] = perf_report
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> Dict[str, str]:
        """Generate model selection recommendations."""
        recommendations = {}
        
        if not report["results"]:
            return recommendations
        
        # Find best model for different use cases
        models = report["results"]
        
        # Speed-critical applications
        fastest = min(models.items(), key=lambda x: x[1]["avg_generation_time"])
        recommendations["speed_critical"] = f"Use {fastest[0]} for time-critical applications (avg {fastest[1]['avg_generation_time']:.1f}s)"
        
        # Quality-critical applications
        highest_quality = max(models.items(), key=lambda x: x[1]["avg_quality_score"])
        recommendations["quality_critical"] = f"Use {highest_quality[0]} for quality-critical applications (score: {highest_quality[1]['avg_quality_score']:.1f})"
        
        # Memory-constrained environments
        lowest_memory = min(models.items(), key=lambda x: x[1]["avg_vram_gb"] if x[1]["avg_vram_gb"] > 0 else float('inf'))
        if lowest_memory[1]["avg_vram_gb"] > 0:
            recommendations["memory_constrained"] = f"Use {lowest_memory[0]} for memory-constrained environments ({lowest_memory[1]['avg_vram_gb']:.1f}GB)"
        
        # Balanced recommendation
        # Score = normalized quality * 0.4 + normalized speed * 0.3 + normalized memory * 0.3
        scores = {}
        for name, metrics in models.items():
            quality_score = metrics["avg_quality_score"] / 100 if metrics["avg_quality_score"] > 0 else 0
            speed_score = 1 - min(metrics["avg_generation_time"] / 60, 1)  # Normalize to 0-1 (60s max)
            memory_score = 1 - min(metrics["avg_vram_gb"] / 24, 1) if metrics["avg_vram_gb"] > 0 else 0.5  # Normalize to 0-1 (24GB max)
            
            balanced_score = quality_score * 0.4 + speed_score * 0.3 + memory_score * 0.3
            scores[name] = balanced_score
        
        best_balanced = max(scores.items(), key=lambda x: x[1])
        recommendations["balanced"] = f"Use {best_balanced[0]} for balanced performance (score: {best_balanced[1]:.2f})"
        
        return recommendations


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description="Run comprehensive 3D model benchmarks")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to benchmark (default: all)",
    )
    parser.add_argument(
        "--category",
        default="all",
        choices=["all", "simple", "medium", "complex", "stylized", "organic", "architectural"],
        help="Prompt category to test",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per prompt",
    )
    parser.add_argument(
        "--ab-test",
        nargs=2,
        metavar=("MODEL_A", "MODEL_B"),
        help="Run A/B test between two models",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuous benchmarking",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create runner
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        models=args.models,
        enable_continuous=args.continuous,
    )
    
    if args.ab_test:
        # Run A/B test
        result = runner.run_ab_test(args.ab_test[0], args.ab_test[1])
        print(f"\nA/B Test Result: {result.summary}")
        
    elif args.continuous:
        # Run continuous benchmarking
        runner.run_continuous_benchmark()
        
    else:
        # Run comprehensive benchmark
        report = runner.run_comprehensive_benchmark(
            prompt_category=args.category,
            num_runs=args.runs,
        )
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for model, results in report["results"].items():
            print(f"\n{model}:")
            print(f"  Success Rate: {results['success_rate']:.1%}")
            print(f"  Avg Generation Time: {results['avg_generation_time']:.1f}s")
            print(f"  Avg Quality Score: {results['avg_quality_score']:.1f}")
            print(f"  Avg VRAM Usage: {results['avg_vram_gb']:.1f}GB")
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        for use_case, recommendation in report["recommendations"].items():
            print(f"\n{use_case.replace('_', ' ').title()}:")
            print(f"  {recommendation}")


if __name__ == "__main__":
    main()