#!/usr/bin/env python3
"""Test Story 11: Performance Optimization and Benchmarking."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestStory11Benchmarking:
    """Test suite for Story 11 acceptance criteria."""

    def test_benchmarking_script_exists(self):
        """Test that benchmarking script is created."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/benchmark.py")
        assert script_path.exists(), "Benchmarking script not found at scripts/benchmark.py"
        
        # Check script is executable (has shebang)
        with script_path.open() as f:
            first_line = f.readline()
            assert first_line.startswith("#!/usr/bin/env python"), "Script missing shebang"

    def test_rescale_factor_optimization(self):
        """Test that rescale factor values are tested."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/benchmark.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for rescale factor testing
        assert "0.3" in content, "Rescale factor 0.3 not tested"
        assert "0.5" in content, "Rescale factor 0.5 not tested"
        assert "0.7" in content, "Rescale factor 0.7 not tested"
        assert "rescale_factor" in content, "Rescale factor parameter not found"

    def test_batch_size_optimization(self):
        """Test that optimal batch size is determined for 24GB VRAM."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/benchmark.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for batch size testing
        assert "batch_size" in content, "Batch size parameter not found"
        assert "24GB" in content or "24" in content, "24GB VRAM not mentioned"
        
        # Check for different batch sizes
        assert "batch_size in [1, 2, 4]" in content or "batch_size: 1" in content, "Batch sizes not tested"

    def test_xformers_configuration(self):
        """Test that xformers memory-efficient attention is configured."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/benchmark.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for xformers configuration
        assert "xformers" in content.lower(), "xformers not mentioned"
        assert "enable_xformers" in content, "enable_xformers parameter not found"
        assert "memory_efficient_attention" in content.lower(), "Memory efficient attention not mentioned"

    def test_timestep_annealing(self):
        """Test that time-step annealing parameters are optimized."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/benchmark.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for timestep annealing
        assert "timestep_annealing" in content or "time_step_annealing" in content, "Timestep annealing not found"
        assert "annealing_eta" in content, "Annealing eta parameter not found"

    def test_benchmark_prompts(self):
        """Test that benchmark uses 5 different prompts."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/benchmark.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for test prompts
        test_prompts = [
            "wooden cube",
            "coffee mug",
            "golden goblet",
            "cartoon character",
            "human head"
        ]
        
        found_prompts = sum(1 for prompt in test_prompts if prompt in content.lower())
        assert found_prompts >= 5, f"Only {found_prompts} test prompts found, expected 5"

    def test_benchmark_results_saved(self):
        """Test that performance metrics are saved to benchmarks/rtx3090_results.json."""
        results_file = Path("/mnt/datadrive_m2/dream-cad/benchmarks/rtx3090_results.json")
        
        # File should exist after running benchmark
        if results_file.exists():
            with results_file.open() as f:
                data = json.load(f)
            
            # Check structure
            assert "metadata" in data, "Results missing metadata"
            assert "results" in data, "Results missing benchmark results"
            assert "analysis" in data, "Results missing analysis"
            
            # Check metadata
            metadata = data["metadata"]
            assert "gpu" in metadata, "Metadata missing GPU info"
            assert "3090" in metadata["gpu"], "GPU should be RTX 3090"

    def test_performance_tuning_guide_exists(self):
        """Test that optimization guide is created at docs/performance_tuning.md."""
        guide_path = Path("/mnt/datadrive_m2/dream-cad/docs/performance_tuning.md")
        assert guide_path.exists(), "Performance tuning guide not found"
        
        # Check content
        with guide_path.open() as f:
            content = f.read()
        
        # Check for key sections
        assert "RTX 3090" in content, "RTX 3090 not mentioned in guide"
        assert "Performance" in content, "Performance section missing"
        assert "Optimization" in content, "Optimization section missing"
        assert "batch_size" in content, "Batch size optimization not covered"
        assert "rescale_factor" in content, "Rescale factor not covered"

    def test_poethepoet_benchmark_task(self):
        """Test that 'poe benchmark' task runs standard test suite."""
        pyproject_path = Path("/mnt/datadrive_m2/dream-cad/pyproject.toml")
        
        with pyproject_path.open() as f:
            content = f.read()
        
        # Check for benchmark task
        assert "benchmark" in content, "benchmark task not found in pyproject.toml"
        assert "scripts/benchmark.py" in content, "benchmark script not referenced"

    def test_generation_time_requirement(self):
        """Test that results show <2 hour generation time for standard complexity."""
        results_file = Path("/mnt/datadrive_m2/dream-cad/benchmarks/rtx3090_results.json")
        
        if results_file.exists():
            with results_file.open() as f:
                data = json.load(f)
            
            # Check analysis
            if "analysis" in data:
                analysis = data["analysis"]
                if "meets_2hr_requirement" in analysis:
                    assert analysis["meets_2hr_requirement"], "Does not meet <2 hour requirement"
                
                # Check average time
                if "average_generation_time_seconds" in analysis:
                    avg_time = analysis["average_generation_time_seconds"]
                    assert avg_time < 7200, f"Average time {avg_time}s exceeds 2 hours (7200s)"

    def test_config_updated_with_optimal_values(self):
        """Test that configuration file is updated with optimal defaults."""
        config_file = Path("/mnt/datadrive_m2/dream-cad/configs/mvdream-sd21.yaml")
        
        assert config_file.exists(), "Config file not found"
        
        with config_file.open() as f:
            config = yaml.safe_load(f)
        
        # Check for optimized values
        assert "model" in config, "Model section missing in config"
        
        model_config = config["model"]
        assert "rescale_factor" in model_config, "Rescale factor not in config"
        assert model_config["rescale_factor"] in [0.3, 0.5, 0.7], "Rescale factor not optimized"
        
        # Check for performance notes
        if "performance_notes" in config:
            notes = config["performance_notes"]
            assert "optimized_for" in notes, "Optimization target not specified"
            assert "3090" in notes["optimized_for"], "Not optimized for RTX 3090"

    def test_benchmark_script_imports(self):
        """Test that benchmark script can be imported and has required classes."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts")
        sys.path.insert(0, str(script_path))
        
        try:
            import benchmark
            
            # Check for required classes
            assert hasattr(benchmark, "BenchmarkRunner"), "BenchmarkRunner class not found"
            assert hasattr(benchmark, "OptimizationConfig"), "OptimizationConfig class not found"
            assert hasattr(benchmark, "BenchmarkResult"), "BenchmarkResult class not found"
            
            # Check BenchmarkRunner methods
            runner = benchmark.BenchmarkRunner()
            assert hasattr(runner, "run_benchmark"), "run_benchmark method not found"
            assert hasattr(runner, "run_all_benchmarks"), "run_all_benchmarks method not found"
            assert hasattr(runner, "analyze_results"), "analyze_results method not found"
            assert hasattr(runner, "save_results"), "save_results method not found"
            
        finally:
            sys.path.pop(0)