#!/usr/bin/env python3
"""
Test suite for Story 8: MVDream 2D Generation Testing
Verifies all acceptance criteria for 2D multi-view generation.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


class TestMVDream2DGeneration(unittest.TestCase):
    """Test MVDream 2D multi-view generation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.output_dir = self.project_root / "outputs" / "2d_test"
        self.results_dir = self.project_root / "tests" / "results"
        os.chdir(self.project_root)
        # Set test mode for all tests
        os.environ["MVDREAM_TEST_MODE"] = "true"
    
    def test_2d_generation_script_exists(self):
        """Test that 2D generation test script exists."""
        script = self.project_root / "scripts" / "test_2d_generation.py"
        
        self.assertTrue(
            script.exists(),
            f"2D generation script not found at {script}"
        )
        
        # Check shebang
        content = script.read_text()
        self.assertTrue(
            content.startswith("#!/usr/bin/env python3"),
            "Script missing shebang"
        )
        
        # Check executable
        self.assertTrue(
            os.access(script, os.X_OK),
            "Script is not executable"
        )
    
    def test_standard_prompt_generates_4_views(self):
        """Test that standard prompt generates 4 views."""
        # Run generation with test prompt
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/test_2d_generation.py",
             "--prompt", "an astronaut riding a horse",
             "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        self.assertEqual(
            result.returncode, 0,
            f"Generation failed:\n{result.stdout}\n{result.stderr}"
        )
        
        # Check that 4 images were created
        images = list(self.output_dir.glob("view_*.png"))
        self.assertEqual(
            len(images), 4,
            f"Expected 4 images, found {len(images)}"
        )
        
        # Check image names
        expected_views = ["front", "back", "left", "right"]
        for i, view in enumerate(expected_views):
            image_path = self.output_dir / f"view_{i:02d}_{view}.png"
            self.assertTrue(
                image_path.exists(),
                f"Missing image for {view} view"
            )
    
    def test_images_saved_to_correct_directory(self):
        """Test that images are saved to outputs/2d_test/."""
        # Directory should exist after generation
        self.assertTrue(
            self.output_dir.exists(),
            f"Output directory not created at {self.output_dir}"
        )
        
        # Should contain PNG files
        images = list(self.output_dir.glob("*.png"))
        self.assertGreater(
            len(images), 0,
            "No PNG files found in output directory"
        )
    
    def test_consistency_verification(self):
        """Test that consistency verification works."""
        # Run generation
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/test_2d_generation.py",
             "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        # Check for consistency verification in output
        self.assertIn(
            "Verifying output consistency",
            result.stdout,
            "Consistency verification not performed"
        )
        
        self.assertIn(
            "Consistency verified",
            result.stdout,
            "Consistency result not reported"
        )
        
        # Check metrics file
        metrics_file = self.output_dir / "metrics.json"
        if metrics_file.exists():
            with metrics_file.open() as f:
                metrics = json.load(f)
            
            self.assertIn("consistency_verified", metrics)
    
    def test_no_cuda_oom_in_test_mode(self):
        """Test that generation completes without CUDA OOM in test mode."""
        # Run generation
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/test_2d_generation.py",
             "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        # Should not have CUDA OOM
        self.assertNotIn(
            "CUDA out of memory",
            result.stdout + result.stderr,
            "CUDA OOM error occurred in test mode"
        )
        
        # Check metrics
        metrics_file = self.output_dir / "metrics.json"
        if metrics_file.exists():
            with metrics_file.open() as f:
                metrics = json.load(f)
            
            self.assertFalse(
                metrics.get("cuda_oom", False),
                "CUDA OOM flag set in metrics"
            )
    
    def test_generation_time_logged(self):
        """Test that generation time is logged."""
        # Run generation
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/test_2d_generation.py",
             "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        # Check for time logging
        self.assertIn(
            "Generation completed in",
            result.stdout,
            "Generation time not logged"
        )
        
        # Check metrics file
        metrics_file = self.output_dir / "metrics.json"
        if metrics_file.exists():
            with metrics_file.open() as f:
                metrics = json.load(f)
            
            self.assertIn("generation_time_seconds", metrics)
            self.assertIn("generation_time_minutes", metrics)
            
            # Should be under 5 minutes in test mode
            self.assertLess(
                metrics["generation_time_minutes"], 5,
                "Generation took more than 5 minutes"
            )
    
    def test_memory_monitoring(self):
        """Test that memory usage is monitored."""
        # Run generation
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/test_2d_generation.py",
             "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        # Check for memory monitoring
        self.assertIn(
            "RAM",
            result.stdout,
            "RAM usage not reported"
        )
        
        # Check metrics file
        metrics_file = self.output_dir / "metrics.json"
        if metrics_file.exists():
            with metrics_file.open() as f:
                metrics = json.load(f)
            
            self.assertIn("initial_memory", metrics)
            self.assertIn("peak_memory", metrics)
            self.assertIn("ram_delta_gb", metrics)
            
            # Peak memory should have RAM info
            peak = metrics.get("peak_memory", {})
            self.assertIn("ram_used_gb", peak)
            self.assertIn("ram_percent", peak)
    
    def test_poe_test_2d_task(self):
        """Test that 'poe test-2d' task is defined."""
        pyproject = self.project_root / "pyproject.toml"
        content = pyproject.read_text()
        
        # Check task is defined
        self.assertIn(
            "test-2d",
            content,
            "test-2d task not found in pyproject.toml"
        )
        
        # Should reference the test script
        self.assertIn(
            "test_2d_generation.py",
            content,
            "test-2d task doesn't reference correct script"
        )
        
        # Should use --test-mode flag
        self.assertIn(
            "--test-mode",
            content,
            "test-2d task doesn't use test mode"
        )
    
    def test_results_documentation(self):
        """Test that results are documented."""
        # Run generation
        subprocess.run(
            ["poetry", "run", "python", "scripts/test_2d_generation.py",
             "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        # Check for results file
        results_file = self.results_dir / "2d_generation.md"
        self.assertTrue(
            results_file.exists(),
            f"Results documentation not created at {results_file}"
        )
        
        # Check content
        content = results_file.read_text()
        required_sections = [
            "Test Configuration",
            "Performance Metrics",
            "Memory Usage",
            "Results",
            "Generated Images",
        ]
        
        for section in required_sections:
            self.assertIn(
                section,
                content,
                f"Results missing section: {section}"
            )
    
    def test_custom_prompt_support(self):
        """Test that custom prompts can be provided."""
        # Run with custom prompt
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/test_2d_generation.py",
             "--prompt", "a red bicycle",
             "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        self.assertEqual(
            result.returncode, 0,
            f"Generation with custom prompt failed:\n{result.stderr}"
        )
        
        # Check metrics for prompt
        metrics_file = self.output_dir / "metrics.json"
        if metrics_file.exists():
            with metrics_file.open() as f:
                metrics = json.load(f)
            
            self.assertEqual(
                metrics.get("prompt"),
                "a red bicycle",
                "Custom prompt not recorded in metrics"
            )
    
    def test_configurable_parameters(self):
        """Test that generation parameters are configurable."""
        # Run with custom parameters
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/test_2d_generation.py",
             "--guidance-scale", "10.0",
             "--num-inference-steps", "30",
             "--seed", "123",
             "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        self.assertEqual(
            result.returncode, 0,
            f"Generation with custom parameters failed:\n{result.stderr}"
        )
        
        # Check metrics
        metrics_file = self.output_dir / "metrics.json"
        if metrics_file.exists():
            with metrics_file.open() as f:
                metrics = json.load(f)
            
            self.assertEqual(metrics.get("guidance_scale"), 10.0)
            self.assertEqual(metrics.get("num_inference_steps"), 30)
            self.assertEqual(metrics.get("seed"), 123)
    
    def test_mock_generation_fallback(self):
        """Test that mock generation works when model unavailable."""
        # Force test mode
        os.environ["MVDREAM_TEST_MODE"] = "true"
        
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/test_2d_generation.py"],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        
        # Should still succeed
        self.assertEqual(
            result.returncode, 0,
            f"Mock generation failed:\n{result.stderr}"
        )
        
        # Should indicate test mode
        self.assertIn(
            "mock",
            result.stdout.lower(),
            "Mock generation not indicated"
        )


def run_tests():
    """Run all 2D generation tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMVDream2DGeneration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)