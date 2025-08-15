#!/usr/bin/env python3
"""
Test suite for Story 9: MVDream 3D Generation Pipeline Setup
Verifies all acceptance criteria for 3D mesh generation.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import yaml


class TestMVDream3DGeneration(unittest.TestCase):
    """Test MVDream 3D generation pipeline functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.config_file = self.project_root / "configs" / "mvdream-sd21.yaml"
        self.output_dir = self.project_root / "outputs" / "3d_test"
        self.results_dir = self.project_root / "tests" / "results"
        os.chdir(self.project_root)
        # Always use test mode for automated tests
        os.environ["MVDREAM_TEST_MODE"] = "true"
    
    def test_config_file_exists(self):
        """Test that mvdream-sd21.yaml config file exists."""
        self.assertTrue(
            self.config_file.exists(),
            f"Config file not found at {self.config_file}"
        )
        
        # Load and validate config
        with self.config_file.open() as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ["model", "generation", "output", "hardware", "logging"]
        for section in required_sections:
            self.assertIn(section, config, f"Config missing section: {section}")
        
        # Check critical settings
        self.assertEqual(config["model"]["device"], "cuda")
        self.assertEqual(config["model"]["precision"], "fp16")
        self.assertLessEqual(config["hardware"]["max_vram_gb"], 20)
    
    def test_memory_efficient_mode_configured(self):
        """Test that memory-efficient settings are configured."""
        with self.config_file.open() as f:
            config = yaml.safe_load(f)
        
        generation = config["generation"]
        
        # Check memory optimization settings
        self.assertTrue(
            generation.get("enable_attention_slicing", False),
            "Attention slicing not enabled for memory efficiency"
        )
        
        self.assertTrue(
            generation.get("enable_vae_slicing", False),
            "VAE slicing not enabled for memory efficiency"
        )
        
        self.assertEqual(
            generation.get("batch_size", 2), 1,
            "Batch size should be 1 for memory efficiency"
        )
        
        # Check VRAM limit
        self.assertLessEqual(
            config["hardware"]["max_vram_gb"], 20,
            "VRAM limit should be ≤20GB for 24GB card"
        )
    
    def test_ceramic_mug_generation(self):
        """Test generation of 'a ceramic coffee mug'."""
        # Run generation with test prompt
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/generate_3d.py",
             "a ceramic coffee mug", "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        self.assertEqual(
            result.returncode, 0,
            f"3D generation failed:\n{result.stdout}\n{result.stderr}"
        )
        
        # Check for success in output
        self.assertIn(
            "Generation Complete",
            result.stdout,
            "Generation completion message not found"
        )
    
    def test_obj_output_format(self):
        """Test that output is saved in OBJ format."""
        # Run generation
        subprocess.run(
            ["poetry", "run", "python", "scripts/generate_3d.py",
             "test object", "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        # Check for OBJ files
        obj_files = list(self.output_dir.glob("**/output.obj"))
        self.assertGreater(
            len(obj_files), 0,
            "No OBJ files found in output directory"
        )
        
        # Validate OBJ format (basic check)
        if obj_files:
            with obj_files[0].open() as f:
                content = f.read()
                # OBJ files should have vertex lines
                self.assertIn("v ", content, "OBJ file missing vertex data")
                # OBJ files should have face lines
                self.assertIn("f ", content, "OBJ file missing face data")
    
    def test_generation_time_tracking(self):
        """Test that generation time is tracked and reasonable."""
        # Run generation
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/generate_3d.py",
             "test", "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        # Check metrics file
        metrics_file = self.results_dir / "3d_generation_metrics.json"
        if metrics_file.exists():
            with metrics_file.open() as f:
                metrics = json.load(f)
            
            self.assertIn("generation_time_minutes", metrics)
            
            # In test mode, should be very fast (< 1 minute)
            self.assertLess(
                metrics["generation_time_minutes"], 1,
                "Test mode generation took too long"
            )
    
    def test_web_interface_config(self):
        """Test that web interface is configured."""
        with self.config_file.open() as f:
            config = yaml.safe_load(f)
        
        interface = config.get("interface", {})
        
        # Check interface settings
        self.assertEqual(
            interface.get("host", ""), "localhost",
            "Web interface should bind to localhost"
        )
        
        self.assertEqual(
            interface.get("port", 0), 7860,
            "Web interface should use port 7860"
        )
        
        self.assertTrue(
            interface.get("enable_gradio", False),
            "Gradio interface should be enabled"
        )
    
    def test_gpu_temperature_monitoring(self):
        """Test that GPU temperature monitoring works."""
        # Run generation with monitoring
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/generate_3d.py",
             "test", "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        # Check for temperature in output
        self.assertIn(
            "GPU Temp",
            result.stdout,
            "GPU temperature not reported"
        )
        
        # Check metrics for temperature data
        metrics_file = self.results_dir / "3d_generation_metrics.json"
        if metrics_file.exists():
            with metrics_file.open() as f:
                metrics = json.load(f)
            
            if "gpu_temperature" in metrics:
                temps = metrics["gpu_temperature"]
                self.assertIn("max", temps)
                self.assertIn("avg", temps)
                
                # Temperature should be under limit
                self.assertLess(
                    temps.get("max", 100), 83,
                    "GPU temperature exceeded 83°C limit"
                )
    
    def test_poe_generate_3d_task(self):
        """Test that 'poe generate-3d' task is defined."""
        pyproject = self.project_root / "pyproject.toml"
        content = pyproject.read_text()
        
        # Check task is defined
        self.assertIn(
            "generate-3d",
            content,
            "generate-3d task not found in pyproject.toml"
        )
        
        # Should reference the 3D generation script
        self.assertIn(
            "generate_3d.py",
            content,
            "generate-3d task doesn't reference correct script"
        )
        
        # Check web interface task
        self.assertIn(
            "generate-3d-web",
            content,
            "generate-3d-web task not found"
        )
    
    def test_results_documentation(self):
        """Test that pipeline test results are documented."""
        # Run generation
        subprocess.run(
            ["poetry", "run", "python", "scripts/generate_3d.py",
             "test", "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        # Check for results file
        results_file = self.results_dir / "3d_generation.md"
        self.assertTrue(
            results_file.exists(),
            f"Results documentation not created at {results_file}"
        )
        
        # Check content
        content = results_file.read_text()
        required_sections = [
            "Configuration",
            "Performance",
            "Memory Usage",
            "GPU Temperature",
            "Validation",
        ]
        
        for section in required_sections:
            self.assertIn(
                section,
                content,
                f"Results missing section: {section}"
            )
    
    def test_custom_prompt_support(self):
        """Test that custom prompts are accepted."""
        # Run with custom prompt
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/generate_3d.py",
             "a red dragon sculpture", "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        self.assertEqual(
            result.returncode, 0,
            f"Generation with custom prompt failed:\n{result.stderr}"
        )
        
        # Check that prompt is recorded
        self.assertIn(
            "a red dragon sculpture",
            result.stdout,
            "Custom prompt not shown in output"
        )
    
    def test_memory_usage_tracking(self):
        """Test that memory usage is tracked."""
        # Run generation
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/generate_3d.py",
             "test", "--test-mode"],
            capture_output=True,
            text=True,
        )
        
        # Check for memory reporting
        self.assertIn(
            "Memory",
            result.stdout,
            "Memory usage not reported"
        )
        
        # Check metrics
        metrics_file = self.results_dir / "3d_generation_metrics.json"
        if metrics_file.exists():
            with metrics_file.open() as f:
                metrics = json.load(f)
            
            self.assertIn("initial_memory", metrics)
            self.assertIn("final_memory", metrics)
            self.assertIn("memory_delta_gb", metrics)
            
            # Check that memory stayed reasonable
            final_mem = metrics.get("final_memory", {})
            if "vram_used_gb" in final_mem:
                self.assertLess(
                    final_mem["vram_used_gb"], 20,
                    "VRAM usage exceeded 20GB limit"
                )
    
    def test_3d_generation_script_exists(self):
        """Test that 3D generation script exists and is executable."""
        script = self.project_root / "scripts" / "generate_3d.py"
        
        self.assertTrue(
            script.exists(),
            f"3D generation script not found at {script}"
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


def run_tests():
    """Run all 3D generation tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMVDream3DGeneration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)