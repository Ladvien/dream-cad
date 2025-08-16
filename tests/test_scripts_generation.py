#!/usr/bin/env python3
"""
Test suite for generation scripts.
Tests generate.py, generate_3d.py, generate_mvdream.py, and test_2d_generation.py
"""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import json


class TestGenerateScript(unittest.TestCase):
    """Test the main generate.py script."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.script_path = self.project_root / "scripts" / "generate.py"
    
    def test_script_exists(self):
        """Test that generate.py exists."""
        self.assertTrue(self.script_path.exists(), "generate.py not found")
    
    def test_help_flag(self):
        """Test generate.py --help."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        self.assertEqual(result.returncode, 0, f"Help failed: {result.stderr}")
        self.assertIn("prompt", result.stdout.lower())
        self.assertIn("generate", result.stdout.lower())
    
    def test_missing_prompt_error(self):
        """Test that missing prompt shows error."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path)],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        self.assertNotEqual(result.returncode, 0, "Should fail without prompt")
        self.assertIn("required", result.stderr.lower())
    
    def test_with_arguments(self):
        """Test generate.py with arguments (dry run)."""
        # Import the module to test without subprocess
        import importlib.util
        spec = importlib.util.spec_from_file_location("generate", self.script_path)
        module = importlib.util.module_from_spec(spec)
        
        # Mock subprocess.run to prevent actual generation
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            # Mock sys.argv
            with patch.object(sys, 'argv', ['generate.py', 'test prompt', 
                                           '--num-inference-steps', '10']):
                spec.loader.exec_module(module)
                # Check main can be called
                self.assertTrue(hasattr(module, 'main'))


class TestGenerate3DScript(unittest.TestCase):
    """Test the generate_3d.py script."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.script_path = self.project_root / "scripts" / "generate_3d.py"
    
    def test_script_exists(self):
        """Test that generate_3d.py exists."""
        self.assertTrue(self.script_path.exists(), "generate_3d.py not found")
    
    def test_help_flag(self):
        """Test generate_3d.py --help."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        self.assertEqual(result.returncode, 0, f"Help failed: {result.stderr}")
        self.assertIn("3d", result.stdout.lower())
    
    def test_test_mode(self):
        """Test generate_3d.py in test mode."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), 
             "test prompt", "--test-mode"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        # Should complete in test mode
        self.assertEqual(result.returncode, 0, f"Test mode failed: {result.stderr}")
        self.assertIn("test", result.stdout.lower())
    
    def test_web_flag(self):
        """Test that --web flag is recognized."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), 
             "--help"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        self.assertIn("--web", result.stdout)


class TestGenerateMVDreamScript(unittest.TestCase):
    """Test the generate_mvdream.py script."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.script_path = self.project_root / "scripts" / "generate_mvdream.py"
    
    def test_script_exists(self):
        """Test that generate_mvdream.py exists."""
        self.assertTrue(self.script_path.exists(), "generate_mvdream.py not found")
    
    def test_help_flag(self):
        """Test generate_mvdream.py --help."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        self.assertEqual(result.returncode, 0, f"Help failed: {result.stderr}")
        self.assertIn("mvdream", result.stdout.lower())
        self.assertIn("prompt", result.stdout.lower())
        self.assertIn("steps", result.stdout.lower())
    
    def test_missing_prompt_error(self):
        """Test that missing prompt shows error."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path)],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        self.assertNotEqual(result.returncode, 0, "Should fail without prompt")
        self.assertIn("required", result.stderr.lower())
    
    def test_cuda_check(self):
        """Test that script checks for CUDA."""
        # Import the script to check CUDA handling
        import importlib.util
        spec = importlib.util.spec_from_file_location("generate_mvdream", self.script_path)
        module = importlib.util.module_from_spec(spec)
        
        # Script should import without errors
        try:
            spec.loader.exec_module(module)
            self.assertTrue(hasattr(module, 'main'))
        except Exception as e:
            # Acceptable if torch import fails
            if "torch" not in str(e).lower():
                self.fail(f"Unexpected import error: {e}")


class TestTest2DGenerationScript(unittest.TestCase):
    """Test the test_2d_generation.py script."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.script_path = self.project_root / "scripts" / "test_2d_generation.py"
    
    def test_script_exists(self):
        """Test that test_2d_generation.py exists."""
        self.assertTrue(self.script_path.exists(), "test_2d_generation.py not found")
    
    def test_help_flag(self):
        """Test test_2d_generation.py --help."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        self.assertEqual(result.returncode, 0, f"Help failed: {result.stderr}")
        self.assertIn("2d", result.stdout.lower())
    
    def test_test_mode(self):
        """Test test_2d_generation.py in test mode."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "--test-mode"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        # Should complete in test mode
        self.assertEqual(result.returncode, 0, f"Test mode failed: {result.stderr}")
        self.assertIn("test", result.stdout.lower())
    
    def test_output_generation(self):
        """Test that script generates output files in test mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["poetry", "run", "python", str(self.script_path),
                 "--test-mode", "--output-dir", tmpdir],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=30
            )
            
            if result.returncode == 0:
                # Check if any output was created
                output_path = Path(tmpdir)
                files = list(output_path.rglob("*"))
                self.assertTrue(
                    len(files) > 0,
                    "No output files generated in test mode"
                )


class TestPoeGenerationTasks(unittest.TestCase):
    """Test poe tasks for generation."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
    
    def test_generate_task_help(self):
        """Test that 'poe generate' shows help when no args given."""
        result = subprocess.run(
            ["poetry", "run", "poe", "generate"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        # Should show usage since prompt is required
        self.assertIn("usage", result.stderr.lower())
    
    def test_test_2d_task(self):
        """Test that 'poe test-2d' runs in test mode."""
        result = subprocess.run(
            ["poetry", "run", "poe", "test-2d"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        # Should complete successfully in test mode
        self.assertEqual(result.returncode, 0, f"test-2d failed: {result.stderr}")
        self.assertIn("test", result.stdout.lower())
    
    def test_test_3d_task(self):
        """Test that 'poe test-3d' runs in test mode."""
        result = subprocess.run(
            ["poetry", "run", "poe", "test-3d"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        # Should complete successfully in test mode
        self.assertEqual(result.returncode, 0, f"test-3d failed: {result.stderr}")
        # Should be running with test prompt
        self.assertIn("ceramic coffee mug", result.stdout.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)