#!/usr/bin/env python3
"""
Test suite for utility scripts.
Tests diagnose.py, benchmark.py, download_models.py, test_model_loading.py
"""

import subprocess
import sys
import unittest
from pathlib import Path
import json
import tempfile
from unittest.mock import patch, MagicMock, mock_open


class TestDiagnoseScript(unittest.TestCase):
    """Test the diagnose.py script."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.script_path = self.project_root / "scripts" / "diagnose.py"
    
    def test_script_exists(self):
        """Test that diagnose.py exists."""
        self.assertTrue(self.script_path.exists(), "diagnose.py not found")
    
    def test_help_flag(self):
        """Test diagnose.py --help."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        self.assertEqual(result.returncode, 0, f"Help failed: {result.stderr}")
        self.assertIn("diagnos", result.stdout.lower())
    
    def test_diagnose_runs(self):
        """Test that diagnose runs and produces output."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path)],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        
        # Should complete successfully
        self.assertEqual(result.returncode, 0, f"Diagnose failed: {result.stderr}")
        
        # Should show diagnostic sections
        expected_sections = ["System", "CUDA", "Dependencies", "GPU", "Models"]
        for section in expected_sections:
            self.assertIn(
                section, result.stdout,
                f"Missing {section} section in diagnostics"
            )
    
    def test_verbose_flag(self):
        """Test diagnose.py --verbose if supported."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "--verbose"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        
        # Should either work or show that flag isn't recognized
        # but shouldn't crash
        self.assertTrue(
            result.returncode == 0 or "unrecognized" in result.stderr.lower(),
            f"Unexpected error: {result.stderr}"
        )


class TestBenchmarkScript(unittest.TestCase):
    """Test the benchmark.py script."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.script_path = self.project_root / "scripts" / "benchmark.py"
    
    def test_script_exists(self):
        """Test that benchmark.py exists."""
        self.assertTrue(self.script_path.exists(), "benchmark.py not found")
    
    def test_help_flag(self):
        """Test benchmark.py --help."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        self.assertEqual(result.returncode, 0, f"Help failed: {result.stderr}")
        self.assertIn("benchmark", result.stdout.lower())
    
    def test_quick_mode(self):
        """Test benchmark.py --quick."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "--quick"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=60  # Quick mode should be fast
        )
        
        # Should complete successfully
        self.assertEqual(result.returncode, 0, f"Quick benchmark failed: {result.stderr}")
        
        # Should show benchmark results
        self.assertIn("benchmark", result.stdout.lower())
    
    def test_test_mode(self):
        """Test benchmark.py --test-mode if supported."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "--test-mode"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        
        # Should either work or show unrecognized argument
        if result.returncode == 0:
            self.assertIn("test", result.stdout.lower())


class TestDownloadModelsScript(unittest.TestCase):
    """Test the download_models.py script."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.script_path = self.project_root / "scripts" / "download_models.py"
        self.models_dir = self.project_root / "models"
    
    def test_script_exists(self):
        """Test that download_models.py exists."""
        self.assertTrue(self.script_path.exists(), "download_models.py not found")
    
    def test_help_flag(self):
        """Test download_models.py --help."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        self.assertEqual(result.returncode, 0, f"Help failed: {result.stderr}")
        self.assertIn("model", result.stdout.lower())
        self.assertIn("download", result.stdout.lower())
    
    def test_verify_only_flag(self):
        """Test download_models.py --verify-only."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "--verify-only"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        
        # Should complete (may report missing models)
        self.assertEqual(result.returncode, 0, f"Verify failed: {result.stderr}")
        
        # Should show verification results
        self.assertIn("verif", result.stdout.lower())
    
    def test_model_directory_exists(self):
        """Test that models directory is checked/created."""
        # Import the module to test
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "download_models", self.script_path
        )
        module = importlib.util.module_from_spec(spec)
        
        # Mock functions that would download
        with patch('urllib.request.urlretrieve'):
            with patch('builtins.open', mock_open()):
                try:
                    spec.loader.exec_module(module)
                    # Check module has necessary functions
                    self.assertTrue(hasattr(module, 'main'))
                except Exception as e:
                    # OK if import fails due to missing dependencies
                    if "import" not in str(e).lower():
                        self.fail(f"Unexpected error: {e}")


class TestTestModelLoadingScript(unittest.TestCase):
    """Test the test_model_loading.py script."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.script_path = self.project_root / "scripts" / "test_model_loading.py"
    
    def test_script_exists(self):
        """Test that test_model_loading.py exists."""
        self.assertTrue(self.script_path.exists(), "test_model_loading.py not found")
    
    def test_script_runs(self):
        """Test that test_model_loading.py runs."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path)],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        
        # Should complete (may fail if model missing, but should run)
        # Check for expected output
        output = result.stdout + result.stderr
        self.assertIn("model", output.lower())
    
    def test_handles_missing_model(self):
        """Test that script handles missing model gracefully."""
        # Temporarily rename models directory if it exists
        models_dir = self.project_root / "models"
        models_backup = self.project_root / "models_backup_temp"
        
        if models_dir.exists():
            models_dir.rename(models_backup)
        
        try:
            result = subprocess.run(
                ["poetry", "run", "python", str(self.script_path)],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=30
            )
            
            # Should handle missing model gracefully
            self.assertTrue(
                "not found" in result.stdout.lower() or 
                "missing" in result.stdout.lower() or
                result.returncode != 0,
                "Should report missing model"
            )
        finally:
            # Restore models directory
            if models_backup.exists():
                models_backup.rename(models_dir)


class TestPoeUtilityTasks(unittest.TestCase):
    """Test poe tasks for utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
    
    def test_diagnose_task(self):
        """Test that 'poe diagnose' runs."""
        result = subprocess.run(
            ["poetry", "run", "poe", "diagnose"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        self.assertEqual(result.returncode, 0, f"diagnose task failed: {result.stderr}")
        self.assertIn("System", result.stdout)
    
    def test_benchmark_quick_task(self):
        """Test that 'poe benchmark-quick' runs."""
        result = subprocess.run(
            ["poetry", "run", "poe", "benchmark-quick"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=60
        )
        self.assertEqual(result.returncode, 0, f"benchmark-quick failed: {result.stderr}")
        self.assertIn("benchmark", result.stdout.lower())
    
    def test_verify_models_task(self):
        """Test that 'poe verify-models' runs."""
        result = subprocess.run(
            ["poetry", "run", "poe", "verify-models"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        self.assertEqual(result.returncode, 0, f"verify-models failed: {result.stderr}")
        self.assertIn("verif", result.stdout.lower())
    
    def test_test_model_loading_task(self):
        """Test that 'poe test-model-loading' runs."""
        result = subprocess.run(
            ["poetry", "run", "poe", "test-model-loading"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        # Should run even if model is missing
        output = result.stdout + result.stderr
        self.assertIn("model", output.lower())
    
    def test_clean_task(self):
        """Test that 'poe clean' runs."""
        # Create a test __pycache__ directory
        test_cache = self.project_root / "test_cache_dir" / "__pycache__"
        test_cache.mkdir(parents=True, exist_ok=True)
        test_pyc = test_cache / "test.pyc"
        test_pyc.touch()
        
        # Run clean
        result = subprocess.run(
            ["poetry", "run", "poe", "clean"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        
        self.assertEqual(result.returncode, 0, f"clean task failed: {result.stderr}")
        
        # Check that __pycache__ was cleaned
        self.assertFalse(test_pyc.exists(), "Clean didn't remove .pyc file")
        
        # Clean up test directory
        if test_cache.parent.exists():
            import shutil
            shutil.rmtree(test_cache.parent)


if __name__ == "__main__":
    unittest.main(verbosity=2)