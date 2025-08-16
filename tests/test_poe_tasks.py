#!/usr/bin/env python3
"""
Comprehensive test suite for all poethepoet (poe) tasks.
Ensures all poe scripts are functional and properly configured.
"""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib.util


class TestPoeTasksConfiguration(unittest.TestCase):
    """Test that all poe tasks are properly configured."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.scripts_dir = self.project_root / "scripts"
        
    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists and is readable."""
        pyproject_path = self.project_root / "pyproject.toml"
        self.assertTrue(pyproject_path.exists(), "pyproject.toml not found")
        
        # Try to parse it
        import tomllib
        with open(pyproject_path, 'rb') as f:
            config = tomllib.load(f)
        
        self.assertIn("tool", config, "No [tool] section in pyproject.toml")
        self.assertIn("poe", config["tool"], "No [tool.poe] section in pyproject.toml")
        self.assertIn("tasks", config["tool"]["poe"], "No tasks defined in [tool.poe.tasks]")
    
    def test_all_scripts_exist(self):
        """Test that all scripts referenced in poe tasks exist."""
        scripts_that_should_exist = [
            "generate.py",
            "generate_3d.py",
            "generate_mvdream.py",
            "download_models.py",
            "test_model_loading.py",
            "diagnose.py",
            "benchmark.py",
            "production_monitor.py",
            "test_2d_generation.py",
        ]
        
        for script_name in scripts_that_should_exist:
            script_path = self.scripts_dir / script_name
            self.assertTrue(
                script_path.exists(),
                f"Script {script_name} not found at {script_path}"
            )
    
    def test_all_scripts_are_executable(self):
        """Test that all Python scripts have proper shebang or are importable."""
        for script_path in self.scripts_dir.glob("*.py"):
            with open(script_path, 'r') as f:
                first_line = f.readline().strip()
            
            # Check for shebang (optional but good practice)
            if first_line.startswith("#!"):
                self.assertIn("python", first_line.lower(), 
                             f"{script_path.name} has invalid shebang")
            
            # Check script is importable
            spec = importlib.util.spec_from_file_location(
                script_path.stem, script_path
            )
            self.assertIsNotNone(spec, f"Cannot create import spec for {script_path.name}")
    
    def test_poetry_is_installed(self):
        """Test that poetry is available for running tasks."""
        result = subprocess.run(
            ["poetry", "--version"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0, "Poetry is not installed or not in PATH")
        self.assertIn("Poetry", result.stdout, "Poetry version output unexpected")
    
    def test_poe_is_installed(self):
        """Test that poethepoet is installed in the environment."""
        result = subprocess.run(
            ["poetry", "run", "poe", "--version"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        self.assertEqual(result.returncode, 0, "poethepoet is not installed")


class TestPoeTasksExecution(unittest.TestCase):
    """Test execution of poe tasks."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
    
    def run_poe_task(self, task_name, *args, check_success=True):
        """Helper to run a poe task."""
        cmd = ["poetry", "run", "poe", task_name] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30  # 30 second timeout for most tasks
        )
        if check_success:
            self.assertEqual(
                result.returncode, 0,
                f"Task '{task_name}' failed with: {result.stderr}"
            )
        return result
    
    def test_help_command(self):
        """Test that poe help works."""
        result = self.run_poe_task("--help")
        self.assertIn("tasks", result.stdout.lower())
    
    def test_list_tasks(self):
        """Test listing all available tasks."""
        result = subprocess.run(
            ["poetry", "run", "poe", "--help"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        self.assertEqual(result.returncode, 0)
        
        # Check that tasks section exists
        self.assertIn("task", result.stdout.lower())
        
        # List actual tasks
        result2 = subprocess.run(
            ["poetry", "run", "poe"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        # Poe returns 1 when no task specified, but lists tasks
        output = result2.stdout + result2.stderr
        
        # Check some key tasks are listed
        expected_tasks = [
            "test", "lint", "format", "generate", 
            "diagnose", "benchmark", "clean"
        ]
        for task in expected_tasks:
            self.assertIn(task, output, f"Task '{task}' not listed")
    
    def test_clean_task(self):
        """Test that clean task runs without errors."""
        # Clean is safe to run
        result = self.run_poe_task("clean")
        # Clean should complete successfully
        self.assertEqual(result.returncode, 0)
    
    def test_format_check_task(self):
        """Test that format-check task runs."""
        result = self.run_poe_task("format-check", check_success=False)
        # Format check may fail if code isn't formatted, but should run
        self.assertIn("ruff", result.stdout.lower() + result.stderr.lower())


class TestScriptHelp(unittest.TestCase):
    """Test that all scripts support --help flag."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.scripts_dir = self.project_root / "scripts"
    
    def test_script_help_flags(self):
        """Test that major scripts support --help."""
        scripts_with_help = [
            "generate.py",
            "generate_3d.py",
            "generate_mvdream.py",
            "download_models.py",
            "diagnose.py",
            "benchmark.py",
            "test_2d_generation.py",
        ]
        
        for script_name in scripts_with_help:
            script_path = self.scripts_dir / script_name
            if not script_path.exists():
                continue
                
            with self.subTest(script=script_name):
                result = subprocess.run(
                    ["poetry", "run", "python", str(script_path), "--help"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=10
                )
                
                # Should either succeed or show usage
                self.assertTrue(
                    result.returncode == 0 or "usage:" in result.stdout.lower(),
                    f"{script_name} doesn't support --help properly"
                )
                
                # Should mention what the script does
                output = result.stdout + result.stderr
                self.assertTrue(
                    len(output) > 50,  # Should have some help text
                    f"{script_name} has insufficient help text"
                )


class TestScriptImports(unittest.TestCase):
    """Test that all scripts can be imported without errors."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.scripts_dir = self.project_root / "scripts"
        sys.path.insert(0, str(self.project_root))
    
    def tearDown(self):
        """Clean up."""
        if str(self.project_root) in sys.path:
            sys.path.remove(str(self.project_root))
    
    def test_import_scripts(self):
        """Test importing each script module."""
        # Scripts that should be importable
        scripts_to_import = [
            "generate",
            "generate_3d",
            "generate_mvdream",
            "download_models",
            "test_model_loading",
            "diagnose",
            "benchmark",
            "test_2d_generation",
        ]
        
        for script_name in scripts_to_import:
            script_path = self.scripts_dir / f"{script_name}.py"
            if not script_path.exists():
                continue
                
            with self.subTest(script=script_name):
                # Try to import the module
                spec = importlib.util.spec_from_file_location(
                    script_name, script_path
                )
                module = importlib.util.module_from_spec(spec)
                
                # This should not raise an exception
                try:
                    # Mock torch if needed to avoid CUDA errors during import
                    with patch.dict('sys.modules', {'torch': MagicMock(), 
                                                   'torch.cuda': MagicMock()}):
                        spec.loader.exec_module(module)
                    
                    # Check module has main function
                    self.assertTrue(
                        hasattr(module, 'main'),
                        f"{script_name} doesn't have a main() function"
                    )
                except Exception as e:
                    self.fail(f"Failed to import {script_name}: {e}")


class TestCriticalTasks(unittest.TestCase):
    """Test critical poe tasks in safe/test mode."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
    
    def test_diagnose_runs(self):
        """Test that diagnose script runs."""
        result = subprocess.run(
            ["poetry", "run", "poe", "diagnose"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        # Diagnose may return 1 if some checks fail, but should complete
        self.assertIn(result.returncode, [0, 1], f"Diagnose crashed: {result.stderr}")
        self.assertIn("Diagnostic", result.stdout)
        self.assertIn("Summary", result.stdout)
    
    def test_test_gpu_runs(self):
        """Test that GPU test runs."""
        result = subprocess.run(
            ["poetry", "run", "poe", "test-gpu"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        # May pass or skip if no GPU, but should not error
        self.assertIn("test", result.stdout.lower(), "test-gpu didn't run tests")
    
    def test_lint_runs(self):
        """Test that lint tasks run."""
        result = subprocess.run(
            ["poetry", "run", "poe", "ruff-check"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=30
        )
        # Ruff should run even if there are lint errors
        # Check that ruff actually ran (output can be in stdout or stderr)
        self.assertTrue(
            "ruff" in result.stdout.lower() or "ruff" in result.stderr.lower() or result.returncode == 0,
            f"ruff-check didn't run properly. stdout: {result.stdout[:200]}, stderr: {result.stderr[:200]}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)