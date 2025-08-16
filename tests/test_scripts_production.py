#!/usr/bin/env python3
"""
Test suite for production monitoring scripts.
Tests production_monitor.py and related queue management.
"""

import subprocess
import sys
import unittest
from pathlib import Path
import json
import tempfile
from unittest.mock import patch, MagicMock


class TestProductionMonitorScript(unittest.TestCase):
    """Test the production_monitor.py script."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.script_path = self.project_root / "scripts" / "production_monitor.py"
        self.queue_file = self.project_root / "generation_queue.json"
    
    def test_script_exists(self):
        """Test that production_monitor.py exists."""
        self.assertTrue(self.script_path.exists(), "production_monitor.py not found")
    
    def test_help_flag(self):
        """Test production_monitor.py --help."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        self.assertEqual(result.returncode, 0, f"Help failed: {result.stderr}")
        self.assertIn("production", result.stdout.lower())
        self.assertIn("monitor", result.stdout.lower())
    
    def test_subcommands(self):
        """Test that subcommands are recognized."""
        subcommands = ["monitor", "start", "status", "queue"]
        
        for subcommand in subcommands:
            with self.subTest(subcommand=subcommand):
                result = subprocess.run(
                    ["poetry", "run", "python", str(self.script_path), 
                     subcommand, "--help"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=10
                )
                # Should show help for subcommand
                self.assertIn(subcommand, result.stdout.lower())
    
    def test_queue_list_command(self):
        """Test queue list command."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), 
             "queue", "list"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        # Should complete successfully
        self.assertEqual(result.returncode, 0, f"Queue list failed: {result.stderr}")
        # Should show queue status
        self.assertIn("queue", result.stdout.lower())
    
    def test_status_command(self):
        """Test status command."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), "status"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        # Should complete successfully
        self.assertEqual(result.returncode, 0, f"Status failed: {result.stderr}")
        # Should show system status
        self.assertIn("status", result.stdout.lower())
    
    def test_queue_add_validation(self):
        """Test that queue add requires prompt."""
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), 
             "queue", "add"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        # Should fail without prompt
        self.assertNotEqual(result.returncode, 0, "Should fail without prompt")
        self.assertIn("required", result.stderr.lower())


class TestProductionQueue(unittest.TestCase):
    """Test production queue functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.script_path = self.project_root / "scripts" / "production_monitor.py"
        
        # Back up existing queue file if it exists
        self.queue_file = self.project_root / "generation_queue.json"
        self.queue_backup = None
        if self.queue_file.exists():
            self.queue_backup = self.queue_file.read_text()
    
    def tearDown(self):
        """Restore queue file."""
        if self.queue_backup is not None:
            self.queue_file.write_text(self.queue_backup)
        elif self.queue_file.exists():
            # Reset to empty queue
            self.queue_file.write_text('{"jobs": []}')
    
    def test_queue_file_structure(self):
        """Test that queue file has correct structure."""
        if self.queue_file.exists():
            with open(self.queue_file, 'r') as f:
                data = json.load(f)
            
            self.assertIn("jobs", data, "Queue file missing 'jobs' key")
            self.assertIsInstance(data["jobs"], list, "'jobs' should be a list")
    
    def test_add_job_to_queue(self):
        """Test adding a job to the queue."""
        # Clear queue first
        self.queue_file.write_text('{"jobs": []}')
        
        # Add a test job
        result = subprocess.run(
            ["poetry", "run", "python", str(self.script_path), 
             "queue", "add", "test prompt for queue"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        
        self.assertEqual(result.returncode, 0, f"Add job failed: {result.stderr}")
        
        # Check job was added
        with open(self.queue_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data["jobs"]), 1, "Job not added to queue")
        self.assertEqual(data["jobs"][0]["prompt"], "test prompt for queue")
        self.assertEqual(data["jobs"][0]["status"], "pending")


class TestPoeProductionTasks(unittest.TestCase):
    """Test poe tasks for production monitoring."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
    
    def test_monitor_task(self):
        """Test that 'poe monitor' can be invoked."""
        # Just test that it starts (will need Ctrl+C to stop normally)
        # So we'll test with timeout and check it starts
        result = subprocess.run(
            ["poetry", "run", "poe", "monitor"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=2  # Short timeout since monitor runs forever
        )
        # Will timeout, but should have started
        self.assertIn("monitor", result.stdout.lower() + result.stderr.lower())
    
    def test_prod_status_task(self):
        """Test that 'poe prod-status' runs."""
        result = subprocess.run(
            ["poetry", "run", "poe", "prod-status"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        self.assertEqual(result.returncode, 0, f"prod-status failed: {result.stderr}")
        self.assertIn("status", result.stdout.lower())
    
    def test_queue_list_task(self):
        """Test that 'poe queue-list' runs."""
        result = subprocess.run(
            ["poetry", "run", "poe", "queue-list"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        self.assertEqual(result.returncode, 0, f"queue-list failed: {result.stderr}")
        self.assertIn("queue", result.stdout.lower())
    
    def test_queue_add_task(self):
        """Test that 'poe queue-add' shows usage."""
        result = subprocess.run(
            ["poetry", "run", "poe", "queue-add"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=10
        )
        # Should fail without prompt
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required", result.stderr.lower())


class TestProductionClasses(unittest.TestCase):
    """Test production monitoring classes can be imported."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.script_path = self.project_root / "scripts" / "production_monitor.py"
        sys.path.insert(0, str(self.project_root / "scripts"))
    
    def tearDown(self):
        """Clean up."""
        scripts_path = str(self.project_root / "scripts")
        if scripts_path in sys.path:
            sys.path.remove(scripts_path)
    
    def test_import_classes(self):
        """Test that main classes can be imported."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "production_monitor", self.script_path
        )
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
            
            # Check main classes exist
            self.assertTrue(hasattr(module, 'GPUMonitor'))
            self.assertTrue(hasattr(module, 'CheckpointManager'))
            self.assertTrue(hasattr(module, 'GenerationQueue'))
            self.assertTrue(hasattr(module, 'ProductionManager'))
            
            # Check main function exists
            self.assertTrue(hasattr(module, 'main'))
            
        except Exception as e:
            self.fail(f"Failed to import production_monitor: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)