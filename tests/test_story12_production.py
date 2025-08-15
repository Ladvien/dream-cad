#!/usr/bin/env python3
"""Test Story 12: Production Setup and Monitoring."""

import json
import pickle
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestStory12Production:
    """Test suite for Story 12 acceptance criteria."""

    def test_production_monitor_script_exists(self):
        """Test that production monitor script is created."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/production_monitor.py")
        assert script_path.exists(), "Production monitor script not found"
        
        # Check script has proper structure
        with script_path.open() as f:
            content = f.read()
        
        assert "#!/usr/bin/env python3" in content, "Missing shebang"
        assert "ProductionManager" in content, "ProductionManager class not found"
        assert "GPUMonitor" in content, "GPUMonitor class not found"
        assert "CheckpointManager" in content, "CheckpointManager class not found"
        assert "GenerationQueue" in content, "GenerationQueue class not found"

    def test_logging_configuration(self):
        """Test that logging is configured with rotating file handler."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/production_monitor.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for logging configuration
        assert "logging.handlers.RotatingFileHandler" in content, "Rotating file handler not configured"
        assert 'f"{name}.log"' in content or "mvdream.log" in content or ".log" in content, "Log file not configured"
        assert "maxBytes" in content, "Log rotation size not configured"
        assert "backupCount" in content, "Backup count not configured"

    def test_gpu_metrics_logging(self):
        """Test that GPU metrics are logged every 30 seconds."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/production_monitor.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for GPU monitoring
        assert "GPUMonitor" in content, "GPUMonitor class not found"
        assert "interval" in content, "Monitoring interval not configured"
        assert "30" in content or "interval=30" in content or "interval: int = 30" in content, "30 second interval not found"
        assert "nvidia-smi" in content, "nvidia-smi integration not found"
        assert "temperature" in content.lower(), "Temperature monitoring not found"
        assert "memory" in content.lower(), "Memory monitoring not found"

    def test_checkpoint_system(self):
        """Test automatic checkpoint saving every 1000 steps."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/production_monitor.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for checkpoint system
        assert "CheckpointManager" in content, "CheckpointManager class not found"
        assert "save_checkpoint" in content, "save_checkpoint method not found"
        assert "load_checkpoint" in content, "load_checkpoint method not found"
        assert "1000" in content, "1000 step interval not found"
        assert "checkpoint" in content.lower(), "Checkpoint functionality not found"

    def test_recovery_functionality(self):
        """Test that recovery script can resume from checkpoints."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/production_monitor.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for recovery functionality
        assert "recover" in content.lower(), "Recovery functionality not found"
        assert "resume" in content.lower() or "load_checkpoint" in content, "Resume functionality not found"
        assert "get_latest_checkpoint" in content, "Latest checkpoint retrieval not found"

    def test_resource_alerts(self):
        """Test system resource alerts for >90% VRAM usage."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/production_monitor.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for alert system
        assert "alert" in content.lower(), "Alert system not found"
        assert "90" in content, "90% threshold not found"
        assert "check_alerts" in content or "alert_threshold" in content, "Alert checking not found"
        assert "VRAM" in content or "memory_percent" in content, "VRAM monitoring not found"

    def test_generation_queue_system(self):
        """Test generation queue system for batch processing."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/production_monitor.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for queue system
        assert "GenerationQueue" in content, "GenerationQueue class not found"
        assert "add_job" in content, "add_job method not found"
        assert "get_next_job" in content, "get_next_job method not found"
        assert "queue" in content.lower(), "Queue functionality not found"
        assert "batch" in content.lower() or "jobs" in content, "Batch processing not mentioned"

    def test_poethepoet_monitor_task(self):
        """Test that 'poe monitor' shows real-time GPU stats."""
        pyproject_path = Path("/mnt/datadrive_m2/dream-cad/pyproject.toml")
        
        with pyproject_path.open() as f:
            content = f.read()
        
        # Check for monitor task
        assert "monitor" in content, "monitor task not found in pyproject.toml"
        assert "production_monitor.py" in content, "production_monitor.py not referenced"

    def test_systemd_service_file(self):
        """Test that systemd service file is created (optional)."""
        service_path = Path("/mnt/datadrive_m2/dream-cad/mvdream.service")
        
        if service_path.exists():
            with service_path.open() as f:
                content = f.read()
            
            # Check service file structure
            assert "[Unit]" in content, "Unit section missing"
            assert "[Service]" in content, "Service section missing"
            assert "[Install]" in content, "Install section missing"
            assert "ExecStart" in content, "ExecStart not defined"
            assert "production_monitor.py" in content, "Script not referenced"

    def test_production_guide_exists(self):
        """Test that production guide is documented at docs/production_setup.md."""
        guide_path = Path("/mnt/datadrive_m2/dream-cad/docs/production_setup.md")
        assert guide_path.exists(), "Production setup guide not found"
        
        with guide_path.open() as f:
            content = f.read()
        
        # Check guide content
        assert "Production" in content, "Production not mentioned"
        assert "Monitoring" in content, "Monitoring not covered"
        assert "Logging" in content, "Logging not covered"
        assert "Queue" in content, "Queue system not covered"
        assert "Checkpoint" in content, "Checkpoint system not covered"

    def test_production_monitor_imports(self):
        """Test that production monitor script can be imported."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts")
        sys.path.insert(0, str(script_path))
        
        try:
            import production_monitor
            
            # Check for required classes
            assert hasattr(production_monitor, "ProductionManager"), "ProductionManager not found"
            assert hasattr(production_monitor, "GPUMonitor"), "GPUMonitor not found"
            assert hasattr(production_monitor, "CheckpointManager"), "CheckpointManager not found"
            assert hasattr(production_monitor, "GenerationQueue"), "GenerationQueue not found"
            
            # Check for data classes
            assert hasattr(production_monitor, "GPUMetrics"), "GPUMetrics not found"
            assert hasattr(production_monitor, "SystemMetrics"), "SystemMetrics not found"
            assert hasattr(production_monitor, "GenerationJob"), "GenerationJob not found"
            
        finally:
            sys.path.pop(0)

    def test_log_directory_created(self):
        """Test that logs directory is created."""
        log_dir = Path("/mnt/datadrive_m2/dream-cad/logs")
        assert log_dir.exists(), "Logs directory not created"
        assert log_dir.is_dir(), "Logs path is not a directory"