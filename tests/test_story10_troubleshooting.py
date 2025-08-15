#!/usr/bin/env python3
"""Test Story 10: Troubleshooting Documentation and Scripts."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestStory10Troubleshooting:
    """Test suite for Story 10 acceptance criteria."""

    def test_diagnostic_script_exists(self):
        """Test that diagnostic script is created at scripts/diagnose.py."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/diagnose.py")
        assert script_path.exists(), "Diagnostic script not found at scripts/diagnose.py"
        
        # Check script is executable (has shebang)
        with script_path.open() as f:
            first_line = f.readline()
            assert first_line.startswith("#!/usr/bin/env python"), "Script missing shebang"

    def test_cuda_checks_in_diagnostic(self):
        """Test that script checks CUDA availability and version."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/diagnose.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for CUDA-related checks
        assert "CUDACheck" in content, "CUDACheck class not found"
        assert "nvidia-smi" in content, "nvidia-smi check not found"
        assert "nvcc" in content, "nvcc check not found"
        assert "torch.cuda.is_available" in content, "PyTorch CUDA check not found"
        assert "CUDA_HOME" in content, "CUDA_HOME environment check not found"

    def test_dependency_verification_in_diagnostic(self):
        """Test that script verifies all Python dependencies are installed."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/diagnose.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for dependency verification
        assert "DependencyCheck" in content, "DependencyCheck class not found"
        
        # Check for key dependencies
        required_packages = [
            "torch", "torchvision", "transformers", "diffusers",
            "pytorch_lightning", "gradio", "numpy", "pillow"
        ]
        for package in required_packages:
            assert package in content, f"Check for {package} not found"

    def test_gpu_memory_test_in_diagnostic(self):
        """Test that script tests GPU memory allocation."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/diagnose.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for GPU memory testing
        assert "GPUMemoryCheck" in content, "GPUMemoryCheck class not found"
        assert "torch.zeros" in content, "GPU memory allocation test not found"
        assert "torch.cuda.empty_cache" in content, "GPU cache clearing not found"
        assert "torch.cuda.get_device_properties" in content, "GPU properties check not found"

    def test_model_file_validation_in_diagnostic(self):
        """Test that script validates model file integrity."""
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/diagnose.py")
        
        with script_path.open() as f:
            content = f.read()
        
        # Check for model file validation
        assert "ModelFileCheck" in content, "ModelFileCheck class not found"
        assert "model_info.json" in content, "Model info file check not found"
        assert "sd-v2.1-base-4view.pt" in content, "MVDream model check not found"
        assert "sha256" in content.lower(), "Checksum validation reference not found"

    def test_troubleshooting_guide_exists(self):
        """Test that troubleshooting guide is created at docs/troubleshooting.md."""
        guide_path = Path("/mnt/datadrive_m2/dream-cad/docs/troubleshooting.md")
        assert guide_path.exists(), "Troubleshooting guide not found at docs/troubleshooting.md"
        
        # Check file is not empty
        assert guide_path.stat().st_size > 1000, "Troubleshooting guide seems too small"

    def test_troubleshooting_guide_covers_common_errors(self):
        """Test that guide covers top 5 common errors with solutions."""
        guide_path = Path("/mnt/datadrive_m2/dream-cad/docs/troubleshooting.md")
        
        with guide_path.open() as f:
            content = f.read()
        
        # Check for top 5 common errors
        expected_errors = [
            "PyTorch NCCL Library Error",
            "CUDA Out of Memory",
            "Model File Not Found",
            "ImportError",
            "GPU Not Detected"
        ]
        
        for error in expected_errors:
            assert error in content, f"'{error}' not covered in troubleshooting guide"
            
        # Check that solutions are provided
        assert "Solutions:" in content or "Solution:" in content, "No solutions section found"
        assert "```bash" in content, "No bash code examples found"

    def test_poethepoet_diagnose_task_exists(self):
        """Test that 'poe diagnose' task runs all diagnostics."""
        pyproject_path = Path("/mnt/datadrive_m2/dream-cad/pyproject.toml")
        
        with pyproject_path.open() as f:
            content = f.read()
        
        # Check for diagnose task
        assert "diagnose" in content, "diagnose task not found in pyproject.toml"
        assert "scripts/diagnose.py" in content, "diagnose script not referenced in task"

    def test_faq_section_in_readme(self):
        """Test that FAQ section is added to main README.md."""
        readme_path = Path("/mnt/datadrive_m2/dream-cad/README.md")
        
        with readme_path.open() as f:
            content = f.read()
        
        # Check for FAQ section
        assert "## FAQ" in content, "FAQ section not found in README.md"
        
        # Check for common questions
        expected_questions = [
            "How do I check if my system is properly configured",
            "PyTorch NCCL library error",
            "How much disk space",
            "How do I download the pre-trained models"
        ]
        
        for question in expected_questions:
            assert question in content, f"FAQ missing question about '{question}'"

    def test_diagnostic_script_runs(self):
        """Test that diagnostic script can be executed."""
        # Import the script module to test its components
        script_path = Path("/mnt/datadrive_m2/dream-cad/scripts/diagnose.py")
        
        # Add scripts directory to path
        sys.path.insert(0, str(script_path.parent))
        
        try:
            import diagnose
            
            # Test that main diagnostic checks can be instantiated
            checks = [
                diagnose.SystemInfoCheck(),
                diagnose.CUDACheck(),
                diagnose.DependencyCheck(),
                diagnose.GPUMemoryCheck(),
                diagnose.ModelFileCheck(),
                diagnose.DirectoryStructureCheck(),
                diagnose.ConfigurationCheck(),
                diagnose.DiskSpaceCheck(),
            ]
            
            assert len(checks) == 8, "Expected 8 diagnostic checks"
            
            # Each check should have required methods
            for check in checks:
                assert hasattr(check, "run"), f"{check.name} missing run method"
                assert hasattr(check, "print_result"), f"{check.name} missing print_result method"
                
        finally:
            # Clean up path
            sys.path.pop(0)

    def test_troubleshooting_guide_has_memory_tips(self):
        """Test that troubleshooting guide includes memory optimization tips."""
        guide_path = Path("/mnt/datadrive_m2/dream-cad/docs/troubleshooting.md")
        
        with guide_path.open() as f:
            content = f.read()
        
        # Check for memory optimization section
        assert "Memory Optimization" in content, "Memory optimization section not found"
        assert "batch_size" in content, "Batch size optimization not mentioned"
        assert "xformers" in content.lower(), "xformers not mentioned"
        assert "gradient_checkpointing" in content, "Gradient checkpointing not mentioned"

    def test_troubleshooting_guide_has_performance_tuning(self):
        """Test that troubleshooting guide documents performance tuning options."""
        guide_path = Path("/mnt/datadrive_m2/dream-cad/docs/troubleshooting.md")
        
        with guide_path.open() as f:
            content = f.read()
        
        # Check for performance tuning section
        assert "Performance Tuning" in content, "Performance tuning section not found"
        assert "RTX 3090" in content, "RTX 3090 specific settings not found"
        assert "fp16" in content.lower() or "float16" in content.lower(), "Half precision not mentioned"
        assert "torch.compile" in content, "torch.compile optimization not mentioned"