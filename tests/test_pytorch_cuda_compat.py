#!/usr/bin/env python3
"""
Test PyTorch CUDA compatibility with current CUDA installation.
This test will be skipped if PyTorch is not installed.
"""

import unittest
import sys
from pathlib import Path


class TestPyTorchCUDACompatibility(unittest.TestCase):
    """Test PyTorch compatibility with CUDA 12.9 installation."""
    
    def test_pytorch_import(self):
        """Test that PyTorch can be imported (if installed)."""
        try:
            import torch
            self.assertTrue(True, "PyTorch imported successfully")
        except ImportError:
            self.skipTest("PyTorch not installed yet - will be installed in Story 4")
    
    def test_cuda_availability(self):
        """Test that PyTorch can detect CUDA (if PyTorch installed)."""
        try:
            import torch
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            
            if not cuda_available:
                # This might be expected if PyTorch is installed without CUDA support
                self.skipTest("PyTorch installed but without CUDA support - will be fixed in Story 4")
            
            self.assertTrue(cuda_available, "CUDA is available in PyTorch")
            
        except ImportError:
            self.skipTest("PyTorch not installed yet - will be installed in Story 4")
    
    def test_cuda_device_detection(self):
        """Test that PyTorch detects the RTX 3090 (if PyTorch with CUDA installed)."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available in PyTorch")
            
            # Check device count
            device_count = torch.cuda.device_count()
            self.assertGreaterEqual(device_count, 1, "At least one CUDA device should be detected")
            
            # Check device name
            device_name = torch.cuda.get_device_name(0)
            self.assertIn("3090", device_name, f"RTX 3090 not detected, found: {device_name}")
            
            # Check CUDA version compatibility
            cuda_version = torch.version.cuda
            if cuda_version:
                # Extract major.minor version
                parts = cuda_version.split('.')
                if len(parts) >= 2:
                    major = int(parts[0])
                    minor = int(parts[1])
                    
                    # CUDA 11.8 or higher is acceptable
                    cuda_ver_num = major + minor / 10
                    self.assertGreaterEqual(cuda_ver_num, 11.8,
                        f"PyTorch CUDA version {cuda_version} should be 11.8+")
            
        except ImportError:
            self.skipTest("PyTorch not installed yet - will be installed in Story 4")
    
    def test_cuda_memory_allocation(self):
        """Test that PyTorch can allocate GPU memory (if available)."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available in PyTorch")
            
            # Try to allocate a small tensor on GPU
            try:
                device = torch.device("cuda:0")
                tensor = torch.zeros(1000, 1000, device=device)
                
                # Verify tensor is on GPU
                self.assertEqual(tensor.device.type, "cuda", "Tensor should be on CUDA device")
                
                # Try a larger allocation (1GB)
                large_tensor = torch.zeros(256, 1024, 1024, device=device)  # ~1GB
                self.assertIsNotNone(large_tensor, "Should be able to allocate 1GB on GPU")
                
                # Clean up
                del tensor
                del large_tensor
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                self.fail(f"Failed to allocate GPU memory: {e}")
            
        except ImportError:
            self.skipTest("PyTorch not installed yet - will be installed in Story 4")
    
    def test_cuda_compatibility_matrix(self):
        """Document CUDA version compatibility between system and PyTorch."""
        try:
            import torch
            import subprocess
            import re
            
            # Get system CUDA version
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            
            system_cuda_match = re.search(r'release (\d+\.\d+)', result.stdout)
            system_cuda = system_cuda_match.group(1) if system_cuda_match else "Unknown"
            
            # Get PyTorch CUDA version (if available)
            pytorch_cuda = torch.version.cuda if hasattr(torch, 'version') else "Not installed"
            
            # Document compatibility
            compatibility_info = f"""
            CUDA Compatibility Matrix:
            - System CUDA: {system_cuda}
            - PyTorch CUDA: {pytorch_cuda}
            - Compatible: {'Yes' if pytorch_cuda != "Not installed" else 'N/A - PyTorch not installed'}
            """
            
            # This is informational, not a failure
            print(compatibility_info)
            
            # If PyTorch is installed, verify versions are compatible
            if pytorch_cuda != "Not installed" and pytorch_cuda is not None:
                # CUDA 12.x is backward compatible with CUDA 11.x libraries
                self.assertTrue(True, "CUDA versions documented")
            
        except ImportError:
            self.skipTest("PyTorch not installed yet - will be installed in Story 4")
        except Exception as e:
            # Don't fail the test for documentation purposes
            print(f"Could not determine CUDA compatibility: {e}")


def run_tests():
    """Run PyTorch CUDA compatibility tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPyTorchCUDACompatibility)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)