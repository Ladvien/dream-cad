#!/usr/bin/env python3
"""
GPU validation test script for MVDream.
Tests PyTorch CUDA functionality and memory allocation.
"""

import torch
import unittest
import sys
from pathlib import Path


class TestCUDAFunctionality(unittest.TestCase):
    """Test suite for GPU/CUDA functionality."""
    
    def test_cuda_available(self):
        """Test that CUDA is available."""
        self.assertTrue(torch.cuda.is_available(), 
                       "CUDA is not available. Check CUDA installation and drivers.")
    
    def test_cuda_device_count(self):
        """Test that at least one CUDA device is detected."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        device_count = torch.cuda.device_count()
        self.assertGreaterEqual(device_count, 1, 
                               f"Expected at least 1 CUDA device, found {device_count}")
    
    def test_rtx_3090_detected(self):
        """Test that RTX 3090 is specifically detected."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        device_name = torch.cuda.get_device_name(0)
        self.assertIn("3090", device_name, 
                     f"RTX 3090 not detected. Found: {device_name}")
    
    def test_cuda_memory_available(self):
        """Test available GPU memory."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Get memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_gb = total_memory / (1024**3)
        
        # RTX 3090 should have approximately 24GB
        self.assertGreater(total_gb, 20, 
                          f"GPU memory ({total_gb:.1f}GB) is less than expected for RTX 3090")
    
    def test_tensor_allocation(self):
        """Test basic tensor allocation on GPU."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Try to allocate a small tensor
        device = torch.device("cuda:0")
        tensor = torch.zeros(1000, 1000, device=device)
        
        self.assertEqual(tensor.device.type, "cuda", 
                        "Tensor not allocated on CUDA device")
        self.assertEqual(tensor.shape, (1000, 1000), 
                        "Tensor shape mismatch")
    
    def test_large_memory_allocation(self):
        """Test ability to allocate 20+ GB of memory."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Clear cache first
        torch.cuda.empty_cache()
        
        try:
            # Try to allocate ~20GB (adjust size based on dtype)
            # float32 = 4 bytes per element
            # 20GB = 20 * 1024^3 bytes / 4 bytes per float = ~5.37 billion elements
            # Use a reasonable shape like (50000, 26843) ≈ 1.34 billion elements ≈ 5GB
            # We'll do 4 allocations of 5GB each
            
            allocated_tensors = []
            target_gb_per_tensor = 5
            elements_per_tensor = int(target_gb_per_tensor * (1024**3) / 4)
            
            # Create a reasonable 2D shape
            height = 50000
            width = elements_per_tensor // height
            
            for i in range(4):  # 4 x 5GB = 20GB
                tensor = torch.zeros(height, width, device='cuda', dtype=torch.float32)
                allocated_tensors.append(tensor)
            
            # Calculate total allocated memory
            total_allocated = sum(t.element_size() * t.nelement() for t in allocated_tensors)
            total_gb = total_allocated / (1024**3)
            
            self.assertGreaterEqual(total_gb, 19, 
                                   f"Could only allocate {total_gb:.1f}GB, expected 20+GB")
            
            # Clean up
            del allocated_tensors
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Get actual available memory
                torch.cuda.empty_cache()
                free_memory = torch.cuda.mem_get_info()[0]
                free_gb = free_memory / (1024**3)
                self.fail(f"Could not allocate 20GB. Available memory: {free_gb:.1f}GB")
            else:
                raise
    
    def test_cuda_operations(self):
        """Test basic CUDA operations (matrix multiplication)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create tensors on GPU
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        
        # Perform matrix multiplication
        c = torch.matmul(a, b)
        
        # Verify result is on GPU
        self.assertEqual(c.device.type, "cuda", 
                        "Result not on CUDA device")
        self.assertEqual(c.shape, (1000, 1000), 
                        "Result shape mismatch")
        
        # Verify computation is correct (spot check)
        # Move small slice to CPU for verification
        a_small = a[:10, :10].cpu()
        b_small = b[:10, :10].cpu()
        c_small = c[:10, :10].cpu()
        c_expected = torch.matmul(a_small, b_small)
        
        self.assertTrue(torch.allclose(c_small, c_expected, rtol=1e-3, atol=1e-5), 
                       "GPU computation result incorrect")
    
    def test_pytorch_version(self):
        """Test PyTorch version compatibility."""
        version = torch.__version__
        
        # Extract base version (remove +cu118 suffix)
        base_version = version.split('+')[0]
        major, minor = map(int, base_version.split('.')[:2])
        
        # We need at least PyTorch 2.0 for modern features
        self.assertGreaterEqual(major, 2, 
                               f"PyTorch version {version} is too old (need 2.0+)")
        
        # Check CUDA version in PyTorch
        if '+cu' in version:
            cuda_version = version.split('+cu')[1][:3]
            self.assertIn(cuda_version, ['118', '121'], 
                         f"Unexpected CUDA version in PyTorch: {cuda_version}")
    
    def test_mixed_precision_support(self):
        """Test mixed precision (FP16) support."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Check if GPU supports FP16
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        
        # RTX 3090 should support FP16 (compute capability >= 7.0)
        major, minor = props.major, props.minor
        compute_capability = major + minor / 10
        
        self.assertGreaterEqual(compute_capability, 7.0, 
                               f"GPU compute capability {compute_capability} too low for FP16")
        
        # Test FP16 tensor creation
        tensor_fp16 = torch.zeros(100, 100, device=device, dtype=torch.float16)
        self.assertEqual(tensor_fp16.dtype, torch.float16, 
                        "FP16 tensor creation failed")
        
        # Test autocast
        with torch.cuda.amp.autocast():
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            c = torch.matmul(a, b)
            # Inside autocast, operations should use FP16
            self.assertTrue(c.dtype in [torch.float16, torch.bfloat16], 
                           f"Autocast not using reduced precision: {c.dtype}")


def run_gpu_tests():
    """Run all GPU validation tests."""
    print("=" * 60)
    print("Running GPU Validation Tests")
    print("=" * 60)
    
    # Print system info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"Total GPU memory: {total_memory / (1024**3):.1f} GB")
    
    print("=" * 60)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCUDAFunctionality)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_gpu_tests()
    sys.exit(0 if success else 1)