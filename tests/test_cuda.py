import torch
import unittest
import sys
from pathlib import Path
class TestCUDAFunctionality(unittest.TestCase):
    def test_cuda_available(self):
        self.assertTrue(torch.cuda.is_available(), 
                       "CUDA is not available. Check CUDA installation and drivers.")
    def test_cuda_device_count(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device_count = torch.cuda.device_count()
        self.assertGreaterEqual(device_count, 1, 
                               f"Expected at least 1 CUDA device, found {device_count}")
    def test_rtx_3090_detected(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device_name = torch.cuda.get_device_name(0)
        self.assertIn("3090", device_name, 
                     f"RTX 3090 not detected. Found: {device_name}")
    def test_cuda_memory_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_gb = total_memory / (1024**3)
        self.assertGreater(total_gb, 20, 
                          f"GPU memory ({total_gb:.1f}GB) is less than expected for RTX 3090")
    def test_tensor_allocation(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda:0")
        tensor = torch.zeros(1000, 1000, device=device)
        self.assertEqual(tensor.device.type, "cuda", 
                        "Tensor not allocated on CUDA device")
        self.assertEqual(tensor.shape, (1000, 1000), 
                        "Tensor shape mismatch")
    def test_large_memory_allocation(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.cuda.empty_cache()
        try:
            allocated_tensors = []
            target_gb_per_tensor = 5
            elements_per_tensor = int(target_gb_per_tensor * (1024**3) / 4)
            height = 50000
            width = elements_per_tensor // height
            for i in range(4):
                tensor = torch.zeros(height, width, device='cuda', dtype=torch.float32)
                allocated_tensors.append(tensor)
            total_allocated = sum(t.element_size() * t.nelement() for t in allocated_tensors)
            total_gb = total_allocated / (1024**3)
            self.assertGreaterEqual(total_gb, 19, 
                                   f"Could only allocate {total_gb:.1f}GB, expected 20+GB")
            del allocated_tensors
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                free_memory = torch.cuda.mem_get_info()[0]
                free_gb = free_memory / (1024**3)
                self.fail(f"Could not allocate 20GB. Available memory: {free_gb:.1f}GB")
            else:
                raise
    def test_cuda_operations(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        c = torch.matmul(a, b)
        self.assertEqual(c.device.type, "cuda", 
                        "Result not on CUDA device")
        self.assertEqual(c.shape, (1000, 1000), 
                        "Result shape mismatch")
        self.assertFalse(torch.isnan(c).any(), "NaN values in GPU result")
        self.assertFalse(torch.isinf(c).any(), "Inf values in GPU result")
        c_cpu = c.cpu()
        self.assertTrue(c_cpu.abs().max() < 1000, 
                       f"Unreasonable values in result: max={c_cpu.abs().max()}")
    def test_pytorch_version(self):
        version = torch.__version__
        base_version = version.split('+')[0]
        major, minor = map(int, base_version.split('.')[:2])
        self.assertGreaterEqual(major, 2, 
                               f"PyTorch version {version} is too old (need 2.0+)")
        if '+cu' in version:
            cuda_version = version.split('+cu')[1][:3]
            self.assertIn(cuda_version[0:2], ['11', '12'], 
                         f"Unexpected CUDA version in PyTorch: {cuda_version}")
    def test_mixed_precision_support(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        major, minor = props.major, props.minor
        compute_capability = major + minor / 10
        self.assertGreaterEqual(compute_capability, 7.0, 
                               f"GPU compute capability {compute_capability} too low for FP16")
        tensor_fp16 = torch.zeros(100, 100, device=device, dtype=torch.float16)
        self.assertEqual(tensor_fp16.dtype, torch.float16, 
                        "FP16 tensor creation failed")
        with torch.amp.autocast('cuda'):
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            c = torch.matmul(a, b)
            self.assertTrue(c.dtype in [torch.float16, torch.bfloat16], 
                           f"Autocast not using reduced precision: {c.dtype}")
def run_gpu_tests():
    print("=" * 60)
    print("Running GPU Validation Tests")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"Total GPU memory: {total_memory / (1024**3):.1f} GB")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCUDAFunctionality)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()
if __name__ == '__main__':
    success = run_gpu_tests()
    sys.exit(0 if success else 1)