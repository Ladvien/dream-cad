import unittest
import sys
from pathlib import Path
class TestPyTorchCUDACompatibility(unittest.TestCase):
    def test_pytorch_import(self):
        try:
            import torch
            self.assertTrue(True, "PyTorch imported successfully")
        except ImportError:
            self.skipTest("PyTorch not installed yet - will be installed in Story 4")
    def test_cuda_availability(self):
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                self.skipTest("PyTorch installed but without CUDA support - will be fixed in Story 4")
            self.assertTrue(cuda_available, "CUDA is available in PyTorch")
        except ImportError:
            self.skipTest("PyTorch not installed yet - will be installed in Story 4")
    def test_cuda_device_detection(self):
        try:
            import torch
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available in PyTorch")
            device_count = torch.cuda.device_count()
            self.assertGreaterEqual(device_count, 1, "At least one CUDA device should be detected")
            device_name = torch.cuda.get_device_name(0)
            self.assertIn("3090", device_name, f"RTX 3090 not detected, found: {device_name}")
            cuda_version = torch.version.cuda
            if cuda_version:
                parts = cuda_version.split('.')
                if len(parts) >= 2:
                    major = int(parts[0])
                    minor = int(parts[1])
                    cuda_ver_num = major + minor / 10
                    self.assertGreaterEqual(cuda_ver_num, 11.8,
                        f"PyTorch CUDA version {cuda_version} should be 11.8+")
        except ImportError:
            self.skipTest("PyTorch not installed yet - will be installed in Story 4")
    def test_cuda_memory_allocation(self):
        try:
            import torch
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available in PyTorch")
            try:
                device = torch.device("cuda:0")
                tensor = torch.zeros(1000, 1000, device=device)
                self.assertEqual(tensor.device.type, "cuda", "Tensor should be on CUDA device")
                large_tensor = torch.zeros(256, 1024, 1024, device=device)
                self.assertIsNotNone(large_tensor, "Should be able to allocate 1GB on GPU")
                del tensor
                del large_tensor
                torch.cuda.empty_cache()
            except RuntimeError as e:
                self.fail(f"Failed to allocate GPU memory: {e}")
        except ImportError:
            self.skipTest("PyTorch not installed yet - will be installed in Story 4")
    def test_cuda_compatibility_matrix(self):
        try:
            import torch
            import subprocess
            import re
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            system_cuda_match = re.search(r'release (\d+\.\d+)', result.stdout)
            system_cuda = system_cuda_match.group(1) if system_cuda_match else "Unknown"
            pytorch_cuda = torch.version.cuda if hasattr(torch, 'version') else "Not installed"
            compatibility_info = f"""
            CUDA Compatibility Matrix:
            - System CUDA: {system_cuda}
            - PyTorch CUDA: {pytorch_cuda}
            - Compatible: {'Yes' if pytorch_cuda != "Not installed" else 'N/A - PyTorch not installed'}
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPyTorchCUDACompatibility)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()
if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)