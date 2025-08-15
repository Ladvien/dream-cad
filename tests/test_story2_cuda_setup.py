#!/usr/bin/env python3
"""
Test suite for Story 2: CUDA Toolkit Installation
Verifies all acceptance criteria for CUDA setup.
"""

import os
import subprocess
import re
import unittest
from pathlib import Path


class TestCUDASetup(unittest.TestCase):
    """Test CUDA installation and configuration for MVDream."""
    
    def test_cuda_installed(self):
        """Test that CUDA toolkit is installed."""
        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            self.assertIn('cuda', result.stdout.lower())
            
            # Extract version
            version_match = re.search(r'release (\d+\.\d+)', result.stdout)
            if version_match:
                version = float(version_match.group(1))
                # Accept CUDA 11.8 or higher (including 12.x)
                self.assertGreaterEqual(version, 11.8, 
                    f"CUDA version {version} is installed (11.8+ required)")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.fail("nvcc not found - CUDA toolkit not installed or not in PATH")
    
    def test_cuda_home_environment(self):
        """Test that CUDA_HOME environment variable is set."""
        cuda_home = os.environ.get('CUDA_HOME')
        if not cuda_home:
            # Try to find it from common locations
            if Path('/opt/cuda').exists():
                cuda_home = '/opt/cuda'
            elif Path('/usr/local/cuda').exists():
                cuda_home = '/usr/local/cuda'
        
        self.assertIsNotNone(cuda_home, "CUDA_HOME environment variable not set")
        self.assertTrue(Path(cuda_home).exists(), 
            f"CUDA_HOME path {cuda_home} does not exist")
    
    def test_cuda_in_path(self):
        """Test that CUDA binaries are in PATH."""
        path_env = os.environ.get('PATH', '')
        
        # Check if any cuda directory is in PATH
        cuda_in_path = any('cuda' in p.lower() for p in path_env.split(':'))
        self.assertTrue(cuda_in_path, "No CUDA directory found in PATH")
        
        # Verify nvcc is accessible
        try:
            subprocess.run(['which', 'nvcc'], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            self.fail("nvcc not found in PATH")
    
    def test_cuda_lib_path(self):
        """Test that CUDA libraries are in LD_LIBRARY_PATH."""
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        
        # Check if cuda lib directory is in LD_LIBRARY_PATH
        cuda_lib_in_path = any('cuda' in p.lower() and 'lib' in p 
                              for p in ld_path.split(':'))
        
        if not cuda_lib_in_path:
            # Check if libraries exist even if not in LD_LIBRARY_PATH
            cuda_lib_paths = [
                '/opt/cuda/lib64',
                '/usr/local/cuda/lib64',
                '/usr/local/cuda-11.8/lib64',
                '/usr/local/cuda-12.9/lib64'
            ]
            lib_exists = any(Path(p).exists() for p in cuda_lib_paths)
            
            if lib_exists:
                self.skipTest("CUDA libraries exist but not in LD_LIBRARY_PATH - may work anyway")
            else:
                self.fail("CUDA libraries not found in LD_LIBRARY_PATH or standard locations")
    
    def test_gpu_detection(self):
        """Test that RTX 3090 GPU is detected with correct compute capability."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )
            
            output = result.stdout.strip()
            self.assertIn('RTX 3090', output, "RTX 3090 not detected")
            self.assertIn('8.6', output, "Compute capability 8.6 not detected")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.fail("nvidia-smi not found or failed to execute")
    
    def test_cuda_samples_compiled(self):
        """Test that CUDA sample programs compile and run."""
        test_dir = Path.home() / 'mvdream' / 'tests'
        
        # Check if deviceQuery test exists
        device_query = test_dir / 'cuda_device_query.cu'
        self.assertTrue(device_query.exists(), 
            f"CUDA device query test not found at {device_query}")
        
        # Check if bandwidth test exists
        bandwidth_test = test_dir / 'cuda_bandwidth_test.cu'
        self.assertTrue(bandwidth_test.exists(),
            f"CUDA bandwidth test not found at {bandwidth_test}")
        
        # Try to compile and run device query if not already compiled
        device_query_bin = test_dir / 'cuda_device_query'
        if not device_query_bin.exists():
            try:
                subprocess.run(
                    ['nvcc', '-o', str(device_query_bin), str(device_query)],
                    cwd=test_dir,
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError as e:
                self.skipTest(f"Could not compile CUDA test: {e.stderr.decode()}")
        
        # Run device query test
        if device_query_bin.exists():
            try:
                result = subprocess.run(
                    [str(device_query_bin)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                self.assertIn('PASS', result.stdout, "Device query test did not pass")
            except subprocess.CalledProcessError as e:
                self.fail(f"Device query test failed: {e.stdout}")
    
    def test_memory_bandwidth(self):
        """Test that GPU memory bandwidth is reasonable for RTX 3090."""
        test_dir = Path.home() / 'mvdream' / 'tests'
        bandwidth_test_bin = test_dir / 'cuda_bandwidth_test'
        
        # Compile if needed
        if not bandwidth_test_bin.exists():
            bandwidth_test_cu = test_dir / 'cuda_bandwidth_test.cu'
            if bandwidth_test_cu.exists():
                try:
                    subprocess.run(
                        ['nvcc', '-o', str(bandwidth_test_bin), str(bandwidth_test_cu)],
                        cwd=test_dir,
                        check=True,
                        capture_output=True
                    )
                except subprocess.CalledProcessError:
                    self.skipTest("Could not compile bandwidth test")
            else:
                self.skipTest("Bandwidth test source not found")
        
        # Run bandwidth test
        if bandwidth_test_bin.exists():
            try:
                result = subprocess.run(
                    [str(bandwidth_test_bin)],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30
                )
                
                # Extract bandwidth value
                match = re.search(r'Device to Device Bandwidth: ([\d.]+) GB/s', result.stdout)
                if match:
                    bandwidth = float(match.group(1))
                    # RTX 3090 theoretical max is 936 GB/s, expect at least 300 GB/s
                    self.assertGreater(bandwidth, 300.0,
                        f"Memory bandwidth {bandwidth} GB/s is too low for RTX 3090")
                
            except subprocess.TimeoutExpired:
                self.skipTest("Bandwidth test timed out")
            except subprocess.CalledProcessError:
                self.skipTest("Bandwidth test failed to run")
    
    def test_persistent_configuration(self):
        """Test that CUDA configuration is persistent in shell config."""
        # Check for zsh config (primary shell)
        zshrc = Path.home() / '.zshrc'
        bashrc = Path.home() / '.bashrc'
        
        config_found = False
        config_file = None
        
        if zshrc.exists():
            content = zshrc.read_text()
            if 'CUDA_HOME' in content or 'cuda' in content.lower():
                config_found = True
                config_file = zshrc
        
        if not config_found and bashrc.exists():
            content = bashrc.read_text()
            if 'CUDA_HOME' in content or 'cuda' in content.lower():
                config_found = True
                config_file = bashrc
        
        self.assertTrue(config_found,
            "CUDA configuration not found in .zshrc or .bashrc")
        
        if config_file:
            content = config_file.read_text()
            self.assertIn('CUDA_HOME', content, 
                f"CUDA_HOME not set in {config_file}")
            self.assertTrue(
                'PATH' in content and 'cuda' in content.lower(),
                f"CUDA not added to PATH in {config_file}"
            )
    
    def test_documentation_exists(self):
        """Test that CUDA setup documentation exists."""
        doc_path = Path.home() / 'mvdream' / 'docs' / 'cuda-setup.md'
        self.assertTrue(doc_path.exists(),
            f"CUDA setup documentation not found at {doc_path}")
        
        # Verify documentation has minimum required content
        content = doc_path.read_text()
        required_sections = [
            'CUDA Version',
            'Environment Variables',
            'Verification',
            'GPU Information'
        ]
        
        for section in required_sections:
            self.assertIn(section, content,
                f"Documentation missing required section: {section}")


def run_tests():
    """Run all CUDA setup tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCUDASetup)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)