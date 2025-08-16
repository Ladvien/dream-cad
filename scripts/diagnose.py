#!/usr/bin/env python3
"""MVDream Diagnostic Tool - Comprehensive system and dependency check."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

# Color codes for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


class DiagnosticCheck:
    """Base class for diagnostic checks."""

    def __init__(self, name: str, description: str):
        """Initialize diagnostic check."""
        self.name = name
        self.description = description
        self.passed = False
        self.message = ""
        self.details: dict[str, Any] = {}

    def run(self) -> bool:
        """Run the diagnostic check. Override in subclasses."""
        raise NotImplementedError

    def print_result(self) -> None:
        """Print the result of the check."""
        status_symbol = f"{GREEN}✓{RESET}" if self.passed else f"{RED}✗{RESET}"
        status_text = f"{GREEN}PASSED{RESET}" if self.passed else f"{RED}FAILED{RESET}"
        
        print(f"\n{status_symbol} {BOLD}{self.name}{RESET}: {status_text}")
        print(f"  {self.description}")
        if self.message:
            color = GREEN if self.passed else YELLOW
            print(f"  {color}{self.message}{RESET}")
        
        if self.details:
            for key, value in self.details.items():
                print(f"    • {key}: {value}")


class SystemInfoCheck(DiagnosticCheck):
    """Check system information."""

    def __init__(self):
        """Initialize system info check."""
        super().__init__(
            "System Information",
            "Gather basic system information"
        )

    def run(self) -> bool:
        """Gather system information."""
        try:
            self.details = {
                "Platform": platform.system(),
                "Platform Release": platform.release(),
                "Platform Version": platform.version()[:50] + "...",
                "Architecture": platform.machine(),
                "Processor": platform.processor()[:50] + "...",
                "Python Version": sys.version.split()[0],
                "Python Executable": sys.executable,
            }
            
            # Check if running in virtual environment
            venv = os.environ.get("VIRTUAL_ENV")
            if venv:
                self.details["Virtual Environment"] = venv
            else:
                self.details["Virtual Environment"] = "Not detected (may cause issues)"
            
            self.passed = True
            self.message = "System information collected successfully"
            return True
            
        except Exception as e:
            self.passed = False
            self.message = f"Error collecting system info: {e}"
            return False


class CUDACheck(DiagnosticCheck):
    """Check CUDA availability and configuration."""

    def __init__(self):
        """Initialize CUDA check."""
        super().__init__(
            "CUDA Configuration",
            "Check CUDA toolkit and GPU availability"
        )

    def run(self) -> bool:
        """Check CUDA configuration."""
        all_checks_passed = True
        
        # Check nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            gpu_info = result.stdout.strip()
            self.details["GPU"] = gpu_info
            
            # Check GPU memory
            if "24576" in gpu_info or "24GB" in gpu_info or "23028" in gpu_info:
                self.details["GPU Memory"] = "24GB detected (RTX 3090)"
            else:
                self.details["GPU Memory"] = f"Detected: {gpu_info}"
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.details["nvidia-smi"] = "Not available"
            all_checks_passed = False
        
        # Check nvcc
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    self.details["CUDA Toolkit"] = line.strip()
                    break
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.details["CUDA Toolkit"] = "nvcc not found"
            all_checks_passed = False
        
        # Check CUDA environment variables
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            self.details["CUDA_HOME"] = cuda_home
        else:
            self.details["CUDA_HOME"] = "Not set (may cause issues)"
            
        # Check PyTorch CUDA support
        try:
            import torch
            self.details["PyTorch Version"] = torch.__version__
            self.details["PyTorch CUDA Available"] = str(torch.cuda.is_available())
            if torch.cuda.is_available():
                self.details["PyTorch CUDA Version"] = torch.version.cuda
                self.details["GPU Count"] = torch.cuda.device_count()
                if torch.cuda.device_count() > 0:
                    self.details["GPU 0 Name"] = torch.cuda.get_device_name(0)
        except ImportError:
            self.details["PyTorch"] = "Not installed or import error"
            all_checks_passed = False
        except Exception as e:
            self.details["PyTorch Error"] = str(e)
            all_checks_passed = False
        
        self.passed = all_checks_passed
        if all_checks_passed:
            self.message = "CUDA configuration is properly set up"
        else:
            self.message = "Some CUDA components are missing or misconfigured"
        
        return all_checks_passed


class DependencyCheck(DiagnosticCheck):
    """Check Python dependencies."""

    def __init__(self):
        """Initialize dependency check."""
        super().__init__(
            "Python Dependencies",
            "Verify all required Python packages are installed"
        )
        
    def run(self) -> bool:
        """Check Python dependencies."""
        required_packages = [
            "torch",
            "torchvision", 
            "transformers",
            "diffusers",
            "pytorch_lightning",
            "omegaconf",
            "einops",
            "gradio",
            "pillow",
            "numpy",
            "scipy",
            "matplotlib",
            "opencv-python",
            "imageio",
            "trimesh",
            "psutil",
            "pyyaml",
            "tqdm",
            "accelerate",
            "safetensors",
        ]
        
        missing = []
        installed = []
        errors = []
        
        for package in required_packages:
            # Map package names to import names
            import_name = package.replace("-", "_")
            if package == "opencv-python":
                import_name = "cv2"
            elif package == "pillow":
                import_name = "PIL"
            elif package == "pytorch-lightning":
                import_name = "pytorch_lightning"
            elif package == "pyyaml":
                import_name = "yaml"
                
            try:
                module = __import__(import_name)
                version = getattr(module, "__version__", "unknown")
                installed.append(f"{package} ({version})")
            except ImportError:
                missing.append(package)
            except Exception as e:
                errors.append(f"{package}: {e}")
        
        self.details["Installed"] = f"{len(installed)}/{len(required_packages)} packages"
        
        if missing:
            self.details["Missing"] = ", ".join(missing)
            
        if errors:
            self.details["Import Errors"] = "; ".join(errors)
        
        self.passed = len(missing) == 0 and len(errors) == 0
        
        if self.passed:
            self.message = "All required dependencies are installed"
        else:
            self.message = f"Missing {len(missing)} packages, {len(errors)} import errors"
            
        return self.passed


class GPUMemoryCheck(DiagnosticCheck):
    """Test GPU memory allocation."""

    def __init__(self):
        """Initialize GPU memory check."""
        super().__init__(
            "GPU Memory Allocation",
            "Test ability to allocate GPU memory"
        )

    def run(self) -> bool:
        """Test GPU memory allocation."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                self.passed = False
                self.message = "CUDA is not available"
                return False
            
            # Try to allocate increasingly large tensors
            test_sizes_gb = [1, 4, 8, 12, 16, 20]
            max_allocated = 0
            
            for size_gb in test_sizes_gb:
                try:
                    # Calculate tensor size for desired GB
                    # float32 = 4 bytes per element
                    elements = (size_gb * 1024 * 1024 * 1024) // 4
                    tensor = torch.zeros(elements, dtype=torch.float32, device="cuda")
                    max_allocated = size_gb
                    del tensor
                    torch.cuda.empty_cache()
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    break
            
            self.details["Max Allocation"] = f"{max_allocated} GB"
            self.details["Total VRAM"] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
            self.details["Reserved VRAM"] = f"{torch.cuda.memory_reserved(0) / (1024**3):.2f} GB"
            
            # Clean up
            torch.cuda.empty_cache()
            
            self.passed = max_allocated >= 16  # Need at least 16GB for MVDream
            
            if self.passed:
                self.message = f"Successfully allocated up to {max_allocated}GB VRAM"
            else:
                self.message = f"Could only allocate {max_allocated}GB VRAM (need 16GB+)"
                
            return self.passed
            
        except ImportError:
            self.passed = False
            self.message = "PyTorch not available for GPU memory test"
            return False
        except Exception as e:
            self.passed = False
            self.message = f"Error during GPU memory test: {e}"
            return False


class ModelFileCheck(DiagnosticCheck):
    """Validate model file integrity."""

    def __init__(self):
        """Initialize model file check."""
        super().__init__(
            "Model Files",
            "Check for required model files and validate integrity"
        )

    def run(self) -> bool:
        """Check model files."""
        models_dir = Path("/mnt/datadrive_m2/dream-cad/models")
        model_info_file = models_dir / "model_info.json"
        
        if not models_dir.exists():
            self.passed = False
            self.message = f"Models directory not found: {models_dir}"
            return False
        
        # Check for model info file
        if model_info_file.exists():
            try:
                with model_info_file.open() as f:
                    model_info = json.load(f)
                
                # Check each model
                for model_name, info in model_info.items():
                    model_path = Path(info["path"])
                    if model_path.exists():
                        size_mb = model_path.stat().st_size / (1024 * 1024)
                        self.details[model_name] = f"{size_mb:.1f} MB"
                        
                        # Verify checksum if available
                        if "sha256" in info and info["sha256"]:
                            self.details[f"{model_name} checksum"] = "Available (run verify-models to check)"
                    else:
                        self.details[model_name] = "File not found"
                        
            except Exception as e:
                self.details["Error"] = f"Failed to read model info: {e}"
        else:
            self.details["model_info.json"] = "Not found"
        
        # Check for common model files
        expected_models = [
            "sd-v2.1-base-4view.pt",
            "sd-v2.1-base-4view.yaml",
        ]
        
        found_models = 0
        for model_name in expected_models:
            model_path = models_dir / model_name
            if model_path.exists():
                found_models += 1
                if model_name not in self.details:
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    self.details[model_name] = f"{size_mb:.1f} MB"
            elif model_name not in self.details:
                self.details[model_name] = "Not found"
        
        # Check HuggingFace cache
        hf_cache = Path("/mnt/datadrive_m2/.huggingface")
        if hf_cache.exists():
            self.details["HuggingFace Cache"] = "Configured"
        else:
            self.details["HuggingFace Cache"] = "Not found"
        
        self.passed = found_models > 0
        
        if self.passed:
            self.message = f"Found {found_models} model files"
        else:
            self.message = "No model files found - run 'poe download-models'"
            
        return self.passed


class DirectoryStructureCheck(DiagnosticCheck):
    """Check project directory structure."""

    def __init__(self):
        """Initialize directory check."""
        super().__init__(
            "Directory Structure", 
            "Verify project directories are properly set up"
        )

    def run(self) -> bool:
        """Check directory structure."""
        base_dir = Path("/mnt/datadrive_m2/dream-cad")
        
        required_dirs = [
            "models",
            "outputs",
            "outputs/2d_test",
            "outputs/3d_test",
            "configs",
            "scripts",
            "tests",
            "tests/results",
            "docs",
            "extern/MVDream",
            "extern/MVDream-threestudio",
            "logs",
        ]
        
        missing = []
        found = []
        
        for dir_name in required_dirs:
            dir_path = base_dir / dir_name
            if dir_path.exists():
                found.append(dir_name)
            else:
                missing.append(dir_name)
        
        self.details["Found"] = f"{len(found)}/{len(required_dirs)} directories"
        
        if missing:
            self.details["Missing"] = ", ".join(missing[:5])  # Show first 5
            if len(missing) > 5:
                self.details["Missing"] += f" ... and {len(missing) - 5} more"
        
        self.passed = len(missing) <= 3  # Allow a few missing dirs
        
        if self.passed:
            self.message = f"Directory structure is mostly complete ({len(found)}/{len(required_dirs)})"
        else:
            self.message = f"Missing {len(missing)} required directories"
            
        return self.passed


class ConfigurationCheck(DiagnosticCheck):
    """Check configuration files."""

    def __init__(self):
        """Initialize configuration check."""
        super().__init__(
            "Configuration Files",
            "Verify configuration files are present and valid"
        )

    def run(self) -> bool:
        """Check configuration files."""
        base_dir = Path("/mnt/datadrive_m2/dream-cad")
        
        config_files = {
            "pyproject.toml": base_dir / "pyproject.toml",
            "ruff.toml": base_dir / "ruff.toml",
            ".gitignore": base_dir / ".gitignore",
            ".bandit": base_dir / ".bandit",
            ".pre-commit-config.yaml": base_dir / ".pre-commit-config.yaml",
            "mvdream-sd21.yaml": base_dir / "configs" / "mvdream-sd21.yaml",
        }
        
        found = []
        missing = []
        invalid = []
        
        for name, path in config_files.items():
            if path.exists():
                found.append(name)
                
                # Try to validate YAML files
                if name.endswith(".yaml"):
                    try:
                        import yaml
                        with path.open() as f:
                            yaml.safe_load(f)
                    except Exception as e:
                        invalid.append(f"{name}: {e}")
                        
            else:
                missing.append(name)
        
        self.details["Found"] = f"{len(found)}/{len(config_files)} config files"
        
        if missing:
            self.details["Missing"] = ", ".join(missing)
            
        if invalid:
            self.details["Invalid"] = "; ".join(invalid)
        
        self.passed = len(missing) == 0 and len(invalid) == 0
        
        if self.passed:
            self.message = "All configuration files present and valid"
        else:
            self.message = f"Missing {len(missing)} files, {len(invalid)} invalid"
            
        return self.passed


class DiskSpaceCheck(DiagnosticCheck):
    """Check available disk space."""

    def __init__(self):
        """Initialize disk space check."""
        super().__init__(
            "Disk Space",
            "Check available disk space for model storage and generation"
        )

    def run(self) -> bool:
        """Check disk space."""
        try:
            import shutil
            
            paths_to_check = [
                ("/mnt/datadrive_m2", "Data Drive"),
                ("/home", "Home"),
                ("/", "Root"),
            ]
            
            min_required_gb = 50
            adequate_space = False
            
            for path, name in paths_to_check:
                if Path(path).exists():
                    stat = shutil.disk_usage(path)
                    free_gb = stat.free / (1024 ** 3)
                    total_gb = stat.total / (1024 ** 3)
                    used_percent = (stat.used / stat.total) * 100
                    
                    self.details[name] = f"{free_gb:.1f}GB free of {total_gb:.1f}GB ({used_percent:.1f}% used)"
                    
                    if path == "/mnt/datadrive_m2" and free_gb >= min_required_gb:
                        adequate_space = True
            
            self.passed = adequate_space
            
            if self.passed:
                self.message = f"Adequate disk space available on data drive"
            else:
                self.message = f"Need at least {min_required_gb}GB free on /mnt/datadrive_m2"
                
            return self.passed
            
        except Exception as e:
            self.passed = False
            self.message = f"Error checking disk space: {e}"
            return False


def run_diagnostics(verbose: bool = False) -> int:
    """Run all diagnostic checks."""
    print(f"\n{BOLD}MVDream Diagnostic Tool{RESET}")
    print("=" * 50)
    
    checks = [
        SystemInfoCheck(),
        CUDACheck(),
        DependencyCheck(),
        GPUMemoryCheck(),
        ModelFileCheck(),
        DirectoryStructureCheck(),
        ConfigurationCheck(),
        DiskSpaceCheck(),
    ]
    
    passed_count = 0
    failed_count = 0
    
    for check in checks:
        try:
            check.run()
            check.print_result()
            
            if check.passed:
                passed_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"\n{RED}Error running {check.name}: {e}{RESET}")
            failed_count += 1
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"{BOLD}Summary:{RESET}")
    print(f"  {GREEN}Passed: {passed_count}{RESET}")
    print(f"  {RED}Failed: {failed_count}{RESET}")
    
    if failed_count == 0:
        print(f"\n{GREEN}{BOLD}All diagnostics passed!{RESET}")
        print("Your MVDream setup appears to be correctly configured.")
        return 0
    else:
        print(f"\n{YELLOW}{BOLD}Some diagnostics failed.{RESET}")
        print("Please check the failed items above and consult docs/troubleshooting.md")
        return 1


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MVDream Diagnostic Tool")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        exit_code = run_diagnostics(verbose=args.verbose)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Diagnostic interrupted by user{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Fatal error: {e}{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()