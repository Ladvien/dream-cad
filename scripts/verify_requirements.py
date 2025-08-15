#!/usr/bin/env python3
"""
System requirements verification script for MVDream setup.
Checks all hardware and software prerequisites.
"""

import shutil
import subprocess
import sys
from pathlib import Path


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_status(message, status):
    """Print colored status message."""
    if status == "pass":
        print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")
    elif status == "fail":
        print(f"{Colors.RED}❌ {message}{Colors.ENDC}")
    else:
        print(f"❓ {message}")


def check_nvidia_gpu():
    """Check for NVIDIA GPU with required VRAM."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_info = result.stdout.strip()

        if "RTX 3090" in gpu_info or "RTX 4090" in gpu_info or "A5000" in gpu_info:
            vram = int(gpu_info.split(",")[1].strip().split()[0])
            driver = gpu_info.split(",")[2].strip()

            if vram >= 24000:
                print_status(f"GPU: {gpu_info.split(',')[0]} with {vram}MB VRAM", "pass")
            else:
                print_status(f"GPU has insufficient VRAM: {vram}MB (need 24GB+)", "fail")
                return False

            driver_major = int(driver.split(".")[0])
            if driver_major >= 470:
                print_status(f"NVIDIA Driver: {driver}", "pass")
            else:
                print_status(f"NVIDIA Driver too old: {driver} (need 470+)", "fail")
                return False

            return True
        else:
            print_status(f"GPU may not meet requirements: {gpu_info}", "warning")
            return False

    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status("NVIDIA driver not found or not accessible", "fail")
        return False


def check_system_memory():
    """Check system RAM meets minimum requirements."""
    try:
        result = subprocess.run(
            ["free", "-b"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        mem_line = lines[1].split()
        total_mem_gb = int(mem_line[1]) / (1024**3)

        if total_mem_gb >= 31:  # Allow for ~31GB as "32GB"
            print_status(f"System RAM: {total_mem_gb:.1f}GB", "pass")
            return True
        else:
            print_status(f"Insufficient RAM: {total_mem_gb:.1f}GB (need 32GB+)", "fail")
            return False

    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status("Could not check system memory", "fail")
        return False


def check_disk_space():
    """Check available disk space."""
    home_path = Path.home()
    mvdream_path = home_path / "mvdream"

    # Check the filesystem where mvdream will be located
    if mvdream_path.exists():
        check_path = mvdream_path
    else:
        check_path = home_path

    total, used, free = shutil.disk_usage(check_path)
    free_gb = free / (1024**3)

    if free_gb >= 50:
        print_status(f"Disk space: {free_gb:.1f}GB free", "pass")
        return True
    elif free_gb >= 20:
        print_status(f"Limited disk space: {free_gb:.1f}GB free (50GB recommended)", "warning")
        return False
    else:
        print_status(f"Insufficient disk space: {free_gb:.1f}GB free (need 50GB+)", "fail")
        return False


def check_cuda():
    """Check if CUDA is installed and accessible."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        if "11.8" in result.stdout or "11.7" in result.stdout or "12." in result.stdout:
            version = result.stdout.split("release")[1].split(",")[0].strip()
            print_status(f"CUDA Toolkit: {version}", "pass")
            return True
        else:
            print_status("CUDA 11.8 not found (will be installed in Story 2)", "warning")
            return False

    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status("CUDA not installed yet (will be installed in Story 2)", "warning")
        return False


def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor in [10, 11]:
        print_status(f"Python: {version.major}.{version.minor}.{version.micro}", "pass")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor} (need 3.10 or 3.11)", "warning")
        return False


def check_project_structure():
    """Check if project directories exist."""
    mvdream_path = Path.home() / "mvdream"
    required_dirs = ["docs", "tests", "outputs", "scripts", "benchmarks", "logs"]

    if not mvdream_path.exists():
        print_status("Project directory ~/mvdream not found", "fail")
        return False

    all_exist = True
    for dir_name in required_dirs:
        dir_path = mvdream_path / dir_name
        if not dir_path.exists():
            print_status(f"Missing directory: {dir_path}", "fail")
            all_exist = False

    if all_exist:
        print_status("Project directory structure complete", "pass")

    return all_exist


def main():
    """Run all verification checks."""
    print(f"\n{Colors.BOLD}=== MVDream System Requirements Verification ==={Colors.ENDC}\n")

    checks = {
        "NVIDIA GPU": check_nvidia_gpu(),
        "System Memory": check_system_memory(),
        "Disk Space": check_disk_space(),
        "CUDA Toolkit": check_cuda(),
        "Python Version": check_python(),
        "Project Structure": check_project_structure(),
    }

    print(f"\n{Colors.BOLD}=== Summary ==={Colors.ENDC}")

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    if passed == total:
        print(f"{Colors.GREEN}All checks passed! ({passed}/{total}){Colors.ENDC}")
        return 0
    else:
        print(f"{Colors.YELLOW}Some checks need attention ({passed}/{total} passed){Colors.ENDC}")

        # Provide actionable next steps
        print(f"\n{Colors.BOLD}=== Next Steps ==={Colors.ENDC}")

        if not checks["Disk Space"]:
            print("1. Free up disk space or mount external storage (50GB needed)")
            print("   - Clean package cache: sudo pacman -Scc")
            print("   - Remove old kernels: sudo pacman -R $(pacman -Qdt)")
            print("   - Check large files: du -h ~ | sort -rh | head -20")

        if not checks["CUDA Toolkit"]:
            print("2. CUDA 11.8 will be installed in Story 2")

        return 1


if __name__ == "__main__":
    sys.exit(main())
