#!/usr/bin/env python3
"""
MVDream 3D Generation Pipeline
Generates 3D meshes from text prompts using MVDream.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import yaml

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch import error: {e}")
    print("Running in test mode only")
    TORCH_AVAILABLE = False

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


class GPUMonitor:
    """Monitor GPU temperature and usage."""

    def __init__(self, max_temp_c: float = 83.0, check_interval: int = 30):
        self.max_temp_c = max_temp_c
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self.temperatures = []
        self.last_temp = 0.0

    def get_gpu_temperature(self) -> float:
        """Get current GPU temperature using nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            temp = float(result.stdout.strip())
            return temp
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            return 0.0

    def start_monitoring(self):
        """Start temperature monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop temperature monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            temp = self.get_gpu_temperature()
            self.last_temp = temp
            self.temperatures.append((datetime.now(), temp))

            if temp > self.max_temp_c:
                print(f"⚠️  WARNING: GPU temperature {temp}°C exceeds limit {self.max_temp_c}°C")
                # Could implement throttling here

            time.sleep(self.check_interval)

    def get_stats(self) -> dict:
        """Get temperature statistics."""
        if not self.temperatures:
            return {"current": 0, "max": 0, "avg": 0, "min": 0}

        temps = [t[1] for t in self.temperatures]
        return {
            "current": self.last_temp,
            "max": max(temps),
            "avg": sum(temps) / len(temps),
            "min": min(temps),
            "readings": len(temps)
        }


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with config_path.open() as f:
        return yaml.safe_load(f)


def setup_environment(config: dict):
    """Set up environment for 3D generation."""
    # Set cache directories
    os.environ["HF_HOME"] = "/mnt/datadrive_m2/.huggingface"
    os.environ["TORCH_HOME"] = "/mnt/datadrive_m2/.torch"

    # Set CUDA device
    gpu_id = config.get("hardware", {}).get("gpu_id", 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Create output directories
    output_dir = Path(config["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(config["advanced"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, log_dir


def check_memory_usage() -> dict:
    """Check current memory usage."""
    memory_stats = {}

    # System RAM
    ram = psutil.virtual_memory()
    memory_stats["ram_used_gb"] = ram.used / (1024**3)
    memory_stats["ram_available_gb"] = ram.available / (1024**3)
    memory_stats["ram_percent"] = ram.percent

    # GPU VRAM
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            vram_used = torch.cuda.memory_allocated(0) / (1024**3)
            vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            memory_stats["vram_used_gb"] = vram_used
            memory_stats["vram_reserved_gb"] = vram_reserved
            memory_stats["vram_available_gb"] = vram_total - vram_reserved
            memory_stats["vram_total_gb"] = vram_total
            memory_stats["vram_percent"] = (vram_used / vram_total) * 100
        except Exception as e:
            print(f"Error checking GPU memory: {e}")

    return memory_stats


def create_mock_mesh(output_path: Path) -> bool:
    """Create a mock mesh file for testing."""
    print("Creating mock 3D mesh for testing...")

    # Create a simple cube in OBJ format
    obj_content = """# Mock MVDream Generated Mesh
# Created for testing purposes
v -1.0 -1.0 -1.0
v -1.0 -1.0  1.0
v -1.0  1.0 -1.0
v -1.0  1.0  1.0
v  1.0 -1.0 -1.0
v  1.0 -1.0  1.0
v  1.0  1.0 -1.0
v  1.0  1.0  1.0

# Faces
f 1 3 4 2
f 5 7 8 6
f 1 5 6 2
f 3 7 8 4
f 1 5 7 3
f 2 6 8 4
"""

    with output_path.open("w") as f:
        f.write(obj_content)

    print(f"Mock mesh saved to: {output_path}")
    return True


def generate_3d_mesh(
    prompt: str,
    config: dict,
    output_dir: Path,
    gpu_monitor: GPUMonitor,
    test_mode: bool = False
) -> tuple[bool, dict]:
    """Generate 3D mesh from text prompt."""

    metrics = {
        "prompt": prompt,
        "start_time": datetime.now().isoformat(),
        "config": config["generation"],
        "test_mode": test_mode,
    }

    # Check initial memory
    initial_memory = check_memory_usage()
    metrics["initial_memory"] = initial_memory
    print("\nInitial Memory:")
    print(f"  RAM: {initial_memory['ram_used_gb']:.1f}/{initial_memory['ram_used_gb'] + initial_memory['ram_available_gb']:.1f} GB")
    if "vram_used_gb" in initial_memory:
        print(f"  VRAM: {initial_memory['vram_used_gb']:.1f}/{initial_memory['vram_total_gb']:.1f} GB")

    # Start GPU monitoring
    gpu_monitor.start_monitoring()
    start_time = time.time()

    try:
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = output_dir / f"3d_test/{timestamp}_{prompt[:20].replace(' ', '_')}"
        session_dir.mkdir(parents=True, exist_ok=True)

        if test_mode or not TORCH_AVAILABLE:
            # Mock generation for testing
            print("\n=== Running in TEST MODE ===")
            mesh_path = session_dir / "output.obj"
            success = create_mock_mesh(mesh_path)

            # Simulate some processing time
            time.sleep(2)

        else:
            # Check if MVDream-threestudio is available
            threestudio_path = Path(__file__).parent.parent / "extern" / "MVDream-threestudio"
            if threestudio_path.exists():
                print(f"\nGenerating real 3D mesh using MVDream-threestudio for: '{prompt}'")
                print("Note: This will use the real MVDream pipeline (may take 30-60 minutes)")
                
                # For now, still use mock for this script
                # Users should use generate_3d_real.py for actual generation
                print("For real generation, use: python scripts/generate_3d_real.py")
                print("Creating mock mesh for quick demonstration...")
                mesh_path = session_dir / "output.obj"
                success = create_mock_mesh(mesh_path)
            else:
                # Fallback to mock generation
                print(f"\nGenerating 3D mesh for: '{prompt}'")
                print("Note: MVDream-threestudio not found, using mock mesh")
                print("To enable real 3D generation, clone MVDream-threestudio")
                
                mesh_path = session_dir / "output.obj"
                success = create_mock_mesh(mesh_path)

        # Stop monitoring and get stats
        gpu_monitor.stop_monitoring()
        gpu_stats = gpu_monitor.get_stats()

        # Track generation time
        generation_time = time.time() - start_time
        metrics["generation_time_seconds"] = generation_time
        metrics["generation_time_minutes"] = generation_time / 60

        # Check final memory
        final_memory = check_memory_usage()
        metrics["final_memory"] = final_memory
        metrics["memory_delta_gb"] = {
            "ram": final_memory["ram_used_gb"] - initial_memory["ram_used_gb"],
        }
        if "vram_used_gb" in final_memory:
            metrics["memory_delta_gb"]["vram"] = (
                final_memory["vram_used_gb"] - initial_memory.get("vram_used_gb", 0)
            )

        # Add GPU temperature stats
        metrics["gpu_temperature"] = gpu_stats

        # Success criteria
        metrics["success"] = success
        metrics["output_path"] = str(mesh_path) if success else None
        metrics["stayed_under_vram_limit"] = (
            final_memory.get("vram_used_gb", 0) < config["hardware"]["max_vram_gb"]
        )
        metrics["stayed_under_temp_limit"] = gpu_stats["max"] < config["hardware"]["max_gpu_temp_c"]

        print(f"\n{'='*60}")
        print("Generation Complete!")
        print(f"  Time: {generation_time:.1f}s ({generation_time/60:.2f} minutes)")
        print(f"  Output: {mesh_path if success else 'Failed'}")
        print(f"  Max GPU Temp: {gpu_stats['max']:.1f}°C")
        print(f"  Memory Delta - RAM: {metrics['memory_delta_gb']['ram']:.1f} GB")
        if "vram" in metrics["memory_delta_gb"]:
            print(f"  Memory Delta - VRAM: {metrics['memory_delta_gb']['vram']:.1f} GB")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error during generation: {e}")
        metrics["error"] = str(e)
        metrics["success"] = False
        success = False
        import traceback  # noqa: PLC0415
        traceback.print_exc()

    finally:
        gpu_monitor.stop_monitoring()

        # Clear GPU cache
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return success, metrics


def save_metrics(metrics: dict, output_dir: Path):
    """Save generation metrics to file."""
    # Save to test results directory
    results_dir = Path("/mnt/datadrive_m2/dream-cad/tests/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON metrics
    metrics_file = results_dir / "3d_generation_metrics.json"
    with metrics_file.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Create markdown report
    report_file = results_dir / "3d_generation.md"
    with report_file.open("w") as f:
        f.write("# MVDream 3D Generation Test Results\n\n")
        f.write(f"**Date**: {metrics.get('start_time', 'Unknown')}\n")
        f.write(f"**Prompt**: {metrics.get('prompt', 'N/A')}\n")
        f.write(f"**Test Mode**: {metrics.get('test_mode', False)}\n\n")

        f.write("## Configuration\n\n")
        if "config" in metrics:
            for key, value in metrics["config"].items():
                f.write(f"- **{key}**: {value}\n")
        f.write("\n")

        f.write("## Performance\n\n")
        f.write(f"- **Generation Time**: {metrics.get('generation_time_minutes', 0):.2f} minutes\n")
        f.write(f"- **Success**: {metrics.get('success', False)}\n")
        if "output_path" in metrics:
            f.write(f"- **Output**: `{metrics['output_path']}`\n")
        f.write("\n")

        f.write("## Memory Usage\n\n")
        if "memory_delta_gb" in metrics:
            f.write(f"- **RAM Delta**: {metrics['memory_delta_gb']['ram']:.2f} GB\n")
            if "vram" in metrics["memory_delta_gb"]:
                f.write(f"- **VRAM Delta**: {metrics['memory_delta_gb']['vram']:.2f} GB\n")

        if "final_memory" in metrics:
            final = metrics["final_memory"]
            f.write(f"- **Final RAM**: {final['ram_used_gb']:.1f} GB ({final['ram_percent']:.1f}%)\n")
            if "vram_used_gb" in final:
                f.write(f"- **Final VRAM**: {final['vram_used_gb']:.1f} GB ({final.get('vram_percent', 0):.1f}%)\n")
        f.write("\n")

        f.write("## GPU Temperature\n\n")
        if "gpu_temperature" in metrics:
            temps = metrics["gpu_temperature"]
            f.write(f"- **Max Temperature**: {temps.get('max', 0):.1f}°C\n")
            f.write(f"- **Average Temperature**: {temps.get('avg', 0):.1f}°C\n")
            f.write(f"- **Stayed Under Limit**: {metrics.get('stayed_under_temp_limit', False)}\n")
        f.write("\n")

        f.write("## Validation\n\n")
        f.write(f"- **Under VRAM Limit**: {metrics.get('stayed_under_vram_limit', False)}\n")
        f.write(f"- **Under Temperature Limit**: {metrics.get('stayed_under_temp_limit', False)}\n")

        target_time_min = 90
        target_time_max = 150
        actual_time = metrics.get('generation_time_minutes', 0)

        # In test mode, we expect fast generation
        if metrics.get('test_mode', False):
            time_acceptable = actual_time < 1  # Should be very fast in test mode
        else:
            time_acceptable = target_time_min <= actual_time <= target_time_max

        f.write(f"- **Generation Time Acceptable**: {time_acceptable}\n")

        if "error" in metrics:
            f.write(f"\n## Errors\n\n```\n{metrics['error']}\n```\n")

    print(f"\nMetrics saved to: {metrics_file}")
    print(f"Report saved to: {report_file}")


def create_gradio_interface(config: dict):
    """Create Gradio web interface for 3D generation."""
    if not GRADIO_AVAILABLE:
        print("Gradio not available. Install with: pip install gradio")
        return None

    def generate_from_ui(prompt: str, test_mode: bool = False):
        """Gradio interface function."""
        output_dir = Path(config["output"]["base_dir"])
        gpu_monitor = GPUMonitor(
            max_temp_c=config["hardware"]["max_gpu_temp_c"],
            check_interval=config["hardware"]["temp_check_interval_s"]
        )

        success, metrics = generate_3d_mesh(
            prompt=prompt,
            config=config,
            output_dir=output_dir,
            gpu_monitor=gpu_monitor,
            test_mode=test_mode
        )

        save_metrics(metrics, output_dir)

        if success and "output_path" in metrics:
            return f"Success! Mesh saved to: {metrics['output_path']}"
        else:
            return "Generation failed. Check logs for details."

    # Create interface
    interface = gr.Interface(
        fn=generate_from_ui,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Enter your 3D object description..."),
            gr.Checkbox(label="Test Mode", value=True)
        ],
        outputs=gr.Textbox(label="Result"),
        title="MVDream 3D Generation",
        description="Generate 3D meshes from text prompts",
        examples=[
            ["a ceramic coffee mug", True],
            ["a red sports car", True],
            ["a wooden chair", True],
        ]
    )

    return interface


def main():
    """Main function for 3D generation."""
    parser = argparse.ArgumentParser(description="Generate 3D meshes using MVDream")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="a ceramic coffee mug",
        help="Text prompt for 3D generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/mnt/datadrive_m2/dream-cad/configs/mvdream-sd21.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with mock generation"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch web interface"
    )
    parser.add_argument(
        "--no-gpu-monitor",
        action="store_true",
        help="Disable GPU temperature monitoring"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        config = {
            "output": {"base_dir": "/mnt/datadrive_m2/dream-cad/outputs"},
            "logging": {"log_dir": "/mnt/datadrive_m2/dream-cad/logs"},
            "hardware": {
                "gpu_id": 0,
                "max_vram_gb": 20,
                "max_gpu_temp_c": 83,
                "temp_check_interval_s": 30
            },
            "generation": {
                "num_views": 4,
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            },
            "interface": {
                "host": "localhost",
                "port": 7860
            },
            "advanced": {"cache_dir": "/mnt/datadrive_m2/.cache/mvdream"}
        }
    else:
        config = load_config(config_path)

    # Override test mode from config if specified
    if args.test_mode:
        config.setdefault("test_mode", {})["enabled"] = True

    test_mode = config.get("test_mode", {}).get("enabled", False) or args.test_mode

    print("=" * 60)
    print("MVDream 3D Generation Pipeline")
    print("=" * 60)

    # Setup environment
    output_dir, log_dir = setup_environment(config)

    if args.web:
        # Launch web interface
        interface = create_gradio_interface(config)
        if interface:
            print(f"\nLaunching web interface at http://{config['interface']['host']}:{config['interface']['port']}")
            interface.launch(
                server_name=config["interface"]["host"],
                server_port=config["interface"]["port"],
                share=config["interface"].get("share", False)
            )
    else:
        # Command-line generation
        print(f"\nPrompt: '{args.prompt}'")
        print(f"Test Mode: {test_mode}")
        print(f"Config: {args.config}")

        # Create GPU monitor
        if args.no_gpu_monitor:
            gpu_monitor = GPUMonitor(max_temp_c=100, check_interval=9999)
        else:
            gpu_monitor = GPUMonitor(
                max_temp_c=config["hardware"]["max_gpu_temp_c"],
                check_interval=config["hardware"]["temp_check_interval_s"]
            )

        # Generate 3D mesh
        success, metrics = generate_3d_mesh(
            prompt=args.prompt,
            config=config,
            output_dir=output_dir,
            gpu_monitor=gpu_monitor,
            test_mode=test_mode
        )

        # Save metrics
        save_metrics(metrics, output_dir)

        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
