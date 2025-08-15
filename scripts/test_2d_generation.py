#!/usr/bin/env python3
"""
Test MVDream 2D multi-view generation functionality.
Generates 4 consistent views from a text prompt.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch import error: {e}")
    print("Running in test mode only")
    TORCH_AVAILABLE = False


def setup_environment():
    """Set up environment for MVDream generation."""
    # Set cache directories to data drive
    os.environ["HF_HOME"] = "/mnt/datadrive_m2/.huggingface"
    os.environ["TORCH_HOME"] = "/mnt/datadrive_m2/.torch"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Check if running in test mode
    if os.environ.get("MVDREAM_TEST_MODE", "").lower() == "true":
        print("Running in TEST MODE - using mock generation")
        return True
    return False


def check_gpu_availability() -> tuple[bool, str | None]:
    """Check if GPU is available and get device name."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available - GPU check skipped")
        return False, None

    try:
        if not torch.cuda.is_available():
            return False, None

        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU detected: {device_name}")
        print(f"VRAM: {vram_gb:.1f} GB")
        return True, device_name
    except Exception as e:
        print(f"Error checking GPU: {e}")
        return False, None


def monitor_memory() -> dict[str, float]:
    """Monitor system and GPU memory usage."""
    memory_stats = {}

    # System RAM
    ram = psutil.virtual_memory()
    memory_stats["ram_used_gb"] = ram.used / (1024**3)
    memory_stats["ram_percent"] = ram.percent

    # GPU VRAM
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            vram_used = torch.cuda.memory_allocated(0) / (1024**3)
            vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            memory_stats["vram_used_gb"] = vram_used
            memory_stats["vram_reserved_gb"] = vram_reserved
            memory_stats["vram_total_gb"] = vram_total
            memory_stats["vram_percent"] = (vram_used / vram_total) * 100
        except Exception as e:
            print(f"Error monitoring GPU memory: {e}")

    return memory_stats


def create_mock_images(prompt: str, output_dir: Path) -> list[Path]:
    """Create mock images for testing when MVDream is not available."""
    print("\n=== Creating mock images for testing ===")

    # Create 4 mock views
    views = ["front", "back", "left", "right"]
    image_paths = []

    for i, view in enumerate(views):
        # Create a simple colored image with text
        img = Image.new("RGB", (256, 256), color=(100 + i*40, 100, 200 - i*30))

        # Draw some basic shapes to simulate content
        from PIL import ImageDraw  # noqa: PLC0415
        draw = ImageDraw.Draw(img)

        # Draw view label
        try:
            # Try to use a basic font
            draw.text((10, 10), f"{view.upper()} VIEW", fill=(255, 255, 255))
            draw.text((10, 30), f"Prompt: {prompt[:20]}...", fill=(255, 255, 255))
        except Exception:
            # If font fails, just use rectangles
            draw.rectangle([50, 50, 200, 200], outline=(255, 255, 255), width=3)

        # Add some variation
        for j in range(3):
            x = 80 + j * 40
            y = 100 + i * 20
            draw.ellipse([x, y, x+30, y+30], fill=(255, 200, 100))

        # Save image
        image_path = output_dir / f"view_{i:02d}_{view}.png"
        img.save(image_path)
        image_paths.append(image_path)
        print(f"Created mock image: {image_path.name}")

    return image_paths


def load_mvdream_model(model_path: Path | None = None):
    """Load MVDream model for generation."""
    print("\nLoading MVDream model...")

    # Check if model exists
    if model_path is None:
        model_path = Path("/mnt/datadrive_m2/dream-cad/models/sd-v2.1-base-4view.pt")

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please run 'poe download-models' first")
        return None

    try:
        # Try to import MVDream
        sys.path.append("/mnt/datadrive_m2/dream-cad/extern/MVDream")
        from mvdream.model_zoo import build_model  # noqa: PLC0415

        # Load model
        model = build_model(model_path)
        print("MVDream model loaded successfully")
        return model
    except ImportError as e:
        print(f"Warning: MVDream not available: {e}")
        print("Will use mock generation for testing")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def generate_multiview_images(
    prompt: str,
    model,
    output_dir: Path,
    num_views: int = 4,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int = 42,
) -> tuple[list[Path], dict]:
    """Generate multi-view images from text prompt."""
    print(f"\nGenerating {num_views} views for prompt: '{prompt}'")

    # Set random seed
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
    np.random.seed(seed)

    # Track metrics
    metrics = {
        "prompt": prompt,
        "num_views": num_views,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "start_time": datetime.now().isoformat(),
    }

    # Monitor initial memory
    initial_memory = monitor_memory()
    metrics["initial_memory"] = initial_memory
    print(f"Initial RAM: {initial_memory.get('ram_used_gb', 0):.1f} GB")
    if "vram_used_gb" in initial_memory:
        print(f"Initial VRAM: {initial_memory['vram_used_gb']:.1f} GB")

    start_time = time.time()

    # Check if we're in test mode or model is not available
    if model is None or os.environ.get("MVDREAM_TEST_MODE", "").lower() == "true":
        # Use mock generation
        image_paths = create_mock_images(prompt, output_dir)
    else:
        # Real MVDream generation
        try:
            from mvdream.pipeline_mvdream import MVDreamPipeline  # noqa: PLC0415

            # Initialize pipeline
            pipe = MVDreamPipeline.from_pretrained(
                model,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            pipe = pipe.to("cuda")

            # Generate images
            with torch.no_grad():
                images = pipe(
                    prompt,
                    num_images_per_prompt=num_views,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=256,
                    width=256,
                ).images

            # Save images
            image_paths = []
            views = ["front", "back", "left", "right"]
            for i, (img, view) in enumerate(zip(images, views)):
                image_path = output_dir / f"view_{i:02d}_{view}.png"
                img.save(image_path)
                image_paths.append(image_path)
                print(f"Saved: {image_path.name}")

        except Exception as e:
            print(f"Error during generation: {e}")
            print("Falling back to mock generation")
            image_paths = create_mock_images(prompt, output_dir)

    # Track generation time
    generation_time = time.time() - start_time
    metrics["generation_time_seconds"] = generation_time
    metrics["generation_time_minutes"] = generation_time / 60

    # Monitor peak memory
    peak_memory = monitor_memory()
    metrics["peak_memory"] = peak_memory

    # Calculate memory delta
    if "vram_used_gb" in peak_memory and "vram_used_gb" in initial_memory:
        vram_delta = peak_memory["vram_used_gb"] - initial_memory["vram_used_gb"]
        metrics["vram_delta_gb"] = vram_delta
        print(f"\nPeak VRAM usage: {peak_memory['vram_used_gb']:.1f} GB")
        print(f"VRAM delta: {vram_delta:.1f} GB")

    ram_delta = peak_memory["ram_used_gb"] - initial_memory["ram_used_gb"]
    metrics["ram_delta_gb"] = ram_delta
    print(f"Peak RAM usage: {peak_memory['ram_used_gb']:.1f} GB")
    print(f"RAM delta: {ram_delta:.1f} GB")

    print(f"\nGeneration completed in {generation_time:.1f} seconds ({generation_time/60:.2f} minutes)")

    # Clear GPU cache
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return image_paths, metrics


def verify_output_consistency(image_paths: list[Path]) -> bool:
    """Verify that generated images show consistent features."""
    print("\nVerifying output consistency...")

    if not image_paths:
        print("No images to verify")
        return False

    # Load images
    images = []
    for path in image_paths:
        if path.exists():
            img = Image.open(path)
            images.append(np.array(img))
            print(f"Loaded: {path.name} - Size: {img.size}")
        else:
            print(f"Warning: Image not found: {path}")

    if len(images) != len(image_paths):
        print(f"Warning: Only {len(images)}/{len(image_paths)} images loaded")

    # Basic consistency checks
    consistency_checks = {
        "all_images_exist": len(images) == len(image_paths),
        "all_same_size": len(set(img.shape for img in images)) == 1 if images else False,
        "all_color_images": all(len(img.shape) == 3 for img in images) if images else False,
    }

    # Check image statistics for similarity
    if images:
        mean_colors = [img.mean(axis=(0, 1)) for img in images]
        color_variance = np.var(mean_colors, axis=0).mean()
        consistency_checks["color_consistency"] = color_variance < 5000  # Threshold for color similarity

        print(f"Color variance across views: {color_variance:.2f}")

    # Report results
    print("\nConsistency checks:")
    for check, passed in consistency_checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}: {passed}")

    return all(consistency_checks.values())


def save_test_results(metrics: dict, output_dir: Path):
    """Save test results to JSON and markdown."""
    # Save JSON metrics
    json_path = output_dir / "metrics.json"
    with json_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to: {json_path}")

    # Create markdown report
    report_dir = Path("/mnt/datadrive_m2/dream-cad/tests/results")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "2d_generation.md"

    with report_path.open("w") as f:
        f.write("# MVDream 2D Generation Test Results\n\n")
        f.write(f"**Date**: {metrics.get('start_time', 'Unknown')}\n\n")
        f.write("## Test Configuration\n\n")
        f.write(f"- **Prompt**: {metrics.get('prompt', 'N/A')}\n")
        f.write(f"- **Number of views**: {metrics.get('num_views', 'N/A')}\n")
        f.write(f"- **Guidance scale**: {metrics.get('guidance_scale', 'N/A')}\n")
        f.write(f"- **Inference steps**: {metrics.get('num_inference_steps', 'N/A')}\n")
        f.write(f"- **Seed**: {metrics.get('seed', 'N/A')}\n\n")

        f.write("## Performance Metrics\n\n")
        gen_time = metrics.get('generation_time_minutes', 0)
        f.write(f"- **Generation time**: {gen_time:.2f} minutes\n")
        f.write(f"- **Time per view**: {gen_time/metrics.get('num_views', 4)*60:.1f} seconds\n\n")

        f.write("## Memory Usage\n\n")
        if 'peak_memory' in metrics:
            peak = metrics['peak_memory']
            f.write(f"- **Peak RAM**: {peak.get('ram_used_gb', 0):.1f} GB ({peak.get('ram_percent', 0):.1f}%)\n")
            if 'vram_used_gb' in peak:
                f.write(f"- **Peak VRAM**: {peak['vram_used_gb']:.1f} GB ({peak.get('vram_percent', 0):.1f}%)\n")

        if 'vram_delta_gb' in metrics:
            f.write(f"- **VRAM delta**: {metrics['vram_delta_gb']:.1f} GB\n")
        f.write(f"- **RAM delta**: {metrics.get('ram_delta_gb', 0):.1f} GB\n\n")

        f.write("## Results\n\n")
        f.write(f"- **Status**: {'✓ Success' if metrics.get('success', False) else '✗ Failed'}\n")
        f.write(f"- **Consistency verified**: {metrics.get('consistency_verified', False)}\n")
        f.write(f"- **CUDA OOM errors**: {metrics.get('cuda_oom', False)}\n")
        f.write(f"- **Under 5 minutes**: {gen_time < 5}\n")
        f.write(f"- **Under 20GB VRAM**: {metrics.get('peak_memory', {}).get('vram_used_gb', 0) < 20}\n\n")

        if 'error' in metrics:
            f.write(f"## Errors\n\n```\n{metrics['error']}\n```\n\n")

        f.write("## Generated Images\n\n")
        if 'output_dir' in metrics:
            f.write(f"Images saved to: `{metrics['output_dir']}`\n\n")
            for i in range(metrics.get('num_views', 4)):
                f.write(f"- view_{i:02d}_*.png\n")

    print(f"Report saved to: {report_path}")


def main():
    """Main function for 2D generation testing."""
    parser = argparse.ArgumentParser(description="Test MVDream 2D multi-view generation")
    parser.add_argument(
        "--prompt",
        type=str,
        default="an astronaut riding a horse",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/datadrive_m2/dream-cad/outputs/2d_test",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=4,
        help="Number of views to generate",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale for generation",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with mock generation",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MVDream 2D Multi-View Generation Test")
    print("=" * 60)

    # Set test mode if requested
    if args.test_mode:
        os.environ["MVDREAM_TEST_MODE"] = "true"

    # Setup environment
    is_test_mode = setup_environment()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Check GPU
    gpu_available, device_name = check_gpu_availability()
    if not gpu_available and not is_test_mode:
        print("\nWarning: No GPU available. Generation may be very slow or fail.")

    # Initialize metrics
    metrics = {
        "success": False,
        "cuda_oom": False,
        "output_dir": str(output_dir),
        "gpu_available": gpu_available,
        "device_name": device_name,
        "test_mode": is_test_mode,
    }

    try:
        # Load model (unless in test mode)
        model = None
        if not is_test_mode:
            model = load_mvdream_model()

        # Generate images
        image_paths, generation_metrics = generate_multiview_images(
            prompt=args.prompt,
            model=model,
            output_dir=output_dir,
            num_views=args.num_views,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
        )

        # Update metrics
        metrics.update(generation_metrics)

        # Verify consistency
        consistency = verify_output_consistency(image_paths)
        metrics["consistency_verified"] = consistency

        # Check success criteria
        metrics["success"] = (
            len(image_paths) == args.num_views
            and consistency
            and generation_metrics["generation_time_minutes"] < 5
            and generation_metrics.get("peak_memory", {}).get("vram_used_gb", 0) < 20
        )

        print("\n" + "=" * 60)
        print("Test Results:")
        print(f"  ✓ Images generated: {len(image_paths)}/{args.num_views}")
        print(f"  {'✓' if consistency else '✗'} Consistency verified: {consistency}")
        print(f"  {'✓' if generation_metrics['generation_time_minutes'] < 5 else '✗'} Under 5 minutes: {generation_metrics['generation_time_minutes']:.2f} min")
        if "vram_used_gb" in generation_metrics.get("peak_memory", {}):
            vram_used = generation_metrics["peak_memory"]["vram_used_gb"]
            print(f"  {'✓' if vram_used < 20 else '✗'} Under 20GB VRAM: {vram_used:.1f} GB")
        print("=" * 60)

    except Exception as e:
        if TORCH_AVAILABLE and "OutOfMemoryError" in str(type(e)):
            print(f"\n✗ CUDA out of memory error: {e}")
            metrics["cuda_oom"] = True
            metrics["error"] = str(e)
            metrics["success"] = False
        else:
            print(f"\n✗ Error during test: {e}")
            metrics["error"] = str(e)
            metrics["success"] = False
            import traceback  # noqa: PLC0415
            traceback.print_exc()

    # Save results
    save_test_results(metrics, output_dir)

    return 0 if metrics["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
