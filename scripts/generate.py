#!/usr/bin/env python3
"""
MVDream 3D generation script.
Generate 3D models from text prompts using MVDream.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main function for MVDream generation."""
    parser = argparse.ArgumentParser(
        description="Generate 3D models from text prompts using MVDream"
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt for 3D generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/3d",
        help="Output directory for generated models (default: outputs/3d)",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=4,
        help="Number of views to generate (default: 4)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale for generation (default: 7.5)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for generation (default: cuda)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating 3D model for prompt: '{args.prompt}'")
    print(f"Output directory: {output_dir}")
    print(f"Number of views: {args.num_views}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")

    # Redirect to the 3D generation script
    print("\nRedirecting to 3D generation pipeline...")
    import subprocess  # noqa: PLC0415
    cmd = [
        sys.executable,
        "/mnt/datadrive_m2/dream-cad/scripts/generate_3d.py",
        args.prompt,
        "--test-mode"  # Remove this when real generation is ready
    ]
    result = subprocess.run(cmd)
    return result.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
