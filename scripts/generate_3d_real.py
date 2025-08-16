#!/usr/bin/env python3
"""
Real 3D generation using MVDream-threestudio pipeline.
This replaces the mock generation with actual text-to-3D using MVDream + threestudio.
"""
import argparse
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json
import shutil
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ["HF_HOME"] = "/mnt/datadrive_m2/.huggingface"
os.environ["TORCH_HOME"] = "/mnt/datadrive_m2/.torch"
os.environ["CUDA_HOME"] = "/opt/cuda"
os.environ["PYTHONWARNINGS"] = "ignore"


def check_mvdream_installation():
    """Check if MVDream and threestudio are properly installed."""
    threestudio_path = project_root / "extern" / "MVDream-threestudio"
    mvdream_path = project_root / "extern" / "MVDream"
    
    if not threestudio_path.exists():
        print("❌ MVDream-threestudio not found. Please initialize submodules:")
        print("   git submodule update --init --recursive")
        return False
    
    if not mvdream_path.exists():
        print("❌ MVDream not found. Please ensure it's installed:")
        print("   Already exists at extern/MVDream")
        return False
    
    # Check if MVDream is installed as a package
    try:
        import mvdream
        print("✅ MVDream package is installed")
    except ImportError:
        print("⚠️  MVDream package not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(mvdream_path)], check=True)
    
    return True


def generate_3d_mesh(prompt: str, config: str = "mvdream-sd21.yaml", gpu: int = 0, 
                     test_mode: bool = False, max_steps: int = 5000):
    """
    Generate 3D mesh using MVDream-threestudio.
    
    Args:
        prompt: Text description of the object
        config: Configuration file to use (mvdream-sd21.yaml or mvdream-sd21-shading.yaml)
        gpu: GPU device index
        test_mode: If True, use minimal steps for testing
        max_steps: Maximum training steps
    """
    if not check_mvdream_installation():
        return None
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = prompt[:30].replace(" ", "_").replace("/", "_")
    output_name = f"{timestamp}_{safe_prompt}"
    
    # Change to threestudio directory
    threestudio_dir = project_root / "extern" / "MVDream-threestudio"
    os.chdir(threestudio_dir)
    
    # Build the command - use poetry run to ensure correct environment
    cmd = [
        "poetry", "run", "python",
        "launch.py",
        "--config", f"configs/{config}",
        "--train",
        "--gpu", str(gpu),
        f"system.prompt_processor.prompt='{prompt}'",  # Quote the prompt
        f"trainer.max_steps={max_steps if not test_mode else 100}",
        f"name={output_name}",
        "data.width=256",
        "data.height=256",
        "data.batch_size=4",  # Must be divisible by n_view (4)
    ]
    
    if test_mode:
        cmd.extend([
            "trainer.val_check_interval=50",
            "trainer.check_val_every_n_epoch=1",
            "system.guidance.guidance_scale=50.0",  # Lower guidance for testing
        ])
    
    print(f"\n{'='*60}")
    print(f"Generating 3D mesh for: '{prompt}'")
    print(f"Configuration: {config}")
    print(f"Max steps: {max_steps if not test_mode else 100}")
    print(f"Output name: {output_name}")
    print(f"{'='*60}\n")
    
    # Run the generation
    try:
        print("Starting MVDream-threestudio generation...")
        print(f"Command: {' '.join(cmd)}\n")
        
        # Run with PYTHONWARNINGS=ignore and capture stderr to filter warnings
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        
        result = subprocess.run(
            cmd, 
            check=True, 
            text=True,
            env=env,
            stderr=subprocess.DEVNULL  # Suppress stderr warnings
        )
        
        # Find the output directory
        output_dir = threestudio_dir / "outputs" / output_name
        if output_dir.exists():
            print(f"\n✅ Generation complete! Output saved to: {output_dir}")
            
            # Look for exported meshes
            mesh_files = list(output_dir.glob("**/*.obj")) + list(output_dir.glob("**/*.ply"))
            if mesh_files:
                print(f"Found {len(mesh_files)} mesh file(s):")
                for mesh_file in mesh_files:
                    print(f"  - {mesh_file}")
                
                # Copy the first mesh to our outputs directory
                our_output_dir = project_root / "outputs" / "3d_real" / output_name
                our_output_dir.mkdir(parents=True, exist_ok=True)
                
                for mesh_file in mesh_files:
                    dest = our_output_dir / mesh_file.name
                    shutil.copy2(mesh_file, dest)
                    print(f"  Copied to: {dest}")
                
                return str(our_output_dir)
            else:
                print("⚠️  No mesh files found in output directory")
                return str(output_dir)
        else:
            print(f"⚠️  Output directory not found: {output_dir}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation failed with error: {e}")
        return None
    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
        return None
    finally:
        # Change back to original directory
        os.chdir(project_root)


def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh using MVDream-threestudio")
    parser.add_argument("prompt", nargs="?", help="Text prompt for 3D generation")
    parser.add_argument("--config", default="mvdream-sd21.yaml", 
                       choices=["mvdream-sd21.yaml", "mvdream-sd21-shading.yaml"],
                       help="Configuration file to use")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--test-mode", action="store_true", 
                       help="Run in test mode with minimal steps")
    parser.add_argument("--max-steps", type=int, default=5000,
                       help="Maximum training steps (default: 5000)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick generation with 1000 steps")
    
    args = parser.parse_args()
    
    # Handle prompt
    if not args.prompt:
        print("Enter your 3D object description:")
        args.prompt = input("> ").strip()
        if not args.prompt:
            print("No prompt provided. Exiting.")
            return
    
    # Adjust steps for quick mode
    if args.quick:
        args.max_steps = 1000
    
    # Run generation
    output_path = generate_3d_mesh(
        args.prompt,
        config=args.config,
        gpu=args.gpu,
        test_mode=args.test_mode,
        max_steps=args.max_steps
    )
    
    if output_path:
        print(f"\n✅ Success! 3D mesh saved to: {output_path}")
    else:
        print("\n❌ Generation failed or was interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()