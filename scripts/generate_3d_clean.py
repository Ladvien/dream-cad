#!/usr/bin/env python3
"""
Clean 3D generation wrapper that suppresses warnings and shows only progress.
"""
import argparse
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ["HF_HOME"] = "/mnt/datadrive_m2/.huggingface"
os.environ["TORCH_HOME"] = "/mnt/datadrive_m2/.torch"
os.environ["CUDA_HOME"] = "/opt/cuda"


def run_3d_generation(prompt: str, quick: bool = False, test_mode: bool = False):
    """Run 3D generation with clean output."""
    
    # Create timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = prompt[:30].replace(" ", "_").replace("/", "_")
    output_name = f"{timestamp}_{safe_prompt}"
    
    # Determine steps
    if test_mode:
        max_steps = 100
        mode = "test"
    elif quick:
        max_steps = 1000
        mode = "quick"
    else:
        max_steps = 5000
        mode = "full"
    
    print(f"\n{'='*60}")
    print(f"🚀 MVDream 3D Generation")
    print(f"{'='*60}")
    print(f"📝 Prompt: '{prompt}'")
    print(f"⚙️  Mode: {mode} ({max_steps} steps)")
    print(f"📁 Output: {output_name}")
    print(f"{'='*60}\n")
    
    # Change to threestudio directory
    threestudio_dir = project_root / "extern" / "MVDream-threestudio"
    os.chdir(threestudio_dir)
    
    # Build the command
    cmd = [
        "poetry", "run", "python", "-W", "ignore",
        "launch.py",
        "--config", "configs/mvdream-sd21.yaml",
        "--train",
        "--gpu", "0",
        f"system.prompt_processor.prompt='{prompt}'",
        f"trainer.max_steps={max_steps}",
        f"name={output_name}",
        "data.width=256",
        "data.height=256",
        "data.batch_size=4",
        "trainer.enable_progress_bar=true",
    ]
    
    if test_mode:
        cmd.extend([
            "trainer.val_check_interval=50",
            "system.guidance.guidance_scale=50.0",
        ])
    
    # Run the generation
    try:
        print("⏳ Initializing MVDream model...")
        print("   This may take a few moments on first run.\n")
        
        # Create environment with all warnings suppressed
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        
        # Run the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        # Filter and display output
        for line in process.stdout:
            # Only show important progress lines
            if any(keyword in line for keyword in [
                "Epoch", "Step", "INFO", "Loading", "Loaded", 
                "Generation", "Saving", "Export", "✓", "✅", "%"
            ]):
                # Clean up the line
                line = line.strip()
                if line and not any(skip in line for skip in [
                    "UserWarning", "FutureWarning", "DeprecationWarning",
                    "pkg_resources", "autocast", "import"
                ]):
                    print(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            # Find output files
            output_dir = threestudio_dir / "outputs" / output_name
            
            print(f"\n{'='*60}")
            print(f"✅ Generation Complete!")
            print(f"{'='*60}")
            
            if output_dir.exists():
                # Look for mesh files
                mesh_files = list(output_dir.glob("**/*.obj")) + \
                           list(output_dir.glob("**/*.ply")) + \
                           list(output_dir.glob("**/*.glb"))
                
                if mesh_files:
                    print(f"📦 Generated {len(mesh_files)} mesh file(s):")
                    for mesh in mesh_files[:5]:  # Show max 5 files
                        print(f"   • {mesh.name}")
                
                print(f"\n📁 Output directory: {output_dir}")
            else:
                print(f"⚠️  Output directory not found: {output_dir}")
            
            return True
        else:
            print(f"\n❌ Generation failed with code {return_code}")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    finally:
        os.chdir(project_root)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D mesh with clean output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "a ceramic coffee mug"
  %(prog)s "a wooden chair" --quick
  %(prog)s "test object" --test
        """
    )
    
    parser.add_argument("prompt", help="Text description of the 3D object")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick generation (1000 steps, ~15 min)")
    parser.add_argument("--test", action="store_true",
                       help="Test mode (100 steps, ~2 min)")
    
    args = parser.parse_args()
    
    # Run generation
    success = run_3d_generation(args.prompt, args.quick, args.test)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()