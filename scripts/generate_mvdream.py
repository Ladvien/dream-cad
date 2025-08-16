#!/usr/bin/env python3
"""Memory-efficient MVDream generation."""

import sys
import os
from pathlib import Path

# Add MVDream to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extern" / "MVDream"))

import torch
import numpy as np
from PIL import Image
import argparse
from datetime import datetime
import gc

def main():
    parser = argparse.ArgumentParser(description="Generate with MVDream (memory efficient)")
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument("--steps", type=int, default=10, help="Diffusion steps")
    parser.add_argument("--size", type=int, default=256, help="Image size")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 MVDream Memory-Efficient Generation")
    print("=" * 80)
    print(f"Prompt: '{args.prompt}'")
    print(f"Steps: {args.steps}, Size: {args.size}x{args.size}")
    print()
    
    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: CUDA is required")
        return 1
    
    print(f"✓ Using: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        from mvdream.model_zoo import build_model
        from mvdream.camera_utils import get_camera
        from mvdream.ldm.models.diffusion.ddim import DDIMSampler
        
        # Load model with fp16 for memory efficiency
        print("\n⏳ Loading MVDream model (fp16 mode)...")
        model_path = Path("models/sd-v2.1-base-4view.pt")
        if not model_path.exists():
            print(f"ERROR: Model not found at {model_path}")
            return 1
        
        model = build_model("sd-v2.1-base-4view", ckpt_path=str(model_path))
        model.device = device
        model.to(device)
        model.eval()
        print("✓ Model loaded!")
        
        # Use smaller batch for memory
        num_frames = 4  # Still 4 views
        batch_size = 4  # Process all at once for consistency
        
        # Get 4-view camera
        print("\n📷 Setting up 4-view camera...")
        camera = get_camera(
            num_frames=num_frames,
            elevation=0,
            azimuth_start=0,
            azimuth_span=360,
            blender_coord=True
        )
        camera = camera.to(device)
        
        # Prepare conditioning
        print("\n📝 Encoding prompt...")
        prompt = args.prompt
        suffix = ", 3d asset"  # MVDream works better with this suffix
        full_prompt = prompt + suffix
        
        c = model.get_learned_conditioning([full_prompt]).to(device)
        uc = model.get_learned_conditioning([""]).to(device)
        
        # Format for MVDream
        cond = {
            "context": c.repeat(batch_size, 1, 1),
            "camera": camera,
            "num_frames": num_frames
        }
        un_cond = {
            "context": uc.repeat(batch_size, 1, 1),
            "camera": camera,
            "num_frames": num_frames
        }
        print("✓ Prompt encoded")
        
        # Sample with fp16
        print(f"\n🎨 Generating latents ({args.steps} steps)...")
        sampler = DDIMSampler(model)
        shape = [4, args.size // 8, args.size // 8]  # latent shape
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                samples, _ = sampler.sample(
                    S=args.steps,
                    conditioning=cond,
                    batch_size=batch_size,
                    shape=shape,
                    verbose=True,
                    unconditional_guidance_scale=7.5,
                    unconditional_conditioning=un_cond,
                    eta=0.0
                )
        
        print("\n✓ Latents generated!")
        
        # Clear some memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Decode one at a time to save memory
        print("\n🖼️ Decoding images (one at a time for memory efficiency)...")
        images = []
        
        for i in range(batch_size):
            print(f"  Decoding view {i+1}/{batch_size}...")
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                # Decode single latent
                single_latent = samples[i:i+1]
                x_sample = model.decode_first_stage(single_latent)
                x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
                
                # Convert to numpy immediately to free GPU memory
                img = x_sample[0].detach().cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                images.append(Image.fromarray(img))
                
                # Clear GPU memory
                del x_sample
                torch.cuda.empty_cache()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_clean = args.prompt[:20].replace(" ", "_").replace("/", "_")
        output_dir = Path("outputs") / "mvdream_real" / f"{timestamp}_{prompt_clean}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create grid
        print("\n🎯 Creating output grid...")
        grid_size = args.size
        grid = Image.new('RGB', (grid_size * 2, grid_size * 2))
        
        view_names = ["Front", "Right", "Back", "Left"]
        for i, (img, name) in enumerate(zip(images, view_names)):
            # Resize if needed and place in grid
            if img.size != (grid_size, grid_size):
                img = img.resize((grid_size, grid_size), Image.LANCZOS)
            x = (i % 2) * grid_size
            y = (i // 2) * grid_size
            grid.paste(img, (x, y))
            
            # Save individual view
            view_path = output_dir / f"{i:02d}_{name.lower()}.png"
            img.save(view_path)
        
        grid_path = output_dir / "grid.png"
        grid.save(grid_path)
        
        print("\n" + "=" * 80)
        print("🎉 SUCCESS! REAL MVDream Generation Complete!")
        print("=" * 80)
        print(f"✓ Generated 4-view images of: '{args.prompt}'")
        print(f"✓ Output directory: {output_dir}")
        print(f"✓ Grid image: {grid_path}")
        print("\n📊 Details:")
        print(f"  - Model: sd-v2.1-base-4view")
        print(f"  - Resolution: {args.size}x{args.size} per view")
        print(f"  - Views: Front, Right, Back, Left")
        print(f"  - Steps: {args.steps}")
        print(f"  - Memory-efficient decoding: Yes")
        print("\n🚀 This is REAL generation using the actual MVDream model!")
        print("=" * 80)
        
        # Final memory stats
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n📈 Final GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())