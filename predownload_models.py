#!/usr/bin/env python3
"""
Pre-download models so the TUI doesn't hang.
Run this BEFORE using the TUI.
"""

print("=" * 60)
print("PRE-DOWNLOAD MODELS FOR DREAMCAD")
print("=" * 60)
print("\nThis will download models BEFORE you use the TUI")
print("so you don't have to stare at a frozen screen.\n")

import sys

model = input("Which model? (1=TripoSR, 2=SF3D, 3=TRELLIS, 4=Hunyuan): ").strip()

models = {
    "1": ("stabilityai/TripoSR", "TripoSR", "1.5GB"),
    "2": ("stabilityai/stable-fast-3d", "Stable-Fast-3D", "2.5GB"),
    "3": ("microsoft/TRELLIS-image-large", "TRELLIS", "4.5GB"),
    "4": ("tencent/Hunyuan3D-2mini", "Hunyuan3D", "4.5GB"),
}

if model not in models:
    print("Invalid choice!")
    sys.exit(1)

repo_id, name, size = models[model]

print(f"\n📥 Downloading {name} ({size})")
print("⏳ This will take 5-15 minutes...")
print("💡 At least now you can see SOMETHING is happening!\n")

try:
    from huggingface_hub import snapshot_download
    import time
    
    start = time.time()
    print(f"Starting download of {repo_id}...")
    print("(No progress bar because HuggingFace doesn't support it)")
    print("But at least this script will tell you when it's done!\n")
    
    # This is the part that hangs with no feedback
    snapshot_download(repo_id)
    
    elapsed = time.time() - start
    print(f"\n✅ SUCCESS! {name} downloaded in {elapsed/60:.1f} minutes")
    print(f"📦 Model is now cached and ready to use in the TUI")
    
except KeyboardInterrupt:
    print("\n\n❌ Download cancelled")
except Exception as e:
    print(f"\n❌ Download failed: {e}")

print("\nYou can now run the TUI and it won't hang on download!")
print("python dreamcad_tui_new.py")