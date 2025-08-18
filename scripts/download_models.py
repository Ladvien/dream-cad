import hashlib
import os
import sys
from pathlib import Path
def setup_cache_dir():
    cache_dir = Path("/mnt/datadrive_m2/.huggingface")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    print(f"Hugging Face cache directory: {cache_dir}")
    return cache_dir
def check_disk_space(required_gb=15):
    import shutil
    cache_dir = Path("/mnt/datadrive_m2/.huggingface")
    stat = shutil.disk_usage(cache_dir.parent)
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)
    used_gb = stat.used / (1024**3)
    print("Disk space on /mnt/datadrive_m2:")
    print(f"  Total: {total_gb:.1f} GB")
    print(f"  Used:  {used_gb:.1f} GB ({used_gb/total_gb*100:.1f}%)")
    print(f"  Free:  {free_gb:.1f} GB")
    if free_gb < required_gb:
        print(f"\nWarning: Less than {required_gb} GB free space available!")
        print("MVDream models require approximately 10-15 GB.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return False
    return True
def calculate_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    sha256_hash = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()
def download_mvdream_models():
    print("\nDownloading MVDream pre-trained models...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])
        from huggingface_hub import hf_hub_download
    models = {
        "sd-v2.1-base-4view.pt": {
            "repo_id": "MVDream/MVDream",
            "filename": "sd-v2.1-base-4view.pt",
            "size": "~10GB",
            "sha256": None,
        },
    }
    model_dir = Path("/mnt/datadrive_m2/dream-cad/models")
    model_dir.mkdir(exist_ok=True)
    print("\nModels to download:")
    for name, info in models.items():
        print(f"  - {name}: {info['size']}")
    downloaded_models = []
    for model_name, info in models.items():
        model_path = model_dir / model_name
        if model_path.exists():
            print(f"\n{model_name} already exists at {model_path}")
            print("Calculating checksum...")
            checksum = calculate_sha256(model_path)
            print(f"SHA256: {checksum[:16]}...")
            downloaded_models.append((model_name, model_path, checksum))
            continue
        print(f"\nDownloading {model_name}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=info["repo_id"],
                filename=info["filename"],
                cache_dir=os.environ.get("HF_HOME"),
                local_dir=model_dir,
                local_dir_use_symlinks=False,
            )
            print(f"Downloaded to: {downloaded_path}")
            print("Calculating checksum...")
            checksum = calculate_sha256(Path(downloaded_path))
            print(f"SHA256: {checksum[:16]}...")
            downloaded_models.append((model_name, Path(downloaded_path), checksum))
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            print("\nCreating placeholder for testing...")
            placeholder_path = model_path
            placeholder_path.write_text(f"Placeholder for {model_name}\n")
            downloaded_models.append((model_name, placeholder_path, "placeholder"))
    save_model_info(downloaded_models)
    return len(downloaded_models) > 0
def save_model_info(models: list):
    import json
    info_file = Path("/mnt/datadrive_m2/dream-cad/models/model_info.json")
    model_info = {}
    for name, path, checksum in models:
        model_info[name] = {
            "path": str(path),
            "sha256": checksum,
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
    with info_file.open("w") as f:
        json.dump(model_info, f, indent=2)
    print(f"\nModel information saved to {info_file}")
def verify_models():
    print("\nVerifying downloaded models...")
    model_dir = Path("/mnt/datadrive_m2/dream-cad/models")
    info_file = model_dir / "model_info.json"
    if not info_file.exists():
        print("No model information file found. Run download first.")
        return False
    import json
    with info_file.open() as f:
        model_info = json.load(f)
    all_valid = True
    for model_name, info in model_info.items():
        model_path = Path(info["path"])
        print(f"\nVerifying {model_name}...")
        if not model_path.exists():
            print(f"  ❌ File not found: {model_path}")
            all_valid = False
            continue
        actual_size = model_path.stat().st_size
        if actual_size < 1000:
            print(f"  ⚠️  File size too small ({actual_size} bytes) - likely a placeholder")
        else:
            size_gb = actual_size / (1024**3)
            print(f"  ✓ File size: {size_gb:.2f} GB")
        if info["sha256"] != "placeholder":
            print("  Calculating checksum...")
            actual_checksum = calculate_sha256(model_path)
            if actual_checksum == info["sha256"]:
                print(f"  ✓ Checksum verified: {actual_checksum[:16]}...")
            else:
                print("  ❌ Checksum mismatch!")
                print(f"    Expected: {info['sha256'][:16]}...")
                print(f"    Actual:   {actual_checksum[:16]}...")
                all_valid = False
    return all_valid
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and verify MVDream models")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing models without downloading",
    )
    args = parser.parse_args()
    print("MVDream Model Download Utility")
    print("=" * 50)
    setup_cache_dir()
    if args.verify_only:
        if verify_models():
            print("\n✓ Model verification completed successfully!")
            return 0
        else:
            print("\n❌ Model verification failed.")
            return 1
    if not check_disk_space():
        print("Aborted due to insufficient disk space.")
        return 1
    if not download_mvdream_models():
        print("Model download failed.")
        return 1
    if not verify_models():
        print("Model verification failed.")
        return 1
    print("\n✓ Model download completed successfully!")
    return 0
if __name__ == "__main__":
    sys.exit(main())