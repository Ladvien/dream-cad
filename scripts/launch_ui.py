import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dream_cad.ui import launch_ui
from dream_cad.models.registry import ModelRegistry
def main():
    parser = argparse.ArgumentParser(
        description="Launch Multi-Model 3D Generation UI"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0 for network access)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the server to (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link"
    )
    parser.add_argument(
        "--registry-path",
        type=str,
        help="Path to model registry file"
    )
    args = parser.parse_args()
    registry = ModelRegistry()
    if args.registry_path:
        registry_path = Path(args.registry_path)
        if registry_path.exists():
            registry.load(registry_path)
            print(f"✅ Loaded registry from {registry_path}")
    try:
        launch_ui(
            host=args.host,
            port=args.port,
            share=args.share,
            registry=registry
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down UI...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        sys.exit(1)
if __name__ == "__main__":
    main()