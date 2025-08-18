#!/usr/bin/env python3

import sys
from pathlib import Path

# CRITICAL: Force reload of ALL dreamcad_tui modules to avoid cache issues
for key in list(sys.modules.keys()):
    if 'dreamcad_tui' in key:
        del sys.modules[key]

sys.path.insert(0, str(Path(__file__).parent))

from dreamcad_tui.core.app import DreamCADApp

def main():
    print("\n" + "="*60)
    print("🎨 DreamCAD Production - 3D Generation Studio")
    print("="*60)
    print("\n✨ Features:")
    print("  • Real model integration with actual 3D generation")
    print("  • Non-blocking UI with progress tracking")
    print("  • Automatic model detection and fallback")
    print("  • System resource monitoring")
    print("  • Demo mode when models unavailable")
    print("\n🚀 Starting production TUI...\n")
    
    try:
        app = DreamCADApp()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nCheck logs at ~/.dreamcad/logs/")
        sys.exit(1)

if __name__ == "__main__":
    main()