#!/usr/bin/env python3
"""
DreamCAD Production TUI - Feature Demo

This script demonstrates the key features of the production-ready TUI:
1. Model detection and status
2. Non-blocking generation
3. Demo mode fallback
4. Real-time monitoring
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("""
╔══════════════════════════════════════════════════════════════╗
║           🎨 DreamCAD Production TUI Features                ║
╚══════════════════════════════════════════════════════════════╝

Key Features Implemented:

1. 🔍 MODEL DETECTION
   - Automatically scans for installed models
   - Checks VRAM compatibility
   - Shows download status and size

2. ⚡ NON-BLOCKING GENERATION
   - Runs in background threads
   - Real-time progress updates
   - Cancellable operations

3. 🎭 SMART FALLBACK
   - Demo mode when models unavailable
   - Generates simple procedural meshes
   - Clear indication of mode

4. 📊 SYSTEM MONITORING
   - Real-time CPU/RAM usage
   - GPU/VRAM tracking (if available)
   - Model cache status

5. 🎯 GENERATION WIZARD
   - Prompt suggestions
   - Model recommendations
   - Output format selection

6. 💾 PERSISTENT CONFIG
   - Saves user preferences
   - Remembers preferred models
   - Maintains generation history

To run the TUI:
  poetry run python dreamcad_production.py

Keyboard Shortcuts:
  Ctrl+Q - Quit
  Ctrl+D - Dashboard
  Ctrl+G - Generate
  Ctrl+U - Queue
  Ctrl+S - Settings

The TUI will automatically:
- Detect available models
- Fall back to demo mode if needed
- Track all generations
- Save preferences on exit
""")

# Quick check of what's available
from dreamcad_tui.core.model_detector import ModelDetector
from dreamcad_tui.core.config import ConfigManager

config = ConfigManager()
detector = ModelDetector(config)

print("\n📍 Current System Status:")
print(f"   VRAM Available: {detector._detect_vram():.1f} GB")
print(f"   Cache Directory: {detector.cache_dir}")

from dreamcad_tui.core.model_detector import KNOWN_MODELS

print("\n🎯 Model Compatibility:")
for model_id, info in KNOWN_MODELS.items():
    vram_ok = detector.vram_gb >= info["min_vram_gb"]
    cached = detector._check_model_cached(info["repo"])
    
    status = "✓ Ready" if cached else "⬇ Need Download" if vram_ok else "✗ Insufficient VRAM"
    print(f"   {info['name']:15} - {status}")
    
print("\n✨ Ready to generate 3D models!")
print("Run: poetry run python dreamcad_production.py")