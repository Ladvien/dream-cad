#!/usr/bin/env python3
"""
DreamCAD - Simple launcher for the production CLI.
"""

import sys
from pathlib import Path

# Add the project directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcad_cli import main

if __name__ == "__main__":
    main()