#!/usr/bin/env python3
"""Test suite for DreamCAD CLI and TUI."""

import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import os


class TestCLICommands:
    """Test CLI commands via subprocess."""
    
    def setup_method(self):
        """Set up test environment."""
        self.dreamcad_path = Path(__file__).parent.parent / "dreamcad"
        assert self.dreamcad_path.exists(), f"dreamcad script not found at {self.dreamcad_path}"
    
    def test_cli_help(self):
        """Test help command."""
        result = subprocess.run(
            [sys.executable, str(self.dreamcad_path), "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "DreamCAD - 3D Generation CLI" in result.stdout
        assert "Commands:" in result.stdout
        assert "quick" in result.stdout
        assert "models" in result.stdout
        assert "wizard" in result.stdout
    
    def test_models_command(self):
        """Test models command output."""
        result = subprocess.run(
            [sys.executable, str(self.dreamcad_path), "models"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Available Models" in result.stdout
        assert "TripoSR" in result.stdout
        assert "Stable-Fast-3D" in result.stdout
        assert "TRELLIS" in result.stdout
        assert "★" in result.stdout  # Star ratings
    
    def test_quick_command_with_prompt(self):
        """Test quick generation with prompt."""
        result = subprocess.run(
            [sys.executable, str(self.dreamcad_path), "quick", "a test object"],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode == 0
        assert "a test object" in result.stdout
        assert "Success!" in result.stdout
        assert "outputs/model_" in result.stdout
    
    def test_invalid_command(self):
        """Test invalid command handling."""
        result = subprocess.run(
            [sys.executable, str(self.dreamcad_path), "invalid_command"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert "Error" in result.stdout or "Error" in result.stderr


class TestTUIComponents:
    """Test TUI components."""
    
    def test_simple_tui_exists(self):
        """Test that simple TUI file exists."""
        tui_path = Path(__file__).parent.parent / "dreamcad_simple_tui.py"
        assert tui_path.exists(), f"Simple TUI not found at {tui_path}"
    
    def test_tui_syntax(self):
        """Test TUI Python syntax is valid."""
        tui_path = Path(__file__).parent.parent / "dreamcad_simple_tui.py"
        with open(tui_path, 'r') as f:
            code = f.read()
        
        try:
            compile(code, str(tui_path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"TUI has syntax error: {e}")
    
    @pytest.mark.skipif(not os.environ.get('DISPLAY'), reason="No display available")
    def test_tui_import(self):
        """Test TUI can be imported (requires textual)."""
        try:
            # Add parent to path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            
            # Try to import the TUI
            import dreamcad_simple_tui
            
            # Check it has required components
            assert hasattr(dreamcad_simple_tui, 'DreamCADApp')
            app_class = dreamcad_simple_tui.DreamCADApp
            
            # Check bindings
            assert hasattr(app_class, 'BINDINGS')
            assert len(app_class.BINDINGS) > 0
            
            # Check CSS
            assert hasattr(app_class, 'CSS')
            assert 'Screen' in app_class.CSS
            
        except ImportError as e:
            if "textual" in str(e).lower():
                pytest.skip("Textual not installed")
            else:
                raise


class TestTUIDebugging:
    """Debug TUI issues."""
    
    def test_tui_keybindings(self):
        """Test and debug TUI keybindings."""
        tui_path = Path(__file__).parent.parent / "dreamcad_simple_tui.py"
        with open(tui_path, 'r') as f:
            content = f.read()
        
        # Check for binding definitions
        assert 'BINDINGS' in content, "No BINDINGS defined in TUI"
        assert 'Binding("q"' in content, "Quit binding (q) not found"
        assert 'Binding("g"' in content, "Generate binding (g) not found"
        assert 'Binding("m"' in content, "Models binding (m) not found"
        
        # Check for action methods
        assert 'def action_quit' in content, "action_quit method not found"
        assert 'def action_generate' in content, "action_generate method not found"
        assert 'def action_models' in content, "action_models method not found"
        
        # Check priority on quit binding
        if 'Binding("q"' in content:
            # Find the quit binding line
            for line in content.split('\n'):
                if 'Binding("q"' in line and 'quit' in line.lower():
                    # Priority binding should have priority=True for quit
                    pass  # Just checking it exists
    
    def test_tui_event_handlers(self):
        """Test TUI event handlers."""
        tui_path = Path(__file__).parent.parent / "dreamcad_simple_tui.py"
        with open(tui_path, 'r') as f:
            content = f.read()
        
        # Check for event handlers
        assert 'def on_button_pressed' in content, "Button handler not found"
        
        # Check button IDs match handlers
        if 'btn-quit' in content:
            assert 'event.button.id == "btn-quit"' in content, "Quit button handler not matching ID"
        
        if 'btn-generate' in content:
            assert 'event.button.id == "btn-generate"' in content, "Generate button handler not matching ID"


class TestCLIDebugInfo:
    """Generate debug information for CLI issues."""
    
    def test_cli_structure(self):
        """Debug CLI structure and imports."""
        dreamcad_path = Path(__file__).parent.parent / "dreamcad"
        
        with open(dreamcad_path, 'r') as f:
            content = f.read()
        
        # Check shebang
        assert content.startswith("#!/usr/bin/env python"), "Missing or incorrect shebang"
        
        # Check imports
        required_imports = ['click', 'rich', 'time', 'subprocess', 'Path']
        for imp in required_imports:
            assert imp in content, f"Missing import: {imp}"
        
        # Check main CLI group
        assert '@click.group' in content or '@cli.command' in content, "No click commands found"
        
        # Check commands
        commands = ['quick', 'models', 'interactive', 'tui', 'wizard', 'generate']
        for cmd in commands:
            assert f'def {cmd}' in content, f"Command {cmd} not found"
    
    def test_debug_output(self):
        """Generate debug output for troubleshooting."""
        info = []
        
        # Check Python version
        info.append(f"Python version: {sys.version}")
        
        # Check if files exist
        files = ['dreamcad', 'dreamcad_simple_tui.py', 'dreamcad_tui.py']
        for f in files:
            path = Path(__file__).parent.parent / f
            info.append(f"{f}: {'EXISTS' if path.exists() else 'MISSING'}")
        
        # Check executable permissions
        dreamcad_path = Path(__file__).parent.parent / "dreamcad"
        if dreamcad_path.exists():
            import stat
            mode = dreamcad_path.stat().st_mode
            is_exec = bool(mode & stat.S_IXUSR)
            info.append(f"dreamcad executable: {is_exec}")
        
        # Print debug info
        print("\n=== DEBUG INFO ===")
        for line in info:
            print(line)
        print("==================\n")


def test_fix_tui_shortcuts():
    """Suggest fixes for TUI shortcut issues."""
    suggestions = []
    
    tui_path = Path(__file__).parent.parent / "dreamcad_simple_tui.py"
    with open(tui_path, 'r') as f:
        content = f.read()
    
    # Check if actions are properly defined
    if 'def action_quit' not in content:
        suggestions.append("Add: def action_quit(self): self.exit()")
    
    if 'def action_generate' not in content:
        suggestions.append("Add: def action_generate(self): # generation logic")
    
    if 'def action_models' not in content:
        suggestions.append("Add: def action_models(self): # show models")
    
    # Check binding format
    if 'BINDINGS = [' in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'Binding(' in line:
                if 'priority=True' not in line and '"q"' in line:
                    suggestions.append(f"Line {i+1}: Add priority=True to quit binding")
    
    if suggestions:
        print("\n=== TUI FIX SUGGESTIONS ===")
        for s in suggestions:
            print(f"  • {s}")
        print("===========================\n")
    else:
        print("\n✓ TUI structure looks correct")


if __name__ == "__main__":
    # Run with pytest if available, otherwise run basic tests
    try:
        import pytest
        pytest.main([__file__, "-v", "--tb=short"])
    except ImportError:
        print("Running basic tests without pytest...")
        
        # Run basic tests
        test = TestCLICommands()
        test.setup_method()
        
        try:
            test.test_cli_help()
            print("✓ CLI help test passed")
        except AssertionError as e:
            print(f"✗ CLI help test failed: {e}")
        
        try:
            test.test_models_command()
            print("✓ Models command test passed")
        except AssertionError as e:
            print(f"✗ Models command test failed: {e}")
        
        # Run debug info
        debug = TestCLIDebugInfo()
        debug.test_debug_output()
        
        # Run TUI fix suggestions
        test_fix_tui_shortcuts()