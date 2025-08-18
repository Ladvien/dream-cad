import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import os
class TestCLICommands:
    def setup_method(self):
        self.dreamcad_path = Path(__file__).parent.parent / "dreamcad"
        assert self.dreamcad_path.exists(), f"dreamcad script not found at {self.dreamcad_path}"
    def test_cli_help(self):
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
        assert "★" in result.stdout
    def test_quick_command_with_prompt(self):
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
        result = subprocess.run(
            [sys.executable, str(self.dreamcad_path), "invalid_command"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert "Error" in result.stdout or "Error" in result.stderr
class TestTUIComponents:
    def test_simple_tui_exists(self):
        tui_path = Path(__file__).parent.parent / "dreamcad_simple_tui.py"
        assert tui_path.exists(), f"Simple TUI not found at {tui_path}"
    def test_tui_syntax(self):
        tui_path = Path(__file__).parent.parent / "dreamcad_simple_tui.py"
        with open(tui_path, 'r') as f:
            code = f.read()
        try:
            compile(code, str(tui_path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"TUI has syntax error: {e}")
    @pytest.mark.skipif(not os.environ.get('DISPLAY'), reason="No display available")
    def test_tui_import(self):
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            import dreamcad_simple_tui
            assert hasattr(dreamcad_simple_tui, 'DreamCADApp')
            app_class = dreamcad_simple_tui.DreamCADApp
            assert hasattr(app_class, 'BINDINGS')
            assert len(app_class.BINDINGS) > 0
            assert hasattr(app_class, 'CSS')
            assert 'Screen' in app_class.CSS
        except ImportError as e:
            if "textual" in str(e).lower():
                pytest.skip("Textual not installed")
            else:
                raise
class TestTUIDebugging:
    def test_tui_keybindings(self):
        tui_path = Path(__file__).parent.parent / "dreamcad_simple_tui.py"
        with open(tui_path, 'r') as f:
            content = f.read()
        assert 'BINDINGS' in content, "No BINDINGS defined in TUI"
        assert 'Binding("q"' in content, "Quit binding (q) not found"
        assert 'Binding("g"' in content, "Generate binding (g) not found"
        assert 'Binding("m"' in content, "Models binding (m) not found"
        assert 'def action_quit' in content, "action_quit method not found"
        assert 'def action_generate' in content, "action_generate method not found"
        assert 'def action_models' in content, "action_models method not found"
        if 'Binding("q"' in content:
            for line in content.split('\n'):
                if 'Binding("q"' in line and 'quit' in line.lower():
                    pass
    def test_tui_event_handlers(self):
        tui_path = Path(__file__).parent.parent / "dreamcad_simple_tui.py"
        with open(tui_path, 'r') as f:
            content = f.read()
        assert 'def on_button_pressed' in content, "Button handler not found"
        if 'btn-quit' in content:
            assert 'event.button.id == "btn-quit"' in content, "Quit button handler not matching ID"
        if 'btn-generate' in content:
            assert 'event.button.id == "btn-generate"' in content, "Generate button handler not matching ID"
class TestCLIDebugInfo:
    def test_cli_structure(self):
        dreamcad_path = Path(__file__).parent.parent / "dreamcad"
        with open(dreamcad_path, 'r') as f:
            content = f.read()
        assert content.startswith("#!/usr/bin/env python"), "Missing or incorrect shebang"
        required_imports = ['click', 'rich', 'time', 'subprocess', 'Path']
        for imp in required_imports:
            assert imp in content, f"Missing import: {imp}"
        assert '@click.group' in content or '@cli.command' in content, "No click commands found"
        commands = ['quick', 'models', 'interactive', 'tui', 'wizard', 'generate']
        for cmd in commands:
            assert f'def {cmd}' in content, f"Command {cmd} not found"
    def test_debug_output(self):
        info = []
        info.append(f"Python version: {sys.version}")
        files = ['dreamcad', 'dreamcad_simple_tui.py', 'dreamcad_tui.py']
        for f in files:
            path = Path(__file__).parent.parent / f
            info.append(f"{f}: {'EXISTS' if path.exists() else 'MISSING'}")
        dreamcad_path = Path(__file__).parent.parent / "dreamcad"
        if dreamcad_path.exists():
            import stat
            mode = dreamcad_path.stat().st_mode
            is_exec = bool(mode & stat.S_IXUSR)
            info.append(f"dreamcad executable: {is_exec}")
        print("\n=== DEBUG INFO ===")
        for line in info:
            print(line)
        print("==================\n")
def test_fix_tui_shortcuts():
    suggestions = []
    tui_path = Path(__file__).parent.parent / "dreamcad_simple_tui.py"
    with open(tui_path, 'r') as f:
        content = f.read()
    if 'def action_quit' not in content:
        suggestions.append("Add: def action_quit(self): self.exit()")
    if 'def action_generate' not in content:
        suggestions.append("Add: def action_generate(self): # generation logic")
    if 'def action_models' not in content:
        suggestions.append("Add: def action_models(self): # show models")
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
    try:
        import pytest
        pytest.main([__file__, "-v", "--tb=short"])
    except ImportError:
        print("Running basic tests without pytest...")
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
        debug = TestCLIDebugInfo()
        debug.test_debug_output()
        test_fix_tui_shortcuts()