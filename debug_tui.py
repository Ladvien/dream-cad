#!/usr/bin/env python3
"""Debug TUI keyboard and interaction issues."""

from pathlib import Path
import sys

def analyze_tui():
    """Analyze TUI for common issues."""
    issues = []
    fixes = []
    
    tui_path = Path(__file__).parent / "dreamcad_simple_tui.py"
    with open(tui_path, 'r') as f:
        lines = f.readlines()
    
    # Check 1: Bindings format
    print("=== CHECKING TUI BINDINGS ===")
    for i, line in enumerate(lines, 1):
        if 'Binding(' in line:
            print(f"Line {i}: {line.strip()}")
            
            # Check if quit has priority
            if '"q"' in line and 'priority=True' not in line:
                issues.append(f"Line {i}: Quit binding should have priority=True")
                fixes.append(f"Change line {i} to: Binding('q', 'quit', 'Quit', priority=True),")
    
    # Check 2: Action methods
    print("\n=== CHECKING ACTION METHODS ===")
    actions = ['quit', 'generate', 'models', 'help']
    for action in actions:
        method_name = f"def action_{action}"
        found = any(method_name in line for line in lines)
        print(f"action_{action}: {'✓ Found' if found else '✗ Missing'}")
        if not found:
            issues.append(f"Missing method: action_{action}")
            fixes.append(f"Add method: def action_{action}(self): ...")
    
    # Check 3: Button event handlers
    print("\n=== CHECKING BUTTON HANDLERS ===")
    has_button_handler = any('def on_button_pressed' in line for line in lines)
    print(f"on_button_pressed: {'✓ Found' if has_button_handler else '✗ Missing'}")
    
    # Check 4: Focus issues
    print("\n=== CHECKING FOCUS HANDLING ===")
    has_focus = any('focus()' in line for line in lines)
    print(f"Focus calls: {'✓ Found' if has_focus else '⚠ Not found (may affect keyboard input)'}")
    
    # Check 5: Widget IDs
    print("\n=== CHECKING WIDGET IDS ===")
    widgets = ['btn-quit', 'btn-generate', 'btn-models', 'content']
    for widget in widgets:
        found = any(f'"{widget}"' in line or f"'{widget}'" in line for line in lines)
        print(f"Widget ID '{widget}': {'✓ Found' if found else '✗ Missing'}")
    
    # Report issues
    if issues:
        print("\n=== ISSUES FOUND ===")
        for issue in issues:
            print(f"  • {issue}")
        
        print("\n=== SUGGESTED FIXES ===")
        for fix in fixes:
            print(f"  • {fix}")
    else:
        print("\n✓ No obvious issues found")
    
    return issues, fixes


def test_keyboard_simulation():
    """Test keyboard input simulation."""
    print("\n=== TESTING KEYBOARD INPUT ===")
    
    # Create a test script to check if Textual is handling keys
    test_code = '''
from textual.app import App
from textual.binding import Binding

class TestApp(App):
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("t", "test", "Test"),
    ]
    
    def action_quit(self):
        print("QUIT ACTION TRIGGERED")
        self.exit()
    
    def action_test(self):
        print("TEST ACTION TRIGGERED")

if __name__ == "__main__":
    app = TestApp()
    # app.run()  # Uncomment to actually run
    print("Test app created successfully")
    print(f"Bindings: {[b.key for b in app.BINDINGS]}")
'''
    
    try:
        exec(test_code)
        print("✓ Basic Textual app structure is valid")
    except Exception as e:
        print(f"✗ Error in basic structure: {e}")


def suggest_improved_tui():
    """Suggest an improved TUI with better keyboard handling."""
    print("\n=== IMPROVED TUI SUGGESTION ===")
    
    improved_code = '''
# Key improvements for TUI keyboard handling:

1. Add priority to quit binding:
   Binding("q", "quit", "Quit", priority=True, show=True)

2. Ensure all actions are defined:
   def action_quit(self):
       """Quit the application."""
       self.exit()
   
   def action_generate(self):
       """Show generation view."""
       self.current_view = "generate"
       self.refresh()
   
   def action_models(self):
       """Show models view."""
       self.current_view = "models"
       self.refresh()

3. Add keyboard event handler:
   def on_key(self, event):
       """Handle key presses."""
       if event.key == "escape":
           self.action_quit()

4. Ensure focus is set on mount:
   def on_mount(self):
       """Set initial focus."""
       self.set_focus(None)  # Let Textual handle focus

5. Use watch methods for reactive updates:
   def watch_current_view(self, old_view, new_view):
       """React to view changes."""
       self.query_one("#content").update(self.get_view(new_view))
'''
    
    print(improved_code)


if __name__ == "__main__":
    print("DreamCAD TUI Debugger")
    print("=" * 50)
    
    # Analyze current TUI
    issues, fixes = analyze_tui()
    
    # Test keyboard simulation
    test_keyboard_simulation()
    
    # Suggest improvements
    if issues:
        suggest_improved_tui()
    
    print("\n" + "=" * 50)
    print("Debug complete. Run './dreamcad tui' to test.")