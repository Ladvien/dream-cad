from pathlib import Path
import sys
def analyze_tui():
    issues = []
    fixes = []
    tui_path = Path(__file__).parent / "dreamcad_simple_tui.py"
    with open(tui_path, 'r') as f:
        lines = f.readlines()
    print("=== CHECKING TUI BINDINGS ===")
    for i, line in enumerate(lines, 1):
        if 'Binding(' in line:
            print(f"Line {i}: {line.strip()}")
            if '"q"' in line and 'priority=True' not in line:
                issues.append(f"Line {i}: Quit binding should have priority=True")
                fixes.append(f"Change line {i} to: Binding('q', 'quit', 'Quit', priority=True),")
    print("\n=== CHECKING ACTION METHODS ===")
    actions = ['quit', 'generate', 'models', 'help']
    for action in actions:
        method_name = f"def action_{action}"
        found = any(method_name in line for line in lines)
        print(f"action_{action}: {'✓ Found' if found else '✗ Missing'}")
        if not found:
            issues.append(f"Missing method: action_{action}")
            fixes.append(f"Add method: def action_{action}(self): ...")
    print("\n=== CHECKING BUTTON HANDLERS ===")
    has_button_handler = any('def on_button_pressed' in line for line in lines)
    print(f"on_button_pressed: {'✓ Found' if has_button_handler else '✗ Missing'}")
    print("\n=== CHECKING FOCUS HANDLING ===")
    has_focus = any('focus()' in line for line in lines)
    print(f"Focus calls: {'✓ Found' if has_focus else '⚠ Not found (may affect keyboard input)'}")
    print("\n=== CHECKING WIDGET IDS ===")
    widgets = ['btn-quit', 'btn-generate', 'btn-models', 'content']
    for widget in widgets:
        found = any(f'"{widget}"' in line or f"'{widget}'" in line for line in lines)
        print(f"Widget ID '{widget}': {'✓ Found' if found else '✗ Missing'}")
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
    print("\n=== TESTING KEYBOARD INPUT ===")
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
    print("Test app created successfully")
    print(f"Bindings: {[b.key for b in app.BINDINGS]}")
1. Add priority to quit binding:
   Binding("q", "quit", "Quit", priority=True, show=True)
2. Ensure all actions are defined:
   def action_quit(self):
       self.exit()
   def action_generate(self):
       self.current_view = "generate"
       self.refresh()
   def action_models(self):
       self.current_view = "models"
       self.refresh()
3. Add keyboard event handler:
   def on_key(self, event):
       if event.key == "escape":
           self.action_quit()
4. Ensure focus is set on mount:
   def on_mount(self):
       self.set_focus(None)
5. Use watch methods for reactive updates:
   def watch_current_view(self, old_view, new_view):
       self.query_one("#content").update(self.get_view(new_view))