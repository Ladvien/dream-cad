from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Label, Button, Static
from textual.containers import Horizontal, Container
from rich.text import Text

HELP_TEXT = """
🎨 DreamCAD Help

Keyboard Shortcuts:
  • Ctrl+Q - Quit application
  • Ctrl+D - Dashboard
  • Ctrl+G - Generate new model
  • Ctrl+U - View queue
  • Ctrl+S - Settings

Models:
  • TripoSR - Ultra fast (0.5s), draft quality
  • Stable-Fast-3D - Fast (3-5s), good quality
  • TRELLIS - Slower (30-60s), excellent quality
  • Hunyuan3D - Production quality with PBR textures

Tips:
  • Models download automatically on first use
  • Each model is 1-5GB (one-time download)
  • Use demo mode to test without models
  • Check dashboard for system resource usage
"""

class HelpScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Label("❓ Help", id="title")
        
        with Container():
            yield Static(Text(HELP_TEXT), id="help-content")
            
            with Horizontal():
                yield Button("🔙 Back", id="btn-back")
                
    def on_button_pressed(self, event):
        if event.button.id == "btn-back":
            self.app.pop_screen()