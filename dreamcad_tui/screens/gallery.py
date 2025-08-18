from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Label, Button, Static
from textual.containers import Horizontal, Container

class GalleryScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Label("📁 Model Gallery", id="title")
        
        with Container():
            yield Static("Gallery view coming soon...", id="gallery-content")
            
            with Horizontal():
                yield Button("🔙 Back", id="btn-back")
                
    def on_button_pressed(self, event):
        if event.button.id == "btn-back":
            self.app.pop_screen()