from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Label, Button, Static, Switch
from textual.containers import Horizontal, Vertical, Container

class SettingsScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Label("⚙️ Settings", id="title")
        
        with Container():
            with Vertical():
                with Horizontal():
                    yield Label("Auto-download models:")
                    yield Switch(value=True, id="auto-download")
                    
                with Horizontal():
                    yield Label("Demo mode fallback:")
                    yield Switch(value=True, id="demo-fallback")
                    
                with Horizontal():
                    yield Label("Notification sounds:")
                    yield Switch(value=False, id="sounds")
                    
            with Horizontal():
                yield Button("💾 Save", variant="primary", id="btn-save")
                yield Button("🔙 Back", id="btn-back")
                
    def on_button_pressed(self, event):
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-save":
            self.app.config.save()
            self.app.notify("Settings saved", severity="information")
            self.app.pop_screen()