from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Label, Button, DataTable
from textual.containers import Horizontal, Vertical, Container

class QueueScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Label("📋 Generation Queue", id="title")
        
        with Container():
            yield DataTable(id="queue-table")
            
            with Horizontal(id="queue-actions"):
                yield Button("⏸️ Pause", id="btn-pause")
                yield Button("▶️ Resume", id="btn-resume")
                yield Button("🗑️ Clear", id="btn-clear")
                yield Button("🔙 Back", id="btn-back")
                
    def on_mount(self):
        table = self.query_one("#queue-table", DataTable)
        table.add_column("ID", width=8)
        table.add_column("Prompt", width=30)
        table.add_column("Model", width=15)
        table.add_column("Status", width=12)
        table.add_column("Progress", width=10)
        
    def on_button_pressed(self, event):
        if event.button.id == "btn-back":
            self.app.pop_screen()