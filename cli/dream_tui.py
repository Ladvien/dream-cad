from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Button, Static, Input, Label, ListView, ListItem, ProgressBar, DataTable
from textual.widget import Widget
from textual.reactive import reactive
from textual import events
from textual.screen import Screen
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich import box
import random
import time
from datetime import datetime
from pathlib import Path
MODELS = {
    "⚡ TripoSR": {"speed": "0.5s", "quality": 3, "vram": "4-6GB"},
    "🎮 Stable-Fast-3D": {"speed": "3s", "quality": 4, "vram": "6-8GB"},
    "💎 TRELLIS": {"speed": "30s", "quality": 5, "vram": "16-24GB"},
    "🏭 Hunyuan3D": {"speed": "5s", "quality": 5, "vram": "12-16GB"},
    "👁️ MVDream": {"speed": "60s", "quality": 4, "vram": "8-12GB"}
}
PROMPTS = [
    "🏰 a medieval castle with towers",
    "🐉 a fierce dragon statue",
    "🚀 a futuristic spaceship",
    "🧙 a wizard's magical staff",
    "🏚️ a cozy cottage in the woods",
    "⚔️ an ornate fantasy sword",
    "🛡️ a viking shield with runes",
    "🎭 a venetian carnival mask",
    "🗿 an ancient stone statue",
    "🌳 a magical tree with glowing leaves"
]
class ModelCard(Static):
    def __init__(self, name: str, info: dict) -> None:
        super().__init__()
        self.name = name
        self.info = info
    def compose(self) -> ComposeResult:
        quality_stars = "★" * self.info["quality"] + "☆" * (5 - self.info["quality"])
        content = f"""[bold cyan]{self.name}[/bold cyan]
Speed: [yellow]{self.info['speed']}[/yellow]
Quality: [green]{quality_stars}[/green]
VRAM: [magenta]{self.info['vram']}[/magenta]"""
        self.update(Panel(content, box=box.ROUNDED))
class GenerationScreen(Screen):
    CSS = """
    GenerationScreen {
        layout: grid;
        grid-size: 2 3;
        grid-rows: 1fr 3fr 1fr;
        grid-columns: 1fr 1fr;
    }
        column-span: 2;
        height: 5;
        margin: 1;
    }
        margin: 1;
        border: solid cyan;
    }
        margin: 1;
        border: solid yellow;
    }
        column-span: 2;
        dock: bottom;
        height: 3;
        margin: 1;
    }
    .model-card {
        height: 7;
        margin: 1;
    }
        yield Header(show_clock=True)
        with Container(id="prompt-container"):
            yield Label("Enter your prompt:")
            yield Input(placeholder="e.g., a medieval castle", id="prompt-input")
        with ScrollableContainer(id="models-list"):
            yield Static("[bold yellow]Select a Model:[/bold yellow]")
            for name, info in MODELS.items():
                yield ModelCard(name, info, classes="model-card")
        with Container(id="preview"):
            yield Static("[bold cyan]Preview:[/bold cyan]\n\nYour 3D model will appear here...", id="preview-text")
        yield Button("🎨 Generate 3D Model", variant="success", id="generate-btn")
        yield Footer()
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "generate-btn":
            prompt = self.query_one("#prompt-input", Input).value
            if prompt:
                self.generate_model(prompt)
    def generate_model(self, prompt: str) -> None:
        preview = self.query_one("#preview-text", Static)
        steps = [
            "🎨 Initializing AI model...",
            "🧠 Processing prompt...",
            "✨ Generating 3D geometry...",
            "🔨 Refining mesh...",
            "💾 Saving model..."
        ]
        preview.update(f"[bold green]Generating:[/bold green] {prompt}\n\n")
        for step in steps:
            preview.update(preview.renderable + f"\n{step}")
            time.sleep(0.1)
        vertices = random.randint(1000, 5000)
        faces = random.randint(2000, 10000)
        result = f"""
[bold green]✅ Generation Complete![/bold green]
[cyan]Prompt:[/cyan] {prompt}
[cyan]Vertices:[/cyan] {vertices:,}
[cyan]Faces:[/cyan] {faces:,}
[cyan]File:[/cyan] output_{int(time.time())}.obj
[yellow]Ready for 3D software import![/yellow]"""
        preview.update(result)
class GalleryScreen(Screen):
    CSS = """
    GalleryScreen {
        layout: vertical;
    }
        height: 100%;
        margin: 1;
    }
        yield Header(show_clock=True)
        yield Static("[bold yellow]🖼️ Model Gallery[/bold yellow]", id="gallery-title")
        table = DataTable(id="gallery-table")
        table.add_columns("Model", "Prompt", "Date", "Size", "Quality")
        sample_data = [
            ("castle_001.obj", "medieval castle", "2024-01-15", "2.3 MB", "★★★★☆"),
            ("dragon_002.obj", "fire dragon", "2024-01-14", "4.1 MB", "★★★★★"),
            ("ship_003.obj", "pirate ship", "2024-01-13", "3.2 MB", "★★★☆☆"),
            ("sword_004.obj", "magic sword", "2024-01-12", "1.1 MB", "★★★★☆"),
            ("tree_005.obj", "ancient oak", "2024-01-11", "5.6 MB", "★★★★★"),
        ]
        for row in sample_data:
            table.add_row(*row)
        yield table
        yield Footer()
class DashboardScreen(Screen):
    CSS = """
    DashboardScreen {
        layout: grid;
        grid-size: 2 2;
        grid-rows: 1fr 1fr;
    }
    .dashboard-panel {
        margin: 1;
        height: 100%;
    }
        yield Header(show_clock=True)
        stats = """[bold cyan]📊 Statistics[/bold cyan]
Total Generations: 42
Success Rate: 95.2%
Avg Generation Time: 8.3s
Favorite Model: ⚡ TripoSR
Total Storage: 124 MB"""
        yield Static(Panel(stats, box=box.DOUBLE), classes="dashboard-panel")
        activity = """[bold yellow]📈 Recent Activity[/bold yellow]
• Generated 'dragon statue' - 2 min ago
• Generated 'castle tower' - 15 min ago
• Generated 'magic staff' - 1 hour ago
• System optimized - 2 hours ago
• Model updated - Yesterday"""
        yield Static(Panel(activity, box=box.DOUBLE), classes="dashboard-panel")
        prompts_text = "[bold green]💡 Quick Prompts[/bold green]\n\n"
        for prompt in random.sample(PROMPTS, 5):
            prompts_text += f"• {prompt}\n"
        yield Static(Panel(prompts_text, box=box.DOUBLE), classes="dashboard-panel")
        status = """[bold magenta]🖥️ System Status[/bold magenta]
CPU Usage: 23%
RAM Usage: 41%
GPU Status: Ready
VRAM Available: 20GB
Models Loaded: 3/5"""
        yield Static(Panel(status, box=box.DOUBLE), classes="dashboard-panel")
        yield Footer()
class DreamCADTUI(App):
    CSS = """
    Screen {
        background: $background;
    }
    Button {
        margin: 1;
    }
        self.title = "🎨 DreamCAD TUI - 3D Generation Magic"
        self.sub_title = "Press 'h' for Dashboard, 'g' for Generation, 'l' for Gallery"
        self.push_screen("dashboard")
    def action_toggle_dark(self) -> None:
        self.dark = not self.dark
    def action_switch_screen(self, screen: str) -> None:
        self.switch_screen(screen)
def main():
    app = DreamCADTUI()
    app.run()
if __name__ == "__main__":
    main()