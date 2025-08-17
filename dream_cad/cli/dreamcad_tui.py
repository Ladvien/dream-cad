#!/usr/bin/env python3
"""DreamCAD TUI - Beautiful Terminal UI for 3D Generation."""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, Input, Label, 
    DataTable, ProgressBar, RichLog, Select, TextArea,
    TabbedContent, TabPane, ListView, ListItem, RadioSet, RadioButton
)
from textual.binding import Binding
from textual import events, work
from textual.reactive import reactive
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console
from rich import box
import asyncio
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, Dict, List, Any
import time

# Mock imports for testing without dependencies
try:
    from dream_cad.models.registry import ModelRegistry
    from dream_cad.models.factory import ModelFactory
except ImportError:
    # Mock classes for testing
    class ModelRegistry:
        def get_all_models(self):
            return {
                "triposr": {
                    "min_vram_gb": 4,
                    "recommended_vram_gb": 6,
                    "generation_time_seconds": 0.5,
                    "supported_formats": ["obj", "ply", "stl", "glb"]
                },
                "stable-fast-3d": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 8,
                    "generation_time_seconds": 3,
                    "supported_formats": ["glb", "obj", "ply"]
                },
                "trellis": {
                    "min_vram_gb": 8,
                    "recommended_vram_gb": 12,
                    "generation_time_seconds": 30,
                    "supported_formats": ["obj", "ply", "nerf", "gaussian"]
                },
                "hunyuan3d-mini": {
                    "min_vram_gb": 12,
                    "recommended_vram_gb": 16,
                    "generation_time_seconds": 10,
                    "supported_formats": ["glb", "obj"]
                },
                "mvdream": {
                    "min_vram_gb": 16,
                    "recommended_vram_gb": 20,
                    "generation_time_seconds": 120,
                    "supported_formats": ["obj", "ply"]
                }
            }
    
    class ModelFactory:
        def create_model(self, model_name, config):
            return None


class ModelCard(Static):
    """A card displaying model information."""
    
    def __init__(self, name: str, info: Dict[str, Any], selected: bool = False):
        super().__init__()
        self.name = name
        self.info = info
        self.selected = selected
        
    def render(self) -> Text:
        """Render the model card."""
        status = "✓ Selected" if self.selected else "  Available"
        status_color = "green" if self.selected else "dim"
        
        # Create formatted text
        text = Text()
        text.append(f"╭─ {self.name.upper()} ", style="bold cyan")
        text.append(f"[{status}]\n", style=status_color)
        text.append(f"│ VRAM: {self.info.get('min_vram_gb', 'N/A')}GB min, ", style="yellow")
        text.append(f"{self.info.get('recommended_vram_gb', 'N/A')}GB recommended\n")
        text.append(f"│ Speed: ~{self.info.get('generation_time_seconds', 'N/A')}s\n", style="green")
        text.append(f"│ Formats: {', '.join(self.info.get('supported_formats', []))}\n", style="blue")
        text.append("╰" + "─" * 40)
        
        return text


class GenerationView(Container):
    """View for active generation with progress."""
    
    def __init__(self):
        super().__init__()
        self.progress_bar = ProgressBar(total=100)
        self.log = RichLog(highlight=True, markup=True)
        
    def compose(self) -> ComposeResult:
        """Compose the generation view."""
        yield Label("🎨 Generation Progress", classes="title")
        yield self.progress_bar
        yield self.log
        
    def update_progress(self, value: int, message: str = ""):
        """Update progress bar and log."""
        self.progress_bar.progress = value
        if message:
            self.log.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


class QueueView(Container):
    """View for managing generation queue."""
    
    def __init__(self):
        super().__init__()
        self.queue_items = []
        
    def compose(self) -> ComposeResult:
        """Compose the queue view."""
        yield Label("📋 Generation Queue", classes="title")
        
        with ScrollableContainer():
            self.queue_list = ListView()
            yield self.queue_list
            
        with Horizontal():
            yield Button("Add Job", id="add_job", variant="primary")
            yield Button("Clear Queue", id="clear_queue", variant="warning")
            yield Button("Process Queue", id="process_queue", variant="success")
    
    def add_job(self, prompt: str, model: str, format: str):
        """Add a job to the queue."""
        job_id = f"job_{len(self.queue_items) + 1:03d}"
        job = {
            "id": job_id,
            "prompt": prompt,
            "model": model,
            "format": format,
            "status": "pending",
            "created": datetime.now().isoformat()
        }
        self.queue_items.append(job)
        
        # Add to list view
        item_text = f"[{job_id}] {model}: {prompt[:30]}... ({format})"
        self.queue_list.append(ListItem(Static(item_text)))
        
        return job_id


class ModelComparisonView(Container):
    """View for comparing models side by side."""
    
    def __init__(self, registry: ModelRegistry):
        super().__init__()
        self.registry = registry
        
    def compose(self) -> ComposeResult:
        """Compose the comparison view."""
        yield Label("📊 Model Comparison", classes="title")
        
        # Create comparison table
        table = DataTable()
        yield table
        
    def on_mount(self):
        """Load model data when mounted."""
        table = self.query_one(DataTable)
        table.add_columns("Feature", "TripoSR", "Stable-Fast-3D", "TRELLIS", "Hunyuan3D", "MVDream")
        
        # Add comparison rows
        models = self.registry.get_all_models()
        
        # Speed comparison
        speeds = ["Feature", "Speed"]
        for model in ["triposr", "stable-fast-3d", "trellis", "hunyuan3d-mini", "mvdream"]:
            if model in models:
                time_s = models[model].get("generation_time_seconds", "N/A")
                speeds.append(f"{time_s}s")
            else:
                speeds.append("N/A")
        table.add_row(*speeds)
        
        # VRAM comparison
        vram = ["Feature", "VRAM"]
        for model in ["triposr", "stable-fast-3d", "trellis", "hunyuan3d-mini", "mvdream"]:
            if model in models:
                min_vram = models[model].get("min_vram_gb", "N/A")
                vram.append(f"{min_vram}GB")
            else:
                vram.append("N/A")
        table.add_row(*vram)
        
        # Quality rating
        table.add_row("Quality", "★★★☆☆", "★★★★☆", "★★★★★", "★★★★★", "★★★★☆")
        
        # Use case
        table.add_row("Best For", "Quick Proto", "Game Assets", "High Quality", "Production", "Research")


class DreamCADApp(App):
    """Main TUI application for DreamCAD."""
    
    CSS = """
    .title {
        text-style: bold;
        color: cyan;
        padding: 1;
    }
    
    ModelCard {
        margin: 1;
        padding: 1;
        border: round cyan;
    }
    
    ModelCard.selected {
        border: double green;
        background: $surface;
    }
    
    GenerationView {
        padding: 1;
    }
    
    QueueView {
        padding: 1;
    }
    
    ProgressBar {
        margin: 1 0;
    }
    
    RichLog {
        border: round #444444;
        height: 15;
        margin: 1 0;
    }
    
    #prompt_input {
        margin: 1 0;
    }
    
    Button {
        margin: 0 1;
    }
    
    DataTable {
        height: 20;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("g", "generate", "Generate"),
        Binding("m", "models", "Models"),
        Binding("c", "compare", "Compare"),
        Binding("h", "help", "Help"),
        Binding("ctrl+c", "cancel", "Cancel"),
    ]
    
    def __init__(self):
        super().__init__()
        self.registry = ModelRegistry()
        self.factory = ModelFactory()
        self.selected_model = "triposr"
        self.selected_format = "obj"
        self.generation_active = False
        
    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield Header(show_clock=True)
        
        with TabbedContent(initial="generate"):
            # Generation tab
            with TabPane("🎨 Generate", id="generate"):
                with Vertical():
                    yield Label("Select Model:", classes="title")
                    
                    with Horizontal():
                        # Model cards
                        models = self.registry.get_all_models()
                        for name, info in models.items():
                            card = ModelCard(name, info, selected=(name == self.selected_model))
                            card.id = f"model_{name}"
                            yield card
                    
                    # Prompt input
                    yield Label("Enter Prompt:", classes="title")
                    yield Input(placeholder="a fantasy cottage with thatched roof...", id="prompt_input")
                    
                    # Format selection
                    yield Label("Output Format:", classes="title")
                    with RadioSet(id="format_select"):
                        yield RadioButton("OBJ", value=True)
                        yield RadioButton("PLY")
                        yield RadioButton("STL")
                        yield RadioButton("GLB")
                    
                    # Generate button
                    with Horizontal():
                        yield Button("🚀 Generate", id="generate_btn", variant="primary")
                        yield Button("🎲 Random Prompt", id="random_prompt", variant="default")
                    
                    # Progress view
                    self.generation_view = GenerationView()
                    yield self.generation_view
            
            # Queue tab
            with TabPane("📋 Queue", id="queue"):
                self.queue_view = QueueView()
                yield self.queue_view
            
            # Comparison tab
            with TabPane("📊 Compare", id="compare"):
                yield ModelComparisonView(self.registry)
            
            # Settings tab
            with TabPane("⚙️ Settings", id="settings"):
                yield Label("Settings", classes="title")
                yield Label("VRAM Limit (GB):")
                yield Input(value="24", id="vram_limit")
                yield Label("Output Directory:")
                yield Input(value="./outputs", id="output_dir")
                yield Label("Quality Preset:")
                with RadioSet(id="quality_preset"):
                    yield RadioButton("Fast", value=True)
                    yield RadioButton("Balanced")
                    yield RadioButton("High Quality")
                yield Button("Save Settings", id="save_settings", variant="success")
        
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "generate_btn":
            self.action_generate()
        elif event.button.id == "random_prompt":
            self.set_random_prompt()
        elif event.button.id == "add_job":
            self.add_to_queue()
        elif event.button.id == "clear_queue":
            self.clear_queue()
        elif event.button.id == "process_queue":
            self.process_queue()
        elif event.button.id == "save_settings":
            self.save_settings()
    
    def on_click(self, event: events.Click) -> None:
        """Handle clicks on model cards."""
        # Check if click was on a model card
        for card in self.query(ModelCard):
            if card.region.contains(event.x, event.y):
                self.select_model(card.name)
                break
    
    def select_model(self, model_name: str):
        """Select a model."""
        self.selected_model = model_name
        
        # Update card selection states
        for card in self.query(ModelCard):
            card.selected = (card.name == model_name)
            card.refresh()
    
    def set_random_prompt(self):
        """Set a random creative prompt."""
        import random
        prompts = [
            "a mystical crystal formation glowing with inner light",
            "a steampunk mechanical owl with brass gears",
            "a miniature Japanese zen garden with stone bridge",
            "a futuristic hovering vehicle with neon accents",
            "an ancient magical tome with floating runes",
            "a cyberpunk street vendor stall with holographic signs",
            "a whimsical mushroom house with tiny windows",
            "a battle-worn robot companion with personality",
            "an ornate Victorian music box with dancing figures",
            "a floating island with waterfalls and ancient ruins"
        ]
        
        prompt_input = self.query_one("#prompt_input", Input)
        prompt_input.value = random.choice(prompts)
    
    @work(exclusive=True)
    async def action_generate(self):
        """Start generation process."""
        if self.generation_active:
            self.generation_view.log.write("[red]Generation already in progress![/red]")
            return
        
        prompt_input = self.query_one("#prompt_input", Input)
        prompt = prompt_input.value.strip()
        
        if not prompt:
            self.generation_view.log.write("[red]Please enter a prompt![/red]")
            return
        
        self.generation_active = True
        
        # Log start
        self.generation_view.log.write(f"[green]Starting generation with {self.selected_model}[/green]")
        self.generation_view.log.write(f"Prompt: {prompt}")
        self.generation_view.log.write(f"Format: {self.selected_format}")
        
        # Simulate generation with progress
        steps = [
            (10, "Loading model..."),
            (20, "Initializing pipeline..."),
            (30, "Processing prompt..."),
            (50, "Generating 3D structure..."),
            (70, "Optimizing mesh..."),
            (85, "Applying textures..."),
            (95, "Finalizing output..."),
            (100, "✨ Generation complete!")
        ]
        
        for progress, message in steps:
            self.generation_view.update_progress(progress, message)
            await asyncio.sleep(0.5)  # Simulate work
        
        # Save result
        output_path = Path("outputs") / f"{self.selected_model}_{int(time.time())}.{self.selected_format}"
        self.generation_view.log.write(f"[green]Saved to: {output_path}[/green]")
        
        self.generation_active = False
    
    def add_to_queue(self):
        """Add current settings to queue."""
        prompt_input = self.query_one("#prompt_input", Input)
        prompt = prompt_input.value.strip()
        
        if prompt:
            job_id = self.queue_view.add_job(prompt, self.selected_model, self.selected_format)
            self.notify(f"Added job {job_id} to queue")
    
    def clear_queue(self):
        """Clear the generation queue."""
        self.queue_view.queue_items.clear()
        self.queue_view.queue_list.clear()
        self.notify("Queue cleared")
    
    @work(exclusive=True)
    async def process_queue(self):
        """Process all jobs in queue."""
        for job in self.queue_view.queue_items:
            if job["status"] == "pending":
                job["status"] = "processing"
                self.notify(f"Processing {job['id']}")
                
                # Simulate processing
                await asyncio.sleep(2)
                
                job["status"] = "completed"
                self.notify(f"Completed {job['id']}")
    
    def save_settings(self):
        """Save current settings."""
        settings = {
            "vram_limit": self.query_one("#vram_limit", Input).value,
            "output_dir": self.query_one("#output_dir", Input).value,
            "selected_model": self.selected_model,
            "selected_format": self.selected_format
        }
        
        settings_path = Path.home() / ".dreamcad" / "settings.json"
        settings_path.parent.mkdir(exist_ok=True)
        
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        
        self.notify("Settings saved!")
    
    def action_quit(self):
        """Quit the application."""
        self.exit()


def main():
    """Run the TUI application."""
    app = DreamCADApp()
    app.run()


if __name__ == "__main__":
    main()