#!/usr/bin/env python3
"""DreamCAD TUI - Complete Terminal User Interface for 3D Generation"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Button, Input, Label, Static, 
    Select, Switch, RadioSet, RadioButton, RichLog
)
from textual.binding import Binding
from textual import work
from rich.panel import Panel
from rich.table import Table
import asyncio
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any, List


# Model configurations with all parameters
MODEL_CONFIGS = {
    "TripoSR": {
        "description": "Fast prototyping, 0.5s generation",
        "vram": "4GB",
        "quality": "★★★☆☆",
        "parameters": {
            "resolution": {
                "type": "select",
                "options": ["256", "512", "1024"],
                "default": "512",
                "label": "Resolution"
            },
            "batch_size": {
                "type": "select",
                "options": ["1", "2", "4"],
                "default": "1",
                "label": "Batch Size"
            },
            "remove_background": {
                "type": "switch",
                "default": True,
                "label": "Remove Background"
            },
            "output_format": {
                "type": "select",
                "options": ["obj", "ply", "stl", "glb"],
                "default": "obj",
                "label": "Output Format"
            }
        }
    },
    "Stable-Fast-3D": {
        "description": "Game assets, PBR materials, 3s generation",
        "vram": "6GB",
        "quality": "★★★★☆",
        "parameters": {
            "target_polycount": {
                "type": "select",
                "options": ["5000", "10000", "20000", "50000"],
                "default": "10000",
                "label": "Target Polycount"
            },
            "texture_size": {
                "type": "select",
                "options": ["512", "1024", "2048"],
                "default": "1024",
                "label": "Texture Size"
            },
            "enable_pbr": {
                "type": "switch",
                "default": True,
                "label": "Enable PBR"
            },
            "delighting": {
                "type": "switch",
                "default": False,
                "label": "Delighting"
            },
            "output_format": {
                "type": "select",
                "options": ["glb", "obj", "ply"],
                "default": "glb",
                "label": "Output Format"
            }
        }
    },
    "TRELLIS": {
        "description": "High quality, multi-representation, 30s generation",
        "vram": "8GB",
        "quality": "★★★★★",
        "parameters": {
            "quality_mode": {
                "type": "select",
                "options": ["fast", "balanced", "hq"],
                "default": "balanced",
                "label": "Quality Mode"
            },
            "representation": {
                "type": "select",
                "options": ["mesh", "nerf", "gaussian"],
                "default": "mesh",
                "label": "Representation"
            },
            "preserve_intermediate": {
                "type": "switch",
                "default": False,
                "label": "Preserve Intermediate"
            },
            "output_format": {
                "type": "select",
                "options": ["obj", "ply", "glb"],
                "default": "obj",
                "label": "Output Format"
            }
        }
    },
    "Hunyuan3D": {
        "description": "Production quality, PBR, 10s generation",
        "vram": "12GB",
        "quality": "★★★★★",
        "parameters": {
            "polycount": {
                "type": "select",
                "options": ["10000", "20000", "30000", "50000"],
                "default": "20000",
                "label": "Polycount"
            },
            "texture_resolution": {
                "type": "select",
                "options": ["1024", "2048", "4096"],
                "default": "2048",
                "label": "Texture Resolution"
            },
            "uv_unwrap_method": {
                "type": "select",
                "options": ["smart", "angle", "conformal"],
                "default": "smart",
                "label": "UV Unwrap Method"
            },
            "output_format": {
                "type": "select",
                "options": ["glb", "obj", "ply"],
                "default": "glb",
                "label": "Output Format"
            }
        }
    }
}


class DreamCADTUI(App):
    """Complete TUI for 3D model generation with all features."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #sidebar {
        width: 40;
        height: 100%;
        dock: left;
        border: solid $primary;
        padding: 1;
    }
    
    #main {
        padding: 1;
    }
    
    #model-info {
        height: 6;
        margin: 1 0;
        padding: 1;
    }
    
    #parameters {
        height: auto;
        margin: 1 0;
        padding: 1;
        border: solid $secondary;
    }
    
    #prompt-container {
        height: 5;
        margin: 1 0;
    }
    
    Input {
        margin: 1 0;
    }
    
    Select {
        margin: 1 0;
    }
    
    Switch {
        margin: 1 0;
    }
    
    Button {
        margin: 1 0;
    }
    
    RichLog {
        height: 20;
        border: solid green;
        margin: 1 0;
    }
    
    .parameter-label {
        margin: 0 0 0 0;
        padding: 0;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+g", "generate", "Generate"),
        Binding("ctrl+c", "clear_log", "Clear Log"),
        Binding("f1", "help", "Help"),
    ]
    
    def __init__(self):
        super().__init__()
        self.current_model = "TripoSR"
        self.current_params = {}
        self.param_widgets = {}
        self._initialized = False  # Flag to prevent initial trigger
    
    def compose(self) -> ComposeResult:
        """Build the UI layout."""
        yield Header()
        
        with Horizontal():
            # Sidebar with model selection
            with Vertical(id="sidebar"):
                yield Label("🎨 DreamCAD TUI", classes="title")
                yield Label("\nSelect Model:")
                
                # Model selection dropdown
                model_options = [(name, name) for name in MODEL_CONFIGS.keys()]
                yield Select(
                    options=model_options,
                    value="TripoSR",
                    id="model-select",
                    allow_blank=False
                )
                
                # Model info panel
                yield Static(id="model-info")
                
                # Parameters container
                yield Label("\nParameters:")
                yield ScrollableContainer(id="parameters")
            
            # Main content area
            with Vertical(id="main"):
                # Prompt input
                with Container(id="prompt-container"):
                    yield Label("Enter Prompt:")
                    yield Input(
                        placeholder="e.g., a crystal sword, a wooden chair, a fantasy cottage",
                        id="prompt"
                    )
                
                # Generate button
                with Horizontal():
                    yield Button("🚀 Generate 3D Model", variant="primary", id="generate-btn")
                    yield Button("🗑️ Clear", variant="warning", id="clear-btn")
                
                # Output log
                yield Label("\nOutput Log:")
                yield RichLog(id="output")
        
        yield Footer()
    
    def on_mount(self):
        """Initialize when app starts."""
        self.update_model_info()
        self.update_parameters()
        self.query_one("#prompt").focus()
        self.log_output("[green]Ready! Select a model, adjust parameters, and enter a prompt.[/green]")
        self._initialized = True  # Mark as initialized after first mount
    
    def on_select_changed(self, event):
        """Handle model selection change."""
        if event.select.id == "model-select":
            # Skip if the value hasn't actually changed
            if event.value == self.current_model:
                return
            self.current_model = event.value
            self.update_model_info()
            self.update_parameters()
            self.log_output(f"[cyan]Selected model: {self.current_model}[/cyan]")
    
    def update_model_info(self):
        """Update the model info display."""
        config = MODEL_CONFIGS[self.current_model]
        info_text = (
            f"[bold]{self.current_model}[/bold]\n"
            f"{config['description']}\n"
            f"VRAM: {config['vram']}\n"
            f"Quality: {config['quality']}"
        )
        panel = Panel(info_text, border_style="cyan")
        self.query_one("#model-info").update(panel)
    
    def update_parameters(self):
        """Update parameter controls based on selected model."""
        container = self.query_one("#parameters")
        
        # Clear existing widgets and registry
        container.remove_children()
        self.param_widgets.clear()
        
        config = MODEL_CONFIGS[self.current_model]
        
        # Create widgets and mount them
        widgets_to_mount = []
        for param_name, param_config in config["parameters"].items():
            # Add label
            label = Label(param_config["label"] + ":", classes="parameter-label")
            widgets_to_mount.append(label)
            
            # Add appropriate widget
            if param_config["type"] == "select":
                options = [(opt, opt) for opt in param_config["options"]]
                widget = Select(
                    options=options,
                    value=param_config["default"],
                    id=f"param-{param_name}-{self.current_model.replace(' ', '_')}",  # Make ID unique per model
                    allow_blank=False
                )
            elif param_config["type"] == "switch":
                widget = Switch(
                    value=param_config["default"],
                    id=f"param-{param_name}-{self.current_model.replace(' ', '_')}"  # Make ID unique per model
                )
            else:
                continue
            
            widgets_to_mount.append(widget)
            self.param_widgets[param_name] = widget
        
        # Mount all widgets at once
        if widgets_to_mount:
            container.mount(*widgets_to_mount)
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        params = {"model": self.current_model}
        
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, Select):
                params[param_name] = widget.value
            elif isinstance(widget, Switch):
                params[param_name] = widget.value
        
        return params
    
    def log_output(self, message: str):
        """Write to output log."""
        output = self.query_one("#output", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        output.write(f"[dim]{timestamp}[/dim] {message}")
    
    @work(exclusive=True)
    async def generate_model(self):
        """Generate 3D model with current settings."""
        prompt = self.query_one("#prompt", Input).value.strip()
        
        if not prompt:
            self.log_output("[red]❌ Please enter a prompt![/red]")
            return
        
        # Disable generate button
        gen_btn = self.query_one("#generate-btn", Button)
        gen_btn.disabled = True
        
        try:
            # Get parameters
            params = self.get_current_parameters()
            
            self.log_output(f"\n[bold cyan]Starting Generation[/bold cyan]")
            self.log_output(f"Prompt: {prompt}")
            self.log_output(f"Model: {params['model']}")
            
            # Log parameters
            for key, value in params.items():
                if key != "model":
                    self.log_output(f"  {key}: {value}")
            
            # Check if we can actually load the model
            try:
                from dream_cad.models.factory import ModelFactory
                from dream_cad.models.registry import ModelRegistry
                
                # Try to get model from factory
                registry = ModelRegistry()
                model_name = self.current_model.lower().replace("-", "_")
                
                if registry.has_model(model_name):
                    self.log_output(f"[yellow]Loading {self.current_model} model...[/yellow]")
                    
                    # Get the model
                    model = ModelFactory.create_model(model_name)
                    
                    # Configure model
                    output_format = params.get("output_format", "obj")
                    
                    self.log_output("[yellow]Generating 3D model...[/yellow]")
                    
                    # Generate
                    result = model.generate_from_text(
                        prompt=prompt,
                        output_format=output_format,
                        **params
                    )
                    
                    if result and result.file_path:
                        self.log_output(f"[green]✅ Success! Generated: {result.file_path}[/green]")
                        if result.metadata:
                            self.log_output(f"Generation time: {result.metadata.get('generation_time', 'N/A')}s")
                    else:
                        self.log_output("[yellow]⚠️ Generation completed but no file was created (mock mode)[/yellow]")
                    
                    # Cleanup
                    model.cleanup()
                else:
                    # Fallback to simulation
                    await self.simulate_generation(prompt, params)
                    
            except ImportError:
                # Models not available, simulate
                await self.simulate_generation(prompt, params)
                
        except Exception as e:
            self.log_output(f"[red]❌ Error: {str(e)}[/red]")
        
        finally:
            gen_btn.disabled = False
    
    async def simulate_generation(self, prompt: str, params: Dict[str, Any]):
        """Simulate generation when models aren't available."""
        self.log_output("[yellow]Running in simulation mode (models not loaded)[/yellow]")
        
        # Simulate steps
        steps = [
            "Loading model...",
            "Processing prompt...",
            "Generating 3D structure...",
            "Applying textures...",
            "Optimizing mesh...",
            "Saving file..."
        ]
        
        for step in steps:
            self.log_output(f"  [dim]{step}[/dim]")
            await asyncio.sleep(0.5)
        
        # Simulate success
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_format = params.get("output_format", "obj")
        filename = f"output_{timestamp}.{output_format}"
        self.log_output(f"[green]✅ Success! (Simulated) Saved to: outputs/{filename}[/green]")
    
    def on_button_pressed(self, event):
        """Handle button presses."""
        if event.button.id == "generate-btn":
            self.generate_model()
        elif event.button.id == "clear-btn":
            self.action_clear_log()
    
    def action_generate(self):
        """Action to generate model."""
        self.generate_model()
    
    def action_clear_log(self):
        """Clear the output log."""
        output = self.query_one("#output", RichLog)
        output.clear()
        self.log_output("[green]Log cleared[/green]")
    
    def action_help(self):
        """Show help information."""
        help_text = """
[bold cyan]DreamCAD TUI Help[/bold cyan]

[bold]Keyboard Shortcuts:[/bold]
  Ctrl+Q : Quit
  Ctrl+G : Generate
  Ctrl+C : Clear Log
  F1     : Show this help
  Tab    : Navigate fields

[bold]Models:[/bold]
  TripoSR       : Fast prototyping (0.5s, 4GB VRAM)
  Stable-Fast-3D: Game assets with PBR (3s, 6GB VRAM)
  TRELLIS       : High quality (30s, 8GB VRAM)
  Hunyuan3D     : Production quality (10s, 12GB VRAM)

[bold]Usage:[/bold]
  1. Select a model from the dropdown
  2. Adjust parameters as needed
  3. Enter your prompt
  4. Click Generate or press Ctrl+G
        """
        self.log_output(help_text)


def main():
    """Run the TUI."""
    app = DreamCADTUI()
    app.run()


if __name__ == "__main__":
    main()