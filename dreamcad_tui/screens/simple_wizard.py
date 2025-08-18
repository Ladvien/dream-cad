from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Button, Label, Input, Select, ProgressBar
from textual.containers import Horizontal, Vertical, Container
from textual import on
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from datetime import datetime
from typing import Optional
import threading
import asyncio
import time

from ..core.generation_worker import GenerationJob, JobStatus, OutputFormat

PROMPT_SUGGESTIONS = [
    "a crystal sword with glowing runes",
    "medieval fantasy cottage with thatched roof",
    "cute robot companion with LED eyes",
    "magical floating orb with particle effects",
    "steampunk mechanical heart",
]

class SimpleWizard(Screen):
    def compose(self) -> ComposeResult:
        yield Label("🎨 Generation Wizard", id="title")
        
        with Container(id="wizard-container"):
            with Vertical(id="wizard-form"):
                yield Label("Step 1: Enter your prompt", classes="step-label")
                yield Input(
                    placeholder="Describe what you want to create...",
                    id="prompt-input"
                )
                
                yield Label("Suggestions:", classes="hint-label")
                with Container(id="suggestions"):
                    for i, suggestion in enumerate(PROMPT_SUGGESTIONS):
                        yield Button(
                            f"💡 {suggestion}",
                            id=f"suggest-{i}",
                            classes="suggestion-btn"
                        )
                        
                yield Label("Step 2: Select model", classes="step-label")
                yield Select(
                    [("🎭 Demo Mode - Simple meshes", "demo")],
                    id="model-select",
                    value="demo"
                )
                
                yield Static(id="model-info")
                
                yield Label("Step 3: Output format", classes="step-label")
                yield Select(
                    [
                        ("OBJ - Universal format", "obj"),
                        ("GLB - With textures", "glb"),
                        ("PLY - Point cloud", "ply")
                    ],
                    id="format-select",
                    value="obj"
                )
                
            with Vertical(id="generation-panel"):
                yield Static(id="generation-status")
                yield ProgressBar(id="generation-progress", show_percentage=True)
                
                with Horizontal(id="wizard-actions"):
                    yield Button("🚀 Generate", variant="primary", id="btn-start")
                    yield Button("❌ Cancel", variant="error", id="btn-cancel", disabled=True)
                    yield Button("🔙 Back", id="btn-back")
                    
    def on_mount(self):
        self._populate_models()
        self._update_status("Ready to generate")
        self.query_one("#generation-progress").update(total=100, progress=0)
        self.current_job = None
        self.generation_thread = None
        
    def _populate_models(self):
        model_select = self.query_one("#model-select", Select)
        
        options = []
        default = None
        
        # Get available models from app
        models = self.app.get_available_models() if hasattr(self.app, 'get_available_models') else {}
        
        # Add all models with their status
        for model_id, info in models.items():
            model_info = info.get('info', {})
            name = model_info.get('name', model_id)
            
            # Determine status and create label
            if info.get('available'):
                label = f"✓ {name} - {model_info.get('speed', 'Ready')}"
                options.append((label, model_id))
                if default is None:
                    default = model_id
            elif info.get('cached'):
                label = f"◉ {name} - Cached"
                options.append((label, model_id))
                if default is None:
                    default = model_id
            elif info.get('vram_compatible', True):
                label = f"⬇ {name} - {model_info.get('size_gb', 0):.1f}GB download"
                options.append((label, model_id))
            else:
                label = f"✗ {name} - Insufficient VRAM"
                options.append((label, model_id))
        
        # Always add demo mode as fallback
        if self.app.is_demo_mode or not options:
            options.insert(0, ("🎭 Demo Mode - Simple meshes", "demo"))
            if default is None:
                default = "demo"
        else:
            options.append(("🎭 Demo Mode - Simple meshes", "demo"))
            
        model_select.set_options(options)
        if default:
            model_select.value = default
            self._update_model_info(default)
            
    @on(Select.Changed, "#model-select")
    def on_model_changed(self, event: Select.Changed):
        self._update_model_info(event.value)
        
    def _update_model_info(self, model_id: str):
        info_widget = self.query_one("#model-info", Static)
        
        if model_id == "demo":
            info_text = Panel(
                Text("Demo mode - generates simple procedural meshes\nNo GPU required\nInstant generation", style="yellow"),
                title="Demo Mode",
                border_style="yellow"
            )
        else:
            # Get model info from app
            model_info = self.app.get_model_info(model_id) if hasattr(self.app, 'get_model_info') else None
            
            if model_info:
                info = model_info.get('info', {})
                
                # Build info table
                table = Table.grid(padding=0)
                table.add_column(style="cyan", width=12)
                table.add_column()
                
                table.add_row("Quality:", info.get('quality', 'Unknown'))
                table.add_row("Speed:", info.get('speed', 'Unknown'))
                table.add_row("VRAM:", f"{info.get('min_vram_gb', 0)}GB minimum")
                table.add_row("Size:", f"{info.get('size_gb', 0):.1f}GB download")
                
                # Add status
                if model_info.get('available'):
                    status = Text("Ready to use", style="green")
                elif model_info.get('cached'):
                    status = Text("Cached, loading required", style="cyan")
                elif model_info.get('vram_compatible', True):
                    status = Text("Will download on first use", style="yellow")
                else:
                    status = Text("Insufficient VRAM", style="red")
                    
                table.add_row("Status:", status)
                
                info_text = Panel(
                    table,
                    title=info.get('description', info.get('name', 'Model Info')),
                    border_style="cyan"
                )
            else:
                info_text = Panel(
                    Text(f"Model: {model_id}", style="cyan"),
                    title="Model Info",
                    border_style="cyan"
                )
                
        info_widget.update(info_text)
        
    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id and event.button.id.startswith("suggest-"):
            idx = int(event.button.id.split("-")[1])
            if idx < len(PROMPT_SUGGESTIONS):
                prompt_input = self.query_one("#prompt-input", Input)
                prompt_input.value = PROMPT_SUGGESTIONS[idx]
        elif event.button.id == "btn-start":
            self.start_generation()
        elif event.button.id == "btn-cancel":
            self.cancel_generation()
        elif event.button.id == "btn-back":
            self.app.pop_screen()
            
    def start_generation(self):
        """Start generation in a background thread"""
        prompt = self.query_one("#prompt-input", Input).value.strip()
        if not prompt:
            self.app.notify("Please enter a prompt", severity="warning")
            return
            
        model_id = self.query_one("#model-select", Select).value
        if not model_id:
            self.app.notify("Please select a model", severity="warning")
            return
                
        format_value = self.query_one("#format-select", Select).value
        output_format = OutputFormat(format_value)
        
        # Disable generate button, enable cancel
        self.query_one("#btn-start", Button).disabled = True
        self.query_one("#btn-cancel", Button).disabled = False
        
        # For demo mode, use triposr as base
        if model_id == "demo":
            model_id = "triposr"
            
        # Create the job
        self.current_job = self.app.generation_worker.create_job(
            prompt=prompt,
            model_name=model_id,
            output_format=output_format
        )
        
        self._update_status(f"Starting generation: {prompt[:50]}...")
        
        # Start generation in a background thread
        self.generation_thread = threading.Thread(
            target=self._run_generation_thread,
            daemon=True
        )
        self.generation_thread.start()
        
        # Start progress monitoring
        self.set_interval(0.1, self._check_progress)
        
    def _run_generation_thread(self):
        """Run generation in a background thread"""
        if not self.current_job:
            return
            
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async generation
            loop.run_until_complete(
                self.app.generation_worker.process_job(self.current_job)
            )
        except Exception as e:
            self.current_job.status = JobStatus.FAILED
            self.current_job.error = str(e)
        finally:
            loop.close()
            # Signal completion
            self.call_from_thread(self._generation_complete)
            
    def _check_progress(self):
        """Check generation progress periodically"""
        if not self.current_job:
            return False  # Stop checking
            
        # Update progress bar
        progress = self.current_job.progress * 100
        self.query_one("#generation-progress").update(progress=int(progress))
        
        # Update status if there's a message
        if self.current_job.progress_message:
            self._update_status(self.current_job.progress_message)
            
        # Check if complete
        if self.current_job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False  # Stop interval
            
        return True  # Continue checking
        
    def _generation_complete(self):
        """Called when generation completes"""
        if not self.current_job:
            return
            
        # Re-enable buttons
        self.query_one("#btn-start", Button).disabled = False
        self.query_one("#btn-cancel", Button).disabled = True
        
        # Update status based on result
        if self.current_job.status == JobStatus.COMPLETED:
            self._update_status(f"✓ Complete! Saved to: {self.current_job.output_path}")
            self.app.notify(
                f"Generation complete: {self.current_job.output_path.name if self.current_job.output_path else 'output'}",
                severity="information"
            )
        elif self.current_job.status == JobStatus.FAILED:
            self._update_status(f"✗ Failed: {self.current_job.error}")
            self.app.notify(
                f"Generation failed: {self.current_job.error}",
                severity="error"
            )
        elif self.current_job.status == JobStatus.CANCELLED:
            self._update_status("✗ Cancelled")
            
        # Reset progress
        self.query_one("#generation-progress").update(progress=0)
        
    def cancel_generation(self):
        """Cancel the current generation"""
        if self.current_job:
            self.app.generation_worker.cancel_job(self.current_job.id)
            self._update_status("Cancelling...")
            self.query_one("#btn-cancel", Button).disabled = True
            
    def _update_status(self, message: str):
        """Update the status display"""
        status_widget = self.query_one("#generation-status", Static)
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_widget.update(
            Panel(
                Text(f"[{timestamp}] {message}"),
                title="Status",
                border_style="cyan"
            )
        )