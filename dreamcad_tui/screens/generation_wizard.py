from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Button, Label, Input, Select, TextArea, ProgressBar
from textual.containers import Horizontal, Vertical, Container
from textual.widget import Widget
from textual import work, on
from textual.worker import Worker, get_current_worker
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from datetime import datetime
from typing import Optional
import asyncio

from ..core.generation_worker import GenerationJob, JobStatus, OutputFormat

PROMPT_SUGGESTIONS = [
    "a crystal sword with glowing runes",
    "medieval fantasy cottage with thatched roof",
    "cute robot companion with LED eyes",
    "magical floating orb with particle effects",
    "steampunk mechanical heart",
    "ancient stone temple ruins",
    "futuristic hover bike",
    "ornate treasure chest",
    "mystical potion bottle",
    "cyberpunk city building"
]

class GenerationWizard(Screen):
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
                    for i, suggestion in enumerate(PROMPT_SUGGESTIONS[:5]):
                        yield Button(
                            f"💡 {suggestion}",
                            id=f"suggest-{i}",
                            classes="suggestion-btn"
                        )
                        
                yield Label("Step 2: Select model", classes="step-label")
                yield Select(
                    [],
                    id="model-select",
                    prompt="Choose a model"
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
        
    def _populate_models(self):
        models = self.app.get_available_models()
        model_select = self.query_one("#model-select", Select)
        
        options = []
        default = None
        
        for model_id, info in models.items():
            name = info['info'].get('name', model_id)
            
            if info['available']:
                label = f"✓ {name} - {info['info'].get('speed', 'N/A')}"
                options.append((label, model_id))
                if default is None:
                    default = model_id
            elif info['cached']:
                label = f"◉ {name} - Cached"
                options.append((label, model_id))
            else:
                label = f"✗ {name} - Not available"
                options.append((label, model_id))
                
        if self.app.is_demo_mode:
            options.insert(0, ("🎭 Demo Mode - Simple meshes", "demo"))
            default = "demo"
            
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
                Text("Demo mode - generates simple procedural meshes\nNo GPU required", style="yellow"),
                title="Demo Mode",
                border_style="yellow"
            )
        else:
            model_info = self.app.get_model_info(model_id)
            if model_info:
                info = model_info['info']
                table = Table.grid(padding=0)
                table.add_column(style="cyan", width=12)
                table.add_column()
                
                table.add_row("Quality:", info.get('quality', 'Unknown'))
                table.add_row("Speed:", info.get('speed', 'Unknown'))
                table.add_row("VRAM:", f"{info.get('min_vram_gb', 0)}GB minimum")
                table.add_row("Size:", f"{info.get('size_gb', 0):.1f}GB download")
                
                if model_info['available']:
                    status = Text("Ready to use", style="green")
                elif model_info['vram_compatible']:
                    status = Text("Will download on first use", style="yellow")
                else:
                    status = Text("Insufficient VRAM", style="red")
                    
                table.add_row("Status:", status)
                
                info_text = Panel(
                    table,
                    title=info.get('description', 'Model Info'),
                    border_style="cyan"
                )
            else:
                info_text = Panel(
                    Text("Model information not available", style="dim"),
                    border_style="dim"
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
        prompt = self.query_one("#prompt-input", Input).value.strip()
        if not prompt:
            self.app.notify("Please enter a prompt", severity="warning")
            return
            
        model_id = self.query_one("#model-select", Select).value
        if not model_id:
            self.app.notify("Please select a model", severity="warning")
            return
            
        if model_id != "demo" and not self.app.is_model_available(model_id):
            if not self.app.config.config.models.fallback_to_mock:
                self.app.notify("Selected model is not available", severity="error")
                return
                
        format_value = self.query_one("#format-select", Select).value
        output_format = OutputFormat(format_value)
        
        self.query_one("#btn-start", Button).disabled = True
        self.query_one("#btn-cancel", Button).disabled = False
        
        if model_id == "demo":
            model_id = "triposr"  # Use triposr as the base for demo
            
        self.current_job = self.app.generation_worker.create_job(
            prompt=prompt,
            model_name=model_id,
            output_format=output_format
        )
        
        self._update_status(f"Starting generation: {prompt[:50]}...")
        
        # Start the generation in a worker thread
        self.run_generation_worker()
        
    def run_generation_worker(self):
        """Start generation in a background worker"""
        worker = self.run_worker(self.execute_generation, thread=True)
        
    def execute_generation(self):
        """Execute the generation synchronously in a thread"""
        if not self.current_job:
            return
            
        # Run the async generation in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self.app.generation_worker.process_job(self.current_job)
            )
        finally:
            loop.close()
    
    def cancel_generation(self):
        if self.current_job:
            self.app.generation_worker.cancel_job(self.current_job.id)
            self._update_status("Cancelling...")
            self.query_one("#btn-cancel", Button).disabled = True
            
    def on_progress_update(self, message):
        if self.current_job and message.job_id == self.current_job.id:
            progress_bar = self.query_one("#generation-progress", ProgressBar)
            progress_bar.update(progress=int(message.progress * 100))
            self._update_status(message.message)
            
    def on_job_complete(self, message):
        if self.current_job and message.job.id == self.current_job.id:
            self.query_one("#btn-start", Button).disabled = False
            self.query_one("#btn-cancel", Button).disabled = True
            
            if message.job.status == JobStatus.COMPLETED:
                self._update_status(f"✓ Complete! Saved to: {message.job.output_path}")
                self.app.notify(
                    f"Generation complete: {message.job.output_path.name}",
                    severity="information"
                )
            elif message.job.status == JobStatus.FAILED:
                self._update_status(f"✗ Failed: {message.job.error}")
                self.app.notify(
                    f"Generation failed: {message.job.error}",
                    severity="error"
                )
            elif message.job.status == JobStatus.CANCELLED:
                self._update_status("✗ Cancelled")
                
            self.query_one("#generation-progress", ProgressBar).update(progress=0)
            
    def _update_status(self, message: str):
        status_widget = self.query_one("#generation-status", Static)
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_widget.update(
            Panel(
                Text(f"[{timestamp}] {message}"),
                title="Status",
                border_style="cyan"
            )
        )