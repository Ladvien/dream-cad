from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Static, Button, Input, Label, TextArea, 
    RadioSet, RadioButton, ProgressBar, Select, Rule
)
from textual.widget import Widget
from textual.reactive import reactive
from textual import work
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
import asyncio
import random
from typing import Optional
class PromptSuggestions(Widget):
    suggestions = [
        "a mystical crystal formation glowing with inner light",
        "a steampunk mechanical owl with brass gears",
        "a miniature Japanese zen garden with stone bridge",
        "a futuristic hovering vehicle with neon accents",
        "an ancient magical tome with floating runes",
        "a cyberpunk street vendor stall with holographic signs",
        "a whimsical mushroom house with tiny windows",
        "a battle-worn robot companion with personality",
    ]
    def render(self):
        table = Table.grid(padding=0)
        table.add_column()
        table.add_row(Text("💡 Suggestions:", style="bold yellow"))
        for i, suggestion in enumerate(self.suggestions[:4], 1):
            table.add_row(Text(f"  {i}. {suggestion[:50]}...", style="dim cyan"))
        return Panel(
            table,
            border_style="yellow",
            expand=True
        )
class ModelRecommendation(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_prompt = ""
    def update_prompt(self, prompt: str):
        self.current_prompt = prompt
        self.refresh()
    def render(self):
        prompt_lower = self.current_prompt.lower()
        if any(word in prompt_lower for word in ["quick", "simple", "basic"]):
            recommended = "TripoSR"
            reason = "Fast generation for simple objects"
        elif any(word in prompt_lower for word in ["game", "asset", "low-poly"]):
            recommended = "Stable-Fast-3D"
            reason = "Optimized for game assets"
        elif any(word in prompt_lower for word in ["detailed", "realistic", "high"]):
            recommended = "TRELLIS"
            reason = "High quality output"
        else:
            recommended = "TripoSR"
            reason = "Balanced speed and quality"
        content = Table.grid(padding=0)
        content.add_column()
        content.add_row(Text("🎯 Recommended Model:", style="bold green"))
        content.add_row(Text(f"  {recommended}", style="bold cyan"))
        content.add_row(Text(f"  {reason}", style="dim"))
        return Panel(
            content,
            border_style="green",
            expand=True
        )
class GenerationProgress(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.progress = 0
        self.status = "Ready"
        self.steps = []
    def start_generation(self):
        self.progress = 0
        self.status = "Initializing..."
        self.steps = [
            (10, "Loading model..."),
            (25, "Processing prompt..."),
            (40, "Generating structure..."),
            (60, "Optimizing mesh..."),
            (80, "Applying textures..."),
            (95, "Finalizing output..."),
            (100, "Complete!")
        ]
        self.set_interval(0.5, self.update_progress)
    def update_progress(self):
        if self.steps and self.progress < 100:
            target, status = self.steps[0]
            if self.progress >= target:
                self.steps.pop(0)
                if self.steps:
                    _, status = self.steps[0]
            self.status = status
            self.progress = min(100, self.progress + random.randint(2, 8))
            self.refresh()
        else:
            self.status = "✨ Generation complete!"
            return False
    def render(self):
        bar_width = 40
        filled = int(self.progress / 100 * bar_width)
        progress_bar = Text()
        progress_bar.append("█" * filled, style="green")
        progress_bar.append("░" * (bar_width - filled), style="dim")
        content = Table.grid(padding=1)
        content.add_column()
        content.add_row(Text(self.status, style="bold cyan"))
        content.add_row(progress_bar)
        content.add_row(Text(f"{self.progress}% Complete", style="yellow"))
        if self.progress > 0 and self.progress < 100:
            spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            spin_char = spinner[self.progress % len(spinner)]
            content.add_row(Text(f"{spin_char} Processing...", style="dim cyan"))
        return Panel(
            content,
            title="🎨 Generation Progress",
            border_style="cyan" if self.progress < 100 else "green",
            expand=True
        )
class GenerationWizard(Container):
    current_step = reactive(1)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt = ""
        self.selected_model = "triposr"
        self.selected_format = "obj"
        self.quality_preset = "balanced"
    def compose(self) -> ComposeResult:
        yield Label("🧙 Generation Wizard", classes="title", id="wizard-title")
        with Horizontal(id="step-indicator"):
            yield Static(self._get_step_indicator(), id="steps")
        yield Rule()
        with ScrollableContainer(id="wizard-content"):
            with Vertical(id="step-1", classes="wizard-step"):
                yield Label("Step 1: Describe what you want to create", classes="subtitle")
                yield TextArea(
                    "Enter your prompt here...\nExample: a magical crystal sword with glowing runes",
                    id="prompt-input"
                )
                yield PromptSuggestions()
            with Vertical(id="step-2", classes="wizard-step hidden"):
                yield Label("Step 2: Choose your model", classes="subtitle")
                yield ModelRecommendation(id="model-recommendation")
                with RadioSet(id="model-select"):
                    yield RadioButton("TripoSR (Fast, 0.5s)", value=True)
                    yield RadioButton("Stable-Fast-3D (Game assets, 3s)")
                    yield RadioButton("TRELLIS (High quality, 30s)")
                    yield RadioButton("Hunyuan3D (Production, 10s)")
                    yield RadioButton("MVDream (Research, 120s)")
            with Vertical(id="step-3", classes="wizard-step hidden"):
                yield Label("Step 3: Configure settings", classes="subtitle")
                yield Label("Output Format:")
                with RadioSet(id="format-select"):
                    yield RadioButton("OBJ (Universal)", value=True)
                    yield RadioButton("PLY (Point cloud)")
                    yield RadioButton("STL (3D printing)")
                    yield RadioButton("GLB (Web/AR)")
                yield Label("Quality Preset:")
                with RadioSet(id="quality-select"):
                    yield RadioButton("Fast (Lower quality)")
                    yield RadioButton("Balanced (Recommended)", value=True)
                    yield RadioButton("High (Best quality)")
            with Vertical(id="step-4", classes="wizard-step hidden"):
                yield Label("Step 4: Review and generate", classes="subtitle")
                yield Static(id="generation-summary")
                yield GenerationProgress(id="generation-progress")
        yield Rule()
        with Horizontal(id="wizard-buttons"):
            yield Button("← Previous", variant="default", id="btn-prev", disabled=True)
            yield Button("→ Next", variant="primary", id="btn-next")
            yield Button("🚀 Generate", variant="success", id="btn-generate", classes="hidden")
            yield Button("✗ Cancel", variant="error", id="btn-cancel")
    def _get_step_indicator(self) -> Text:
        steps = ["Prompt", "Model", "Settings", "Generate"]
        indicator = Text()
        for i, step in enumerate(steps, 1):
            if i == self.current_step:
                indicator.append(f"● {step}", style="bold cyan")
            elif i < self.current_step:
                indicator.append(f"✓ {step}", style="green")
            else:
                indicator.append(f"○ {step}", style="dim")
            if i < len(steps):
                indicator.append(" → ", style="dim")
        return indicator
    def on_button_pressed(self, event):
        if event.button.id == "btn-next":
            self.next_step()
        elif event.button.id == "btn-prev":
            self.prev_step()
        elif event.button.id == "btn-generate":
            self.start_generation()
    def next_step(self):
        if self.current_step < 4:
            self.query_one(f"#step-{self.current_step}").add_class("hidden")
            self.current_step += 1
            self.query_one(f"#step-{self.current_step}").remove_class("hidden")
            self.query_one("#btn-prev").disabled = False
            if self.current_step == 4:
                self.query_one("#btn-next").add_class("hidden")
                self.query_one("#btn-generate").remove_class("hidden")
                self.update_summary()
            self.query_one("#steps").update(self._get_step_indicator())
    def prev_step(self):
        if self.current_step > 1:
            self.query_one(f"#step-{self.current_step}").add_class("hidden")
            self.current_step -= 1
            self.query_one(f"#step-{self.current_step}").remove_class("hidden")
            if self.current_step == 1:
                self.query_one("#btn-prev").disabled = True
            if self.current_step < 4:
                self.query_one("#btn-next").remove_class("hidden")
                self.query_one("#btn-generate").add_class("hidden")
            self.query_one("#steps").update(self._get_step_indicator())
    def update_summary(self):
        prompt = self.query_one("#prompt-input").text or "No prompt entered"
        summary = Table.grid(padding=1)
        summary.add_column(style="yellow", width=12)
        summary.add_column()
        summary.add_row("Prompt:", prompt[:50] + "..." if len(prompt) > 50 else prompt)
        summary.add_row("Model:", self.selected_model.upper())
        summary.add_row("Format:", self.selected_format.upper())
        summary.add_row("Quality:", self.quality_preset.capitalize())
        self.query_one("#generation-summary").update(
            Panel(summary, title="📋 Generation Summary", border_style="cyan")
        )
    @work(exclusive=True)
    async def start_generation(self):
        progress_widget = self.query_one("#generation-progress", GenerationProgress)
        progress_widget.start_generation()
        self.query_one("#btn-generate").disabled = True
        await asyncio.sleep(10)
        self.query_one("#btn-generate").disabled = False