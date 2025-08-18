from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Button, Label
from textual.containers import Horizontal, Vertical, Container, Grid
from textual.widget import Widget
from textual.reactive import reactive
from textual import work, on
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from datetime import datetime
from typing import List, Dict, Optional
import psutil
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

class SystemMonitor(Widget):
    cpu_usage = reactive(0.0)
    ram_usage = reactive(0.0)
    gpu_usage = reactive(0.0)
    vram_usage = reactive(0.0)
    
    def on_mount(self):
        self.set_interval(2.0, self.update_stats)
        
    def update_stats(self):
        self.cpu_usage = psutil.cpu_percent(interval=0.1)
        self.ram_usage = psutil.virtual_memory().percent
        
        if TORCH_AVAILABLE:
            try:
                self.gpu_usage = torch.cuda.utilization()
                allocated = torch.cuda.memory_allocated() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.vram_usage = (allocated / total) * 100
            except:
                pass
        self.refresh()
        
    def render(self):
        monitor = Table.grid(padding=1)
        monitor.add_column(style="yellow", width=12)
        monitor.add_column(width=25)
        
        cpu_bar = self._create_bar(self.cpu_usage, "green" if self.cpu_usage < 80 else "red")
        monitor.add_row("CPU:", cpu_bar)
        
        ram_bar = self._create_bar(self.ram_usage, "cyan" if self.ram_usage < 80 else "orange")
        monitor.add_row("RAM:", ram_bar)
        
        if TORCH_AVAILABLE:
            gpu_bar = self._create_bar(self.gpu_usage, "green" if self.gpu_usage < 80 else "red")
            monitor.add_row("GPU:", gpu_bar)
            
            vram_bar = self._create_bar(self.vram_usage, "cyan" if self.vram_usage < 80 else "orange")
            monitor.add_row("VRAM:", vram_bar)
        else:
            monitor.add_row("GPU:", Text("Not Available", style="dim"))
            monitor.add_row("VRAM:", Text("Not Available", style="dim"))
            
        monitor.add_row("", "")
        
        cached_models = len(self.app.generation_worker.model_cache) if hasattr(self.app, 'generation_worker') else 0
        monitor.add_row("Cached:", Text(f"{cached_models} models", style="green"))
        
        mode = "Demo Mode" if self.app.is_demo_mode else "Production"
        style = "yellow" if self.app.is_demo_mode else "green"
        monitor.add_row("Mode:", Text(mode, style=style))
        
        return Panel(
            monitor,
            title="🖥️ System Monitor",
            border_style="green",
            expand=True
        )
        
    def _create_bar(self, value: float, color: str) -> Text:
        percentage = min(100, value)
        filled = int(percentage / 5)
        bar = Text()
        bar.append("█" * filled, style=color)
        bar.append("░" * (20 - filled), style="dim")
        bar.append(f" {percentage:.0f}%", style=color)
        return bar

class ModelStatusWidget(Widget):
    def on_mount(self):
        self.refresh()
        
    def render(self):
        models = self.app.get_available_models() if hasattr(self.app, 'get_available_models') else {}
        
        table = Table(show_header=True, header_style="bold cyan", box=None, expand=True)
        table.add_column("Model", style="cyan", width=15)
        table.add_column("Status", style="green", width=12)
        table.add_column("Size", style="yellow", width=8)
        table.add_column("VRAM", style="magenta", width=8)
        
        for model_id, info in models.items():
            if info['available']:
                status = "✓ Ready"
                status_style = "green"
            elif info['downloading']:
                status = "⬇ Loading"
                status_style = "yellow"
            elif info['cached']:
                status = "◉ Cached"
                status_style = "cyan"
            else:
                status = "✗ Not Ready"
                status_style = "red"
                
            size = f"{info['info'].get('size_gb', 0):.1f}GB"
            vram = f"{info['info'].get('min_vram_gb', 0):.0f}GB"
            
            if info['vram_compatible']:
                vram_display = Text(vram, style="green")
            else:
                vram_display = Text(vram, style="red")
                
            table.add_row(
                info['info'].get('name', model_id),
                Text(status, style=status_style),
                size,
                vram
            )
            
        if not models:
            table.add_row("Scanning...", "-", "-", "-")
            
        return Panel(
            table,
            title="🎯 Model Status",
            border_style="cyan",
            expand=True
        )

class RecentGenerationsWidget(Widget):
    def __init__(self):
        super().__init__()
        self.generations = []
        
    def add_generation(self, job):
        self.generations.insert(0, {
            "time": job.completed_at.strftime("%H:%M") if job.completed_at else "-",
            "prompt": job.prompt[:20] + "..." if len(job.prompt) > 20 else job.prompt,
            "model": job.model_name.split('-')[0].upper() if '-' in job.model_name else job.model_name.upper(),
            "status": "✓" if job.status.value == "completed" else "✗"
        })
        self.generations = self.generations[:5]
        self.refresh()
        
    def render(self):
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Time", style="dim", width=8)
        table.add_column("Prompt", style="white", width=20)
        table.add_column("Model", style="yellow", width=10)
        table.add_column("", style="green", width=3)
        
        if self.generations:
            for gen in self.generations:
                table.add_row(
                    gen["time"],
                    gen["prompt"],
                    gen["model"],
                    gen["status"]
                )
        else:
            table.add_row("-", "No generations yet", "-", "-")
            
        return Panel(
            table,
            title="📜 Recent Generations",
            border_style="blue",
            expand=True
        )

class ActivityFeed(Widget):
    def __init__(self):
        super().__init__()
        self.messages: List[tuple] = []
        self.max_messages = 8
        
    def add_message(self, message: str, style: str = "white"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.messages.append((f"[{timestamp}] {message}", style))
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
        self.refresh()
        
    def render(self):
        feed = Text()
        for message, style in self.messages:
            feed.append(message + "\n", style=style)
            
        if not feed:
            feed = Text("Waiting for activity...", style="dim")
            
        return Panel(
            feed,
            title="📡 Activity Feed",
            border_style="purple",
            expand=True,
            height=10
        )

class DashboardScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Label("🎨 DreamCAD - 3D Generation Studio", id="title")
        
        with Grid(id="dashboard-grid"):
            with Horizontal():
                yield SystemMonitor()
                yield ModelStatusWidget()
                
            yield RecentGenerationsWidget()
            yield ActivityFeed()
            
        with Horizontal(id="quick-actions"):
            yield Button("🚀 Generate", variant="primary", id="btn-generate")
            yield Button("📋 Queue", id="btn-queue")
            yield Button("📁 Gallery", id="btn-gallery")
            yield Button("⚙️ Settings", id="btn-settings")
            
    def on_mount(self):
        activity = self.query_one(ActivityFeed)
        activity.add_message("✓ DreamCAD initialized", "green")
        
        if self.app.is_demo_mode:
            activity.add_message("⚠ Running in demo mode", "yellow")
        else:
            models = self.app.get_available_models() if hasattr(self.app, 'get_available_models') else {}
            model_count = sum(1 for m in models.values() if m.get('available', False))
            if model_count > 0:
                activity.add_message(f"✓ {model_count} models available", "green")
            
    @on(Button.Pressed, "#btn-generate")
    def on_generate(self):
        self.app.switch_screen("wizard")
        
    @on(Button.Pressed, "#btn-queue")
    def on_queue(self):
        self.app.switch_screen("queue")
        
    @on(Button.Pressed, "#btn-gallery")
    def on_gallery(self):
        self.app.switch_screen("gallery")
        
    @on(Button.Pressed, "#btn-settings")
    def on_settings(self):
        self.app.switch_screen("settings")
        
    def on_job_complete(self, message):
        recent = self.query_one(RecentGenerationsWidget)
        recent.add_generation(message.job)
        
        activity = self.query_one(ActivityFeed)
        if message.job.status.value == "completed":
            activity.add_message(f"✓ Generation complete: {message.job.prompt[:30]}", "green")
        else:
            activity.add_message(f"✗ Generation failed: {message.job.error}", "red")