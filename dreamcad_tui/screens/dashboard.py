"""Interactive dashboard screen for DreamCAD TUI."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Static, Button, Label, ProgressBar, DataTable, Sparkline
from textual.widget import Widget
from textual.reactive import reactive
from textual import work
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from datetime import datetime
import asyncio
import random
from typing import List, Dict


class SystemMonitor(Widget):
    """Live system monitoring widget."""
    
    gpu_usage = reactive(0)
    vram_usage = reactive(0)
    temperature = reactive(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_timer = None
        
    def on_mount(self):
        """Start monitoring when mounted."""
        self.update_timer = self.set_interval(1.0, self.update_stats)
    
    def update_stats(self):
        """Update system statistics."""
        # Simulate stats for demo
        self.gpu_usage = min(100, max(0, self.gpu_usage + random.randint(-10, 15)))
        self.vram_usage = min(24, max(0, self.vram_usage + random.uniform(-2, 3)))
        self.temperature = min(85, max(30, self.temperature + random.randint(-3, 5)))
        self.refresh()
    
    def render(self):
        """Render the system monitor."""
        # Create monitoring display
        monitor = Table.grid(padding=1)
        monitor.add_column(style="yellow", width=12)
        monitor.add_column(width=20)
        
        # GPU usage bar
        gpu_bar = self._create_bar(self.gpu_usage, 100, "green" if self.gpu_usage < 80 else "red")
        monitor.add_row("GPU Usage:", gpu_bar)
        
        # VRAM usage
        vram_bar = self._create_bar(self.vram_usage, 24, "cyan" if self.vram_usage < 20 else "orange")
        monitor.add_row("VRAM:", f"{vram_bar} {self.vram_usage:.1f}/24 GB")
        
        # Temperature
        temp_color = "green" if self.temperature < 70 else "yellow" if self.temperature < 80 else "red"
        monitor.add_row("Temperature:", Text(f"{self.temperature}°C", style=temp_color))
        
        # Model status
        monitor.add_row("", "")
        monitor.add_row("Model:", Text("TripoSR Ready", style="green"))
        monitor.add_row("Queue:", Text("3 jobs pending", style="yellow"))
        
        return Panel(
            monitor,
            title="🖥️  System Monitor",
            border_style="green",
            expand=True
        )
    
    def _create_bar(self, value: float, max_value: float, color: str) -> Text:
        """Create a text progress bar."""
        percentage = min(100, value / max_value * 100)
        filled = int(percentage / 5)
        bar = Text()
        bar.append("█" * filled, style=color)
        bar.append("░" * (20 - filled), style="dim")
        bar.append(f" {percentage:.0f}%", style=color)
        return bar


class QuickActions(Widget):
    """Quick action buttons panel."""
    
    def render(self):
        """Render quick actions."""
        actions = Table.grid(padding=1)
        actions.add_column()
        
        # Create action items
        actions.add_row(Text("🚀 Quick Generate", style="bold cyan"))
        actions.add_row(Text("   Press 'G' or click below", style="dim"))
        actions.add_row("")
        actions.add_row(Text("📊 View Models", style="bold yellow"))
        actions.add_row(Text("   Press 'M' for comparison", style="dim"))
        actions.add_row("")
        actions.add_row(Text("📁 Browse Gallery", style="bold green"))
        actions.add_row(Text("   Press 'B' for past work", style="dim"))
        actions.add_row("")
        actions.add_row(Text("⚙️  Settings", style="bold magenta"))
        actions.add_row(Text("   Press 'S' to configure", style="dim"))
        
        return Panel(
            actions,
            title="⚡ Quick Actions",
            border_style="cyan",
            expand=True
        )


class RecentGenerations(Widget):
    """Display recent generations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generations = [
            {"time": "2 min ago", "prompt": "crystal sword", "model": "TripoSR", "status": "✓"},
            {"time": "15 min ago", "prompt": "fantasy cottage", "model": "Stable-Fast-3D", "status": "✓"},
            {"time": "1 hour ago", "prompt": "robot companion", "model": "TRELLIS", "status": "✓"},
            {"time": "3 hours ago", "prompt": "magical orb", "model": "TripoSR", "status": "⚠"},
        ]
    
    def render(self):
        """Render recent generations."""
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Time", style="dim", width=12)
        table.add_column("Prompt", style="white")
        table.add_column("Model", style="yellow")
        table.add_column("", style="green", width=3)
        
        for gen in self.generations:
            table.add_row(
                gen["time"],
                gen["prompt"][:20] + "..." if len(gen["prompt"]) > 20 else gen["prompt"],
                gen["model"],
                gen["status"]
            )
        
        return Panel(
            table,
            title="📜 Recent Generations",
            border_style="blue",
            expand=True
        )


class ActivityFeed(Widget):
    """Live activity feed."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: List[str] = []
        self.max_messages = 10
        
    def on_mount(self):
        """Start activity simulation."""
        self.set_interval(3.0, self.add_random_activity)
        
    def add_random_activity(self):
        """Add random activity message."""
        activities = [
            ("✓", "green", "Model loaded successfully"),
            ("ℹ", "cyan", "Queue processing started"),
            ("⚡", "yellow", "GPU acceleration enabled"),
            ("✓", "green", "Generation completed"),
            ("↓", "blue", "Downloading model weights"),
            ("⚠", "orange", "High VRAM usage detected"),
        ]
        
        icon, color, message = random.choice(activities)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.add_message(f"[{timestamp}] {icon} {message}", color)
    
    def add_message(self, message: str, style: str = "white"):
        """Add a message to the feed."""
        self.messages.append((message, style))
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
        self.refresh()
    
    def render(self):
        """Render the activity feed."""
        feed = Text()
        for message, style in self.messages:
            feed.append(message + "\n", style=style)
        
        return Panel(
            feed if feed else Text("No activity yet...", style="dim"),
            title="📡 Activity Feed",
            border_style="purple",
            expand=True,
            height=12
        )


class Dashboard(Container):
    """Main dashboard screen."""
    
    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        yield Label("🎨 DreamCAD - 3D Generation Suite", classes="title", id="dashboard-title")
        
        # Main content grid
        with Grid(id="dashboard-grid"):
            # Top row - System monitor and quick actions
            with Horizontal(classes="dashboard-row"):
                yield SystemMonitor(classes="dashboard-widget system-monitor")
                yield QuickActions(classes="dashboard-widget quick-actions")
            
            # Middle row - Recent generations
            yield RecentGenerations(classes="dashboard-widget recent-generations")
            
            # Bottom row - Activity feed
            yield ActivityFeed(classes="dashboard-widget activity-feed")
        
        # Action buttons
        with Horizontal(id="dashboard-buttons"):
            yield Button("🚀 Generate", variant="primary", id="btn-generate")
            yield Button("📊 Models", variant="default", id="btn-models")
            yield Button("📁 Gallery", variant="default", id="btn-gallery")
            yield Button("📋 Queue", variant="default", id="btn-queue")
            yield Button("⚙️ Settings", variant="default", id="btn-settings")