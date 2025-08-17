"""Model card widget for displaying model information."""

from textual.widget import Widget
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Group
from rich.align import Align
from typing import Dict, Any


class ModelCard(Widget):
    """Interactive model card with hover effects."""
    
    selected = reactive(False)
    hovered = reactive(False)
    
    def __init__(self, name: str, info: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.info = info
        self.can_focus = True
        
    def render(self):
        """Render the model card."""
        # Determine style based on state
        if self.selected:
            border_style = "bold green"
            title_style = "bold green"
            badge = "✓ SELECTED"
            badge_style = "bold green"
        elif self.hovered:
            border_style = "bold cyan"
            title_style = "bold cyan"
            badge = "→ HOVER"
            badge_style = "cyan"
        else:
            border_style = "dim"
            title_style = "bold white"
            badge = "AVAILABLE"
            badge_style = "dim"
        
        # Create content
        content = []
        
        # Model name with badge
        title_text = Text()
        title_text.append(f"{self.name.upper()} ", style=title_style)
        title_text.append(f"[{badge}]", style=badge_style)
        content.append(Align.center(title_text))
        content.append("")
        
        # Create specs table
        specs = Table.grid(padding=0)
        specs.add_column(style="yellow", width=8)
        specs.add_column()
        
        # Add specifications
        specs.add_row("VRAM:", f"{self.info.get('min_vram_gb', 'N/A')}GB")
        specs.add_row("Speed:", f"~{self.info.get('generation_time_seconds', 'N/A')}s")
        specs.add_row("Quality:", self._get_quality_stars())
        
        content.append(specs)
        content.append("")
        
        # Supported formats
        formats = self.info.get('supported_formats', [])
        if formats:
            format_text = Text("Formats: ", style="blue")
            format_text.append(", ".join(formats[:3]).upper(), style="cyan")
            content.append(Align.center(format_text))
        
        # Create panel
        return Panel(
            Group(*content),
            border_style=border_style,
            expand=True,
            height=9
        )
    
    def _get_quality_stars(self) -> str:
        """Get quality rating as stars."""
        # Determine quality based on model characteristics
        vram = self.info.get('min_vram_gb', 0)
        if vram >= 12:
            return "★★★★★"
        elif vram >= 8:
            return "★★★★☆"
        elif vram >= 6:
            return "★★★☆☆"
        else:
            return "★★☆☆☆"
    
    def on_mouse_move(self, event):
        """Handle mouse hover."""
        self.hovered = True
    
    def on_leave(self, event):
        """Handle mouse leave."""
        self.hovered = False
    
    def on_click(self, event):
        """Handle click to select."""
        self.selected = not self.selected
        if self.selected:
            # Notify parent to deselect others
            self.post_message(ModelSelected(self.name))


class ModelSelected:
    """Message sent when a model is selected."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name