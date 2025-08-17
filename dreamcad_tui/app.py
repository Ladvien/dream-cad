#!/usr/bin/env python3
"""Main DreamCAD TUI Application."""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, TabbedContent, TabPane, Label
from textual.binding import Binding
from textual.screen import Screen
from textual import events

from dreamcad_tui.screens.dashboard import Dashboard
from dreamcad_tui.screens.wizard import GenerationWizard
from dreamcad_tui.themes.dark import THEME, COLORS

# ASCII art logo
LOGO = """
╔══════════════════════════════════════════════════════════════════╗
║  ____                           ____    _    ____               ║
║ |  _ \ _ __ ___  __ _ _ __ ___ / ___|  / \  |  _ \              ║
║ | | | | '__/ _ \/ _` | '_ ` _ \ |     / _ \ | | | |             ║
║ | |_| | | |  __/ (_| | | | | | | |___/ ___ \| |_| |             ║
║ |____/|_|  \___|\__,_|_| |_| |_|\____/_/   \_\____/             ║
║                                                                  ║
║           Transform Text to 3D with AI Magic ✨                 ║
╚══════════════════════════════════════════════════════════════════╝
"""


class SplashScreen(Screen):
    """Splash screen with logo."""
    
    def compose(self) -> ComposeResult:
        """Compose splash screen."""
        yield Container(
            Label(LOGO, id="logo"),
            Label("Press any key to continue...", classes="subtitle"),
            id="splash-container"
        )
    
    def on_key(self, event: events.Key):
        """Dismiss splash on any key."""
        self.app.pop_screen()


class DreamCADApp(App):
    """Main DreamCAD TUI Application."""
    
    CSS = """
    /* Global styles */
    Screen {
        background: #0a0a0f;
    }
    
    #logo {
        color: $accent;
        text-align: center;
        padding: 2;
    }
    
    #splash-container {
        align: center middle;
    }
    
    .title {
        color: $accent;
        text-style: bold;
        text-align: center;
        padding: 1;
    }
    
    .subtitle {
        color: $text-muted;
        text-align: center;
        padding: 1;
    }
    
    /* Dashboard styles */
    #dashboard-grid {
        layout: grid;
        grid-size: 2 3;
        grid-gutter: 1;
        padding: 1;
    }
    
    .dashboard-widget {
        height: 100%;
        border: round $primary;
        background: $panel;
        padding: 1;
    }
    
    .system-monitor {
        column-span: 1;
        row-span: 1;
    }
    
    .quick-actions {
        column-span: 1;
        row-span: 1;
    }
    
    .recent-generations {
        column-span: 2;
        row-span: 1;
    }
    
    .activity-feed {
        column-span: 2;
        row-span: 1;
        height: 12;
    }
    
    #dashboard-buttons {
        dock: bottom;
        height: 3;
        align: center middle;
        background: $panel;
    }
    
    /* Wizard styles */
    #step-indicator {
        height: 3;
        align: center middle;
        padding: 1;
    }
    
    #wizard-content {
        height: 100%;
        padding: 1;
    }
    
    .wizard-step {
        height: 100%;
    }
    
    .wizard-step.hidden {
        display: none;
    }
    
    #wizard-buttons {
        dock: bottom;
        height: 3;
        align: center middle;
        background: $panel;
    }
    
    .hidden {
        display: none;
    }
    
    /* Button styles */
    Button {
        margin: 0 1;
        min-width: 12;
    }
    
    Button:hover {
        text-style: bold;
    }
    
    Button:focus {
        text-style: bold reverse;
    }
    
    /* Tab styles */
    TabbedContent {
        background: $panel;
    }
    
    TabPane {
        padding: 1;
    }
    
    /* Input styles */
    Input, TextArea {
        background: $panel-darken-1;
        border: tall $primary;
    }
    
    Input:focus, TextArea:focus {
        border: double $accent;
    }
    
    /* Progress bar */
    ProgressBar {
        color: $success;
        background: $panel;
    }
    
    /* Rule */
    Rule {
        color: $primary;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("d", "dashboard", "Dashboard"),
        Binding("g", "generate", "Generate"),
        Binding("m", "models", "Models"),
        Binding("b", "gallery", "Gallery"),
        Binding("s", "settings", "Settings"),
        Binding("h", "help", "Help"),
        Binding("ctrl+c", "cancel", "Cancel"),
        Binding("f1", "help", "Help", show=False),
        Binding("escape", "back", "Back"),
    ]
    
    def compose(self) -> ComposeResult:
        """Compose the main application."""
        yield Header(show_clock=True)
        
        with TabbedContent(initial="dashboard"):
            with TabPane("📊 Dashboard", id="dashboard"):
                yield Dashboard()
            
            with TabPane("🎨 Generate", id="generate"):
                yield GenerationWizard()
            
            with TabPane("🤖 Models", id="models"):
                yield Label("Model Selector - Coming Soon", classes="title")
            
            with TabPane("📁 Gallery", id="gallery"):
                yield Label("Gallery View - Coming Soon", classes="title")
            
            with TabPane("📋 Queue", id="queue"):
                yield Label("Queue Manager - Coming Soon", classes="title")
            
            with TabPane("⚙️ Settings", id="settings"):
                yield Label("Settings - Coming Soon", classes="title")
        
        yield Footer()
    
    def on_mount(self):
        """Show splash screen on mount."""
        self.push_screen(SplashScreen())
    
    def action_quit(self):
        """Quit the application."""
        self.exit()
    
    def action_dashboard(self):
        """Switch to dashboard."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "dashboard"
    
    def action_generate(self):
        """Switch to generation wizard."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "generate"
    
    def action_models(self):
        """Switch to models view."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "models"
    
    def action_gallery(self):
        """Switch to gallery."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "gallery"
    
    def action_settings(self):
        """Switch to settings."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "settings"
    
    def action_help(self):
        """Show help."""
        self.notify("Help: Use Tab to navigate, Enter to select, Q to quit", severity="information")
    
    def action_back(self):
        """Go back to dashboard."""
        self.action_dashboard()


def main():
    """Run the DreamCAD TUI."""
    app = DreamCADApp()
    app.run()


if __name__ == "__main__":
    main()