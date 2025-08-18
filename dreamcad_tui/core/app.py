from contextlib import contextmanager
from typing import Optional, Dict, Any
from pathlib import Path
import sys

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer
from textual.screen import Screen
from textual.containers import Container
from textual import work, on
from textual.message import Message

from .config import ConfigManager
from .logger import TUILogger
from .model_detector import ModelDetector
from .generation_worker import GenerationWorker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class ModelNotFoundError(Exception):
    pass

class OutOfMemoryError(Exception):
    pass

class ErrorMessage(Message):
    def __init__(self, error: str, severity: str = "error"):
        super().__init__()
        self.error = error
        self.severity = severity

class DreamCADApp(App):
    TITLE = "DreamCAD - 3D Generation Studio"
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #title {
        height: 3;
        content-align: center middle;
        text-style: bold;
        background: $primary;
    }
    
    #dashboard-grid {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
        padding: 1;
    }
    
    Button {
        margin: 0 1;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+d", "dashboard", "Dashboard"),
        Binding("ctrl+g", "generate", "Generate"),
        Binding("ctrl+u", "queue", "Queue"),
        Binding("ctrl+h", "help", "Help"),
        Binding("ctrl+s", "settings", "Settings"),
    ]
    
    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        self.logger = TUILogger()
        self.model_detector = ModelDetector(self.config)
        self.generation_worker = GenerationWorker(self)
        
        self.available_models: Dict[str, Any] = {}
        self.current_screen = "dashboard"
        self.is_demo_mode = False
        
    def on_mount(self) -> None:
        self.logger.info("DreamCAD TUI starting")
        self.detect_models()
        from ..screens.dashboard_new import DashboardScreen
        self.push_screen(DashboardScreen())
        
    @work(thread=True)
    async def detect_models(self) -> None:
        try:
            self.logger.info("Detecting available models...")
            self.available_models = await self.model_detector.scan_models()
            
            if not any(m['available'] for m in self.available_models.values()):
                self.is_demo_mode = True
                self.notify(
                    "No models found - running in demo mode",
                    severity="warning",
                    timeout=5
                )
            else:
                model_count = sum(1 for m in self.available_models.values() if m['available'])
                self.notify(
                    f"Found {model_count} available models",
                    severity="information",
                    timeout=3
                )
                
        except Exception as e:
            self.logger.error(f"Model detection failed: {e}")
            self.is_demo_mode = True
            self.notify(
                "Model detection failed - running in demo mode",
                severity="error",
                timeout=5
            )
    
    @contextmanager
    def error_boundary(self):
        try:
            yield
        except ModelNotFoundError as e:
            self.logger.warning(f"Model not found: {e}")
            self.notify(f"Model not available: {e}", severity="warning")
        except OutOfMemoryError as e:
            self.logger.error(f"Out of memory: {e}")
            self.notify("Out of memory - try a smaller model", severity="error")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.notify("An error occurred - check logs", severity="error")
            
    def action_quit(self) -> None:
        self.logger.info("DreamCAD TUI shutting down")
        self.config.save()
        self.exit()
        
    def action_dashboard(self) -> None:
        self.switch_screen("dashboard")
        
    def action_generate(self) -> None:
        self.switch_screen("wizard")
        
    def action_queue(self) -> None:
        self.switch_screen("queue")
        
    def action_help(self) -> None:
        self.switch_screen("help")
        
    def action_settings(self) -> None:
        self.switch_screen("settings")
        
    def switch_screen(self, screen_name: str) -> None:
        self.logger.info(f"Switching to {screen_name} screen")
        
        if screen_name == "dashboard":
            from ..screens.dashboard_new import DashboardScreen
            self.push_screen(DashboardScreen())
        elif screen_name == "wizard":
            from ..screens.wizard_new import WizardScreen
            self.push_screen(WizardScreen())
        elif screen_name == "queue":
            from ..screens.queue import QueueScreen
            self.push_screen(QueueScreen())
        elif screen_name == "gallery":
            from ..screens.gallery import GalleryScreen
            self.push_screen(GalleryScreen())
        elif screen_name == "settings":
            from ..screens.settings import SettingsScreen
            self.push_screen(SettingsScreen())
        elif screen_name == "help":
            from ..screens.help import HelpScreen
            self.push_screen(HelpScreen())
                
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(id="main")
        yield Footer()
        
    def on_error_message(self, message: ErrorMessage) -> None:
        self.notify(message.error, severity=message.severity)
        
    def get_available_models(self) -> Dict[str, Any]:
        return self.available_models
        
    def is_model_available(self, model_name: str) -> bool:
        return self.available_models.get(model_name, {}).get('available', False)
        
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        return self.available_models.get(model_name)