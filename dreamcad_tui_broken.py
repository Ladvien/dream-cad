from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Button, Input, Label, Static, 
    Select, RichLog, ProgressBar
)
from textual.binding import Binding
from textual import work, on
from rich.text import Text
from rich.panel import Panel
from datetime import datetime
from pathlib import Path
import asyncio
import sys
import os
import time
from typing import Dict, Any, Optional
sys.path.insert(0, str(Path(__file__).parent))
from dream_cad.models.async_download import download_model_async
from dream_cad.models.factory import ModelFactory
from dream_cad.models.base import ModelConfig, OutputFormat
MODELS = {
    "TripoSR": {
        "name": "triposr",
        "repo": "stabilityai/TripoSR",
        "description": "Fast draft quality (0.5s)",
        "vram": "4GB",
        "size_gb": 1.5,
        "quality": "⭐⭐",
        "speed": "⚡⚡⚡⚡⚡"
    },
    "Stable-Fast-3D": {
        "name": "stable-fast-3d",
        "repo": "stabilityai/stable-fast-3d", 
        "description": "Game-ready assets (3-5s)",
        "vram": "6GB",
        "size_gb": 2.5,
        "quality": "⭐⭐⭐",
        "speed": "⚡⚡⚡⚡"
    },
    "TRELLIS": {
        "name": "trellis",
        "repo": "microsoft/TRELLIS-image-large",
        "description": "High quality (30-60s)",
        "vram": "8GB", 
        "size_gb": 4.5,
        "quality": "⭐⭐⭐⭐⭐",
        "speed": "⚡⚡"
    },
    "Hunyuan3D": {
        "name": "hunyuan3d-mini",
        "repo": "tencent/Hunyuan3D-2mini",
        "description": "Production quality (10-20s)",
        "vram": "12GB",
        "size_gb": 4.5,
        "quality": "⭐⭐⭐⭐",
        "speed": "⚡⚡⚡"
    }
}
class DreamCADCLI(App):
    CSS = """
    Screen {
        background: $surface;
    }
    #sidebar {
        width: 25;
        height: 100%;
        dock: left;
        background: $panel;
    }
    #main {
        background: $surface;
    }
    #output {
        height: 100%;
        border: solid $primary;
    }
        self.update_model_info()
        self.log_message("✨ Welcome to DreamCAD Production CLI!")
        self.log_message("📍 This version shows real progress and actually generates 3D models")
        self.log_message("🚀 Select a model and enter a prompt to begin")
    def update_model_info(self):
        model = MODELS[self.current_model]
        info_text = (
            f"[bold]{self.current_model}[/bold]\n"
            f"{model['description']}\n"
            f"Quality: {model['quality']}  Speed: {model['speed']}\n"
            f"VRAM: {model['vram']}  Download: {model['size_gb']}GB"
        )
        self.query_one("#model-info").update(Panel(info_text, border_style="cyan"))
    def log_message(self, message: str, style: str = ""):
        log_widget = self.query_one("#log", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        if style:
            log_widget.write(f"[dim]{timestamp}[/dim] [{style}]{message}[/{style}]")
        else:
            log_widget.write(f"[dim]{timestamp}[/dim] {message}")
    @on(Select.Changed, "#model-select")
    def on_model_changed(self, event: Select.Changed):
        self.current_model = event.value
        self.update_model_info()
        self.log_message(f"Selected model: {self.current_model}", "cyan")
    @on(Button.Pressed, "#generate-btn")
    def on_generate(self):
        self.action_generate()
    @on(Button.Pressed, "#clear-btn")
    def on_clear(self):
        self.action_clear()
    @on(Button.Pressed, "#open-btn")
    def on_open_output(self):
        import subprocess
        import platform
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        if platform.system() == "Darwin":
            subprocess.run(["open", str(output_dir)])
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", str(output_dir)])
        elif platform.system() == "Windows":
            subprocess.run(["explorer", str(output_dir)])
        self.log_message(f"Opened output directory: {output_dir}", "green")
    @work(exclusive=True)
    async def action_generate(self):
        if self.generating:
            self.log_message("Already generating! Please wait...", "yellow")
            return
        prompt_input = self.query_one("#prompt-input")
        prompt = prompt_input.value.strip()
        if not prompt:
            self.log_message("Please enter a prompt!", "red")
            return
        self.generating = True
        gen_btn = self.query_one("#generate-btn")
        gen_btn.disabled = True
        try:
            self.log_message(f"\n[bold]Starting generation:[/bold] {prompt}", "cyan")
            model_info = MODELS[self.current_model]
            self.log_message("Checking model cache...", "dim")
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            cache_name = f"models--{model_info['repo'].replace('/', '--')}"
            model_cache_dir = cache_dir / cache_name
            is_cached = model_cache_dir.exists() and self._check_model_complete(model_cache_dir)
            if not is_cached:
                self.log_message(f"Model not cached. Downloading {model_info['size_gb']}GB...", "yellow")
                progress_container = self.query_one("#progress-container")
                progress_container.remove_class("hidden")
                progress_bar = self.query_one("#progress", ProgressBar)
                progress_label = self.query_one("#progress-label")
                async def progress_callback(msg: str):
                    self.log_message(msg, "cyan")
                    if "Progress:" in msg and "%" in msg:
                        try:
                            percent_str = msg.split("%")[0].split(":")[-1].strip()
                            percent = float(percent_str)
                            progress_bar.update(progress=percent)
                            if "MB/s" in msg and "ETA:" in msg:
                                speed = msg.split("MB/s")[0].split("-")[-1].strip() + "MB/s"
                                eta = msg.split("ETA:")[-1].strip()
                                progress_label.update(f"Downloading: {percent:.1f}% - {speed} - ETA: {eta}")
                        except:
                            pass
                try:
                    await download_model_async(
                        model_info['repo'],
                        progress_callback=progress_callback,
                        estimated_size_gb=model_info['size_gb']
                    )
                    self.log_message("✅ Model downloaded successfully!", "green")
                except Exception as e:
                    self.log_message(f"❌ Download failed: {e}", "red")
                    return
                finally:
                    progress_container.add_class("hidden")
            else:
                self.log_message("✅ Model already cached and ready!", "green")
            self.log_message("Loading model into memory...", "dim")
            def model_progress(msg: str):
                pass
            config = ModelConfig(
                model_name=model_info['name'],
                output_dir=Path("outputs"),
                cache_dir=cache_dir,
                progress_callback=model_progress
            )
            loop = asyncio.get_event_loop()
            def load_model():
                try:
                    return ModelFactory.create_model(model_info['name'], config=config)
                except Exception as e:
                    return f"Error: {e}"
            result = await loop.run_in_executor(None, load_model)
            if isinstance(result, str) and result.startswith("Error"):
                self.log_message(f"❌ Failed to load model: {result}", "red")
                return
            self.model_instance = result
            self.log_message("✅ Model loaded successfully!", "green")
            self.log_message("Generating 3D model...", "yellow")
            self.log_message(f"Parameters: {self.current_model}, prompt='{prompt}'", "dim")
            def generate():
                try:
                    return self.model_instance.generate_from_text(
                        prompt=prompt,
                        output_format=OutputFormat.OBJ
                    )
                except Exception as e:
                    return f"Error: {e}"
            start_time = time.time()
            result = await loop.run_in_executor(None, generate)
            elapsed = time.time() - start_time
            if isinstance(result, str) and result.startswith("Error"):
                self.log_message(f"❌ Generation failed: {result}", "red")
                return
            if result and hasattr(result, 'mesh_path'):
                output_path = result.mesh_path
                self.log_message(f"✅ Success! Generated 3D model in {elapsed:.1f}s", "green")
                self.log_message(f"📁 Saved to: {output_path}", "green")
                self.log_message(f"💾 File size: {self._get_file_size(output_path)}", "dim")
            else:
                self.log_message("⚠️ Generation completed but no output file", "yellow")
            if self.model_instance:
                self.model_instance.cleanup()
                self.model_instance = None
        except Exception as e:
            self.log_message(f"❌ Unexpected error: {e}", "red")
            import traceback
            self.log_message(traceback.format_exc(), "dim")
        finally:
            self.generating = False
            gen_btn.disabled = False
            prompt_input.value = ""
    def _check_model_complete(self, cache_dir: Path) -> bool:
        has_safetensors = any(cache_dir.rglob("*.safetensors"))
        has_bin = any(cache_dir.rglob("*.bin"))
        has_pth = any(cache_dir.rglob("*.pth"))
        return has_safetensors or has_bin or has_pth
    def _get_file_size(self, path: Path) -> str:
        if not path.exists():
            return "0 bytes"
        size = path.stat().st_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"
    def action_clear(self):
        log_widget = self.query_one("#log", RichLog)
        log_widget.clear()
        self.log_message("Log cleared", "dim")
def main():
    print("\n" + "="*60)
    print("🎨 DreamCAD Production CLI")
    print("="*60)
    print("\n✨ Features:")
    print("  • Real download progress with speed and ETA")
    print("  • Actual 3D model generation (no mocks)")
    print("  • Clear status updates at every step")
    print("  • Support for 4 production models")
    print("  • No UI freezing or hanging")
    print("\n🚀 Starting CLI...\n")
    app = DreamCADCLI()
    app.run()
if __name__ == "__main__":
    main()