import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
from ..models.factory import ModelFactory
from ..models.registry import ModelRegistry
from ..models.base import ModelConfig
@dataclass
class PresetConfig:
    name: str
    description: str
    model: str
    quality_mode: str
    extra_params: Dict[str, Any]
class HardwareMonitor:
    def __init__(self):
        self.has_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
        self.gpu_name = self._get_gpu_name()
        self.total_vram = self._get_total_vram()
    def _get_gpu_name(self) -> str:
        if self.has_gpu:
            try:
                return torch.cuda.get_device_name(0)
            except:
                return "Unknown GPU"
        return "No GPU Available"
    def _get_total_vram(self) -> float:
        if self.has_gpu:
            try:
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                return 0.0
        return 0.0
    def get_available_vram(self) -> float:
        if self.has_gpu:
            try:
                used = torch.cuda.memory_allocated(0) / (1024**3)
                return self.total_vram - used
            except:
                return 0.0
        return 0.0
    def get_system_ram(self) -> Tuple[float, float]:
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                return mem.used / (1024**3), mem.total / (1024**3)
            except:
                pass
        return 0.0, 0.0
    def get_hardware_info(self) -> Dict[str, Any]:
        used_ram, total_ram = self.get_system_ram()
        return {
            "gpu_available": self.has_gpu,
            "gpu_name": self.gpu_name,
            "total_vram_gb": round(self.total_vram, 2),
            "available_vram_gb": round(self.get_available_vram(), 2),
            "total_ram_gb": round(total_ram, 2),
            "used_ram_gb": round(used_ram, 2),
            "cpu_count": psutil.cpu_count() if PSUTIL_AVAILABLE else 0,
            "cpu_percent": psutil.cpu_percent() if PSUTIL_AVAILABLE else 0
        }
class ModelSelectionUI:
    PRESETS = [
        PresetConfig(
            name="⚡ Ultra Fast",
            description="Fastest generation, lower quality (TripoSR)",
            model="triposr",
            quality_mode="fast",
            extra_params={"resolution": 256, "use_fp16": True}
        ),
        PresetConfig(
            name="⚖️ Balanced",
            description="Good balance of speed and quality (Stable-Fast-3D)",
            model="stable-fast-3d",
            quality_mode="balanced",
            extra_params={"polycount": 15000, "generate_pbr": True}
        ),
        PresetConfig(
            name="🎮 Game Ready",
            description="Production game assets with PBR (Hunyuan3D-Mini)",
            model="hunyuan3d-mini",
            quality_mode="production",
            extra_params={"polycount": 25000, "texture_resolution": 1024}
        ),
        PresetConfig(
            name="🎨 High Quality",
            description="Best quality, slower generation (TRELLIS)",
            model="trellis",
            quality_mode="hq",
            extra_params={"representation": "mesh", "mesh_resolution": 256}
        ),
        PresetConfig(
            name="🔮 NeRF/Gaussian",
            description="Advanced representations (TRELLIS)",
            model="trellis",
            quality_mode="balanced",
            extra_params={"representation": "gaussian_splatting"}
        )
    ]
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self.hardware_monitor = HardwareMonitor()
        self.factory = ModelFactory
        self.factory.set_registry(self.registry)
        self.current_config = None
        self.current_model = None
        self.config_history = []
        try:
            config_dir = Path.home() / ".dream_cad"
            config_dir.mkdir(parents=True, exist_ok=True)
            self.config_save_path = config_dir / "saved_configs.json"
        except (OSError, PermissionError):
            import tempfile
            self.config_save_path = Path(tempfile.gettempdir()) / "dream_cad_configs.json"
        self.load_saved_configs()
    def load_saved_configs(self) -> None:
        if self.config_save_path.exists():
            try:
                with open(self.config_save_path, "r") as f:
                    self.saved_configs = json.load(f)
            except:
                self.saved_configs = {}
        else:
            self.saved_configs = {}
    def save_config(self, name: str, config: Dict) -> str:
        self.saved_configs[name] = {
            "config": config,
            "timestamp": time.time(),
            "hardware": self.hardware_monitor.get_hardware_info()
        }
        try:
            with open(self.config_save_path, "w") as f:
                json.dump(self.saved_configs, f, indent=2)
            return f"✅ Configuration '{name}' saved successfully"
        except Exception as e:
            return f"❌ Failed to save configuration: {str(e)}"
    def get_model_recommendations(self) -> List[Tuple[str, str, float]]:
        available_vram = self.hardware_monitor.get_available_vram()
        recommendations = []
        for model_name in self.registry.list_models():
            caps = self.registry.get_capabilities(model_name)
            if caps:
                if available_vram >= caps.recommended_vram_gb:
                    score = 1.0
                    status = "✅ Recommended"
                elif available_vram >= caps.min_vram_gb:
                    score = 0.7
                    status = "⚠️ Possible"
                else:
                    score = 0.3
                    status = "❌ Not Recommended"
                recommendations.append((model_name, status, score))
        recommendations.sort(key=lambda x: x[2], reverse=True)
        return recommendations
    def format_hardware_info(self) -> str:
        info = self.hardware_monitor.get_hardware_info()
        lines = [
            "### 🖥️ System Hardware",
            f"**GPU:** {info['gpu_name']}",
            f"**VRAM:** {info['available_vram_gb']:.1f} / {info['total_vram_gb']:.1f} GB available",
            f"**RAM:** {info['used_ram_gb']:.1f} / {info['total_ram_gb']:.1f} GB used",
            f"**CPU:** {info['cpu_count']} cores ({info['cpu_percent']:.1f}% usage)"
        ]
        return "\n".join(lines)
    def format_model_capabilities(self, model_name: str) -> str:
        caps = self.registry.get_capabilities(model_name)
        if not caps:
            return "No information available"
        lines = [
            f"### 📊 {caps.name} Capabilities",
            f"**Min VRAM:** {caps.min_vram_gb} GB",
            f"**Recommended VRAM:** {caps.recommended_vram_gb} GB",
            f"**GPU Required:** {'Yes' if caps.requires_gpu else 'No'}",
            f"**Batch Support:** {'Yes' if caps.supports_batch else 'No'}",
            "",
            "**Generation Types:**",
            *[f"  • {gt.value}" for gt in caps.generation_types],
            "",
            "**Output Formats:**",
            *[f"  • {fmt.value.upper()}" for fmt in caps.output_formats],
            "",
            "**Estimated Times:**",
            *[f"  • {k}: {v:.1f}s" for k, v in caps.estimated_time_seconds.items()]
        ]
        return "\n".join(lines)
    def get_model_comparison_table(self) -> str:
        models = self.registry.list_models()
        if not models:
            return "No models available for comparison"
        headers = ["Model", "Min VRAM", "Speed", "Quality", "Formats", "Special Features"]
        rows = []
        for model_name in models:
            caps = self.registry.get_capabilities(model_name)
            if caps:
                avg_time = np.mean(list(caps.estimated_time_seconds.values()))
                if avg_time < 5:
                    speed = "⚡⚡⚡"
                elif avg_time < 20:
                    speed = "⚡⚡"
                else:
                    speed = "⚡"
                quality_map = {
                    "triposr": "⭐⭐",
                    "stable-fast-3d": "⭐⭐⭐",
                    "hunyuan3d-mini": "⭐⭐⭐⭐",
                    "trellis": "⭐⭐⭐⭐⭐",
                    "mvdream": "⭐⭐⭐⭐"
                }
                quality = quality_map.get(model_name, "⭐⭐⭐")
                formats = ", ".join([f.value.upper() for f in caps.output_formats[:3]])
                features = []
                if "hunyuan" in model_name:
                    features.append("PBR Materials")
                if "trellis" in model_name:
                    features.append("NeRF/Gaussian")
                if "triposr" in model_name:
                    features.append("Ultra Fast")
                if "stable-fast" in model_name:
                    features.append("Game Optimized")
                rows.append([
                    caps.name,
                    f"{caps.min_vram_gb}GB",
                    speed,
                    quality,
                    formats,
                    ", ".join(features)
                ])
        table_lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|"
        ]
        for row in rows:
            table_lines.append("| " + " | ".join(row) + " |")
        return "\n".join(table_lines)
    def update_model_info(self, model_name: str) -> Tuple[str, Dict, str]:
        if not model_name:
            return "", {}, ""
        caps_text = self.format_model_capabilities(model_name)
        recommendations = self.get_model_recommendations()
        rec_status = "Unknown"
        for name, status, _ in recommendations:
            if name == model_name:
                rec_status = status
                break
        params = self.get_model_specific_params(model_name)
        return caps_text, params, rec_status
    def get_model_specific_params(self, model_name: str) -> Dict:
        params = {}
        if model_name == "triposr":
            params = {
                "resolution": 512,
                "use_fp16": True,
                "remove_background": True
            }
        elif model_name == "stable-fast-3d":
            params = {
                "polycount": 5000,
                "texture_resolution": 1024,
                "generate_pbr": True,
                "delight": True,
                "target_engine": "universal"
            }
        elif model_name == "hunyuan3d-mini":
            params = {
                "polycount": 20000,
                "texture_resolution": 1024,
                "quality_mode": "balanced",
                "generate_pbr": True,
                "num_views": 4,
                "commercial_license": False
            }
        elif model_name == "trellis":
            params = {
                "representation": "mesh",
                "quality_mode": "balanced",
                "mesh_resolution": 256,
                "nerf_resolution": 128,
                "gaussian_count": 100000,
                "view_consistency": True
            }
        elif model_name == "mvdream":
            params = {
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
                "elevation_deg": 0,
                "camera_distance": 1.2
            }
        return params
    def apply_preset(self, preset_name: str) -> Tuple[str, Dict, str]:
        for preset in self.PRESETS:
            if preset.name == preset_name:
                model_name = preset.model
                params = self.get_model_specific_params(model_name)
                params.update(preset.extra_params)
                params["quality_mode"] = preset.quality_mode
                caps_text, _, rec_status = self.update_model_info(model_name)
                return model_name, params, f"Applied preset: {preset.description}"
        return "", {}, "Preset not found"
    def generate_with_config(
        self,
        prompt: str,
        model_name: str,
        params: Dict,
        output_format: str
    ) -> Tuple[Any, str]:
        if not prompt or not prompt.strip():
            return None, "❌ Please provide a valid prompt"
        if not model_name:
            return None, "❌ Please select a model"
        try:
            config = ModelConfig(
                model_name=model_name,
                output_dir=Path("outputs") / model_name,
                extra_params=params
            )
            model = self.factory.create_model(model_name, config)
            if not model._initialized:
                model.initialize()
            if prompt.startswith("image:"):
                image_path = prompt[6:].strip()
                from PIL import Image
                image = Image.open(image_path)
                result = model.generate_from_image(image)
            else:
                result = model.generate_from_text(prompt)
            if result.success:
                return result.output_path, f"✅ Generated successfully in {result.generation_time:.1f}s"
            else:
                return None, f"❌ Generation failed: {result.error_message}"
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    def create_interface(self) -> gr.Blocks:
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required for the UI")
        with gr.Blocks(title="Multi-Model 3D Generation") as interface:
            gr.Markdown("# 🎨 Multi-Model 3D Generation System")
            gr.Markdown("Select and configure different 3D generation models with hardware-aware recommendations.")
            with gr.Tabs():
                with gr.Tab("🚀 Generate"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Model Selection")
                            hardware_info = gr.Markdown(self.format_hardware_info())
                            model_choices = self.registry.list_models()
                            model_dropdown = gr.Dropdown(
                                choices=model_choices,
                                label="Select Model",
                                value=model_choices[0] if model_choices else None
                            )
                            rec_status = gr.Markdown("Select a model to see recommendation")
                            gr.Markdown("### Quick Presets")
                            preset_buttons = []
                            for preset in self.PRESETS:
                                btn = gr.Button(preset.name, variant="secondary")
                                preset_buttons.append(btn)
                            refresh_hw_btn = gr.Button("🔄 Refresh Hardware", variant="secondary")
                        with gr.Column(scale=2):
                            gr.Markdown("### Model Configuration")
                            model_caps = gr.Markdown("Select a model to see capabilities")
                            with gr.Accordion("Parameters", open=True):
                                quality_mode = gr.Radio(
                                    choices=["fast", "balanced", "production", "hq"],
                                    label="Quality Mode",
                                    value="balanced"
                                )
                                resolution = gr.Slider(
                                    minimum=128,
                                    maximum=2048,
                                    step=128,
                                    label="Resolution",
                                    value=512
                                )
                                extra_params = gr.JSON(
                                    label="Additional Parameters",
                                    value={}
                                )
                            output_format = gr.Dropdown(
                                choices=["obj", "ply", "glb", "stl"],
                                label="Output Format",
                                value="obj"
                            )
                            with gr.Row():
                                config_name = gr.Textbox(
                                    label="Configuration Name",
                                    placeholder="my_config"
                                )
                                save_config_btn = gr.Button("💾 Save Config", variant="secondary")
                                load_config_dropdown = gr.Dropdown(
                                    choices=list(self.saved_configs.keys()),
                                    label="Load Config"
                                )
                    with gr.Row():
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter text prompt or 'image:path/to/image.jpg'",
                            lines=2
                        )
                        generate_btn = gr.Button("🎨 Generate 3D Model", variant="primary")
                    with gr.Row():
                        output_model = gr.Model3D(label="Generated Model")
                        output_status = gr.Markdown("Ready to generate")
                with gr.Tab("📊 Compare Models"):
                    gr.Markdown("### Model Comparison Matrix")
                    comparison_table = gr.Markdown(self.get_model_comparison_table())
                    gr.Markdown("### 🎯 Hardware-Based Recommendations")
                    with gr.Row():
                        for model_name, status, score in self.get_model_recommendations()[:5]:
                            with gr.Column():
                                gr.Markdown(f"**{model_name}**")
                                gr.Markdown(f"{status}")
                                gr.Progress(value=score, label="Compatibility")
                with gr.Tab("📜 History"):
                    gr.Markdown("### Configuration History")
                    history_table = gr.Dataframe(
                        headers=["Timestamp", "Model", "Config", "Status"],
                        value=[],
                        interactive=False
                    )
                    clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary")
                with gr.Tab("📚 Documentation"):
                    gr.Markdown("""
                    1. **Check Hardware**: Review your system capabilities in the hardware info panel
                    2. **Select Model**: Choose a model based on recommendations
                    3. **Configure**: Adjust parameters or use a preset
                    4. **Generate**: Enter a prompt and click Generate
                    - **TripoSR**: Ultra-fast, good for prototyping
                    - **Stable-Fast-3D**: Game-optimized with PBR materials
                    - **Hunyuan3D-Mini**: Production quality, requires license for commercial use
                    - **TRELLIS**: Highest quality, supports NeRF and Gaussian Splatting
                    - **MVDream**: Original high-quality model
                    - Start with presets for optimal configurations
                    - Monitor VRAM usage to avoid out-of-memory errors
                    - Save successful configurations for reuse
                    - Use lower resolutions for faster iteration
    Args:
        host: Host to bind to
        port: Port to bind to
        share: Whether to create a public share link
        registry: Model registry to use