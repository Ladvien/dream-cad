import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import json
import subprocess
import platform

from dream_cad.models.factory import ModelFactory
from dream_cad.models.base import ModelConfig as BaseModelConfig

KNOWN_MODELS = {
    "triposr": {
        "name": "TripoSR",
        "repo": "stabilityai/TripoSR",
        "min_vram_gb": 4.0,
        "recommended_vram_gb": 6.0,
        "size_gb": 1.5,
        "speed": "0.5s",
        "quality": "Draft",
        "description": "Ultra-fast draft quality"
    },
    "stable-fast-3d": {
        "name": "Stable-Fast-3D", 
        "repo": "stabilityai/stable-fast-3d",
        "min_vram_gb": 6.0,
        "recommended_vram_gb": 8.0,
        "size_gb": 2.5,
        "speed": "3-5s",
        "quality": "Good",
        "description": "Fast game-ready assets"
    },
    "trellis": {
        "name": "TRELLIS",
        "repo": "microsoft/TRELLIS-image-large",
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 12.0,
        "size_gb": 4.5,
        "speed": "30-60s",
        "quality": "Excellent",
        "description": "High quality detailed models"
    },
    "hunyuan3d-mini": {
        "name": "Hunyuan3D",
        "repo": "tencent/Hunyuan3D-2mini",
        "min_vram_gb": 12.0,
        "recommended_vram_gb": 16.0,
        "size_gb": 4.5,
        "speed": "10-20s",
        "quality": "Production",
        "description": "Production quality with PBR"
    }
}

class ModelStatus:
    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.cached = False
        self.downloading = False
        self.error = None
        self.cache_size = 0
        self.vram_compatible = False
        self.info = KNOWN_MODELS.get(name, {})
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "cached": self.cached,
            "downloading": self.downloading,
            "error": self.error,
            "cache_size": self.cache_size,
            "vram_compatible": self.vram_compatible,
            "info": self.info
        }

class ModelDetector:
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config.config.cache_dir) / "hub"
        self.vram_gb = self._detect_vram()
        
    async def scan_models(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        
        for model_id, info in KNOWN_MODELS.items():
            status = ModelStatus(model_id)
            status.info = info
            
            status.cached = self._check_model_cached(info["repo"])
            
            if status.cached:
                status.cache_size = self._get_cache_size(info["repo"])
                status.available = self._test_model_load(model_id)
            
            status.vram_compatible = self.vram_gb >= info["min_vram_gb"]
            
            if not status.available and not status.cached:
                if self.config.config.models.auto_download and status.vram_compatible:
                    status.downloading = False
                else:
                    status.error = "Not downloaded"
                    
            results[model_id] = status.to_dict()
            
        return results
        
    def _check_model_cached(self, repo_id: str) -> bool:
        cache_name = f"models--{repo_id.replace('/', '--')}"
        model_cache_dir = self.cache_dir / cache_name
        
        if not model_cache_dir.exists():
            return False
            
        has_weights = any(
            model_cache_dir.rglob("*.safetensors") or
            model_cache_dir.rglob("*.bin") or
            model_cache_dir.rglob("*.pth")
        )
        
        return has_weights
        
    def _get_cache_size(self, repo_id: str) -> float:
        cache_name = f"models--{repo_id.replace('/', '--')}"
        model_cache_dir = self.cache_dir / cache_name
        
        if not model_cache_dir.exists():
            return 0.0
            
        total_size = 0
        for file in model_cache_dir.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
                
        return total_size / (1024 ** 3)
        
    def _test_model_load(self, model_id: str) -> bool:
        try:
            config = BaseModelConfig(
                model_name=model_id,
                output_dir=Path("outputs"),
                cache_dir=Path(self.config.config.cache_dir),
                device="cpu"
            )
            
            model = ModelFactory.create_model(model_id, config=config)
            return True
        except Exception:
            return False
            
    def _detect_vram(self) -> float:
        try:
            if platform.system() == "Windows":
                return self._detect_vram_windows()
            elif platform.system() == "Linux":
                return self._detect_vram_linux()
            elif platform.system() == "Darwin":
                return self._detect_vram_macos()
        except Exception:
            pass
        return 0.0
        
    def _detect_vram_linux(self) -> float:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                vram_mb = float(result.stdout.strip().split('\n')[0])
                return vram_mb / 1024.0
        except Exception:
            pass
        return 0.0
        
    def _detect_vram_windows(self) -> float:
        try:
            import wmi
            c = wmi.WMI()
            for gpu in c.Win32_VideoController():
                if gpu.AdapterRAM:
                    return gpu.AdapterRAM / (1024 ** 3)
        except Exception:
            pass
        return self._detect_vram_linux()
        
    def _detect_vram_macos(self) -> float:
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'VRAM' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            vram_str = parts[1].strip()
                            if 'GB' in vram_str:
                                return float(vram_str.replace('GB', '').strip())
                            elif 'MB' in vram_str:
                                return float(vram_str.replace('MB', '').strip()) / 1024.0
        except Exception:
            pass
        return 8.0
        
    async def download_model(self, model_id: str, progress_callback=None) -> bool:
        from dream_cad.models.async_download import download_model_async
        
        info = KNOWN_MODELS.get(model_id)
        if not info:
            return False
            
        try:
            await download_model_async(
                info["repo"],
                progress_callback=progress_callback,
                estimated_size_gb=info["size_gb"]
            )
            return True
        except Exception as e:
            return False