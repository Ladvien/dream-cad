import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field

@dataclass
class UIConfig:
    theme: str = "dark"
    show_gpu_monitor: bool = True
    notification_sound: bool = True
    auto_save_queue: bool = True
    max_history_items: int = 100
    default_output_format: str = "obj"
    
@dataclass
class ModelConfig:
    preferred_model: str = "triposr"
    auto_download: bool = True
    cache_models: bool = True
    max_cached_models: int = 2
    fallback_to_mock: bool = True
    
@dataclass
class GenerationConfig:
    default_quality: str = "balanced"
    max_queue_size: int = 50
    auto_process_queue: bool = True
    save_metadata: bool = True
    thumbnail_size: int = 256
    
@dataclass
class AppConfig:
    ui: UIConfig = field(default_factory=UIConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output_dir: str = "outputs"
    cache_dir: str = ""
    
class ConfigManager:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".dreamcad" / "config.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = self.load()
        
    def load(self) -> AppConfig:
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    return AppConfig(
                        ui=UIConfig(**data.get('ui', {})),
                        models=ModelConfig(**data.get('models', {})),
                        generation=GenerationConfig(**data.get('generation', {})),
                        output_dir=data.get('output_dir', 'outputs'),
                        cache_dir=data.get('cache_dir', '')
                    )
            except Exception:
                pass
        
        config = AppConfig()
        if not config.cache_dir:
            config.cache_dir = str(Path.home() / ".cache" / "huggingface")
        return config
        
    def save(self) -> None:
        data = {
            'ui': asdict(self.config.ui),
            'models': asdict(self.config.models),
            'generation': asdict(self.config.generation),
            'output_dir': self.config.output_dir,
            'cache_dir': self.config.cache_dir
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split('.')
        obj = self.config
        
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default
        return obj
        
    def set(self, key: str, value: Any) -> None:
        parts = key.split('.')
        obj = self.config
        
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return
                
        if hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], value)
            self.save()