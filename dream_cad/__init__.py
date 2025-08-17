"""
Dream-CAD: Multi-model 3D generation system.
"""

__version__ = "0.2.0"

from .models.base import Model3D, ModelCapabilities, ModelConfig
from .models.factory import ModelFactory
from .models.registry import ModelRegistry

__all__ = [
    "Model3D",
    "ModelCapabilities", 
    "ModelConfig",
    "ModelFactory",
    "ModelRegistry",
]