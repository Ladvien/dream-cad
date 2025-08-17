#!/usr/bin/env python
"""Simple test for Hunyuan3D without imports issues."""

import sys
import warnings

# Mock torch
class MockTorch:
    class Tensor:
        pass
    
    class cuda:
        @staticmethod
        def is_available():
            return False
        @staticmethod
        def memory_allocated():
            return 0
    
    @staticmethod
    def device(name):
        return name
    
    @staticmethod
    def is_tensor(obj):
        return False
    
    float16 = None

sys.modules['torch'] = MockTorch()

# Now import our modules
from dream_cad.models.hunyuan3d import Hunyuan3DMini, ProductionPBRMaterial, UVMapConfig
from dream_cad.models.base import ModelConfig, GenerationType, OutputFormat
from dream_cad.models.factory import ModelFactory
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile

def test_capabilities():
    """Test model capabilities."""
    config = ModelConfig(model_name="hunyuan3d-mini")
    model = Hunyuan3DMini(config)
    
    caps = model.capabilities
    assert caps.name == "Hunyuan3D-2-Mini"
    assert GenerationType.IMAGE_TO_3D in caps.generation_types
    assert GenerationType.MULTIVIEW_TO_3D in caps.generation_types
    assert caps.min_vram_gb == 12.0
    print("✓ Capabilities test passed")

def test_factory_registration():
    """Test factory registration."""
    assert "hunyuan3d-mini" in ModelFactory.list_models()
    model = ModelFactory.create_model("hunyuan3d-mini")
    assert isinstance(model, Hunyuan3DMini)
    print("✓ Factory registration test passed")

def test_polycount_control():
    """Test polycount control."""
    config = ModelConfig(
        model_name="hunyuan3d-mini",
        extra_params={"polycount": 30000}
    )
    model = Hunyuan3DMini(config)
    assert model.polycount == 30000
    
    # Test bounds
    config_low = ModelConfig(
        model_name="hunyuan3d-mini",
        extra_params={"polycount": 5000}
    )
    model_low = Hunyuan3DMini(config_low)
    assert model_low.polycount == 10000  # Clamped to minimum
    print("✓ Polycount control test passed")

def test_uv_config():
    """Test UV configuration."""
    config = UVMapConfig()
    assert config.method == "smart"
    assert config.island_margin == 0.02
    assert config.pack_islands == True
    print("✓ UV config test passed")

def test_pbr_material():
    """Test PBR material."""
    material = ProductionPBRMaterial(
        albedo=np.ones((1024, 1024, 3)),
        roughness_value=0.7
    )
    assert material.roughness_value == 0.7
    size = material.get_texture_size()
    assert size == (1024, 1024)
    print("✓ PBR material test passed")

def test_generation():
    """Test generation pipeline."""
    config = ModelConfig(
        model_name="hunyuan3d-mini",
        output_dir=Path(tempfile.mkdtemp())
    )
    model = Hunyuan3DMini(config)
    
    # Initialize mock components
    model.model = model._create_mock_model()
    model._initialize_texture_generator()
    model._initialize_uv_unwrapper()
    model._initialize_mesh_optimizer()
    model._initialized = True
    
    # Test image generation
    image = Image.new("RGB", (512, 512), color=(128, 128, 128))
    result = model.generate_from_image(image)
    
    if not result.success:
        print(f"Generation failed: {result.error_message}")
    
    assert result.success
    assert result.output_path is not None
    # Check file exists
    if not result.output_path.exists():
        print(f"Output path doesn't exist: {result.output_path}")
        # Try to list parent directory
        if result.output_path.parent.exists():
            print(f"Parent directory contents: {list(result.output_path.parent.iterdir())}")
    assert result.output_path.exists()
    assert "uvs" in result.mesh_data
    print("✓ Generation test passed")

def test_multiview():
    """Test multi-view generation."""
    config = ModelConfig(
        model_name="hunyuan3d-mini",
        output_dir=Path(tempfile.mkdtemp())
    )
    model = Hunyuan3DMini(config)
    model.model = model._create_mock_model()
    model._initialize_texture_generator()
    model._initialize_uv_unwrapper() 
    model._initialize_mesh_optimizer()
    model._initialized = True
    
    # Test with multiple views
    images = [
        Image.new("RGB", (256, 256), color=(100, 0, 0)),
        Image.new("RGB", (256, 256), color=(0, 100, 0)),
        Image.new("RGB", (256, 256), color=(0, 0, 100))
    ]
    
    result = model.generate_from_multiview(images)
    assert result.success
    assert result.metadata["num_views"] == 3
    print("✓ Multi-view test passed")

def test_commercial_license_warning():
    """Test commercial license warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = ModelConfig(
            model_name="hunyuan3d-mini",
            extra_params={"commercial_license": False}
        )
        model = Hunyuan3DMini(config)
        
        assert len(w) == 1
        assert "commercial license" in str(w[0].message).lower()
    print("✓ Commercial license warning test passed")

if __name__ == "__main__":
    print("Running Hunyuan3D-2 Mini tests...")
    print("-" * 40)
    
    test_capabilities()
    test_factory_registration()
    test_polycount_control()
    test_uv_config()
    test_pbr_material()
    test_generation()
    test_multiview()
    test_commercial_license_warning()
    
    print("-" * 40)
    print("All tests passed! ✅")