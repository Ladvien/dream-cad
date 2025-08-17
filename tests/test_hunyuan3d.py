"""
Tests for Hunyuan3D-2 Mini model integration.
"""

import pytest
import tempfile
import json
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from dream_cad.models.base import (
    ModelConfig, GenerationType, OutputFormat
)
from dream_cad.models.hunyuan3d import (
    Hunyuan3DMini, ProductionPBRMaterial, UVMapConfig
)
from dream_cad.models.factory import ModelFactory


class TestHunyuan3DCapabilities:
    """Test Hunyuan3D-2 Mini capabilities and configuration."""
    
    def test_capabilities(self):
        """Test model capabilities."""
        config = ModelConfig(model_name="hunyuan3d-mini")
        model = Hunyuan3DMini(config)
        
        caps = model.capabilities
        assert caps.name == "Hunyuan3D-2-Mini"
        assert GenerationType.IMAGE_TO_3D in caps.generation_types
        assert GenerationType.TEXT_TO_3D in caps.generation_types
        assert GenerationType.MULTIVIEW_TO_3D in caps.generation_types
        assert OutputFormat.GLB in caps.output_formats
        assert caps.min_vram_gb == 12.0  # Mini version
        assert caps.recommended_vram_gb == 16.0
        assert caps.model_size_gb == 8.0
    
    def test_factory_registration(self):
        """Test factory registration."""
        assert "hunyuan3d-mini" in ModelFactory.list_models()
        
        model = ModelFactory.create_model("hunyuan3d-mini")
        assert isinstance(model, Hunyuan3DMini)
    
    def test_polycount_configuration(self):
        """Test polycount control configuration."""
        config = ModelConfig(
            model_name="hunyuan3d-mini",
            extra_params={"polycount": 30000}
        )
        model = Hunyuan3DMini(config)
        
        assert model.polycount == 30000
        
        # Test bounds
        config_low = ModelConfig(
            model_name="hunyuan3d-mini",
            extra_params={"polycount": 5000}  # Below minimum
        )
        model_low = Hunyuan3DMini(config_low)
        assert model_low.polycount == 10000  # Clamped to minimum
        
        config_high = ModelConfig(
            model_name="hunyuan3d-mini",
            extra_params={"polycount": 60000}  # Above maximum
        )
        model_high = Hunyuan3DMini(config_high)
        assert model_high.polycount == 50000  # Clamped to maximum
    
    def test_quality_modes(self):
        """Test quality mode configuration."""
        for mode in ["fast", "balanced", "production"]:
            config = ModelConfig(
                model_name="hunyuan3d-mini",
                extra_params={"quality_mode": mode}
            )
            model = Hunyuan3DMini(config)
            assert model.quality_mode == mode
    
    def test_commercial_license_warning(self):
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


class TestProductionPBRMaterial:
    """Test production PBR material dataclass."""
    
    def test_pbr_material_creation(self):
        """Test PBR material creation."""
        material = ProductionPBRMaterial(
            albedo=np.ones((1024, 1024, 3)),
            normal=np.ones((1024, 1024, 3)),
            roughness=np.ones((1024, 1024)),
            metallic=np.zeros((1024, 1024)),
            roughness_value=0.7,
            metallic_value=0.2,
            ior=1.45
        )
        
        assert material.albedo is not None
        assert material.roughness_value == 0.7
        assert material.metallic_value == 0.2
        assert material.ior == 1.45
    
    def test_texture_size(self):
        """Test texture size detection."""
        material = ProductionPBRMaterial(
            albedo=np.ones((2048, 2048, 3))
        )
        
        size = material.get_texture_size()
        assert size == (2048, 2048)
        
        # Test default
        empty_material = ProductionPBRMaterial()
        default_size = empty_material.get_texture_size()
        assert default_size == (1024, 1024)


class TestUVMapConfig:
    """Test UV map configuration."""
    
    def test_uv_config_defaults(self):
        """Test UV configuration defaults."""
        config = UVMapConfig()
        
        assert config.method == "smart"
        assert config.island_margin == 0.02
        assert config.angle_limit == 66.0
        assert config.correct_aspect == True
        assert config.pack_islands == True
    
    def test_uv_config_custom(self):
        """Test custom UV configuration."""
        config = UVMapConfig(
            method="angle_based",
            island_margin=0.05,
            angle_limit=45.0,
            pack_islands=False
        )
        
        assert config.method == "angle_based"
        assert config.island_margin == 0.05
        assert config.angle_limit == 45.0
        assert config.pack_islands == False


class TestHunyuan3DInitialization:
    """Test model initialization."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_initialization(self):
        """Test model initialization."""
        config = ModelConfig(model_name="hunyuan3d-mini")
        model = Hunyuan3DMini(config)
        
        model.initialize()
        assert model._initialized
        assert model.model is not None
        assert model.texture_generator is not None
        assert model.uv_unwrapper is not None
        assert model.mesh_optimizer is not None
    
    def test_mock_model_creation(self):
        """Test mock model creation."""
        config = ModelConfig(model_name="hunyuan3d-mini")
        model = Hunyuan3DMini(config)
        
        mock_model = model._create_mock_model()
        assert mock_model is not None
        
        # Test mock generation
        result = mock_model.generate(None)
        assert "vertices" in result
        assert "faces" in result
        assert "uvs" in result
        assert "normals" in result
        
        # Test multi-view
        images = [None, None, None]
        mv_result = mock_model.generate_multiview(images)
        assert "vertices" in mv_result
    
    @patch('huggingface_hub.snapshot_download')
    def test_model_download(self, mock_download):
        """Test model download handling."""
        mock_download.return_value = "/tmp/mock_model"
        
        config = ModelConfig(model_name="hunyuan3d-mini")
        model = Hunyuan3DMini(config)
        
        model_path = model._download_model()
        assert model_path is not None


class TestHunyuan3DGeneration:
    """Test production-quality mesh generation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = ModelConfig(
            model_name="hunyuan3d-mini",
            output_dir=Path(tempfile.mkdtemp()),
            extra_params={
                "polycount": 20000,
                "texture_resolution": 1024,
                "quality_mode": "balanced"
            }
        )
        self.model = Hunyuan3DMini(self.config)
        # Use mock components
        self.model.model = self.model._create_mock_model()
        self.model._initialize_texture_generator()
        self.model._initialize_uv_unwrapper()
        self.model._initialize_mesh_optimizer()
        self.model._initialized = True
    
    def test_image_preprocessing(self):
        """Test image preprocessing for production."""
        image = Image.new("RGB", (512, 512), color=(255, 0, 0))
        
        processed = self.model._preprocess_image(image)
        
        assert processed.size == (1024, 1024)  # High resolution
        assert processed.mode == "RGB"
    
    def test_generate_from_image(self):
        """Test production mesh generation from image."""
        image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        
        result = self.model.generate_from_image(image)
        
        assert result.success
        assert result.output_path is not None
        assert result.output_path.exists()
        assert result.mesh_data is not None
        assert "uvs" in result.mesh_data
        assert result.metadata["has_pbr"] == True
        assert result.metadata["polycount"] > 0
        assert result.metadata["texture_resolution"] == 1024
        assert result.metadata["commercial_license"] == False
    
    def test_generate_without_pbr(self):
        """Test generation without PBR materials."""
        self.model.generate_pbr = False
        image = Image.new("RGB", (256, 256))
        
        result = self.model.generate_from_image(image)
        
        assert result.success
        assert result.metadata["has_pbr"] == False
    
    def test_generate_from_text(self):
        """Test generation from text prompt."""
        result = self.model.generate_from_text("a detailed character model")
        
        assert result.success
        assert result.output_path is not None
        assert result.metadata["prompt"] == "a detailed character model"
    
    def test_multiview_generation(self):
        """Test multi-view generation."""
        images = [
            Image.new("RGB", (512, 512), color=(100, 0, 0)),
            Image.new("RGB", (512, 512), color=(0, 100, 0)),
            Image.new("RGB", (512, 512), color=(0, 0, 100)),
            Image.new("RGB", (512, 512), color=(100, 100, 0))
        ]
        
        result = self.model.generate_from_multiview(images)
        
        assert result.success
        assert result.output_path is not None
        assert result.metadata["method"] == "multiview"
        assert result.metadata["num_views"] == 4
        assert result.metadata["fused"] == True
    
    def test_multiview_validation(self):
        """Test multi-view input validation."""
        # Test with too few views
        result = self.model.generate_from_multiview([Image.new("RGB", (256, 256))])
        assert not result.success
        assert "at least 2 views" in result.error_message.lower()
        
        # Test with valid number
        images = [Image.new("RGB", (256, 256)) for _ in range(3)]
        result = self.model.generate_from_multiview(images)
        assert result.success


class TestUVUnwrapping:
    """Test professional UV unwrapping."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = ModelConfig(model_name="hunyuan3d-mini")
        self.model = Hunyuan3DMini(self.config)
        self.model._initialize_uv_unwrapper()
    
    def test_uv_unwrapping(self):
        """Test UV unwrapping with different methods."""
        mesh_data = {
            "vertices": np.random.randn(1000, 3),
            "faces": np.random.randint(0, 1000, (500, 3))
        }
        
        uvs = self.model.uv_unwrapper.unwrap(mesh_data)
        
        assert uvs.shape == (1000, 2)
        assert uvs.min() >= 0
        assert uvs.max() <= 1
    
    def test_smart_uv_projection(self):
        """Test smart UV projection."""
        self.model.uv_config.method = "smart"
        self.model._initialize_uv_unwrapper()
        
        mesh_data = {
            "vertices": np.array([
                [1, 0, 0],  # X dominant
                [0, 1, 0],  # Y dominant
                [0, 0, 1]   # Z dominant
            ])
        }
        
        uvs = self.model.uv_unwrapper.unwrap(mesh_data)
        assert uvs.shape == (3, 2)


class TestMeshOptimization:
    """Test mesh optimization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = ModelConfig(model_name="hunyuan3d-mini")
        self.model = Hunyuan3DMini(self.config)
        self.model._initialize_mesh_optimizer()
    
    def test_mesh_optimization(self):
        """Test mesh topology optimization."""
        # Create high-poly mesh
        mesh_data = {
            "vertices": np.random.randn(10000, 3),
            "faces": np.random.randint(0, 10000, (30000, 3)),
            "uvs": np.random.rand(10000, 2)
        }
        
        # Optimize to target polycount
        optimized = self.model.mesh_optimizer.optimize(
            mesh_data,
            target_polycount=10000,
            preserve_uvs=True
        )
        
        assert len(optimized["faces"]) <= 10000
        assert len(optimized["vertices"]) <= len(mesh_data["vertices"])
        assert "uvs" in optimized  # UVs preserved
    
    def test_optimization_preserves_quality(self):
        """Test that optimization preserves mesh quality."""
        # Small mesh that doesn't need optimization
        mesh_data = {
            "vertices": np.random.randn(100, 3),
            "faces": np.random.randint(0, 100, (50, 3))
        }
        
        optimized = self.model.mesh_optimizer.optimize(
            mesh_data,
            target_polycount=1000,  # Higher than current
            preserve_uvs=False
        )
        
        # Should return unchanged
        assert len(optimized["faces"]) == len(mesh_data["faces"])


class TestProductionAssetSaving:
    """Test production asset saving."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = ModelConfig(
            model_name="hunyuan3d-mini",
            output_dir=Path(tempfile.mkdtemp())
        )
        self.model = Hunyuan3DMini(self.config)
        
        # Create test mesh data
        self.mesh_data = {
            "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            "faces": np.array([[0, 1, 2]], dtype=np.int32),
            "uvs": np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32),
            "normals": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        }
        
        # Create test materials
        self.materials = ProductionPBRMaterial(
            albedo=np.ones((256, 256, 3), dtype=np.float32),
            normal=np.ones((256, 256, 3), dtype=np.float32) * 0.5,
            roughness=np.ones((256, 256), dtype=np.float32) * 0.5,
            metallic=np.zeros((256, 256), dtype=np.float32),
            ambient_occlusion=np.ones((256, 256), dtype=np.float32) * 0.9,
            roughness_value=0.5,
            metallic_value=0.0
        )
    
    def test_save_glb_production(self):
        """Test GLB format saving with PBR."""
        output_path = self.model._save_production_asset(
            self.mesh_data,
            self.materials,
            format=OutputFormat.GLB
        )
        
        assert output_path.exists()
        # GLB or fallback OBJ should exist
        assert output_path.suffix in [".glb", ".obj"]
    
    def test_save_obj_production(self):
        """Test OBJ saving with MTL and textures."""
        output_path = self.model._save_production_asset(
            self.mesh_data,
            self.materials,
            format=OutputFormat.OBJ
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".obj"
        
        # Check MTL file
        mtl_path = output_path.with_suffix(".mtl")
        assert mtl_path.exists()
        
        # Check OBJ content
        with open(output_path, "r") as f:
            content = f.read()
            assert "mtllib" in content
            assert "usemtl" in content
            assert "vt" in content  # UV coordinates
            assert "vn" in content  # Normals
    
    def test_save_pbr_textures(self):
        """Test PBR texture saving."""
        output_dir = self.config.output_dir / "test_textures"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model._save_pbr_textures(output_dir, self.materials)
        
        # Check texture files
        assert (output_dir / "albedo.png").exists()
        assert (output_dir / "normal.png").exists()
        assert (output_dir / "roughness.png").exists()
        assert (output_dir / "metallic.png").exists()
        assert (output_dir / "ao.png").exists()
    
    def test_save_production_metadata(self):
        """Test metadata saving."""
        output_dir = self.config.output_dir / "test_metadata"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model._save_production_metadata(output_dir, self.mesh_data, self.materials)
        
        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            assert metadata["model"] == "hunyuan3d-mini"
            assert metadata["mesh"]["vertices"] == 3
            assert metadata["mesh"]["faces"] == 1
            assert metadata["mesh"]["has_uvs"] == True
            assert metadata["materials"]["has_pbr"] == True
            assert "albedo" in metadata["materials"]["maps"]


class TestTextureGeneration:
    """Test PBR texture generation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = ModelConfig(
            model_name="hunyuan3d-mini",
            extra_params={"texture_resolution": 512}
        )
        self.model = Hunyuan3DMini(self.config)
        self.model._initialize_texture_generator()
    
    def test_texture_generation(self):
        """Test PBR texture generation."""
        mesh_data = {"vertices": np.random.randn(100, 3)}
        image = Image.new("RGB", (512, 512))
        
        materials = self.model.texture_generator.generate_pbr_textures(
            mesh_data,
            image,
            resolution=512
        )
        
        assert isinstance(materials, ProductionPBRMaterial)
        assert materials.albedo.shape == (512, 512, 3)
        assert materials.normal.shape == (512, 512, 3)
        assert materials.roughness.shape == (512, 512)
        assert materials.metallic.shape == (512, 512)
        assert materials.ambient_occlusion.shape == (512, 512)


class TestOptimizations:
    """Test memory and performance optimizations."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_fp16_optimization(self):
        """Test FP16 optimization."""
        config = ModelConfig(
            model_name="hunyuan3d-mini",
            dtype=torch.float16 if torch else None
        )
        model = Hunyuan3DMini(config)
        
        assert model.use_fp16 == True
    
    def test_sequential_processing(self):
        """Test sequential processing configuration."""
        config = ModelConfig(
            model_name="hunyuan3d-mini",
            extra_params={"sequential_processing": True}
        )
        model = Hunyuan3DMini(config)
        
        assert model.sequential_processing == True
    
    def test_cpu_offloading(self):
        """Test CPU offloading configuration."""
        config = ModelConfig(
            model_name="hunyuan3d-mini",
            extra_params={"offload_to_cpu": True}
        )
        model = Hunyuan3DMini(config)
        
        assert model.offload_to_cpu == True
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        model = Hunyuan3DMini(ModelConfig(model_name="hunyuan3d-mini"))
        
        # Test different quality modes
        model.quality_mode = "fast"
        model.polycount = 10000
        assert model._estimate_memory_usage() <= 5.0
        
        model.quality_mode = "production"
        model.polycount = 40000
        memory = model._estimate_memory_usage()
        assert memory > 10.0  # Higher for production mode


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_image_handling(self):
        """Test handling of invalid images."""
        config = ModelConfig(model_name="hunyuan3d-mini")
        model = Hunyuan3DMini(config)
        model.model = model._create_mock_model()
        model._initialized = True
        
        result = model.generate_from_image(None)
        assert not result.success
        assert "cannot be None" in result.error_message
    
    def test_invalid_multiview_input(self):
        """Test handling of invalid multi-view input."""
        config = ModelConfig(model_name="hunyuan3d-mini")
        model = Hunyuan3DMini(config)
        model._initialized = True
        
        # Test with empty list
        result = model.generate_from_multiview([])
        assert not result.success
        assert "at least 2 views" in result.error_message.lower()
    
    def test_cleanup(self):
        """Test resource cleanup."""
        config = ModelConfig(model_name="hunyuan3d-mini")
        model = Hunyuan3DMini(config)
        model.model = model._create_mock_model()
        model._initialize_texture_generator()
        model._initialize_uv_unwrapper()
        model._initialize_mesh_optimizer()
        model._initialized = True
        
        model.cleanup()
        
        assert model.model is None
        assert model.texture_generator is None
        assert model.uv_unwrapper is None
        assert model.mesh_optimizer is None
        assert not model._initialized


class TestIntegration:
    """Test integration with the system."""
    
    def test_registry_integration(self):
        """Test registry integration."""
        from dream_cad.models.registry import ModelRegistry
        
        registry = ModelRegistry()
        ModelFactory.set_registry(registry)
        
        assert "hunyuan3d-mini" in registry
        
        caps = registry.get_capabilities("hunyuan3d-mini")
        assert caps is not None
        assert caps.name == "Hunyuan3D-2-Mini"
    
    def test_hardware_compatibility(self):
        """Test hardware compatibility."""
        config = ModelConfig(model_name="hunyuan3d-mini")
        model = Hunyuan3DMini(config)
        
        caps = model.capabilities
        
        # Should work with 16GB VRAM (recommended)
        assert caps.can_run_on_hardware(16.0)
        
        # Should work with 12GB (minimum for mini)
        assert caps.can_run_on_hardware(12.0)
        
        # Should not work with 8GB
        assert not caps.can_run_on_hardware(8.0)
    
    def test_multiview_support(self):
        """Test multi-view generation support."""
        config = ModelConfig(model_name="hunyuan3d-mini")
        model = Hunyuan3DMini(config)
        
        caps = model.capabilities
        assert GenerationType.MULTIVIEW_TO_3D in caps.generation_types
    
    def test_production_quality_features(self):
        """Test production quality features."""
        config = ModelConfig(
            model_name="hunyuan3d-mini",
            extra_params={
                "quality_mode": "production",
                "polycount": 40000,
                "texture_resolution": 2048,
                "commercial_license": True
            }
        )
        model = Hunyuan3DMini(config)
        
        assert model.quality_mode == "production"
        assert model.polycount == 40000
        assert model.texture_resolution == 2048
        assert model.commercial_license == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])