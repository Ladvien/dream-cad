import pytest
import tempfile
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
from dream_cad.models.stable_fast_3d import StableFast3D, PBRMaterial
from dream_cad.models.factory import ModelFactory
class TestStableFast3DCapabilities:
    def test_capabilities(self):
        config = ModelConfig(model_name="stable-fast-3d")
        model = StableFast3D(config)
        caps = model.capabilities
        assert caps.name == "Stable-Fast-3D"
        assert GenerationType.IMAGE_TO_3D in caps.generation_types
        assert GenerationType.TEXT_TO_3D in caps.generation_types
        assert OutputFormat.GLB in caps.output_formats
        assert OutputFormat.OBJ in caps.output_formats
        assert caps.min_vram_gb == 6.0
        assert caps.recommended_vram_gb == 7.0
        assert caps.estimated_time_seconds["image_to_3d"] == 3.0
    def test_factory_registration(self):
        assert "stable-fast-3d" in ModelFactory.list_models()
        model = ModelFactory.create_model("stable-fast-3d")
        assert isinstance(model, StableFast3D)
    def test_game_optimization_config(self):
        config = ModelConfig(
            model_name="stable-fast-3d",
            extra_params={
                "polycount": 3000,
                "texture_resolution": 2048,
                "target_engine": "unity",
                "optimize_for_mobile": True
            }
        )
        model = StableFast3D(config)
        assert model.target_polycount == 3000
        assert model.texture_resolution == 2048
        assert model.target_engine == "unity"
        assert model.optimize_for_mobile == True
    def test_pbr_configuration(self):
        config = ModelConfig(
            model_name="stable-fast-3d",
            extra_params={
                "generate_pbr": True,
                "delight": True,
                "material_quality": "high"
            }
        )
        model = StableFast3D(config)
        assert model.generate_pbr == True
        assert model.delight == True
        assert model.material_quality == "high"
class TestPBRMaterial:
    def test_pbr_material_creation(self):
        material = PBRMaterial(
            albedo=np.ones((512, 512, 3)),
            roughness=np.ones((512, 512)),
            metallic=np.zeros((512, 512))
        )
        assert material.albedo is not None
        assert material.roughness is not None
        assert material.metallic is not None
        assert material.normal is None
    def test_pbr_material_to_dict(self):
        material = PBRMaterial(
            albedo=np.array([[1, 0, 0]]),
            roughness=np.array([[0.5]])
        )
        material_dict = material.to_dict()
        assert "albedo" in material_dict
        assert "roughness" in material_dict
        assert isinstance(material_dict["albedo"], list)
class TestStableFast3DInitialization:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_initialization(self):
        config = ModelConfig(model_name="stable-fast-3d")
        model = StableFast3D(config)
        model.initialize()
        assert model._initialized
        assert model.model is not None
        assert model.material_generator is not None
        assert model.uv_unwrapper is not None
    def test_mock_model_creation(self):
        config = ModelConfig(model_name="stable-fast-3d")
        model = StableFast3D(config)
        mock_model = model._create_mock_model()
        assert mock_model is not None
        result = mock_model.generate(None)
        assert "vertices" in result
        assert "faces" in result
        assert "uvs" in result
    @patch('huggingface_hub.snapshot_download')
    def test_model_download(self, mock_download):
        mock_download.return_value = "/tmp/mock_model"
        config = ModelConfig(model_name="stable-fast-3d")
        model = StableFast3D(config)
        model_path = model._download_model()
        assert model_path is not None
class TestStableFast3DGeneration:
    def setup_method(self):
        self.config = ModelConfig(
            model_name="stable-fast-3d",
            output_dir=Path(tempfile.mkdtemp()),
            extra_params={"polycount": 1000}
        )
        self.model = StableFast3D(self.config)
        self.model.model = self.model._create_mock_model()
        self.model._initialize_material_generator()
        self.model._initialize_uv_unwrapper()
        self.model._initialized = True
    def test_image_preprocessing(self):
        image = Image.new("RGB", (1024, 1024), color=(255, 0, 0))
        processed = self.model._preprocess_image(image)
        assert processed.size == (self.model.resolution, self.model.resolution)
        assert processed.mode == "RGB"
    def test_delighting(self):
        image = Image.new("RGB", (512, 512), color=(200, 150, 100))
        delighted = self.model._delight_image(image)
        assert delighted.size == image.size
        assert np.array(delighted).mean() != np.array(image).mean()
    def test_topology_optimization(self):
        mesh_data = {
            "vertices": np.random.randn(2000, 3),
            "faces": np.random.randint(0, 2000, (5000, 3))
        }
        optimized = self.model._optimize_topology(mesh_data)
        assert len(optimized["faces"]) <= self.model.target_polycount
        assert len(optimized["vertices"]) <= len(mesh_data["vertices"])
    def test_uv_generation(self):
        mesh_data = {
            "vertices": np.random.randn(100, 3),
            "faces": np.random.randint(0, 100, (50, 3))
        }
        uvs = self.model._generate_uvs(mesh_data)
        assert uvs.shape == (100, 2)
        assert uvs.min() >= 0
        assert uvs.max() <= 1
    def test_generate_from_image(self):
        image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        result = self.model.generate_from_image(image)
        assert result.success
        assert result.output_path is not None
        assert result.output_path.exists()
        assert result.generation_time < 10.0
        assert result.metadata["has_pbr"] == True
        assert result.metadata["polycount"] > 0
    def test_generate_without_pbr(self):
        self.model.generate_pbr = False
        image = Image.new("RGB", (256, 256))
        result = self.model.generate_from_image(image)
        assert result.success
        assert result.metadata["has_pbr"] == False
    def test_generate_from_text(self):
        result = self.model.generate_from_text("a low-poly sword for a game")
        assert result.success
        assert result.output_path is not None
        assert result.metadata["prompt"] == "a low-poly sword for a game"
class TestGameAssetSaving:
    def setup_method(self):
        self.config = ModelConfig(
            model_name="stable-fast-3d",
            output_dir=Path(tempfile.mkdtemp())
        )
        self.model = StableFast3D(self.config)
        self.mesh_data = {
            "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            "faces": np.array([[0, 1, 2]], dtype=np.int32),
            "uvs": np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
        }
        self.materials = PBRMaterial(
            albedo=np.ones((256, 256, 3), dtype=np.float32),
            roughness=np.ones((256, 256), dtype=np.float32) * 0.5,
            metallic=np.zeros((256, 256), dtype=np.float32)
        )
    def test_save_glb_format(self):
        output_path = self.model._save_game_asset(
            self.mesh_data,
            self.materials,
            format=OutputFormat.GLB
        )
        assert output_path.exists()
        assert output_path.suffix in [".glb", ".obj"]
    def test_save_obj_with_materials(self):
        output_path = self.config.output_dir / "test.obj"
        self.model._save_obj_with_materials(output_path, self.mesh_data, self.materials)
        assert output_path.exists()
        with open(output_path, "r") as f:
            content = f.read()
            assert "mtllib" in content
            assert "usemtl" in content
            assert "vt" in content
    def test_save_mtl_file(self):
        mtl_path = self.config.output_dir / "test.mtl"
        self.model._save_mtl_file(mtl_path, self.materials, self.config.output_dir)
        assert mtl_path.exists()
        with open(mtl_path, "r") as f:
            content = f.read()
            assert "newmtl" in content
            assert "map_Kd" in content
    def test_save_material_textures(self):
        self.model._save_material_textures(self.config.output_dir, self.materials)
        assert (self.config.output_dir / "albedo.png").exists()
        assert (self.config.output_dir / "roughness.png").exists()
        assert (self.config.output_dir / "metallic.png").exists()
    def test_save_basic_formats(self):
        for format in ["ply", "stl", "obj"]:
            output_path = self.config.output_dir / f"test.{format}"
            self.model._save_basic_mesh(output_path, self.mesh_data, format)
            assert output_path.exists()
class TestPBRMaterialGeneration:
    def setup_method(self):
        self.config = ModelConfig(model_name="stable-fast-3d")
        self.model = StableFast3D(self.config)
        self.model._initialize_material_generator()
    def test_material_generation(self):
        mesh_data = {"vertices": np.random.randn(100, 3)}
        image = Image.new("RGB", (512, 512))
        materials = self.model._generate_pbr_materials(mesh_data, image)
        assert isinstance(materials, PBRMaterial)
        assert materials.albedo is not None
        assert materials.roughness is not None
        assert materials.metallic is not None
    def test_texture_saving(self):
        texture = np.random.rand(256, 256, 3)
        output_path = Path(tempfile.mkdtemp()) / "test_texture.png"
        self.model._save_texture(output_path, texture)
        assert output_path.exists()
        img = Image.open(output_path)
        assert img.size == (256, 256)
class TestOptimizations:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_fp16_optimization(self):
        config = ModelConfig(
            model_name="stable-fast-3d",
            dtype=torch.float16 if torch else None
        )
        model = StableFast3D(config)
        assert model.use_fp16 == True
    def test_sequential_processing(self):
        config = ModelConfig(
            model_name="stable-fast-3d",
            extra_params={"sequential_processing": True}
        )
        model = StableFast3D(config)
        assert model.sequential_processing == True
    def test_mobile_optimization(self):
        config = ModelConfig(
            model_name="stable-fast-3d",
            extra_params={
                "optimize_for_mobile": True,
                "polycount": 1000
            }
        )
        model = StableFast3D(config)
        assert model.optimize_for_mobile == True
        assert model.target_polycount == 1000
class TestErrorHandling:
    def test_invalid_image_handling(self):
        config = ModelConfig(model_name="stable-fast-3d")
        model = StableFast3D(config)
        model.model = model._create_mock_model()
        model._initialized = True
        result = model.generate_from_image(None)
        assert not result.success
        assert "cannot be None" in result.error_message
    def test_cleanup(self):
        config = ModelConfig(model_name="stable-fast-3d")
        model = StableFast3D(config)
        model.model = model._create_mock_model()
        model._initialize_material_generator()
        model._initialize_uv_unwrapper()
        model._initialized = True
        model.cleanup()
        assert model.model is None
        assert model.material_generator is None
        assert model.uv_unwrapper is None
        assert not model._initialized
class TestIntegration:
    def test_registry_integration(self):
        from dream_cad.models.registry import ModelRegistry
        registry = ModelRegistry()
        ModelFactory.set_registry(registry)
        assert "stable-fast-3d" in registry
        caps = registry.get_capabilities("stable-fast-3d")
        assert caps is not None
        assert caps.name == "Stable-Fast-3D"
    def test_hardware_compatibility(self):
        config = ModelConfig(model_name="stable-fast-3d")
        model = StableFast3D(config)
        caps = model.capabilities
        assert caps.can_run_on_hardware(7.0)
        assert caps.can_run_on_hardware(6.0)
        assert not caps.can_run_on_hardware(4.0)
    def test_game_engine_compatibility(self):
        for engine in ["unity", "unreal", "universal"]:
            config = ModelConfig(
                model_name="stable-fast-3d",
                extra_params={"target_engine": engine}
            )
            model = StableFast3D(config)
            assert model.target_engine == engine
if __name__ == "__main__":
    pytest.main([__file__, "-v"])