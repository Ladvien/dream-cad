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
from dream_cad.models.trellis import (
    TRELLIS, RepresentationType, NeRFRepresentation,
    GaussianSplatting, SLATRepresentation
)
from dream_cad.models.factory import ModelFactory
class TestTRELLISCapabilities:
    def test_capabilities(self):
        config = ModelConfig(model_name="trellis")
        model = TRELLIS(config)
        caps = model.capabilities
        assert caps.name == "TRELLIS"
        assert GenerationType.IMAGE_TO_3D in caps.generation_types
        assert GenerationType.TEXT_TO_3D in caps.generation_types
        assert GenerationType.MULTIVIEW_TO_3D in caps.generation_types
        assert OutputFormat.MESH in caps.output_formats
        assert OutputFormat.OBJ in caps.output_formats
        assert caps.min_vram_gb == 8.0
        assert caps.recommended_vram_gb == 16.0
        assert not caps.supports_batch
    def test_factory_registration(self):
        assert "trellis" in ModelFactory.list_models()
        model = ModelFactory.create_model("trellis")
        assert isinstance(model, TRELLIS)
    def test_quality_modes(self):
        for mode in ["fast", "balanced", "hq"]:
            config = ModelConfig(
                model_name="trellis",
                extra_params={"quality_mode": mode}
            )
            model = TRELLIS(config)
            assert model.quality_mode == mode
    def test_representation_configuration(self):
        config = ModelConfig(
            model_name="trellis",
            extra_params={
                "representation": "nerf",
                "nerf_resolution": 256,
                "gaussian_count": 50000
            }
        )
        model = TRELLIS(config)
        assert model.representation_type == "nerf"
        assert model.nerf_resolution == 256
        assert model.gaussian_count == 50000
class TestRepresentationTypes:
    def test_nerf_representation(self):
        nerf = NeRFRepresentation(
            density_grid=np.ones((128, 128, 128)),
            feature_grid=np.ones((128, 128, 128, 32)),
            resolution=128
        )
        assert nerf.density_grid is not None
        assert nerf.feature_grid.shape[-1] == 32
        assert nerf.resolution == 128
    def test_gaussian_splatting(self):
        gaussian = GaussianSplatting(
            positions=np.random.randn(1000, 3),
            scales=np.ones((1000, 3)),
            rotations=np.random.randn(1000, 4),
            opacities=np.ones(1000),
            features=np.random.randn(1000, 32)
        )
        assert gaussian.positions.shape == (1000, 3)
        assert gaussian.rotations.shape == (1000, 4)
        gaussian_dict = gaussian.to_dict()
        assert gaussian_dict["num_gaussians"] == 1000
        assert gaussian_dict["has_features"] == True
    def test_slat_representation(self):
        slat = SLATRepresentation(
            latent_features=np.random.randn(64, 64, 64, 16),
            structure_params={"type": "grid"},
            resolution=(64, 64, 64)
        )
        assert slat.latent_features.shape == (64, 64, 64, 16)
        assert slat.resolution == (64, 64, 64)
        assert slat.structure_params["type"] == "grid"
class TestTRELLISInitialization:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_initialization(self):
        config = ModelConfig(model_name="trellis")
        model = TRELLIS(config)
        model.initialize()
        assert model._initialized
        assert model.model is not None
        assert model.nerf_generator is not None
        assert model.gaussian_generator is not None
    def test_mock_model_creation(self):
        config = ModelConfig(model_name="trellis")
        model = TRELLIS(config)
        mock_model = model._create_mock_model()
        assert mock_model is not None
        slat = mock_model.generate_slat(None)
        assert isinstance(slat, SLATRepresentation)
        mesh = mock_model.slat_to_mesh(slat)
        assert "vertices" in mesh
        assert "faces" in mesh
        nerf = mock_model.slat_to_nerf(slat)
        assert isinstance(nerf, NeRFRepresentation)
        gaussian = mock_model.slat_to_gaussian(slat)
        assert isinstance(gaussian, GaussianSplatting)
    @patch('huggingface_hub.snapshot_download')
    def test_model_download_optimized(self, mock_download):
        mock_download.return_value = "/tmp/mock_model"
        config = ModelConfig(
            model_name="trellis",
            extra_params={"use_optimized_fork": True}
        )
        model = TRELLIS(config)
        model_path = model._download_model()
        assert model_path is not None
class TestTRELLISGeneration:
    def setup_method(self):
        self.config = ModelConfig(
            model_name="trellis",
            output_dir=Path(tempfile.mkdtemp())
        )
        self.model = TRELLIS(self.config)
        self.model.model = self.model._create_mock_model()
        self.model._initialize_generators()
        self.model._initialized = True
    def test_image_preprocessing(self):
        image = Image.new("RGB", (1024, 1024), color=(255, 0, 0))
        processed = self.model._preprocess_image(image)
        assert processed.size == (self.model.resolution, self.model.resolution)
        assert processed.mode == "RGB"
    def test_generate_mesh_from_image(self):
        image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        result = self.model.generate_from_image(
            image,
            representation="mesh"
        )
        assert result.success
        assert result.output_path is not None
        assert result.output_path.exists()
        assert result.mesh_data is not None
        assert result.metadata["representation"] == "mesh"
    def test_generate_nerf_from_image(self):
        image = Image.new("RGB", (512, 512))
        result = self.model.generate_from_image(
            image,
            representation="nerf"
        )
        assert result.success
        assert result.output_path is not None
        assert result.output_path.suffix == ".npz"
        assert result.metadata["representation"] == "nerf"
    def test_generate_gaussian_from_image(self):
        image = Image.new("RGB", (512, 512))
        result = self.model.generate_from_image(
            image,
            representation="gaussian_splatting"
        )
        assert result.success
        assert result.output_path is not None
        assert result.metadata["representation"] == "gaussian_splatting"
    def test_generate_from_text(self):
        result = self.model.generate_from_text("a detailed sculpture")
        assert result.success
        assert result.output_path is not None
        assert result.metadata["prompt"] == "a detailed sculpture"
    def test_multiview_generation(self):
        images = [
            Image.new("RGB", (256, 256), color=(100, 0, 0)),
            Image.new("RGB", (256, 256), color=(0, 100, 0)),
            Image.new("RGB", (256, 256), color=(0, 0, 100)),
            Image.new("RGB", (256, 256), color=(100, 100, 0))
        ]
        result = self.model.generate_from_multiview(images)
        assert result.success
        assert result.output_path is not None
        assert result.metadata["method"] == "multiview"
        assert result.metadata["num_views"] == 4
    def test_view_consistency(self):
        self.model.view_consistency = True
        image = Image.new("RGB", (512, 512))
        result = self.model.generate_from_image(image)
        assert result.success
        assert result.images is not None
        assert len(result.images) > 0
        assert result.metadata["multi_view"] == True
class TestSequentialProcessing:
    def setup_method(self):
        self.config = ModelConfig(
            model_name="trellis",
            output_dir=Path(tempfile.mkdtemp()),
            extra_params={"sequential_processing": True}
        )
        self.model = TRELLIS(self.config)
        self.model.model = self.model._create_mock_model()
        self.model._initialize_generators()
        self.model._initialized = True
    def test_sequential_slat_generation(self):
        image = Image.new("RGB", (512, 512))
        slat = self.model._generate_slat_sequential(image)
        assert isinstance(slat, SLATRepresentation)
        assert slat.latent_features is not None
    def test_memory_efficient_generation(self):
        image = Image.new("RGB", (512, 512))
        result = self.model.generate_from_image(image)
        assert result.success
        assert result.memory_used_gb <= 16.0
class TestFormatConversion:
    def setup_method(self):
        self.config = ModelConfig(
            model_name="trellis",
            output_dir=Path(tempfile.mkdtemp())
        )
        self.model = TRELLIS(self.config)
        self.model.model = self.model._create_mock_model()
        self.model._initialize_generators()
        self.model._initialized = True
    def test_nerf_to_mesh_conversion(self):
        nerf_path = self.config.output_dir / "test_nerf.npz"
        nerf = NeRFRepresentation(
            density_grid=np.random.rand(64, 64, 64),
            feature_grid=np.random.randn(64, 64, 64, 32)
        )
        np.savez_compressed(
            nerf_path,
            density_grid=nerf.density_grid,
            feature_grid=nerf.feature_grid
        )
        output_path = self.model.convert_representation(
            nerf_path,
            RepresentationType.NERF,
            RepresentationType.MESH
        )
        assert output_path.exists()
        assert output_path.parent.name == "converted_mesh"
    def test_preserve_intermediate(self):
        self.model.preserve_intermediate = True
        image = Image.new("RGB", (256, 256))
        result = self.model.generate_from_image(image)
        assert result.success
class TestRepresentationSaving:
    def setup_method(self):
        self.config = ModelConfig(
            model_name="trellis",
            output_dir=Path(tempfile.mkdtemp())
        )
        self.model = TRELLIS(self.config)
    def test_save_nerf(self):
        nerf = NeRFRepresentation(
            density_grid=np.random.rand(128, 128, 128),
            feature_grid=np.random.randn(128, 128, 128, 32),
            resolution=128
        )
        output_path = self.model._save_nerf(self.config.output_dir, nerf)
        assert output_path.exists()
        assert output_path.suffix == ".npz"
        meta_path = self.config.output_dir / "nerf_meta.json"
        assert meta_path.exists()
        with open(meta_path, "r") as f:
            meta = json.load(f)
            assert meta["type"] == "nerf"
            assert meta["resolution"] == 128
    def test_save_gaussian(self):
        gaussian = GaussianSplatting(
            positions=np.random.randn(5000, 3),
            scales=np.ones((5000, 3)),
            rotations=np.random.randn(5000, 4),
            opacities=np.ones(5000),
            features=np.random.randn(5000, 32)
        )
        output_path = self.model._save_gaussian(self.config.output_dir, gaussian)
        assert output_path.exists()
        assert output_path.suffix == ".npz"
        meta_path = self.config.output_dir / "gaussian_meta.json"
        assert meta_path.exists()
    def test_save_slat(self):
        slat = SLATRepresentation(
            latent_features=np.random.randn(64, 64, 64, 16),
            structure_params={"type": "grid"},
            resolution=(64, 64, 64)
        )
        output_path = self.model._save_slat(self.config.output_dir, slat)
        assert output_path.exists()
        assert output_path.suffix == ".npz"
        meta_path = self.config.output_dir / "slat_meta.json"
        assert meta_path.exists()
    def test_save_mesh_formats(self):
        mesh_data = {
            "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            "faces": np.array([[0, 1, 2]])
        }
        for format in ["obj", "ply"]:
            output_path = self.model._save_mesh(
                self.config.output_dir,
                mesh_data,
                format=format
            )
            assert output_path.exists()
            assert output_path.suffix == f".{format}"
class TestOptimizations:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_fp16_optimization(self):
        config = ModelConfig(
            model_name="trellis",
            dtype=torch.float16 if torch else None
        )
        model = TRELLIS(config)
        assert model.use_fp16 == True
    def test_gradient_checkpointing(self):
        config = ModelConfig(
            model_name="trellis",
            extra_params={"gradient_checkpointing": True}
        )
        model = TRELLIS(config)
        assert model.use_checkpointing == True
    def test_optimized_fork_usage(self):
        config = ModelConfig(
            model_name="trellis",
            extra_params={"use_optimized_fork": True}
        )
        model = TRELLIS(config)
        assert model.use_optimized_fork == True
    def test_quality_mode_memory(self):
        model = TRELLIS(ModelConfig(model_name="trellis"))
        model.quality_mode = "fast"
        assert model._estimate_memory_usage() <= 8.0
        model.quality_mode = "balanced"
        assert model._estimate_memory_usage() <= 12.0
        model.quality_mode = "hq"
        assert model._estimate_memory_usage() <= 16.0
class TestErrorHandling:
    def test_invalid_image_handling(self):
        config = ModelConfig(model_name="trellis")
        model = TRELLIS(config)
        model.model = model._create_mock_model()
        model._initialize_generators()
        model._initialized = True
        result = model.generate_from_image(None)
        assert not result.success
        assert "cannot be None" in result.error_message
    def test_invalid_multiview_input(self):
        config = ModelConfig(model_name="trellis")
        model = TRELLIS(config)
        model._initialized = True
        result = model.generate_from_multiview([Image.new("RGB", (256, 256))])
        assert not result.success
        assert "at least 2 views" in result.error_message.lower()
    def test_cleanup(self):
        config = ModelConfig(model_name="trellis")
        model = TRELLIS(config)
        model.model = model._create_mock_model()
        model._initialize_generators()
        model._initialized = True
        model.cleanup()
        assert model.model is None
        assert model.nerf_generator is None
        assert model.gaussian_generator is None
        assert model.slat_encoder is None
        assert not model._initialized
class TestIntegration:
    def test_registry_integration(self):
        from dream_cad.models.registry import ModelRegistry
        registry = ModelRegistry()
        ModelFactory.set_registry(registry)
        assert "trellis" in registry
        caps = registry.get_capabilities("trellis")
        assert caps is not None
        assert caps.name == "TRELLIS"
    def test_hardware_compatibility(self):
        config = ModelConfig(model_name="trellis")
        model = TRELLIS(config)
        caps = model.capabilities
        assert caps.can_run_on_hardware(16.0)
        assert caps.can_run_on_hardware(8.0)
        assert not caps.can_run_on_hardware(6.0)
    def test_multiview_support(self):
        config = ModelConfig(model_name="trellis")
        model = TRELLIS(config)
        caps = model.capabilities
        assert GenerationType.MULTIVIEW_TO_3D in caps.generation_types
if __name__ == "__main__":
    pytest.main([__file__, "-v"])