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
from dream_cad.models.triposr import TripoSR
from dream_cad.models.factory import ModelFactory
class TestTripoSRCapabilities:
    def test_capabilities(self):
        config = ModelConfig(model_name="triposr")
        model = TripoSR(config)
        caps = model.capabilities
        assert caps.name == "TripoSR"
        assert GenerationType.IMAGE_TO_3D in caps.generation_types
        assert GenerationType.TEXT_TO_3D in caps.generation_types
        assert OutputFormat.OBJ in caps.output_formats
        assert OutputFormat.PLY in caps.output_formats
        assert OutputFormat.GLB in caps.output_formats
        assert caps.min_vram_gb == 4.0
        assert caps.recommended_vram_gb == 6.0
        assert caps.estimated_time_seconds["image_to_3d"] == 0.5
    def test_factory_registration(self):
        assert "triposr" in ModelFactory.list_models()
        model = ModelFactory.create_model("triposr")
        assert isinstance(model, TripoSR)
    def test_configuration(self):
        config = ModelConfig(
            model_name="triposr",
            extra_params={
                "resolution": 256,
                "chunk_size": 4096,
                "remove_background": False
            }
        )
        model = TripoSR(config)
        assert model.resolution == 256
        assert model.chunk_size == 4096
        assert model.remove_background == False
class TestTripoSRInitialization:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_initialization(self):
        config = ModelConfig(model_name="triposr")
        model = TripoSR(config)
        model.initialize()
        assert model._initialized
        assert model.model is not None
    def test_mock_model_creation(self):
        config = ModelConfig(model_name="triposr")
        model = TripoSR(config)
        mock_model = model._create_mock_model()
        assert mock_model is not None
        result = mock_model.generate(None)
        assert "vertices" in result
        assert "faces" in result
        assert result["vertices"].shape[1] == 3
        assert result["faces"].shape[1] == 3
    @patch('huggingface_hub.snapshot_download')
    def test_model_download(self, mock_download):
        mock_download.return_value = "/tmp/mock_model"
        config = ModelConfig(model_name="triposr")
        model = TripoSR(config)
        model_path = model._download_model()
        assert model_path is not None
class TestTripoSRGeneration:
    def setup_method(self):
        self.config = ModelConfig(
            model_name="triposr",
            output_dir=Path(tempfile.mkdtemp())
        )
        self.model = TripoSR(self.config)
        self.model.initialize()
    def test_image_preprocessing(self):
        image = Image.new("RGB", (1024, 1024), color=(255, 0, 0))
        processed = self.model._preprocess_image(image)
        assert processed.size == (self.model.resolution, self.model.resolution)
        assert processed.mode == "RGB"
    def test_numpy_array_input(self):
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        processed = self.model._preprocess_image(arr)
        assert isinstance(processed, Image.Image)
        assert processed.size == (self.model.resolution, self.model.resolution)
    def test_generate_from_image(self):
        image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        result = self.model.generate_from_image(image)
        assert result.success
        assert result.output_path is not None
        assert result.output_path.exists()
        assert result.generation_time < 5.0
        assert result.memory_used_gb < 8.0
    def test_generate_from_text(self):
        result = self.model.generate_from_text("a red chair")
        assert result.success
        assert result.output_path is not None
        assert result.metadata["prompt"] == "a red chair"
    def test_output_formats(self):
        image = Image.new("RGB", (256, 256))
        for format in [OutputFormat.OBJ, OutputFormat.PLY, OutputFormat.STL]:
            result = self.model.generate_from_image(image, format=format)
            assert result.success
            assert result.output_path.suffix == f".{format.value}"
class TestTripoSRMeshSaving:
    def setup_method(self):
        self.config = ModelConfig(
            model_name="triposr",
            output_dir=Path(tempfile.mkdtemp())
        )
        self.model = TripoSR(self.config)
        self.mesh_data = {
            "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            "faces": np.array([[0, 1, 2]], dtype=np.int32),
            "vertex_colors": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        }
    def test_save_obj(self):
        output_path = self.config.output_dir / "test.obj"
        self.model._save_obj(output_path, self.mesh_data)
        assert output_path.exists()
        with open(output_path, "r") as f:
            content = f.read()
            assert "v 0" in content
            assert "f 1 2 3" in content
    def test_save_ply(self):
        output_path = self.config.output_dir / "test.ply"
        self.model._save_ply(output_path, self.mesh_data)
        assert output_path.exists()
        with open(output_path, "rb") as f:
            content = f.read()
            assert b"ply" in content
            assert b"element vertex 3" in content
            assert b"element face 1" in content
    def test_save_stl(self):
        output_path = self.config.output_dir / "test.stl"
        self.model._save_stl(output_path, self.mesh_data)
        assert output_path.exists()
        with open(output_path, "r") as f:
            content = f.read()
            assert "solid mesh" in content
            assert "facet normal" in content
            assert "vertex" in content
            assert "endsolid" in content
class TestTripoSROptimizations:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_fp16_optimization(self):
        config = ModelConfig(
            model_name="triposr",
            dtype=torch.float16 if torch else None
        )
        model = TripoSR(config)
        assert model.use_fp16 == True
    def test_gradient_checkpointing(self):
        config = ModelConfig(
            model_name="triposr",
            extra_params={"gradient_checkpointing": True}
        )
        model = TripoSR(config)
        assert model.use_gradient_checkpointing == True
    def test_cpu_offloading(self):
        config = ModelConfig(
            model_name="triposr",
            extra_params={"offload_to_cpu": True}
        )
        model = TripoSR(config)
        assert model.offload_to_cpu == True
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_memory_estimation(self):
        config = ModelConfig(model_name="triposr")
        model = TripoSR(config)
        model.initialize()
        memory = model._estimate_memory_usage()
        assert memory >= 0
        assert memory < 8.0
class TestTripoSRErrorHandling:
    def test_invalid_image_format(self):
        config = ModelConfig(model_name="triposr")
        model = TripoSR(config)
        model.model = model._create_mock_model()
        model._initialized = True
        result = model.generate_from_image(None)
        assert not result.success
        assert "failed" in result.error_message.lower()
    def test_oom_handling(self):
        config = ModelConfig(
            model_name="triposr",
            extra_params={"resolution": 4096}
        )
        model = TripoSR(config)
        image = Image.new("RGB", (256, 256))
        result = model.generate_from_image(image)
        assert result.success or "memory" in str(result.error_message).lower()
    def test_cleanup(self):
        config = ModelConfig(model_name="triposr")
        model = TripoSR(config)
        model.model = model._create_mock_model()
        model._initialized = True
        model.cleanup()
        assert model.model is None
        assert not model._initialized
class TestTripoSRIntegration:
    def test_registry_integration(self):
        from dream_cad.models.registry import ModelRegistry
        registry = ModelRegistry()
        ModelFactory.set_registry(registry)
        assert "triposr" in registry
        caps = registry.get_capabilities("triposr")
        assert caps is not None
        assert caps.name == "TripoSR"
    def test_hardware_compatibility(self):
        config = ModelConfig(model_name="triposr")
        model = TripoSR(config)
        caps = model.capabilities
        assert caps.can_run_on_hardware(6.0)
        assert caps.can_run_on_hardware(4.0)
        assert not caps.can_run_on_hardware(2.0)
    def test_batch_processing(self):
        config = ModelConfig(model_name="triposr")
        model = TripoSR(config)
        caps = model.capabilities
        assert caps.supports_batch
        assert caps.max_batch_size == 4
if __name__ == "__main__":
    pytest.main([__file__, "-v"])