"""
Tests for the 3D model integration architecture.
"""

import pytest
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

# Handle import issues gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from dream_cad.models.base import (
    Model3D, ModelCapabilities, ModelConfig, GenerationResult,
    GenerationType, OutputFormat
)
from dream_cad.models.factory import ModelFactory, register_model
from dream_cad.models.registry import ModelRegistry, ModelRequirement


class TestModelBase:
    """Test the base model classes."""
    
    def test_model_capabilities(self):
        """Test ModelCapabilities functionality."""
        caps = ModelCapabilities(
            name="TestModel",
            generation_types=[GenerationType.TEXT_TO_3D, GenerationType.IMAGE_TO_3D],
            output_formats=[OutputFormat.OBJ, OutputFormat.PLY],
            min_vram_gb=8.0,
            recommended_vram_gb=12.0,
            supports_batch=True,
            max_batch_size=4
        )
        
        assert caps.supports_generation_type(GenerationType.TEXT_TO_3D)
        assert caps.supports_generation_type(GenerationType.IMAGE_TO_3D)
        assert not caps.supports_generation_type(GenerationType.MULTIVIEW_TO_3D)
        
        assert caps.supports_output_format(OutputFormat.OBJ)
        assert not caps.supports_output_format(OutputFormat.GLB)
        
        assert caps.can_run_on_hardware(8.0)
        assert caps.can_run_on_hardware(10.0)
        assert not caps.can_run_on_hardware(6.0)
    
    def test_model_config(self):
        """Test ModelConfig initialization."""
        config = ModelConfig(
            model_name="test",
            batch_size=2,
            seed=42
        )
        
        assert config.model_name == "test"
        assert config.batch_size == 2
        assert config.seed == 42
        assert config.device == "cuda"
        assert config.output_dir.exists()
        assert config.cache_dir.exists()
    
    def test_generation_result(self):
        """Test GenerationResult dataclass."""
        result = GenerationResult(
            success=True,
            output_path=Path("/tmp/test.obj"),
            generation_time=120.5,
            memory_used_gb=8.3
        )
        
        assert result.success
        assert result.output_path == Path("/tmp/test.obj")
        assert result.generation_time == 120.5
        assert result.memory_used_gb == 8.3
        assert result.error_message is None


class TestModelFactory:
    """Test the model factory pattern."""
    
    def setup_method(self):
        """Clear factory before each test."""
        ModelFactory.clear()
    
    def test_register_model(self):
        """Test model registration."""
        
        class TestModel(Model3D):
            @property
            def capabilities(self):
                return ModelCapabilities(
                    name="Test",
                    generation_types=[GenerationType.TEXT_TO_3D],
                    output_formats=[OutputFormat.OBJ],
                    min_vram_gb=4.0,
                    recommended_vram_gb=8.0
                )
            
            def initialize(self):
                pass
            
            def generate_from_text(self, prompt, **kwargs):
                return GenerationResult(success=True)
            
            def generate_from_image(self, image, **kwargs):
                return GenerationResult(success=False)
        
        ModelFactory.register_model("test", TestModel)
        
        assert "test" in ModelFactory.list_models()
        assert ModelFactory.get_model_class("test") == TestModel
    
    def test_create_model(self):
        """Test model creation."""
        
        class TestModel(Model3D):
            @property
            def capabilities(self):
                return ModelCapabilities(
                    name="Test",
                    generation_types=[GenerationType.TEXT_TO_3D],
                    output_formats=[OutputFormat.OBJ],
                    min_vram_gb=4.0,
                    recommended_vram_gb=8.0
                )
            
            def initialize(self):
                self._initialized = True
            
            def generate_from_text(self, prompt, **kwargs):
                return GenerationResult(success=True)
            
            def generate_from_image(self, image, **kwargs):
                return GenerationResult(success=False)
        
        ModelFactory.register_model("test", TestModel)
        
        config = ModelConfig(model_name="test")
        model = ModelFactory.create_model("test", config)
        
        assert isinstance(model, TestModel)
        assert model.config.model_name == "test"
    
    def test_invalid_model_name(self):
        """Test error handling for invalid model names."""
        with pytest.raises(ValueError, match="Model 'invalid' not found"):
            ModelFactory.create_model("invalid")
    
    def test_decorator_registration(self):
        """Test decorator-based model registration."""
        
        @register_model("decorated")
        class DecoratedModel(Model3D):
            @property
            def capabilities(self):
                return ModelCapabilities(
                    name="Decorated",
                    generation_types=[GenerationType.TEXT_TO_3D],
                    output_formats=[OutputFormat.OBJ],
                    min_vram_gb=4.0,
                    recommended_vram_gb=8.0
                )
            
            def initialize(self):
                pass
            
            def generate_from_text(self, prompt, **kwargs):
                return GenerationResult(success=True)
            
            def generate_from_image(self, image, **kwargs):
                return GenerationResult(success=False)
        
        assert "decorated" in ModelFactory.list_models()


class TestModelRegistry:
    """Test the model registry."""
    
    def setup_method(self):
        """Create temporary registry for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "registry.json"
        self.registry = ModelRegistry(self.registry_path)
    
    def test_register_model(self):
        """Test registering a model in the registry."""
        caps = ModelCapabilities(
            name="TestModel",
            generation_types=[GenerationType.TEXT_TO_3D],
            output_formats=[OutputFormat.OBJ],
            min_vram_gb=8.0,
            recommended_vram_gb=12.0
        )
        
        self.registry.register("test", caps)
        
        assert "test" in self.registry
        assert self.registry.get_capabilities("test") == caps
        assert len(self.registry) == 1
    
    def test_find_models_by_generation_type(self):
        """Test finding models by generation type."""
        caps1 = ModelCapabilities(
            name="Model1",
            generation_types=[GenerationType.TEXT_TO_3D],
            output_formats=[OutputFormat.OBJ],
            min_vram_gb=8.0,
            recommended_vram_gb=12.0
        )
        
        caps2 = ModelCapabilities(
            name="Model2",
            generation_types=[GenerationType.IMAGE_TO_3D],
            output_formats=[OutputFormat.OBJ],
            min_vram_gb=6.0,
            recommended_vram_gb=8.0
        )
        
        self.registry.register("model1", caps1)
        self.registry.register("model2", caps2)
        
        text_models = self.registry.find_models_for_generation_type(GenerationType.TEXT_TO_3D)
        assert text_models == ["model1"]
        
        image_models = self.registry.find_models_for_generation_type(GenerationType.IMAGE_TO_3D)
        assert image_models == ["model2"]
    
    def test_find_compatible_models(self):
        """Test finding hardware-compatible models."""
        caps1 = ModelCapabilities(
            name="LightModel",
            generation_types=[GenerationType.TEXT_TO_3D],
            output_formats=[OutputFormat.OBJ],
            min_vram_gb=4.0,
            recommended_vram_gb=6.0
        )
        
        caps2 = ModelCapabilities(
            name="HeavyModel",
            generation_types=[GenerationType.TEXT_TO_3D],
            output_formats=[OutputFormat.OBJ],
            min_vram_gb=12.0,
            recommended_vram_gb=16.0
        )
        
        self.registry.register("light", caps1)
        self.registry.register("heavy", caps2)
        
        # Test with 8GB VRAM
        compatible = self.registry.find_compatible_models(8.0)
        assert compatible == ["light"]
        
        # Test with 16GB VRAM
        compatible = self.registry.find_compatible_models(16.0)
        assert set(compatible) == {"light", "heavy"}
    
    def test_recommend_model(self):
        """Test model recommendation."""
        caps1 = ModelCapabilities(
            name="FastModel",
            generation_types=[GenerationType.TEXT_TO_3D],
            output_formats=[OutputFormat.OBJ],
            min_vram_gb=4.0,
            recommended_vram_gb=6.0,
            estimated_time_seconds={"text_to_3d": 60.0}
        )
        
        caps2 = ModelCapabilities(
            name="QualityModel",
            generation_types=[GenerationType.TEXT_TO_3D],
            output_formats=[OutputFormat.OBJ],
            min_vram_gb=8.0,
            recommended_vram_gb=12.0,
            estimated_time_seconds={"text_to_3d": 300.0}
        )
        
        self.registry.register("fast", caps1)
        self.registry.register("quality", caps2)
        
        # Recommend for speed
        model = self.registry.recommend_model(
            GenerationType.TEXT_TO_3D,
            vram_gb=16.0,
            prioritize_speed=True
        )
        assert model == "fast"
        
        # Recommend for quality
        model = self.registry.recommend_model(
            GenerationType.TEXT_TO_3D,
            vram_gb=16.0,
            prioritize_speed=False
        )
        assert model == "quality"
    
    def test_save_and_load(self):
        """Test saving and loading registry to/from file."""
        caps = ModelCapabilities(
            name="TestModel",
            generation_types=[GenerationType.TEXT_TO_3D],
            output_formats=[OutputFormat.OBJ],
            min_vram_gb=8.0,
            recommended_vram_gb=12.0
        )
        
        req = ModelRequirement(
            min_vram_gb=8.0,
            recommended_vram_gb=12.0,
            min_ram_gb=16.0,
            dependencies=["torch", "numpy"]
        )
        
        self.registry.register("test", caps, req)
        self.registry.save_to_file()
        
        # Create new registry and load
        new_registry = ModelRegistry(self.registry_path)
        
        assert "test" in new_registry
        loaded_caps = new_registry.get_capabilities("test")
        assert loaded_caps.name == "TestModel"
        assert loaded_caps.min_vram_gb == 8.0
        
        loaded_req = new_registry.get_requirements("test")
        assert loaded_req.min_ram_gb == 16.0
        assert "torch" in loaded_req.dependencies
    
    def test_model_comparison(self):
        """Test getting model comparison data."""
        caps1 = ModelCapabilities(
            name="Model1",
            generation_types=[GenerationType.TEXT_TO_3D],
            output_formats=[OutputFormat.OBJ],
            min_vram_gb=8.0,
            recommended_vram_gb=12.0,
            supports_fp16=True
        )
        
        caps2 = ModelCapabilities(
            name="Model2",
            generation_types=[GenerationType.IMAGE_TO_3D],
            output_formats=[OutputFormat.PLY],
            min_vram_gb=6.0,
            recommended_vram_gb=8.0,
            supports_int8=True
        )
        
        self.registry.register("model1", caps1)
        self.registry.register("model2", caps2)
        
        comparison = self.registry.get_model_comparison()
        
        assert "model1" in comparison
        assert "model2" in comparison
        
        assert comparison["model1"]["min_vram_gb"] == 8.0
        assert comparison["model1"]["supports_fp16"] == True
        assert comparison["model2"]["supports_int8"] == True


class TestMVDreamAdapter:
    """Test MVDream adapter integration."""
    
    @pytest.mark.skipif(not Path("/mnt/datadrive_m2/dream-cad/scripts/generate_3d_real.py").exists(),
                        reason="MVDream script not found")
    def test_mvdream_capabilities(self):
        """Test MVDream adapter capabilities."""
        from dream_cad.models.mvdream_adapter import MVDreamAdapter
        
        config = ModelConfig(model_name="mvdream")
        model = MVDreamAdapter(config)
        
        caps = model.capabilities
        assert caps.name == "MVDream"
        assert GenerationType.TEXT_TO_3D in caps.generation_types
        assert OutputFormat.OBJ in caps.output_formats
        assert caps.min_vram_gb == 20.0
        assert caps.recommended_vram_gb == 24.0
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_hardware_validation(self):
        """Test hardware validation."""
        from dream_cad.models.mvdream_adapter import MVDreamAdapter
        
        config = ModelConfig(model_name="mvdream")
        model = MVDreamAdapter(config)
        
        # This will check if CUDA is available and VRAM is sufficient
        # May fail on systems without proper GPU
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            can_run = model.validate_hardware()
            assert can_run == (vram_gb >= 20.0)


class TestEndToEndIntegration:
    """Test end-to-end integration of the architecture."""
    
    def setup_method(self):
        """Setup for integration tests."""
        ModelFactory.clear()
        
        # Create a mock model for testing
        @register_model("mock")
        class MockModel(Model3D):
            @property
            def capabilities(self):
                return ModelCapabilities(
                    name="MockModel",
                    generation_types=[GenerationType.TEXT_TO_3D],
                    output_formats=[OutputFormat.OBJ],
                    min_vram_gb=4.0,
                    recommended_vram_gb=8.0
                )
            
            def initialize(self):
                self._initialized = True
            
            def generate_from_text(self, prompt, **kwargs):
                return GenerationResult(
                    success=True,
                    output_path=Path("/tmp/mock.obj"),
                    generation_time=1.0,
                    memory_used_gb=4.0,
                    metadata={"prompt": prompt}
                )
            
            def generate_from_image(self, image, **kwargs):
                return GenerationResult(
                    success=False,
                    error_message="Not implemented"
                )
    
    def test_full_workflow(self):
        """Test complete workflow from factory to generation."""
        # Create registry
        registry = ModelRegistry()
        ModelFactory.set_registry(registry)
        
        # Model should be auto-registered
        assert "mock" in ModelFactory.list_models()
        assert "mock" in registry
        
        # Create model
        model = ModelFactory.create_model("mock")
        
        # Initialize and generate
        with model:
            result = model.generate_from_text("a test object")
            assert result.success
            assert result.metadata["prompt"] == "a test object"
            assert result.generation_time == 1.0
    
    def test_model_selection_workflow(self):
        """Test model selection based on requirements."""
        registry = ModelRegistry()
        
        # Register multiple mock models
        @register_model("light")
        class LightModel(Model3D):
            @property
            def capabilities(self):
                return ModelCapabilities(
                    name="LightModel",
                    generation_types=[GenerationType.TEXT_TO_3D],
                    output_formats=[OutputFormat.OBJ],
                    min_vram_gb=2.0,
                    recommended_vram_gb=4.0,
                    estimated_time_seconds={"text_to_3d": 30.0}
                )
            
            def initialize(self):
                pass
            
            def generate_from_text(self, prompt, **kwargs):
                return GenerationResult(success=True)
            
            def generate_from_image(self, image, **kwargs):
                return GenerationResult(success=False)
        
        @register_model("heavy")
        class HeavyModel(Model3D):
            @property
            def capabilities(self):
                return ModelCapabilities(
                    name="HeavyModel",
                    generation_types=[GenerationType.TEXT_TO_3D],
                    output_formats=[OutputFormat.OBJ],
                    min_vram_gb=16.0,
                    recommended_vram_gb=24.0,
                    estimated_time_seconds={"text_to_3d": 300.0}
                )
            
            def initialize(self):
                pass
            
            def generate_from_text(self, prompt, **kwargs):
                return GenerationResult(success=True)
            
            def generate_from_image(self, image, **kwargs):
                return GenerationResult(success=False)
        
        ModelFactory.set_registry(registry)
        
        # Test recommendation for low VRAM
        recommended = registry.recommend_model(
            GenerationType.TEXT_TO_3D,
            vram_gb=8.0,
            prioritize_speed=True
        )
        assert recommended == "light"
        
        # Test recommendation for high VRAM
        recommended = registry.recommend_model(
            GenerationType.TEXT_TO_3D,
            vram_gb=24.0,
            prioritize_speed=False
        )
        assert recommended == "heavy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])