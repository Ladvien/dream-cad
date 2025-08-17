"""
Tests for Model Selection and Configuration UI.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Mock gradio if not available
import sys
sys.modules['gradio'] = MagicMock()

# Mock torch
class MockTorch:
    class Tensor:
        pass
    
    class cuda:
        @staticmethod
        def is_available():
            return True
        
        @staticmethod
        def get_device_name(idx):
            return "NVIDIA GeForce RTX 3090"
        
        @staticmethod
        def get_device_properties(idx):
            class Props:
                total_memory = 24 * 1024**3  # 24GB
            return Props()
        
        @staticmethod
        def memory_allocated(idx):
            return 4 * 1024**3  # 4GB used
    
    @staticmethod
    def device(name):
        return name
    
    @staticmethod
    def is_tensor(obj):
        return False
    
    float16 = None

sys.modules['torch'] = MockTorch()

# Mock psutil
class MockPsutil:
    @staticmethod
    def virtual_memory():
        class Memory:
            total = 32 * 1024**3  # 32GB
            used = 16 * 1024**3   # 16GB used
        return Memory()
    
    @staticmethod
    def cpu_count():
        return 8
    
    @staticmethod
    def cpu_percent():
        return 25.0

sys.modules['psutil'] = MockPsutil()

from dream_cad.ui.model_selection_ui import (
    ModelSelectionUI, HardwareMonitor, PresetConfig
)
from dream_cad.models.registry import ModelRegistry
from dream_cad.models.base import ModelCapabilities, GenerationType, OutputFormat


class TestHardwareMonitor:
    """Test hardware monitoring functionality."""
    
    def test_initialization(self):
        """Test hardware monitor initialization."""
        monitor = HardwareMonitor()
        
        assert monitor.has_gpu == True
        assert monitor.gpu_name == "NVIDIA GeForce RTX 3090"
        assert monitor.total_vram == 24.0
    
    def test_available_vram(self):
        """Test available VRAM calculation."""
        monitor = HardwareMonitor()
        available = monitor.get_available_vram()
        
        assert available == 20.0  # 24GB total - 4GB used
    
    def test_system_ram(self):
        """Test system RAM monitoring."""
        monitor = HardwareMonitor()
        used, total = monitor.get_system_ram()
        
        assert total == 32.0
        assert used == 16.0
    
    def test_hardware_info(self):
        """Test comprehensive hardware info."""
        monitor = HardwareMonitor()
        info = monitor.get_hardware_info()
        
        assert info["gpu_available"] == True
        assert info["gpu_name"] == "NVIDIA GeForce RTX 3090"
        assert info["total_vram_gb"] == 24.0
        assert info["available_vram_gb"] == 20.0
        assert info["total_ram_gb"] == 32.0
        assert info["used_ram_gb"] == 16.0
        assert info["cpu_count"] == 8
        assert info["cpu_percent"] == 25.0


class TestPresetConfig:
    """Test preset configurations."""
    
    def test_preset_creation(self):
        """Test preset configuration creation."""
        preset = PresetConfig(
            name="Test Preset",
            description="Test description",
            model="test_model",
            quality_mode="balanced",
            extra_params={"param1": "value1"}
        )
        
        assert preset.name == "Test Preset"
        assert preset.model == "test_model"
        assert preset.quality_mode == "balanced"
        assert preset.extra_params["param1"] == "value1"


class TestModelSelectionUI:
    """Test model selection UI."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create mock registry
        self.registry = ModelRegistry()
        
        # Add test models
        self._add_test_models()
        
        # Create UI with temp config path
        self.temp_dir = tempfile.mkdtemp()
        self.ui = ModelSelectionUI(self.registry)
        self.ui.config_save_path = Path(self.temp_dir) / "configs.json"
    
    def _add_test_models(self):
        """Add test models to registry."""
        # Add TripoSR
        caps1 = ModelCapabilities(
            name="TripoSR",
            generation_types=[GenerationType.IMAGE_TO_3D],
            output_formats=[OutputFormat.OBJ, OutputFormat.PLY],
            min_vram_gb=4.0,
            recommended_vram_gb=6.0,
            supports_batch=False,
            max_batch_size=1,
            estimated_time_seconds={"fast": 0.5},
            requires_gpu=True,
            supports_fp16=True,
            supports_int8=False,
            model_size_gb=1.5
        )
        self.registry.register("triposr", caps1)
        
        # Add TRELLIS
        caps2 = ModelCapabilities(
            name="TRELLIS",
            generation_types=[GenerationType.IMAGE_TO_3D, GenerationType.MULTIVIEW_TO_3D],
            output_formats=[OutputFormat.MESH, OutputFormat.OBJ],
            min_vram_gb=8.0,
            recommended_vram_gb=16.0,
            supports_batch=False,
            max_batch_size=1,
            estimated_time_seconds={"balanced": 30.0},
            requires_gpu=True,
            supports_fp16=True,
            supports_int8=False,
            model_size_gb=4.5
        )
        self.registry.register("trellis", caps2)
    
    def test_initialization(self):
        """Test UI initialization."""
        assert self.ui.registry is not None
        assert self.ui.hardware_monitor is not None
        assert len(self.ui.PRESETS) > 0
    
    def test_save_config(self):
        """Test configuration saving."""
        config = {
            "model": "triposr",
            "params": {"resolution": 512}
        }
        
        result = self.ui.save_config("test_config", config)
        assert "successfully" in result
        
        # Check saved
        assert "test_config" in self.ui.saved_configs
        assert self.ui.saved_configs["test_config"]["config"] == config
    
    def test_load_saved_configs(self):
        """Test loading saved configurations."""
        # Save a config first
        config = {"model": "test", "params": {}}
        self.ui.save_config("test", config)
        
        # Create new UI instance
        ui2 = ModelSelectionUI(self.registry)
        ui2.config_save_path = self.ui.config_save_path
        ui2.load_saved_configs()
        
        assert "test" in ui2.saved_configs
    
    def test_model_recommendations(self):
        """Test model recommendations based on hardware."""
        recommendations = self.ui.get_model_recommendations()
        
        assert len(recommendations) == 2  # We added 2 models
        
        # Check ordering (should be sorted by score)
        # With 20GB available VRAM:
        # - TripoSR: 20GB >= 6GB recommended -> score 1.0
        # - TRELLIS: 20GB >= 16GB recommended -> score 1.0
        for name, status, score in recommendations:
            if name == "triposr":
                assert status == "✅ Recommended"
                assert score == 1.0
            elif name == "trellis":
                assert status == "✅ Recommended"
                assert score == 1.0
    
    def test_format_hardware_info(self):
        """Test hardware info formatting."""
        info_text = self.ui.format_hardware_info()
        
        assert "RTX 3090" in info_text
        assert "20.0 / 24.0 GB" in info_text
        assert "16.0 / 32.0 GB" in info_text
        assert "8 cores" in info_text
    
    def test_format_model_capabilities(self):
        """Test model capabilities formatting."""
        caps_text = self.ui.format_model_capabilities("triposr")
        
        assert "TripoSR" in caps_text
        assert "4.0 GB" in caps_text  # Min VRAM
        assert "6.0 GB" in caps_text  # Recommended VRAM
        assert "image_to_3d" in caps_text
        assert "OBJ" in caps_text
    
    def test_model_comparison_table(self):
        """Test model comparison table generation."""
        table = self.ui.get_model_comparison_table()
        
        assert "TripoSR" in table
        assert "TRELLIS" in table
        assert "4.0GB" in table  # Min VRAM for TripoSR
        assert "⚡⚡⚡" in table  # Speed rating for fast model
    
    def test_update_model_info(self):
        """Test updating model information."""
        caps_text, params, rec_status = self.ui.update_model_info("triposr")
        
        assert "TripoSR" in caps_text
        assert isinstance(params, dict)
        assert "✅ Recommended" in rec_status
    
    def test_get_model_specific_params(self):
        """Test getting model-specific parameters."""
        # Test TripoSR params
        params = self.ui.get_model_specific_params("triposr")
        assert "resolution" in params
        assert "use_fp16" in params
        
        # Test TRELLIS params
        params = self.ui.get_model_specific_params("trellis")
        assert "representation" in params
        assert "quality_mode" in params
        
        # Test Hunyuan params
        params = self.ui.get_model_specific_params("hunyuan3d-mini")
        assert "polycount" in params
        assert "commercial_license" in params
    
    def test_apply_preset(self):
        """Test applying preset configurations."""
        # Test Ultra Fast preset
        model, params, message = self.ui.apply_preset("⚡ Ultra Fast")
        
        assert model == "triposr"
        assert params["resolution"] == 256
        assert params["use_fp16"] == True
        assert "TripoSR" in message
        
        # Test invalid preset
        model, params, message = self.ui.apply_preset("Invalid")
        assert model == ""
        assert "not found" in message


class TestUIIntegration:
    """Test UI integration with models."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.registry = ModelRegistry()
        self.ui = ModelSelectionUI(self.registry)
    
    @patch('dream_cad.ui.model_selection_ui.ModelFactory')
    def test_generate_with_config(self, mock_factory):
        """Test generation with configuration."""
        # Create mock model
        mock_model = Mock()
        mock_model._initialized = False
        mock_model.initialize = Mock()
        
        # Create mock result
        mock_result = Mock()
        mock_result.success = True
        mock_result.output_path = Path("/tmp/output.obj")
        mock_result.generation_time = 5.0
        mock_model.generate_from_text = Mock(return_value=mock_result)
        
        # Mock the factory as a class with create_model method
        mock_factory.create_model = Mock(return_value=mock_model)
        
        # Patch the factory in the UI module
        self.ui.factory = mock_factory
        
        # Test generation
        output, status = self.ui.generate_with_config(
            "a chair",
            "test_model",
            {"param1": "value1"},
            "obj"
        )
        
        assert output == Path("/tmp/output.obj")
        assert "successfully" in status
        assert "5.0s" in status
    
    def test_presets_valid(self):
        """Test that all presets are valid."""
        for preset in self.ui.PRESETS:
            assert preset.name
            assert preset.description
            assert preset.model in [
                "triposr", "stable-fast-3d", "hunyuan3d-mini", 
                "trellis", "mvdream"
            ]
            assert preset.quality_mode in ["fast", "balanced", "production", "hq"]
            assert isinstance(preset.extra_params, dict)


class TestUICreation:
    """Test UI interface creation."""
    
    def test_create_interface(self):
        """Test Gradio interface creation."""
        registry = ModelRegistry()
        ui = ModelSelectionUI(registry)
        
        # Mock gradio
        with patch('dream_cad.ui.model_selection_ui.gr') as mock_gr:
            mock_gr.Blocks = Mock(return_value=MagicMock())
            mock_gr.themes = Mock()
            mock_gr.themes.Soft = Mock()
            
            interface = ui.create_interface()
            
            # Check that interface was created
            assert interface is not None
            mock_gr.Blocks.assert_called_once()


class TestConfigPersistence:
    """Test configuration persistence."""
    
    def test_config_save_and_load(self):
        """Test saving and loading configurations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create UI with custom path
            ui = ModelSelectionUI()
            ui.config_save_path = Path(tmpdir) / "test_configs.json"
            
            # Save config
            config = {
                "model": "test_model",
                "parameters": {
                    "param1": "value1",
                    "param2": 123
                }
            }
            
            result = ui.save_config("test_config", config)
            assert "successfully" in result
            
            # Load in new instance
            ui2 = ModelSelectionUI()
            ui2.config_save_path = ui.config_save_path
            ui2.load_saved_configs()
            
            assert "test_config" in ui2.saved_configs
            loaded = ui2.saved_configs["test_config"]
            assert loaded["config"] == config
            assert "timestamp" in loaded
            assert "hardware" in loaded


class TestErrorHandling:
    """Test error handling in UI."""
    
    def test_invalid_model_info(self):
        """Test handling of invalid model."""
        ui = ModelSelectionUI()
        
        caps_text, params, rec_status = ui.update_model_info("invalid_model")
        assert caps_text == "No information available"
        assert params == {}
        assert rec_status == "Unknown"
    
    def test_save_config_error(self):
        """Test config save error handling."""
        ui = ModelSelectionUI()
        ui.config_save_path = Path("/invalid/path/configs.json")
        
        result = ui.save_config("test", {})
        assert "Failed" in result
    
    @patch('dream_cad.ui.model_selection_ui.ModelFactory')
    def test_generate_error(self, mock_factory):
        """Test generation error handling."""
        ui = ModelSelectionUI()
        
        # Make factory raise exception
        mock_factory.create_model.side_effect = Exception("Model error")
        
        output, status = ui.generate_with_config(
            "test prompt",
            "test_model",
            {},
            "obj"
        )
        
        assert output is None
        assert "Error" in status
        assert "Model error" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])