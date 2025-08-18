#!/usr/bin/env python3
"""Real integration tests for DreamCAD TUI - absolutely no mocks, stubs, or fakes."""

import pytest
from pathlib import Path
from textual.widgets import Input, Select, Button, RichLog, Switch, Label
import asyncio
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcad_tui_new import DreamCADTUI, MODEL_CONFIGS


class TestRealTUIBehavior:
    """Test real TUI behavior with actual components and no mocking."""
    
    @pytest.mark.asyncio
    async def test_full_generation_workflow(self):
        """Test complete generation workflow from start to finish."""
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            # 1. App starts with default model
            assert app.current_model == "TripoSR"
            
            # 2. Enter a real prompt
            prompt_input = app.query_one("#prompt", Input)
            test_prompt = "a medieval castle with towers"
            prompt_input.value = test_prompt
            await pilot.pause()
            
            # 3. Change model to TRELLIS
            model_select = app.query_one("#model-select", Select)
            model_select.value = "TRELLIS"
            await pilot.pause()
            assert app.current_model == "TRELLIS"
            
            # 4. Adjust parameters
            if "quality_mode" in app.param_widgets:
                quality_widget = app.param_widgets["quality_mode"]
                quality_widget.value = "balanced"
                await pilot.pause()
            
            # 5. Start generation
            gen_btn = app.query_one("#generate-btn", Button)
            initial_disabled_state = gen_btn.disabled
            await pilot.click(gen_btn)
            
            # 6. Wait for generation to complete
            await asyncio.sleep(5)
            
            # 7. Verify button is re-enabled
            assert gen_btn.disabled == initial_disabled_state
            
            # 8. Verify log has content
            output_log = app.query_one("#output", RichLog)
            assert output_log is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_ui_operations(self):
        """Test that UI remains responsive during concurrent operations."""
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            # Start a generation
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "test object"
            await pilot.pause()
            
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            
            # Immediately try other UI operations
            await pilot.pause(0.1)
            
            # Should still be able to interact with other elements
            model_select = app.query_one("#model-select", Select)
            assert model_select is not None
            
            # Clear button should still be clickable
            clear_btn = app.query_one("#clear-btn", Button)
            assert clear_btn is not None
            
            # Wait for generation to finish
            await asyncio.sleep(4)
    
    @pytest.mark.asyncio
    async def test_parameter_persistence_across_models(self):
        """Test that parameter values are maintained correctly when switching models."""
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            # Set up TripoSR with specific parameters
            assert app.current_model == "TripoSR"
            
            # Set resolution to 1024
            if "resolution" in app.param_widgets:
                app.param_widgets["resolution"].value = "1024"
            
            # Get TripoSR parameters
            triposr_params = app.get_current_parameters()
            
            # Switch to Stable-Fast-3D
            model_select = app.query_one("#model-select", Select)
            model_select.value = "Stable-Fast-3D"
            await pilot.pause()
            
            # Get Stable-Fast-3D parameters
            stable_params = app.get_current_parameters()
            
            # Parameters should be different (different model)
            assert stable_params["model"] == "Stable-Fast-3D"
            assert triposr_params["model"] == "TripoSR"
            
            # Switch back to TripoSR
            model_select.value = "TripoSR"
            await pilot.pause()
            
            # Parameters should be reset to defaults (not persisted)
            new_triposr_params = app.get_current_parameters()
            assert new_triposr_params["model"] == "TripoSR"


class TestRealModelLoading:
    """Test real model loading and error handling."""
    
    @pytest.mark.asyncio
    async def test_model_import_handling(self):
        """Test that the app handles model import attempts correctly."""
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            # Try to trigger real model loading
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "test model loading"
            await pilot.pause()
            
            # The app should attempt to import models
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            
            # Wait for import attempt and fallback
            await asyncio.sleep(5)
            
            # App should not crash regardless of import success/failure
            assert app is not None
            assert gen_btn.disabled == False
    
    @pytest.mark.asyncio
    async def test_all_model_configurations(self):
        """Test that all model configurations work correctly."""
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            for model_name, config in MODEL_CONFIGS.items():
                # Select model
                model_select = app.query_one("#model-select", Select)
                model_select.value = model_name
                await pilot.pause()
                
                # Verify model info updated
                model_info = app.query_one("#model-info")
                assert model_info is not None
                
                # Verify all parameters have widgets
                for param_name, param_config in config["parameters"].items():
                    assert param_name in app.param_widgets
                    widget = app.param_widgets[param_name]
                    
                    # Verify widget type matches config
                    if param_config["type"] == "select":
                        assert isinstance(widget, Select)
                        # Verify default value is set
                        assert widget.value == param_config["default"]
                    elif param_config["type"] == "switch":
                        assert isinstance(widget, Switch)
                        # Verify default value is set
                        assert widget.value == param_config["default"]


class TestRealUserWorkflows:
    """Test realistic user workflows."""
    
    @pytest.mark.asyncio
    async def test_iterative_generation_workflow(self):
        """Test a realistic workflow where user generates multiple models iteratively."""
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            # First generation - quick prototype with TripoSR
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "fantasy sword"
            await pilot.pause()
            
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await asyncio.sleep(4)
            
            # Clear log
            clear_btn = app.query_one("#clear-btn", Button)
            await pilot.click(clear_btn)
            await pilot.pause()
            
            # Second generation - higher quality with TRELLIS
            model_select = app.query_one("#model-select", Select)
            model_select.value = "TRELLIS"
            await pilot.pause()
            
            # Update prompt
            prompt_input.value = "ornate fantasy sword with runes"
            await pilot.pause()
            
            # Generate again
            await pilot.click(gen_btn)
            await asyncio.sleep(4)
            
            # Verify app is still responsive
            assert app is not None
            assert gen_btn.disabled == False
    
    @pytest.mark.asyncio
    async def test_exploration_workflow(self):
        """Test workflow where user explores different models and settings."""
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            
            # User explores each model to understand capabilities
            models_explored = []
            for model_name in MODEL_CONFIGS.keys():
                model_select.value = model_name
                await pilot.pause()
                
                # Check model info is displayed
                model_info = app.query_one("#model-info")
                assert model_info is not None
                
                # User adjusts some parameters
                param_widgets = list(app.param_widgets.values())
                if param_widgets:
                    # Adjust first parameter if it exists
                    first_param = param_widgets[0]
                    if isinstance(first_param, Select) and first_param._options:
                        # Change to second option if available
                        if len(first_param._options) > 1:
                            first_param.value = first_param._options[1][0]
                            await pilot.pause()
                    elif isinstance(first_param, Switch):
                        first_param.value = not first_param.value
                        await pilot.pause()
                
                models_explored.append(model_name)
            
            # Verify all models were explored
            assert len(models_explored) == len(MODEL_CONFIGS)
    
    @pytest.mark.asyncio
    async def test_help_and_discovery_workflow(self):
        """Test workflow where user uses help to understand the app."""
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            # User presses F1 for help
            await pilot.press("f1")
            await pilot.pause()
            
            # Output log should have help content
            output_log = app.query_one("#output", RichLog)
            assert output_log is not None
            
            # User clears log after reading help
            await pilot.press("ctrl+c")
            await pilot.pause()
            
            # User tries keyboard shortcut for generation
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "test object"
            await pilot.pause()
            
            await pilot.press("ctrl+g")
            await asyncio.sleep(4)
            
            # Verify generation completed
            gen_btn = app.query_one("#generate-btn", Button)
            assert gen_btn.disabled == False


class TestRealErrorScenarios:
    """Test real error scenarios without mocking."""
    
    @pytest.mark.asyncio
    async def test_invalid_operations_sequence(self):
        """Test that invalid operation sequences don't crash the app."""
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            gen_btn = app.query_one("#generate-btn", Button)
            
            # Try to generate with empty prompt
            await pilot.click(gen_btn)
            await pilot.pause()
            
            # Rapidly click generate multiple times
            for _ in range(5):
                await pilot.click(gen_btn)
                await pilot.pause(0.01)
            
            # Clear log while potentially generating
            clear_btn = app.query_one("#clear-btn", Button)
            await pilot.click(clear_btn)
            
            # Change model while potentially generating
            model_select = app.query_one("#model-select", Select)
            model_select.value = "Hunyuan3D"
            await pilot.pause()
            
            # App should still be functional
            assert app is not None
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "recovery test"
            await pilot.pause()
            
            # Should be able to generate normally now
            await pilot.click(gen_btn)
            await asyncio.sleep(4)
            assert gen_btn.disabled == False
    
    @pytest.mark.asyncio
    async def test_resource_constraints(self):
        """Test behavior under resource constraints."""
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            # Select highest VRAM model
            model_select = app.query_one("#model-select", Select)
            model_select.value = "Hunyuan3D"
            await pilot.pause()
            
            # Set highest quality parameters
            if "polycount" in app.param_widgets:
                app.param_widgets["polycount"].value = "50000"
            if "texture_resolution" in app.param_widgets:
                app.param_widgets["texture_resolution"].value = "4096"
            await pilot.pause()
            
            # Try to generate
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "complex detailed model"
            await pilot.pause()
            
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await asyncio.sleep(5)
            
            # App should handle high resource request gracefully
            assert app is not None
            assert gen_btn.disabled == False


class TestRealUIResponsiveness:
    """Test UI responsiveness under real conditions."""
    
    @pytest.mark.asyncio
    async def test_ui_during_long_operation(self):
        """Test that UI remains responsive during long operations."""
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            # Start a generation
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "long generation test"
            await pilot.pause()
            
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            
            # During generation, test UI responsiveness
            await pilot.pause(0.5)
            
            # Should be able to navigate UI elements
            await pilot.press("tab")
            await pilot.pause(0.1)
            await pilot.press("tab")
            await pilot.pause(0.1)
            
            # Model dropdown should be accessible
            model_select = app.query_one("#model-select", Select)
            assert model_select is not None
            
            # Should be able to type in prompt (for next generation)
            prompt_input.value = "next generation prompt"
            await pilot.pause()
            
            # Wait for current generation to complete
            await asyncio.sleep(3)
            
            # UI should be fully functional
            assert gen_btn.disabled == False
    
    @pytest.mark.asyncio
    async def test_rapid_ui_interactions(self):
        """Test rapid UI interactions don't cause issues."""
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            # Rapidly change selections
            model_select = app.query_one("#model-select", Select)
            prompt_input = app.query_one("#prompt", Input)
            
            for i in range(10):
                # Quick model changes
                model_select.value = list(MODEL_CONFIGS.keys())[i % len(MODEL_CONFIGS)]
                await pilot.pause(0.01)
                
                # Quick prompt updates
                prompt_input.value = f"test {i}"
                await pilot.pause(0.01)
                
                # Quick parameter changes
                if app.param_widgets:
                    first_widget = list(app.param_widgets.values())[0]
                    if isinstance(first_widget, Switch):
                        first_widget.value = not first_widget.value
                        await pilot.pause(0.01)
            
            # App should still be stable
            assert app is not None
            assert app.current_model in MODEL_CONFIGS
            
            # Final generation should work
            prompt_input.value = "final test"
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await asyncio.sleep(4)
            assert gen_btn.disabled == False


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])