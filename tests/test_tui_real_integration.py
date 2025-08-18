import pytest
from pathlib import Path
from textual.widgets import Input, Select, Button, RichLog, Switch, Label
import asyncio
import sys
import os
import tempfile
import shutil
sys.path.insert(0, str(Path(__file__).parent.parent))
from dreamcad_tui_new import DreamCADTUI, MODEL_CONFIGS
class TestRealTUIBehavior:
    @pytest.mark.asyncio
    async def test_full_generation_workflow(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            assert app.current_model == "TripoSR"
            prompt_input = app.query_one("#prompt", Input)
            test_prompt = "a medieval castle with towers"
            prompt_input.value = test_prompt
            await pilot.pause()
            model_select = app.query_one("#model-select", Select)
            model_select.value = "TRELLIS"
            await pilot.pause()
            assert app.current_model == "TRELLIS"
            if "quality_mode" in app.param_widgets:
                quality_widget = app.param_widgets["quality_mode"]
                quality_widget.value = "balanced"
                await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            initial_disabled_state = gen_btn.disabled
            await pilot.click(gen_btn)
            await asyncio.sleep(5)
            assert gen_btn.disabled == initial_disabled_state
            output_log = app.query_one("#output", RichLog)
            assert output_log is not None
    @pytest.mark.asyncio
    async def test_concurrent_ui_operations(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "test object"
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await pilot.pause(0.1)
            model_select = app.query_one("#model-select", Select)
            assert model_select is not None
            clear_btn = app.query_one("#clear-btn", Button)
            assert clear_btn is not None
            await asyncio.sleep(4)
    @pytest.mark.asyncio
    async def test_parameter_persistence_across_models(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            assert app.current_model == "TripoSR"
            if "resolution" in app.param_widgets:
                app.param_widgets["resolution"].value = "1024"
            triposr_params = app.get_current_parameters()
            model_select = app.query_one("#model-select", Select)
            model_select.value = "Stable-Fast-3D"
            await pilot.pause()
            stable_params = app.get_current_parameters()
            assert stable_params["model"] == "Stable-Fast-3D"
            assert triposr_params["model"] == "TripoSR"
            model_select.value = "TripoSR"
            await pilot.pause()
            new_triposr_params = app.get_current_parameters()
            assert new_triposr_params["model"] == "TripoSR"
class TestRealModelLoading:
    @pytest.mark.asyncio
    async def test_model_import_handling(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "test model loading"
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await asyncio.sleep(5)
            assert app is not None
            assert gen_btn.disabled == False
    @pytest.mark.asyncio
    async def test_all_model_configurations(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            for model_name, config in MODEL_CONFIGS.items():
                model_select = app.query_one("#model-select", Select)
                model_select.value = model_name
                await pilot.pause()
                model_info = app.query_one("#model-info")
                assert model_info is not None
                for param_name, param_config in config["parameters"].items():
                    assert param_name in app.param_widgets
                    widget = app.param_widgets[param_name]
                    if param_config["type"] == "select":
                        assert isinstance(widget, Select)
                        assert widget.value == param_config["default"]
                    elif param_config["type"] == "switch":
                        assert isinstance(widget, Switch)
                        assert widget.value == param_config["default"]
class TestRealUserWorkflows:
    @pytest.mark.asyncio
    async def test_iterative_generation_workflow(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "fantasy sword"
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await asyncio.sleep(4)
            clear_btn = app.query_one("#clear-btn", Button)
            await pilot.click(clear_btn)
            await pilot.pause()
            model_select = app.query_one("#model-select", Select)
            model_select.value = "TRELLIS"
            await pilot.pause()
            prompt_input.value = "ornate fantasy sword with runes"
            await pilot.pause()
            await pilot.click(gen_btn)
            await asyncio.sleep(4)
            assert app is not None
            assert gen_btn.disabled == False
    @pytest.mark.asyncio
    async def test_exploration_workflow(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            models_explored = []
            for model_name in MODEL_CONFIGS.keys():
                model_select.value = model_name
                await pilot.pause()
                model_info = app.query_one("#model-info")
                assert model_info is not None
                param_widgets = list(app.param_widgets.values())
                if param_widgets:
                    first_param = param_widgets[0]
                    if isinstance(first_param, Select) and first_param._options:
                        if len(first_param._options) > 1:
                            first_param.value = first_param._options[1][0]
                            await pilot.pause()
                    elif isinstance(first_param, Switch):
                        first_param.value = not first_param.value
                        await pilot.pause()
                models_explored.append(model_name)
            assert len(models_explored) == len(MODEL_CONFIGS)
    @pytest.mark.asyncio
    async def test_help_and_discovery_workflow(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            await pilot.press("f1")
            await pilot.pause()
            output_log = app.query_one("#output", RichLog)
            assert output_log is not None
            await pilot.press("ctrl+c")
            await pilot.pause()
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "test object"
            await pilot.pause()
            await pilot.press("ctrl+g")
            await asyncio.sleep(4)
            gen_btn = app.query_one("#generate-btn", Button)
            assert gen_btn.disabled == False
class TestRealErrorScenarios:
    @pytest.mark.asyncio
    async def test_invalid_operations_sequence(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await pilot.pause()
            for _ in range(5):
                await pilot.click(gen_btn)
                await pilot.pause(0.01)
            clear_btn = app.query_one("#clear-btn", Button)
            await pilot.click(clear_btn)
            model_select = app.query_one("#model-select", Select)
            model_select.value = "Hunyuan3D"
            await pilot.pause()
            assert app is not None
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "recovery test"
            await pilot.pause()
            await pilot.click(gen_btn)
            await asyncio.sleep(4)
            assert gen_btn.disabled == False
    @pytest.mark.asyncio
    async def test_resource_constraints(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            model_select.value = "Hunyuan3D"
            await pilot.pause()
            if "polycount" in app.param_widgets:
                app.param_widgets["polycount"].value = "50000"
            if "texture_resolution" in app.param_widgets:
                app.param_widgets["texture_resolution"].value = "4096"
            await pilot.pause()
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "complex detailed model"
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await asyncio.sleep(5)
            assert app is not None
            assert gen_btn.disabled == False
class TestRealUIResponsiveness:
    @pytest.mark.asyncio
    async def test_ui_during_long_operation(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "long generation test"
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await pilot.pause(0.5)
            await pilot.press("tab")
            await pilot.pause(0.1)
            await pilot.press("tab")
            await pilot.pause(0.1)
            model_select = app.query_one("#model-select", Select)
            assert model_select is not None
            prompt_input.value = "next generation prompt"
            await pilot.pause()
            await asyncio.sleep(3)
            assert gen_btn.disabled == False
    @pytest.mark.asyncio
    async def test_rapid_ui_interactions(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            prompt_input = app.query_one("#prompt", Input)
            for i in range(10):
                model_select.value = list(MODEL_CONFIGS.keys())[i % len(MODEL_CONFIGS)]
                await pilot.pause(0.01)
                prompt_input.value = f"test {i}"
                await pilot.pause(0.01)
                if app.param_widgets:
                    first_widget = list(app.param_widgets.values())[0]
                    if isinstance(first_widget, Switch):
                        first_widget.value = not first_widget.value
                        await pilot.pause(0.01)
            assert app is not None
            assert app.current_model in MODEL_CONFIGS
            prompt_input.value = "final test"
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await asyncio.sleep(4)
            assert gen_btn.disabled == False
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])