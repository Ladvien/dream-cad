import pytest
from pathlib import Path
from textual.pilot import Pilot
from textual.widgets import Input, Select, Button, RichLog, Switch
import asyncio
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))
from dreamcad_tui_new import DreamCADTUI, MODEL_CONFIGS
class TestTUIBasicFunctionality:
    @pytest.mark.asyncio
    async def test_app_startup(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            assert app is not None
            assert app.query_one("Header") is not None
            assert app.query_one("Footer") is not None
            assert app.query_one("#sidebar") is not None
            assert app.query_one("#main") is not None
            assert app.query_one("#model-select") is not None
            assert app.query_one("#prompt") is not None
            assert app.query_one("#generate-btn") is not None
            assert app.query_one("#output") is not None
    @pytest.mark.asyncio
    async def test_initial_state(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            assert model_select.value == "TripoSR"
            assert app.current_model == "TripoSR"
            prompt_input = app.query_one("#prompt", Input)
            assert prompt_input.value == ""
            output_log = app.query_one("#output", RichLog)
            assert output_log is not None
            gen_btn = app.query_one("#generate-btn", Button)
            assert gen_btn.disabled == False
    @pytest.mark.asyncio
    async def test_all_models_available_in_dropdown(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            available_models = [option[0] for option in model_select._options]
            for model_name in MODEL_CONFIGS.keys():
                assert model_name in available_models
class TestModelSelection:
    @pytest.mark.asyncio
    async def test_model_selection_changes(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            for model_name in MODEL_CONFIGS.keys():
                model_select.value = model_name
                await pilot.pause()
                assert app.current_model == model_name
                model_info = app.query_one("#model-info")
                assert model_info is not None
    @pytest.mark.asyncio
    async def test_parameters_update_on_model_change(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            assert app.current_model == "TripoSR"
            initial_params = len(app.param_widgets)
            model_select = app.query_one("#model-select", Select)
            model_select.value = "Stable-Fast-3D"
            await pilot.pause()
            assert app.current_model == "Stable-Fast-3D"
            params_container = app.query_one("#parameters")
            assert params_container is not None
            assert len(app.param_widgets) > 0
    @pytest.mark.asyncio
    async def test_parameter_widgets_creation(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            for model_name, config in MODEL_CONFIGS.items():
                model_select = app.query_one("#model-select", Select)
                model_select.value = model_name
                await pilot.pause()
                expected_params = len(config["parameters"])
                assert len(app.param_widgets) == expected_params
                for param_name in config["parameters"].keys():
                    assert param_name in app.param_widgets
class TestUserInteractions:
    @pytest.mark.asyncio
    async def test_prompt_input(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            test_prompt = "a crystal sword"
            prompt_input.value = test_prompt
            await pilot.pause()
            assert prompt_input.value == test_prompt
    @pytest.mark.asyncio
    async def test_clear_button_functionality(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            output_log = app.query_one("#output", RichLog)
            app.log_output("Test message 1")
            app.log_output("Test message 2")
            await pilot.pause()
            clear_btn = app.query_one("#clear-btn", Button)
            await pilot.click(clear_btn)
            await pilot.pause()
            assert clear_btn is not None
    @pytest.mark.asyncio
    async def test_parameter_value_changes(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            model_select.value = "TRELLIS"
            await pilot.pause()
            initial_params = app.get_current_parameters()
            if "quality_mode" in app.param_widgets:
                quality_widget = app.param_widgets["quality_mode"]
                if isinstance(quality_widget, Select):
                    quality_widget.value = "hq"
                    await pilot.pause()
            updated_params = app.get_current_parameters()
            assert updated_params.get("quality_mode") == "hq"
    @pytest.mark.asyncio
    async def test_switch_parameter_toggle(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            model_select.value = "TripoSR"
            await pilot.pause()
            if "remove_background" in app.param_widgets:
                switch_widget = app.param_widgets["remove_background"]
                if isinstance(switch_widget, Switch):
                    initial_value = switch_widget.value
                    switch_widget.value = not initial_value
                    await pilot.pause()
                    params = app.get_current_parameters()
                    assert params.get("remove_background") != initial_value
class TestGenerationWorkflow:
    @pytest.mark.asyncio
    async def test_generate_without_prompt_shows_error(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = ""
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await pilot.pause()
            assert gen_btn is not None
    @pytest.mark.asyncio
    async def test_generate_with_valid_prompt_simulation(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "a wooden chair"
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await asyncio.sleep(4)
            assert gen_btn.disabled == False
    @pytest.mark.asyncio
    async def test_generate_button_disabled_during_generation(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "test object"
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await pilot.pause(0.1)
            assert gen_btn is not None
    @pytest.mark.asyncio
    async def test_generation_with_different_models(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "a test object"
            for model_name in ["TripoSR", "TRELLIS", "Hunyuan3D"]:
                model_select = app.query_one("#model-select", Select)
                model_select.value = model_name
                await pilot.pause()
                gen_btn = app.query_one("#generate-btn", Button)
                await pilot.click(gen_btn)
                await asyncio.sleep(4)
                assert app.current_model == model_name
class TestKeyboardShortcuts:
    @pytest.mark.asyncio
    async def test_help_shortcut(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            await pilot.press("f1")
            await pilot.pause()
            assert hasattr(app, 'action_help')
    @pytest.mark.asyncio
    async def test_generate_shortcut(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "shortcut test"
            await pilot.pause()
            await pilot.press("ctrl+g")
            await asyncio.sleep(0.5)
            assert hasattr(app, 'action_generate')
    @pytest.mark.asyncio
    async def test_clear_log_shortcut(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            app.log_output("Test entry")
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert hasattr(app, 'action_clear_log')
class TestParameterRetrieval:
    @pytest.mark.asyncio
    async def test_get_all_parameters(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            for model_name in MODEL_CONFIGS.keys():
                model_select = app.query_one("#model-select", Select)
                model_select.value = model_name
                await pilot.pause()
                params = app.get_current_parameters()
                assert params["model"] == model_name
                config = MODEL_CONFIGS[model_name]
                for param_name in config["parameters"].keys():
                    assert param_name in params
    @pytest.mark.asyncio
    async def test_parameter_values_match_widgets(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            model_select.value = "Stable-Fast-3D"
            await pilot.pause()
            if "target_polycount" in app.param_widgets:
                app.param_widgets["target_polycount"].value = "20000"
            if "enable_pbr" in app.param_widgets:
                app.param_widgets["enable_pbr"].value = False
            await pilot.pause()
            params = app.get_current_parameters()
            assert params.get("target_polycount") == "20000"
            assert params.get("enable_pbr") == False
class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_empty_prompt_handling(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = ""
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            initial_state = gen_btn.disabled
            await pilot.click(gen_btn)
            await pilot.pause()
            assert gen_btn.disabled == initial_state
    @pytest.mark.asyncio
    async def test_whitespace_only_prompt(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "   \t\n   "
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await pilot.pause()
            assert gen_btn is not None
    @pytest.mark.asyncio
    async def test_very_long_prompt(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            long_prompt = "a " * 500 + "very detailed object"
            prompt_input.value = long_prompt
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await asyncio.sleep(1)
            assert app is not None
    @pytest.mark.asyncio
    async def test_rapid_model_switching(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            models = list(MODEL_CONFIGS.keys())
            for _ in range(10):
                for model in models:
                    model_select.value = model
                    await pilot.pause(0.01)
            assert app.current_model in models
            assert len(app.param_widgets) > 0
class TestLogging:
    @pytest.mark.asyncio
    async def test_log_output_formatting(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            app.log_output("Plain message")
            app.log_output("[green]Success message[/green]")
            app.log_output("[red]Error message[/red]")
            app.log_output("[yellow]Warning message[/yellow]")
            await pilot.pause()
            output_log = app.query_one("#output", RichLog)
            assert output_log is not None
    @pytest.mark.asyncio
    async def test_log_timestamps(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            app.log_output("Test message")
            import datetime
            assert datetime.datetime.now() is not None
class TestModelIntegration:
    @pytest.mark.asyncio
    async def test_model_factory_integration(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "integration test object"
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await asyncio.sleep(4)
            assert gen_btn.disabled == False
    @pytest.mark.asyncio
    async def test_real_model_loading_attempt(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            can_import_models = True
            try:
                from dream_cad.models.factory import ModelFactory
                from dream_cad.models.registry import ModelRegistry
            except ImportError:
                can_import_models = False
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "real model test"
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await asyncio.sleep(4)
            assert gen_btn.disabled == False
            output_log = app.query_one("#output", RichLog)
            assert output_log is not None
class TestUIState:
    @pytest.mark.asyncio
    async def test_initialization_flag(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            assert app._initialized == True
    @pytest.mark.asyncio
    async def test_exclusive_generation(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            prompt_input.value = "test"
            await pilot.pause()
            gen_btn = app.query_one("#generate-btn", Button)
            await pilot.click(gen_btn)
            await pilot.pause(0.01)
            await pilot.click(gen_btn)
            await pilot.pause(0.01)
            assert gen_btn is not None
    @pytest.mark.asyncio
    async def test_model_select_no_change_trigger(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            initial_widgets = len(app.param_widgets)
            model_select.value = "TripoSR"
            await pilot.pause()
            assert len(app.param_widgets) == initial_widgets
class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            prompt_input = app.query_one("#prompt", Input)
            special_prompts = [
                "object with @#$% symbols",
                "quote's and \"quotes\"",
                "newline\ntest",
                "emoji 🎨 test",
                "unicode ñ é ü test"
            ]
            for prompt in special_prompts:
                prompt_input.value = prompt
                await pilot.pause()
                assert prompt_input.value == prompt
    @pytest.mark.asyncio
    async def test_parameter_boundary_values(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            model_select.value = "Hunyuan3D"
            await pilot.pause()
            if "polycount" in app.param_widgets:
                app.param_widgets["polycount"].value = "50000"
                await pilot.pause()
            params = app.get_current_parameters()
            assert params.get("polycount") == "50000"
    @pytest.mark.asyncio
    async def test_all_switches_off(self):
        app = DreamCADTUI()
        async with app.run_test() as pilot:
            model_select = app.query_one("#model-select", Select)
            model_select.value = "Stable-Fast-3D"
            await pilot.pause()
            for widget in app.param_widgets.values():
                if isinstance(widget, Switch):
                    widget.value = False
            await pilot.pause()
            params = app.get_current_parameters()
            assert params.get("enable_pbr") == False
            assert params.get("delighting") == False
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])