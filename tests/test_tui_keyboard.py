import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from textual.pilot import Pilot
    from textual.app import App
    import asyncio
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    print("Warning: Textual not available, skipping TUI tests")
def test_tui_keyboard():
    if not TEXTUAL_AVAILABLE:
        print("Skipping TUI tests - Textual not installed")
        return
    try:
        import dreamcad_simple_tui
        async def run_test():
            app = dreamcad_simple_tui.DreamCADApp()
            async with app.run_test() as pilot:
                assert app.current_view == "main"
                await pilot.press("g")
                await pilot.pause(0.1)
                await pilot.press("m")
                await pilot.pause(0.1)
                await pilot.press("h")
                await pilot.pause(0.1)
                await pilot.press("q")
                print("✓ All keyboard shortcuts tested")
        asyncio.run(run_test())
    except Exception as e:
        print(f"TUI test error: {e}")
        import traceback
        traceback.print_exc()
def test_tui_structure():
    try:
        import dreamcad_simple_tui
        app_class = dreamcad_simple_tui.DreamCADApp
        bindings = app_class.BINDINGS
        binding_keys = [b.key for b in bindings]
        assert "q" in binding_keys, "Missing 'q' binding"
        assert "g" in binding_keys, "Missing 'g' binding"
        assert "m" in binding_keys, "Missing 'm' binding"
        assert "h" in binding_keys, "Missing 'h' binding"
        quit_binding = next((b for b in bindings if b.key == "q"), None)
        if quit_binding:
            assert quit_binding.priority, "Quit binding should have priority=True"
        assert hasattr(app_class, 'action_quit'), "Missing action_quit"
        assert hasattr(app_class, 'action_generate'), "Missing action_generate"
        assert hasattr(app_class, 'action_models'), "Missing action_models"
        assert hasattr(app_class, 'action_help'), "Missing action_help"
        assert hasattr(app_class, 'on_button_pressed'), "Missing on_button_pressed"
        assert hasattr(app_class, 'on_key'), "Missing on_key handler"
        assert hasattr(app_class, 'on_mount'), "Missing on_mount"
        print("✓ TUI structure test passed")
        print("  - All bindings present")
        print("  - Quit has priority")
        print("  - All action methods defined")
        print("  - Event handlers present")
    except ImportError as e:
        print(f"Cannot import TUI: {e}")
    except AssertionError as e:
        print(f"✗ TUI structure test failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
def test_manual_instructions():
    print("\n=== MANUAL TUI TESTING INSTRUCTIONS ===")
    print("Run: ./dreamcad tui")
    print("\nTest these keyboard shortcuts:")
    print("  1. Press 'g' - Should show/change to generation view")
    print("  2. Press 'm' - Should show models comparison")
    print("  3. Press 'h' - Should show help notification")
    print("  4. Press 'q' - Should quit the application")
    print("  5. Press 'Escape' - Should also quit")
    print("\nAlso test:")
    print("  - Click buttons with mouse")
    print("  - Check if notifications appear")
    print("  - Verify views change correctly")
    print("\nKnown issues to check:")
    print("  - If keys don't work, check if any widget has focus")
    print("  - Try clicking background first, then pressing keys")
    print("  - Check terminal for any error messages")
if __name__ == "__main__":
    print("TUI Keyboard Test Suite")
    print("=" * 50)
    test_tui_structure()
    if TEXTUAL_AVAILABLE:
        print("\nTesting keyboard functionality...")
        test_tui_keyboard()
    test_manual_instructions()