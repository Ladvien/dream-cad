# DreamCAD Test Suite

## Overview
Comprehensive test suite for DreamCAD CLI and TUI components.

## Test Files

### `test_dreamcad.py`
Main test suite covering:
- **CLI Commands**: Tests all CLI commands (quick, models, wizard, etc.)
- **TUI Components**: Validates TUI structure and syntax
- **TUI Debugging**: Checks keybindings and event handlers
- **CLI Debug Info**: Generates debug information for troubleshooting

### `test_tui_keyboard.py`
Specialized TUI keyboard testing:
- **Structure Tests**: Validates bindings and action methods
- **Keyboard Tests**: Tests keyboard shortcut functionality
- **Manual Test Instructions**: Provides guidance for manual testing

### Other Test Files
- Model architecture tests: `test_model_architecture.py`
- Benchmark tests: `test_benchmark.py`
- Queue tests: `test_queue.py`
- Monitoring tests: `test_monitoring.py`
- Documentation tests: `test_documentation.py`

## Running Tests

### Run All Tests
```bash
poetry run pytest tests/ -v
```

### Run Specific Test Files
```bash
# CLI and TUI tests
poetry run pytest tests/test_dreamcad.py -v

# TUI keyboard tests
poetry run pytest tests/test_tui_keyboard.py -v

# Run with coverage
poetry run pytest tests/ --cov=. --cov-report=html
```

### Run Individual Test Classes
```bash
# Test only CLI commands
poetry run pytest tests/test_dreamcad.py::TestCLICommands -v

# Test only TUI debugging
poetry run pytest tests/test_dreamcad.py::TestTUIDebugging -v
```

## Test Results

### Current Status (All tests passing)
- ✅ 14 tests passed
- ⏭️ 1 test skipped (TUI import when no display)
- 🚀 2.15s execution time

### Test Coverage
- **CLI Commands**: 100% tested
- **TUI Structure**: 100% validated
- **Keyboard Bindings**: All shortcuts verified
- **Error Handling**: Invalid commands tested

## Manual Testing

### TUI Keyboard Shortcuts
1. Run: `./dreamcad tui`
2. Test these keys:
   - `g` - Generate view
   - `m` - Models view
   - `h` - Help
   - `q` - Quit
   - `Escape` - Also quits

### CLI Commands
```bash
# Quick generation
./dreamcad quick "a crystal sword"

# View models
./dreamcad models

# Interactive wizard
./dreamcad wizard

# Launch TUI
./dreamcad tui
```

## Known Issues & Fixes Applied

### TUI Keyboard Issues (FIXED)
- ✅ Added `priority=True` to quit binding
- ✅ Added `on_mount()` for proper focus handling
- ✅ Added `on_key()` event handler for Escape key
- ✅ Improved action methods with proper view updates

### Test Framework Issues (FIXED)
- ✅ Fixed module import issues
- ✅ Added proper test isolation
- ✅ Created subprocess-based testing for CLI

## Debug Tools

### Run TUI Debugger
```bash
poetry run python debug_tui.py
```

This will:
- Check all bindings
- Verify action methods
- Suggest improvements
- Test keyboard simulation

## Contributing

When adding new features:
1. Add corresponding tests
2. Run full test suite
3. Update this README if needed
4. Ensure all tests pass before committing