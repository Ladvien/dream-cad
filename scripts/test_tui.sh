#!/bin/bash
# Test runner for TUI end-to-end tests

echo "🧪 Running DreamCAD TUI End-to-End Tests"
echo "========================================"
echo ""

# Run tests with coverage if requested
if [ "$1" == "--coverage" ]; then
    echo "Running with coverage..."
    poetry run pytest tests/test_tui_e2e.py \
        --asyncio-mode=auto \
        --cov=dreamcad_tui_new \
        --cov-report=term-missing \
        --cov-report=html \
        -v
    echo ""
    echo "Coverage report generated at htmlcov/index.html"
else
    # Run tests normally
    poetry run pytest tests/test_tui_e2e.py \
        --asyncio-mode=auto \
        -v \
        "$@"
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All TUI tests passed!"
else
    echo ""
    echo "❌ Some tests failed. Please review the output above."
    exit 1
fi