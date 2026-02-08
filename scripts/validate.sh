#!/bin/bash
set -e

echo "🔍 Running quality gates..."

echo ""
echo "📦 Installing dependencies..."
uv sync

echo ""
echo "🧪 Running unit tests..."
python -m pytest tests/ -v --tb=short

echo ""
echo "🔬 Running e2e tests..."
python -m pytest -m e2e tests/ -v --tb=short

echo ""
echo "🔎 Running mypy..."
python -m mypy src/

echo ""
echo "🔧 Running ruff check..."
python -m ruff check src/

echo ""
echo "🎨 Running ruff format check..."
python -m ruff format src/ --check

echo ""
echo "✅ All quality gates passed!"
