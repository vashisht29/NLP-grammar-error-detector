#!/usr/bin/env bash
# ==============================================================================
# Independent Self-Hosted LLM & Linguistic Reasoning Backend
# Runs locally on Mac (Apple Silicon MPS / CPU) or on independent Linux GPU (CUDA)
# Zero dependence on third-party cloud APIs (No OpenAI, No Google Gemini)
# ==============================================================================

set -e

PORT=${1:-8000}
HOST="0.0.0.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "======================================================================"
echo "🚀 Starting Independent LLM Backend Server"
echo "📍 Location: http://${HOST}:${PORT}"
echo "🧠 Hardware: $(python3 -c "import torch; print('CUDA' if torch.cuda.is_available() else ('Apple Silicon MPS' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'Multi-core CPU'))" 2>/dev/null || echo "Host CPU")"
echo "======================================================================"

python3 -m src.api.llm_server --host "$HOST" --port "$PORT"
