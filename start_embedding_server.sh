#!/bin/bash

# Script to start the Qwen3 embedding server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"
MODEL_PATH="$LLAMA_DIR/models/qwen3-embedding-0.6b-q8_0.gguf"
BINARY_PATH="$LLAMA_DIR/build/bin/llama-server"

echo "🚀 Starting Qwen3 Embedding Server..."

# Check if binary exists
if [ ! -f "$BINARY_PATH" ]; then
    echo "❌ Error: llama-server binary not found at $BINARY_PATH"
    echo "Please run: cd llama.cpp/build && make llama-server"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    echo "Please download the model first"
    exit 1
fi

# Check if server is already running
if curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "✅ Embedding server is already running on port 8080"
    exit 0
fi

echo "📂 Model: $MODEL_PATH"
echo "🌐 Starting server on http://127.0.0.1:8080"
echo "💡 Press Ctrl+C to stop the server"
echo ""

cd "$LLAMA_DIR/build"
exec ./bin/llama-server \
    -m "$MODEL_PATH" \
    --embedding \
    --pooling last \
    -ub 8192 \
    --verbose-prompt \
    --port 8080