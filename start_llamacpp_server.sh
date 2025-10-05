#!/bin/bash

# Start llama-server with SmolVLM GGUF for real-time inference
# This achieves sub-1-second inference (vs 30-36s with HuggingFace Transformers)

echo "🚀 Starting llama-server with SmolVLM GGUF..."
echo "📊 Expected performance: 250-500ms per frame"
echo "🎯 Target: Real-time video analysis (<1s)"
echo ""

# Kill any existing llama-server on port 8080
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Killing existing llama-server on port 8080..."
    kill -9 $(lsof -Pi :8080 -sTCP:LISTEN -t)
    sleep 1
fi

# Start llama-server
llama-server \
  -hf ggml-org/SmolVLM-500M-Instruct-GGUF \
  -ngl 99 \
  --port 8080 \
  --ctx-size 2048 \
  --n-predict 150 \
  --host 127.0.0.1 \
  --alias smolvlm \
  --log-disable \
  2>&1 | tee /tmp/llama-server.log

# Flags explained:
# -hf: Download model from HuggingFace (includes multimodal projector)
# -ngl 99: Offload all layers to Metal GPU for maximum speed
# --port 8080: Different from FastAPI backend (8001)
# --ctx-size 2048: Context window size
# --n-predict 150: Maximum tokens to generate
# --host 127.0.0.1: Localhost only (security)
# --alias smolvlm: Model name for API calls
# --log-disable: Reduce log verbosity
