#!/bin/bash
# Clone TTS model repositories

set -e

echo "=== Cloning TTS Model Repositories ==="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# GLM-TTS
if [ -d "GLM-TTS" ]; then
    echo "[SKIP] GLM-TTS already exists"
else
    echo "[CLONE] GLM-TTS..."
    git clone https://github.com/THUDM/GLM-TTS.git
fi

# CosyVoice (with submodules)
if [ -d "CosyVoice" ]; then
    echo "[SKIP] CosyVoice already exists"
else
    echo "[CLONE] CosyVoice..."
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
fi

# Qwen3-TTS (optional, for reference)
if [ -d "Qwen3-TTS" ]; then
    echo "[SKIP] Qwen3-TTS already exists"
else
    echo "[CLONE] Qwen3-TTS..."
    git clone https://github.com/QwenLM/Qwen3-TTS.git
fi

echo ""
echo "=== Done ==="
echo "Next step: Run ./scripts/download_models.sh to download model weights"
