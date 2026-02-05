#!/bin/bash
# Download TTS model weights

set -e

echo "=== Downloading TTS Model Weights ==="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check for modelscope
if ! python -c "import modelscope" 2>/dev/null; then
    echo "[INSTALL] Installing modelscope..."
    pip install modelscope
fi

# GLM-TTS weights
echo ""
echo "[1/2] Downloading GLM-TTS weights..."
if [ -d "GLM-TTS/ckpt/glm-4-voice-tokenizer" ]; then
    echo "[SKIP] GLM-TTS weights already exist"
else
    python -c "
from modelscope import snapshot_download
print('Downloading GLM-TTS from ModelScope...')
snapshot_download('ZhipuAI/GLM-TTS', local_dir='./GLM-TTS/ckpt')
print('Done!')
"
fi

# CosyVoice weights
echo ""
echo "[2/2] Downloading CosyVoice weights..."
if [ -d "CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/llm.pt" ]; then
    echo "[SKIP] CosyVoice weights already exist"
else
    python -c "
from modelscope import snapshot_download
print('Downloading CosyVoice from ModelScope...')
snapshot_download('iic/Fun-CosyVoice3-0.5B', local_dir='./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')
print('Done!')
"
fi

echo ""
echo "=== Download Complete ==="
echo ""
echo "Note: Qwen3-TTS weights are downloaded automatically from Hugging Face"
echo "when first used. If you need to pre-download:"
echo ""
echo "  python -c \"from qwen_tts import Qwen3TTSModel; Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice')\""
echo ""
echo "Next step: Run 'python setup_models.py --all' to create conda environments"
