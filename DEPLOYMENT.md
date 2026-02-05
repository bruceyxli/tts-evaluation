# TTS Evaluation Pipeline - Linux Deployment Guide

This guide covers deploying the TTS evaluation pipeline on a remote Linux server with NVIDIA GPU.

## For Claude Code

If you are Claude Code helping to deploy this project, follow the step-by-step commands below. Execute each section in order and verify the output before proceeding.

---

## Prerequisites Check

First, verify the system meets requirements:

```bash
# Check OS
cat /etc/os-release

# Check NVIDIA driver and GPU
nvidia-smi

# Check available disk space (need ~50GB)
df -h .

# Check if conda is installed
conda --version || echo "Conda not installed"

# Check if git is installed
git --version
```

**Required:**
- Linux (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA support (check nvidia-smi output)
- At least 50GB free disk space
- Git installed

---

## Step 1: Install System Dependencies

```bash
# Update package lists
sudo apt update

# Install required system packages
sudo apt install -y build-essential git curl wget ffmpeg libsndfile1 sox

# Verify ffmpeg installation
ffmpeg -version | head -1
```

---

## Step 2: Install Miniconda (if not installed)

Skip this step if `conda --version` already works.

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

# Install Miniconda (non-interactive)
bash /tmp/miniconda.sh -b -p $HOME/miniconda3

# Initialize conda for bash
$HOME/miniconda3/bin/conda init bash

# IMPORTANT: Reload shell configuration
source ~/.bashrc

# Verify conda installation
conda --version
```

**Note:** After `conda init bash`, you may need to open a new terminal or run `source ~/.bashrc` for conda to work.

---

## Step 3: Clone the Repository

```bash
# Navigate to your preferred directory
cd ~

# Clone the tts-evaluation repository
# Replace YOUR_USERNAME with actual GitHub username
git clone https://github.com/YOUR_USERNAME/tts-evaluation.git

# Enter project directory
cd tts-evaluation

# Verify files are present
ls -la
```

**Expected files:** `main.py`, `model_runner.py`, `setup_models.py`, `config.yaml`, `envs/`, `scripts/`

---

## Step 4: Clone TTS Model Repositories

Each TTS model has its own source repository that must be cloned:

```bash
# Ensure we're in the project directory
cd ~/tts-evaluation

# Clone GLM-TTS
git clone https://github.com/THUDM/GLM-TTS.git
echo "GLM-TTS cloned: $(ls GLM-TTS/ | wc -l) files"

# Clone CosyVoice (with submodules for Matcha-TTS)
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
echo "CosyVoice cloned: $(ls CosyVoice/ | wc -l) files"

# (Optional) Clone Qwen3-TTS for reference
git clone https://github.com/QwenLM/Qwen3-TTS.git
echo "Qwen3-TTS cloned: $(ls Qwen3-TTS/ | wc -l) files"
```

**Verify:**
```bash
ls -d GLM-TTS CosyVoice Qwen3-TTS
```

---

## Step 5: Create Conda Environments

The setup script creates isolated conda environments for each TTS model:

```bash
cd ~/tts-evaluation

# View available options
python setup_models.py --help

# Create ALL environments (recommended)
python setup_models.py --all

# Or create specific environments:
# python setup_models.py --glm        # GLM-TTS only
# python setup_models.py --cosyvoice  # CosyVoice only
# python setup_models.py --qwen       # Qwen3-TTS only
# python setup_models.py --qwen-vllm  # Qwen3-TTS with vLLM (faster, Linux only)
```

**This will take 10-30 minutes** depending on network speed.

**Verify environments were created:**
```bash
python setup_models.py --list

# Or manually check
conda env list | grep tts
```

**Expected environments:**
- `tts-glm` - for GLM-TTS
- `tts-cosyvoice` - for CosyVoice
- `tts-qwen` - for Qwen3-TTS
- `tts-qwen-vllm` - for Qwen3-TTS with vLLM acceleration (optional)

---

## Step 6: Download Model Weights

### 6.1 Install modelscope (for downloading from Chinese mirrors)

```bash
pip install modelscope huggingface_hub
```

### 6.2 Download GLM-TTS Weights (~5GB)

```bash
cd ~/tts-evaluation

python -c "
from modelscope import snapshot_download
print('Downloading GLM-TTS weights from ModelScope...')
snapshot_download('ZhipuAI/GLM-TTS', local_dir='./GLM-TTS/ckpt')
print('Done!')
"

# Verify download
ls -la GLM-TTS/ckpt/
```

**Expected files in `GLM-TTS/ckpt/`:**
- `glm-4-voice-tokenizer/`
- `glm-4-voice-decoder/`
- `speech_tokenizer/` or similar

### 6.3 Download CosyVoice Weights (~3GB)

```bash
cd ~/tts-evaluation

python -c "
from modelscope import snapshot_download
print('Downloading CosyVoice weights from ModelScope...')
snapshot_download('iic/Fun-CosyVoice3-0.5B', local_dir='./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')
print('Done!')
"

# Verify download
ls -la CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/
```

**Expected files:**
- `llm.pt`
- `flow.pt`
- `hift.pt`
- `speech_tokenizer/`

### 6.4 Qwen3-TTS Weights (Auto-download)

Qwen3-TTS weights are automatically downloaded from Hugging Face on first use. No manual download needed.

**Optional: Pre-download to avoid delays during evaluation:**
```bash
conda run -n tts-qwen python -c "
from qwen_tts import Qwen3TTSModel
print('Pre-downloading Qwen3-TTS CustomVoice model...')
model = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice')
print('Done!')
"
```

---

## Step 7: Verify Each Environment

Test that each environment works correctly:

### 7.1 Test GLM-TTS Environment

```bash
conda run -n tts-glm python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"
```

### 7.2 Test CosyVoice Environment

```bash
conda run -n tts-cosyvoice python -c "
import torch
import pynini  # CosyVoice requires pynini
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('pynini OK')
"
```

### 7.3 Test Qwen-TTS Environment

```bash
conda run -n tts-qwen python -c "
import torch
from qwen_tts import Qwen3TTSModel
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('qwen-tts import OK')
"
```

### 7.4 Test vLLM Environment (if installed)

```bash
conda run -n tts-qwen-vllm python -c "
import torch
from vllm import LLM
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('vLLM import OK')
" 2>/dev/null || echo "vLLM environment not installed or has issues"
```

---

## Step 8: Prepare Reference Audio (Voice Cloning)

For voice cloning, you need reference audio files:

```bash
cd ~/tts-evaluation

# Create directory for processed audio
mkdir -p processed_audio

# If you have reference audio, copy it here:
# cp /path/to/your/speaker.wav processed_audio/

# Normalize audio (recommended for better quality)
# ffmpeg -i processed_audio/speaker.wav -af loudnorm -ar 16000 processed_audio/normalized_speaker.wav
```

**Reference audio requirements:**
- Format: WAV (recommended) or MP3
- Duration: 5-15 seconds (optimal for voice cloning)
- Quality: Clear speech, minimal background noise
- Sample rate: 16kHz or higher

**If no reference audio is available:** The evaluation will still work with models that have preset voices (Qwen3-TTS CustomVoice).

---

## Step 9: Configure the Evaluation

Edit `config.yaml` to customize which models to evaluate:

```bash
cd ~/tts-evaluation

# View current configuration
cat config.yaml

# Edit configuration (use your preferred editor)
# nano config.yaml
# or
# vim config.yaml
```

**Key settings to check:**

```yaml
# Device settings
device: "cuda"  # Use "cpu" if no GPU
gpu_index: 0    # Change if using different GPU

# Enable/disable models
models:
  - name: "glm_tts"
    enabled: true   # Set to false to skip this model

  - name: "cosyvoice"
    enabled: true

  - name: "qwen_tts"
    enabled: true
```

**If you have reference audio for voice cloning, update paths:**

```yaml
models:
  - name: "glm_tts"
    prompt_wav: "./processed_audio/normalized_speaker.wav"
    prompt_text: "The exact transcript of your reference audio."
```

---

## Step 10: Run a Quick Test

Test with a single model first:

```bash
cd ~/tts-evaluation

# Test GLM-TTS with debug script
conda run -n tts-glm python debug_single_model.py --model glm_tts

# Check output
ls -la outputs/debug_*.wav
```

**If successful, you should see:**
- Audio file generated in `outputs/`
- No error messages

---

## Step 11: Run Full Evaluation

```bash
cd ~/tts-evaluation

# Run evaluation on all enabled models
python main.py

# Or run specific models only
python main.py -m glm_tts cosyvoice

# Or use a specific test script
python main.py -s scripts/tts_evaluation_dataset.json
```

**Output will be saved to:** `outputs/<timestamp>_<model_names>/`

**Check results:**
```bash
# Find latest output directory
LATEST_OUTPUT=$(ls -td outputs/*/ | head -1)
echo "Latest output: $LATEST_OUTPUT"

# View metrics
cat "${LATEST_OUTPUT}metrics.json"

# View report
cat "${LATEST_OUTPUT}report.md"

# List generated audio files
find "$LATEST_OUTPUT" -name "*.wav" | head -10
```

---

## RTX 50 Series (Blackwell sm_120) Support

**IMPORTANT:** If the remote machine has an RTX 5080 or 5090 GPU, standard PyTorch will NOT work. You need PyTorch nightly with CUDA 12.8.

### Check GPU Architecture

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

If compute capability shows `12.0` (sm_120), you have a Blackwell GPU.

### Upgrade PyTorch in Each Environment

```bash
# Upgrade tts-glm
conda run -n tts-glm pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Upgrade tts-cosyvoice
conda run -n tts-cosyvoice pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Upgrade tts-qwen
conda run -n tts-qwen pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Upgrade tts-qwen-vllm (if installed)
conda run -n tts-qwen-vllm pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Verify PyTorch Nightly

```bash
conda run -n tts-glm python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU works: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    # Test actual GPU computation
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'GPU compute test: OK')
"
```

### Known Issue: torchaudio.load() with PyTorch Nightly

PyTorch nightly's torchaudio may fail with `torchcodec` errors. The codebase uses `soundfile` as a workaround. If you see errors like:

```
ImportError: TorchCodec is required for load_with_torchcodec
```

The code should automatically use soundfile instead. If not, install soundfile:

```bash
conda run -n tts-glm pip install soundfile
conda run -n tts-cosyvoice pip install soundfile
```

---

## Troubleshooting

### Problem: "CUDA out of memory"

```bash
# Check GPU memory usage
nvidia-smi

# Kill zombie Python processes
pkill -f "python.*model_runner"

# Clear GPU memory cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Problem: CosyVoice import errors

```bash
# Ensure submodules are initialized
cd ~/tts-evaluation/CosyVoice
git submodule update --init --recursive

# Check if Matcha-TTS exists
ls third_party/Matcha-TTS/
```

### Problem: pynini installation fails

pynini MUST be installed via conda, not pip:

```bash
conda activate tts-cosyvoice
conda install -c conda-forge pynini=2.1.5
```

### Problem: Model download fails (network issues)

Try using a mirror or manual download:

```bash
# For GLM-TTS, try huggingface instead of modelscope
huggingface-cli download ZhipuAI/GLM-TTS --local-dir ./GLM-TTS/ckpt

# For CosyVoice
huggingface-cli download FunAudioLLM/CosyVoice-300M-SFT --local-dir ./CosyVoice/pretrained_models/CosyVoice-300M-SFT
```

### Problem: "No module named 'xxx'"

Ensure you're using the correct conda environment:

```bash
# List all environments
conda env list

# Activate specific environment and check
conda activate tts-glm
pip list | grep torch
```

---

## Environment Variables (Optional)

Add these to `~/.bashrc` for persistent configuration:

```bash
# Hugging Face cache directory (saves disk space on home)
export HF_HOME=/data/hf_cache

# Specify GPU (if multiple GPUs)
export CUDA_VISIBLE_DEVICES=0

# Suppress tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# ModelScope cache
export MODELSCOPE_CACHE=/data/modelscope_cache
```

Apply changes:
```bash
source ~/.bashrc
```

---

## Quick Reference Commands

```bash
# Check environment status
python setup_models.py --list

# Run specific model
python main.py -m glm_tts

# Run all models
python main.py

# Debug single model
conda run -n tts-glm python debug_single_model.py --model glm_tts
conda run -n tts-cosyvoice python debug_single_model.py --model cosyvoice
conda run -n tts-qwen python debug_single_model.py --model qwen_tts

# View latest results
cat outputs/$(ls -t outputs/ | head -1)/report.md
```

---

## Summary Checklist

- [ ] System dependencies installed (ffmpeg, libsndfile1)
- [ ] Miniconda installed and initialized
- [ ] Repository cloned
- [ ] Model repositories cloned (GLM-TTS, CosyVoice)
- [ ] Conda environments created (`python setup_models.py --all`)
- [ ] Model weights downloaded
- [ ] Each environment verified working
- [ ] Reference audio prepared (optional, for voice cloning)
- [ ] config.yaml customized
- [ ] Test run successful
- [ ] (If RTX 50 series) PyTorch nightly installed

---

## Contact

For issues, provide:
1. Output of `nvidia-smi`
2. Output of `python setup_models.py --list`
3. Full error traceback
4. Contents of `config.yaml`
