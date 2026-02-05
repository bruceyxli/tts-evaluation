# TTS Evaluation Pipeline - Linux Deployment Guide

This guide covers deploying the TTS evaluation pipeline on a remote Linux server with NVIDIA GPU.

## Prerequisites

- **Linux** (Ubuntu 20.04+ recommended)
- **NVIDIA GPU** with CUDA support
- **CUDA 12.1+** (or CUDA 12.8 for RTX 50 series)
- **Miniconda or Anaconda**
- **Git**
- **~50GB disk space** (for models and environments)

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/tts-evaluation.git
cd tts-evaluation

# 2. Run setup script (creates all conda environments)
python setup_models.py --all

# 3. Clone model repositories
./scripts/clone_repos.sh

# 4. Download model weights
./scripts/download_models.sh

# 5. Run evaluation
python main.py
```

## Detailed Setup

### Step 1: System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essentials
sudo apt install -y build-essential git curl wget ffmpeg libsndfile1

# Verify NVIDIA driver
nvidia-smi
```

### Step 2: Install Miniconda

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install (follow prompts)
bash Miniconda3-latest-Linux-x86_64.sh

# Initialize conda
source ~/.bashrc
conda init bash
```

### Step 3: Clone Model Repositories

The TTS models require their source repositories. Clone them into the project directory:

```bash
cd tts-evaluation

# GLM-TTS
git clone https://github.com/THUDM/GLM-TTS.git

# CosyVoice
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

# (Optional) Qwen3-TTS repo for reference
git clone https://github.com/QwenLM/Qwen3-TTS.git
```

### Step 4: Download Model Weights

#### GLM-TTS

Download from ModelScope or Hugging Face:

```bash
# Using modelscope
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('ZhipuAI/GLM-TTS', local_dir='./GLM-TTS/ckpt')
"

# Or using huggingface-cli
huggingface-cli download ZhipuAI/GLM-TTS --local-dir ./GLM-TTS/ckpt
```

#### CosyVoice

```bash
# Using modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('iic/Fun-CosyVoice3-0.5B', local_dir='./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')
"
```

#### Qwen3-TTS

Qwen3-TTS models are automatically downloaded from Hugging Face when first used. Ensure you have HF access:

```bash
# Login to Hugging Face (optional, for gated models)
huggingface-cli login
```

### Step 5: Create Conda Environments

Run the setup script to create all required environments:

```bash
# Create all environments
python setup_models.py --all

# Or create specific environments
python setup_models.py --glm       # GLM-TTS
python setup_models.py --cosyvoice # CosyVoice
python setup_models.py --qwen      # Qwen3-TTS
python setup_models.py --qwen-vllm # Qwen3-TTS with vLLM (Linux only)
```

### Step 6: Verify Installation

```bash
# Check all environments
python setup_models.py --list

# Test each environment
conda run -n tts-glm python -c "import torch; print(f'GLM: CUDA={torch.cuda.is_available()}')"
conda run -n tts-cosyvoice python -c "import torch; print(f'CosyVoice: CUDA={torch.cuda.is_available()}')"
conda run -n tts-qwen python -c "from qwen_tts import Qwen3TTSModel; print('Qwen OK')"
```

## Configuration

### Edit config.yaml

Customize the evaluation settings:

```yaml
# Device settings
device: "cuda"
gpu_index: 0

# Enable/disable models
models:
  - name: "glm_tts"
    type: "glm_tts"
    enabled: true  # Set to false to skip
    model_path: "./GLM-TTS/ckpt"
    # ... other settings
```

### Prepare Reference Audio (Voice Cloning)

For voice cloning, place your reference audio in `processed_audio/`:

```bash
mkdir -p processed_audio

# Copy your reference audio
cp /path/to/your/speaker.wav processed_audio/

# Recommended: normalize audio
ffmpeg -i processed_audio/speaker.wav -af loudnorm processed_audio/normalized_speaker.wav
```

Update `config.yaml` with your audio paths:

```yaml
models:
  - name: "glm_tts"
    prompt_wav: "./processed_audio/normalized_speaker.wav"
    prompt_text: "The transcript of your reference audio."
```

## Running Evaluations

### Full Evaluation

```bash
# Run all enabled models
python main.py

# Run specific models
python main.py -m glm_tts cosyvoice

# Use custom test scripts
python main.py -s scripts/my_scripts.json
```

### Debug Single Model

```bash
# Test GLM-TTS
conda run -n tts-glm python debug_single_model.py --model glm_tts

# Test CosyVoice
conda run -n tts-cosyvoice python debug_single_model.py --model cosyvoice

# Test Qwen3-TTS with vLLM
conda run -n tts-qwen-vllm python debug_single_model.py --model qwen_vllm
```

### Output Structure

Results are saved to `outputs/<timestamp>_<models>/`:

```
outputs/
└── 20240201_143052_glm_tts-cosyvoice/
    ├── scripts.json           # Test texts used
    ├── metrics.json           # Summary metrics (RTF, latency)
    ├── detailed_results.json  # Per-sample results
    ├── report.md              # Markdown comparison table
    ├── glm_tts/
    │   ├── test_001.wav
    │   └── test_002.wav
    └── cosyvoice/
        ├── test_001.wav
        └── test_002.wav
```

## RTX 50 Series (Blackwell) Support

For RTX 5080/5090 GPUs (sm_120 architecture), you need PyTorch nightly with CUDA 12.8:

```bash
# In each conda environment, upgrade PyTorch
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Note**: The `torchaudio.load()` function may have compatibility issues with PyTorch nightly. The codebase uses `soundfile` as a workaround.

## Troubleshooting

### CUDA Out of Memory

- Models are run sequentially with GPU memory cleared between runs
- Reduce batch size or use CPU for evaluation
- Check for zombie processes: `nvidia-smi` and `kill` stuck processes

### CosyVoice Import Errors

Ensure the repository path is set correctly in config:

```yaml
- name: "cosyvoice"
  repo_path: "./CosyVoice"  # Must point to the cloned repo
```

### pynini Installation Fails

pynini must be installed via conda, not pip:

```bash
conda install -c conda-forge pynini=2.1.5
```

### vLLM Not Working

vLLM only works on Linux. On Windows, use WSL2:

```bash
# From Windows, run in WSL
wsl -e bash -c "cd /mnt/c/path/to/tts-evaluation && python main.py -m qwen_tts_vllm"
```

## Environment Variables

```bash
# Optional: Set Hugging Face cache directory
export HF_HOME=/path/to/hf_cache

# Optional: CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Optional: Disable tokenizers parallelism warnings
export TOKENIZERS_PARALLELISM=false
```

## Performance Tips

1. **Use vLLM for Qwen3-TTS** - Significantly faster RTF (~0.1-0.2 vs ~0.3-0.5)
2. **SSD storage** - Model loading is much faster on SSD
3. **Sufficient VRAM** - 16GB+ recommended for smooth operation
4. **Longer reference audio** - 10-15 seconds gives better voice cloning quality

## Support

For issues, please open a GitHub issue with:
- Output of `nvidia-smi`
- Output of `python setup_models.py --list`
- Full error traceback
- Your `config.yaml` (sanitized)
