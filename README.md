# TTS Evaluation Pipeline

A unified evaluation pipeline for comparing multiple Text-to-Speech (TTS) models.

## Supported Models

| Model | Type | Features |
|-------|------|----------|
| GLM-TTS | `glm_tts` | Zero-shot voice cloning |
| GLM-TTS RL | `glm_tts_rl` | + RL fine-tuning |
| Qwen3-TTS | `qwen_tts` | CustomVoice (preset speakers) |
| Qwen3-TTS VC | `qwen_tts_vc` | Voice cloning (Base model) |
| Qwen3-TTS vLLM | `qwen_tts_vllm` | vLLM-accelerated (Linux only) |
| CosyVoice | `cosyvoice` | Zero-shot voice cloning |
| CosyVoice RL | `cosyvoice_rl` | + RL fine-tuning |

## Quick Start

```bash
# Install all environments
python setup_models.py --all

# Run evaluation with all enabled models
python main.py

# Run specific models
python main.py -m glm_tts qwen_tts cosyvoice
```

## Output Structure

```
outputs/
├── latest/           # Current run results
│   ├── glm_tts/      # Model audio outputs
│   ├── qwen_tts/
│   ├── cosyvoice/
│   ├── metrics.json
│   ├── detailed_results.json
│   └── report.md
└── history/          # Archived runs
    └── 20260206_120000/
```

## Key Metrics

- **RTF (Real-Time Factor)**: < 1.0 means faster than real-time
- **First Token Latency**: Time to first audio chunk (streaming models)
- **GPU Memory**: Peak memory usage during synthesis

## Configuration

Edit `config.yaml` to customize models and settings. See `CLAUDE.md` for detailed documentation.

## Requirements

- CUDA-capable GPU
- Conda (Miniconda or Anaconda)
- 50GB+ disk space for models

## Deployment

See `DEPLOYMENT.md` for detailed setup instructions.

---

## Claude Code Deployment Prompt

```
Please help me deploy this TTS evaluation project. Follow the steps in DEPLOYMENT.md:

1. Check system environment (GPU, disk space, conda)
2. Install system dependencies
3. If conda is not installed, install Miniconda
4. Clone model repositories (GLM-TTS, CosyVoice)
5. Create conda environments: python setup_models.py --all
6. Download model weights (GLM-TTS and CosyVoice)
7. Verify each environment works correctly
8. Run test: python main.py -m glm_tts

If the GPU is RTX 5080/5090, install PyTorch nightly (refer to the "RTX 50 Series" section in DEPLOYMENT.md).

Report results after each step. If errors occur, try to resolve them using the Troubleshooting section in DEPLOYMENT.md first.
```
