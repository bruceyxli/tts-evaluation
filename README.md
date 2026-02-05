Prompt for Claude Code to deploy:
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
