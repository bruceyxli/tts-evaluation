"""Setup script for TTS models with isolated conda environments."""

import subprocess
import os
import platform
from pathlib import Path

# Environment configurations
ENVS = {
    "glm": {
        "name": "tts-glm",
        "yaml": "envs/glm_tts.yaml",
        "repo": "https://github.com/zai-org/GLM-TTS.git",
        "repo_dir": "GLM-TTS",
        "model_download": "huggingface-cli download zai-org/GLM-TTS --local-dir {ckpt_path}",
    },
    "qwen": {
        "name": "tts-qwen",
        "yaml": "envs/qwen_tts.yaml",
        "repo": None,  # Uses pip package
        "model_download": None,  # Auto-downloads on first use
    },
    "qwen-vllm": {
        "name": "tts-qwen-vllm",
        "yaml": "envs/qwen_tts_vllm.yaml",
        "repo": "https://github.com/vllm-project/vllm-omni.git",
        "repo_dir": "vllm-omni",
        "model_download": None,  # Auto-downloads on first use
        "linux_only": True,  # vLLM only supports Linux
    },
    "cosyvoice": {
        "name": "tts-cosyvoice",
        "yaml": "envs/cosyvoice.yaml",
        "repo": "https://github.com/FunAudioLLM/CosyVoice.git",
        "repo_dir": "CosyVoice",
        "model_download": None,  # Uses modelscope
    },
}

IS_WINDOWS = platform.system() == "Windows"


def get_conda_exe() -> str:
    """Get conda executable path."""
    # Check environment variable first
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    # Common Windows locations
    if IS_WINDOWS:
        common_paths = [
            Path.home() / "anaconda3" / "Scripts" / "conda.exe",
            Path.home() / "miniconda3" / "Scripts" / "conda.exe",
            Path("C:/ProgramData/anaconda3/Scripts/conda.exe"),
            Path("C:/ProgramData/miniconda3/Scripts/conda.exe"),
        ]
        for p in common_paths:
            if p.exists():
                return str(p)

    return "conda"  # Fallback to PATH


def run_cmd(cmd: str, cwd: str = None, capture: bool = False) -> tuple:
    """Run a command and return (success, output)."""
    print(f"\n>>> {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            cwd=cwd,
            capture_output=capture,
            text=True,
        )
        return True, result.stdout if capture else ""
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False, str(e)


def env_exists(env_name: str) -> bool:
    """Check if conda environment exists."""
    conda = get_conda_exe()
    success, output = run_cmd(f'"{conda}" env list', capture=True)
    if success:
        return env_name in output
    return False


def create_env(env_name: str, yaml_path: str) -> bool:
    """Create conda environment from yaml file."""
    conda = get_conda_exe()

    if env_exists(env_name):
        print(f"Environment '{env_name}' already exists.")
        return True

    print(f"\nCreating conda environment: {env_name}")
    yaml_abs = Path(yaml_path).resolve()

    success, _ = run_cmd(f'"{conda}" env create -f "{yaml_abs}"')
    return success


def get_env_python(env_name: str) -> str:
    """Get the Python executable for a conda environment."""
    conda = get_conda_exe()

    # Get conda info to find envs path
    success, output = run_cmd(f'"{conda}" info --json', capture=True)
    if success:
        import json
        info = json.loads(output)
        envs_dirs = info.get("envs_dirs", [])

        for envs_dir in envs_dirs:
            env_path = Path(envs_dir) / env_name
            if env_path.exists():
                if IS_WINDOWS:
                    python = env_path / "python.exe"
                else:
                    python = env_path / "bin" / "python"
                if python.exists():
                    return str(python)

    # Fallback: try to get from conda run
    return f'conda run -n {env_name} python'


def run_in_env(env_name: str, cmd: str, cwd: str = None) -> bool:
    """Run a command in a specific conda environment."""
    conda = get_conda_exe()
    full_cmd = f'"{conda}" run -n {env_name} {cmd}'
    success, _ = run_cmd(full_cmd, cwd=cwd)
    return success


def setup_glm():
    """Setup GLM-TTS with isolated environment."""
    config = ENVS["glm"]
    env_name = config["name"]

    print("\n" + "=" * 60)
    print("Setting up GLM-TTS (isolated environment)")
    print("=" * 60)

    # Create environment
    if not create_env(env_name, config["yaml"]):
        print("Failed to create environment")
        return False

    # Clone repository
    repo_dir = Path(config["repo_dir"])
    if not repo_dir.exists():
        print("\nCloning GLM-TTS repository...")
        run_cmd(f'git clone {config["repo"]}')
    else:
        print("\nGLM-TTS directory exists, pulling latest...")
        run_cmd("git pull", cwd=str(repo_dir))

    # Install repo-specific requirements in the env
    req_file = repo_dir / "requirements.txt"
    if req_file.exists():
        print("\nInstalling GLM-TTS requirements in environment...")
        run_in_env(env_name, f'pip install -r "{req_file.resolve()}"')

    # Download model weights
    ckpt_path = repo_dir / "ckpt"
    ckpt_path.mkdir(exist_ok=True)

    if not any(ckpt_path.iterdir()):
        print("\nDownloading GLM-TTS model weights...")
        download_cmd = config["model_download"].format(ckpt_path=ckpt_path.resolve())
        run_in_env(env_name, download_cmd)
    else:
        print(f"\nModel weights already exist in {ckpt_path}")

    print("\nGLM-TTS setup complete!")
    print(f"  Environment: {env_name}")
    print(f"  Repository: {repo_dir.resolve()}")
    return True


def setup_qwen():
    """Setup Qwen3-TTS with isolated environment."""
    config = ENVS["qwen"]
    env_name = config["name"]

    print("\n" + "=" * 60)
    print("Setting up Qwen3-TTS (isolated environment)")
    print("=" * 60)

    # Create environment (qwen-tts is already in the yaml)
    if not create_env(env_name, config["yaml"]):
        print("Failed to create environment")
        return False

    # Optional: Install flash attention
    print("\nAttempting to install flash-attn (optional)...")
    run_in_env(env_name, "pip install flash-attn --no-build-isolation")

    print("\nQwen3-TTS setup complete!")
    print(f"  Environment: {env_name}")
    print("  Model weights will be downloaded automatically on first use.")
    return True


def setup_cosyvoice():
    """Setup CosyVoice with isolated environment."""
    config = ENVS["cosyvoice"]
    env_name = config["name"]

    print("\n" + "=" * 60)
    print("Setting up CosyVoice (isolated environment)")
    print("=" * 60)

    # Create environment
    if not create_env(env_name, config["yaml"]):
        print("Failed to create environment")
        return False

    # Clone repository
    repo_dir = Path(config["repo_dir"])
    if not repo_dir.exists():
        print("\nCloning CosyVoice repository...")
        run_cmd(f'git clone --recursive {config["repo"]}')
    else:
        print("\nCosyVoice directory exists, pulling latest...")
        run_cmd("git pull", cwd=str(repo_dir))
        run_cmd("git submodule update --init --recursive", cwd=str(repo_dir))

    # Install repo-specific requirements
    req_file = repo_dir / "requirements.txt"
    if req_file.exists():
        print("\nInstalling CosyVoice requirements in environment...")
        run_in_env(env_name, f'pip install -r "{req_file.resolve()}"')

    print("\nCosyVoice setup complete!")
    print(f"  Environment: {env_name}")
    print(f"  Repository: {repo_dir.resolve()}")
    return True


def setup_qwen_vllm():
    """Setup Qwen3-TTS with vLLM-Omni (Linux only)."""
    config = ENVS["qwen-vllm"]
    env_name = config["name"]

    print("\n" + "=" * 60)
    print("Setting up Qwen3-TTS with vLLM-Omni (accelerated)")
    print("=" * 60)

    # Check if Linux
    if IS_WINDOWS:
        print("\n[WARNING] vLLM-Omni only supports Linux!")
        print("You can use this environment in WSL2 or on a Linux server.")
        print("Skipping vLLM setup on Windows.")
        return False

    # Create environment
    if not create_env(env_name, config["yaml"]):
        print("Failed to create environment")
        return False

    # Install vLLM
    print("\nInstalling vLLM...")
    run_in_env(env_name, "pip install vllm==0.15.0")

    # Clone vllm-omni repository
    repo_dir = Path(config["repo_dir"])
    if not repo_dir.exists():
        print("\nCloning vllm-omni repository...")
        run_cmd(f'git clone {config["repo"]}')
    else:
        print("\nvllm-omni directory exists, pulling latest...")
        run_cmd("git pull", cwd=str(repo_dir))

    # Install vllm-omni in editable mode
    print("\nInstalling vllm-omni...")
    run_in_env(env_name, f'pip install -e "{repo_dir.resolve()}"')

    print("\nQwen3-TTS with vLLM-Omni setup complete!")
    print(f"  Environment: {env_name}")
    print(f"  Repository: {repo_dir.resolve()}")
    print("  Model weights will be downloaded automatically on first use.")
    print("\n  [NOTE] Use model type 'qwen_tts_vllm' in config.yaml to use vLLM acceleration")
    return True


def list_envs():
    """List all TTS environments and their status."""
    print("\n" + "=" * 60)
    print("TTS Environments Status")
    print("=" * 60)

    for key, config in ENVS.items():
        env_name = config["name"]
        exists = env_exists(env_name)
        status = "[OK] Installed" if exists else "[--] Not installed"
        print(f"  {key:12} ({env_name:15}) : {status}")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup TTS models with isolated conda environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_models.py --all        # Setup all models (excluding vLLM)
  python setup_models.py --glm        # Setup GLM-TTS only
  python setup_models.py --qwen       # Setup Qwen3-TTS only
  python setup_models.py --qwen-vllm  # Setup Qwen3-TTS with vLLM (Linux only)
  python setup_models.py --cosyvoice  # Setup CosyVoice only
  python setup_models.py --list       # Show environment status
        """
    )
    parser.add_argument("--all", action="store_true", help="Setup all models (excluding vLLM)")
    parser.add_argument("--glm", action="store_true", help="Setup GLM-TTS")
    parser.add_argument("--qwen", action="store_true", help="Setup Qwen3-TTS")
    parser.add_argument("--qwen-vllm", action="store_true", help="Setup Qwen3-TTS with vLLM-Omni (Linux only)")
    parser.add_argument("--cosyvoice", action="store_true", help="Setup CosyVoice")
    parser.add_argument("--list", action="store_true", help="List environment status")
    args = parser.parse_args()

    if args.list:
        list_envs()
        return

    if args.glm or args.all:
        setup_glm()

    if args.qwen or args.all:
        setup_qwen()

    if getattr(args, 'qwen_vllm', False):
        setup_qwen_vllm()

    if args.cosyvoice or args.all:
        setup_cosyvoice()

    if not any([args.all, args.glm, args.qwen, getattr(args, 'qwen_vllm', False), args.cosyvoice, args.list]):
        parser.print_help()


if __name__ == "__main__":
    main()
