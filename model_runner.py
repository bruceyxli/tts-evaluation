"""
Model runner script - executed in isolated conda environments.

This script is called by the main pipeline via subprocess to run
TTS synthesis in the model's dedicated conda environment.

Usage:
    conda run -n tts-glm python model_runner.py --model glm_tts --config config.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def _torchaudio_load_soundfile(filepath, *args, **kwargs):
    """Replacement for torchaudio.load using soundfile backend.

    Workaround for torchcodec ABI mismatch with PyTorch 2.9.0+cu128.
    """
    data, sr = sf.read(str(filepath))
    waveform = torch.from_numpy(data).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    return waveform, sr


# Monkey-patch torchaudio.load if torchcodec is broken
try:
    import torchaudio
    torchaudio.load("/dev/null")
except Exception:
    try:
        import torchaudio
        torchaudio.load = _torchaudio_load_soundfile
    except ImportError:
        pass


class GLMTTSWrapper:
    """Wrapper class for GLM-TTS to provide a unified interface."""

    def __init__(self, use_phoneme=False, use_cache=True, sample_rate=24000, device="cuda",
                 prompt_wav=None, prompt_text=None):
        import os
        import torch
        import torchaudio

        self.original_dir = os.getcwd()
        os.chdir(Path("./GLM-TTS").resolve())

        from glmtts_inference import load_models, generate_long, DEVICE

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.use_phoneme = use_phoneme
        self.use_cache = use_cache

        # Load all model components
        self.frontend, self.text_frontend, self.speech_tokenizer, self.llm, self.token2wav = load_models(
            use_phoneme=use_phoneme, sample_rate=sample_rate
        )
        self.generate_long = generate_long

        # Default prompt audio and text
        self._default_prompt_text = prompt_text or "Hello everyone, welcome back."
        self._flow_prompt_token = None
        self._speech_feat = None
        self._embedding = None
        self._cache = None

        # Load prompt audio if provided
        if prompt_wav:
            prompt_path = Path(self.original_dir) / prompt_wav
            print(f"  Prompt path: {prompt_path} (exists: {prompt_path.exists()})", file=sys.stderr)
            if prompt_path.exists():
                self._load_prompt_audio(str(prompt_path), self._default_prompt_text)
            else:
                print(f"  Warning: Prompt audio not found at {prompt_path}", file=sys.stderr)

        # If no prompt audio provided, use a default from the repo
        if self._flow_prompt_token is None:
            default_prompt = Path("./ckpt/example_prompt.wav")
            if default_prompt.exists():
                self._load_prompt_audio(str(default_prompt), self._default_prompt_text)
            else:
                # Try to find any wav file in ckpt directory
                ckpt_wavs = list(Path("./ckpt").glob("*.wav"))
                if ckpt_wavs:
                    self._load_prompt_audio(str(ckpt_wavs[0]), self._default_prompt_text)

    def _load_prompt_audio(self, prompt_wav_path: str, prompt_text: str):
        """Load and extract features from prompt audio."""
        import torch

        print(f"  Loading prompt audio from: {prompt_wav_path}", file=sys.stderr)
        try:
            # Load audio using soundfile (more reliable)
            import soundfile as sf
            speech_np, sr = sf.read(prompt_wav_path)
            print(f"  Audio loaded: shape={speech_np.shape}, sr={sr}", file=sys.stderr)

            # Convert to tensor
            if speech_np.ndim == 1:
                speech = torch.tensor(speech_np, dtype=torch.float32).unsqueeze(0)
            else:
                # stereo to mono
                speech = torch.tensor(speech_np.mean(axis=1), dtype=torch.float32).unsqueeze(0)
            print(f"  Tensor shape: {speech.shape}", file=sys.stderr)

            # Resample if needed
            if sr != self.sample_rate:
                import torchaudio.functional as F
                speech = F.resample(speech, sr, self.sample_rate)
                print(f"  Resampled shape: {speech.shape}", file=sys.stderr)

            speech = speech.to(self.device)
            print(f"  Speech on device: {speech.device}", file=sys.stderr)

            # Extract speech features (expects tensor)
            self._speech_feat = self.frontend._extract_speech_feat(speech, sample_rate=self.sample_rate)
            print(f"  Speech feat shape: {self._speech_feat.shape}", file=sys.stderr)

            # Extract speaker embedding (expects tensor or path)
            self._embedding = self.frontend._extract_spk_embedding(speech.cpu())
            print(f"  Speaker embedding shape: {self._embedding.shape}", file=sys.stderr)

            # Extract speech tokens (expects list of tuples: [(tensor, sample_rate)])
            # Note: speech_tokenizer expects CPU tensor
            prompt_speech_token = self.frontend._extract_speech_token([(speech.cpu(), self.sample_rate)])
            print(f"  Speech token shape: {prompt_speech_token.shape}", file=sys.stderr)
            cache_speech_token = [prompt_speech_token.squeeze().tolist()]
            self._flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(self.device)
            print(f"  Flow prompt token shape: {self._flow_prompt_token.shape}", file=sys.stderr)

            # Build cache
            prompt_text_tn = self.text_frontend.text_normalize(prompt_text)
            prompt_text_token = self.frontend._extract_text_token(prompt_text_tn + " ")

            self._cache = {
                "cache_text": [prompt_text_tn],
                "cache_text_token": [prompt_text_token],
                "cache_speech_token": cache_speech_token,
                "use_cache": self.use_cache,
            }

        except Exception as e:
            import traceback
            print(f"Warning: Failed to load prompt audio: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self._flow_prompt_token = None
            self._speech_feat = None
            self._embedding = None
            self._cache = None

    def _init_cache(self, prompt_text=None):
        """Initialize cache for generation."""
        if self._cache is not None:
            # Return a copy of the pre-loaded cache
            return {
                "cache_text": self._cache["cache_text"].copy(),
                "cache_text_token": self._cache["cache_text_token"].copy(),
                "cache_speech_token": [x.copy() if isinstance(x, list) else x for x in self._cache["cache_speech_token"]],
                "use_cache": self.use_cache,
            }

        # Fallback to basic cache (this will likely fail without speech features)
        if prompt_text is None:
            prompt_text = self._default_prompt_text

        prompt_text_tn = self.text_frontend.text_normalize(prompt_text)
        prompt_text_token = self.frontend._extract_text_token(prompt_text_tn + " ")

        cache = {
            "cache_text": [prompt_text_tn],
            "cache_text_token": [prompt_text_token],
            "cache_speech_token": [[0]],  # Placeholder
            "use_cache": self.use_cache,
        }
        return cache

    def synthesize(self, text: str, prompt_text=None):
        """Synthesize speech from text."""
        import torch

        # Normalize text
        synth_text = self.text_frontend.text_normalize(text)
        print(f"    Normalized text: '{synth_text[:100]}...' (len={len(synth_text)})", file=sys.stderr)

        # Initialize cache
        cache = self._init_cache(prompt_text)
        print(f"    Cache speech token len: {len(cache['cache_speech_token'])}", file=sys.stderr)
        print(f"    Flow prompt token: {self._flow_prompt_token is not None}, Speech feat: {self._speech_feat is not None}", file=sys.stderr)

        # Generate audio
        tts_speech, _, _, _ = self.generate_long(
            frontend=self.frontend,
            text_frontend=self.text_frontend,
            llm=self.llm,
            flow=self.token2wav,
            text_info=["synth", synth_text],
            cache=cache,
            embedding=self._embedding,
            seed=0,
            flow_prompt_token=self._flow_prompt_token,
            speech_feat=self._speech_feat,
            device=self.device,
            use_phoneme=self.use_phoneme,
        )

        return tts_speech.squeeze()


def load_glm_tts(config: dict):
    """Load GLM-TTS model."""
    sys.path.insert(0, str(Path("./GLM-TTS").resolve()))

    device = config.get("device", "cuda")
    use_phoneme = config.get("use_phoneme", False)
    use_cache = config.get("use_cache", True)
    prompt_wav = config.get("prompt_wav", "processed_audio/normalized_Yang.wav")
    prompt_text = config.get("prompt_text", "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision.")

    model = GLMTTSWrapper(
        use_phoneme=use_phoneme,
        use_cache=use_cache,
        sample_rate=24000,
        device=device,
        prompt_wav=prompt_wav,
        prompt_text=prompt_text,
    )
    return model, 24000  # GLM-TTS uses 24kHz


def load_glm_tts_rl(config: dict):
    """Load GLM-TTS with RL checkpoint."""
    import torch
    sys.path.insert(0, str(Path("./GLM-TTS").resolve()))

    device = config.get("device", "cuda")
    use_phoneme = config.get("use_phoneme", False)
    use_cache = config.get("use_cache", True)
    rl_ckpt = config.get("rl_ckpt_path")
    prompt_wav = config.get("prompt_wav", "processed_audio/normalized_Yang.wav")
    prompt_text = config.get("prompt_text", "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision.")

    model = GLMTTSWrapper(
        use_phoneme=use_phoneme,
        use_cache=use_cache,
        sample_rate=24000,
        device=device,
        prompt_wav=prompt_wav,
        prompt_text=prompt_text,
    )

    # Load RL weights if specified
    if rl_ckpt:
        rl_path = Path(rl_ckpt)
        if rl_path.exists():
            state_dict = torch.load(rl_path, map_location=device)
            model.llm.llama.load_state_dict(state_dict, strict=False)

    return model, 24000


def load_qwen_tts(config: dict):
    """Load Qwen3-TTS model."""
    import torch
    from qwen_tts import Qwen3TTSModel

    model_id = config.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    device = config.get("device", "cuda")
    use_flash_attn = config.get("use_flash_attn", True)

    # Build kwargs for optimized loading
    load_kwargs = {
        "device_map": device,
        "dtype": torch.bfloat16,
    }

    # Use flash attention if available and enabled
    if use_flash_attn:
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print(f"  Using Flash Attention 2", file=sys.stderr)
        except ImportError:
            print(f"  flash-attn not installed, using default attention", file=sys.stderr)

    model = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
    return model, 24000  # Qwen-TTS uses 24kHz


def load_qwen_tts_vc(config: dict):
    """Load Qwen3-TTS Base model for voice cloning."""
    import torch
    from qwen_tts import Qwen3TTSModel

    # Voice cloning requires Base model, not CustomVoice
    model_id = config.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    device = config.get("device", "cuda")
    use_flash_attn = config.get("use_flash_attn", True)

    # Build kwargs for optimized loading
    load_kwargs = {
        "device_map": device,
        "dtype": torch.bfloat16,
    }

    # Use flash attention if available and enabled
    if use_flash_attn:
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print(f"  Using Flash Attention 2", file=sys.stderr)
        except ImportError:
            print(f"  flash-attn not installed, using default attention", file=sys.stderr)

    model = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
    return model, 24000


def load_cosyvoice(config: dict):
    """Load CosyVoice model (Fun-CosyVoice3-0.5B)."""
    repo_path = Path(config.get("repo_path", "./CosyVoice")).resolve()
    sys.path.insert(0, str(repo_path))
    sys.path.insert(0, str(repo_path / "third_party" / "Matcha-TTS"))

    from cosyvoice.cli.cosyvoice import AutoModel

    model_path = config.get("model_path", "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")
    model = AutoModel(model_dir=str(model_path))
    return model, model.sample_rate  # Usually 24000


def load_cosyvoice_rl(config: dict):
    """Load CosyVoice with RL weights (llm.rl.pt)."""
    import torch
    repo_path = Path(config.get("repo_path", "./CosyVoice")).resolve()
    sys.path.insert(0, str(repo_path))
    sys.path.insert(0, str(repo_path / "third_party" / "Matcha-TTS"))

    from cosyvoice.cli.cosyvoice import AutoModel

    model_path = Path(config.get("model_path", "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"))
    model = AutoModel(model_dir=str(model_path))

    # Load RL weights
    rl_weights = model_path / "llm.rl.pt"
    if rl_weights.exists():
        device = config.get("device", "cuda")
        state_dict = torch.load(rl_weights, map_location=device)
        model.model.llm.load_state_dict(state_dict, strict=False)

    return model, model.sample_rate


def load_cosyvoice_vllm(config: dict):
    """Load CosyVoice with vLLM acceleration (Linux only).

    Note: vLLM backend is incompatible with inference_bistream().
    Standard inference_zero_shot() with stream=True works fine.
    """
    repo_path = Path(config.get("repo_path", "./CosyVoice")).resolve()
    sys.path.insert(0, str(repo_path))
    sys.path.insert(0, str(repo_path / "third_party" / "Matcha-TTS"))

    # Set PYTHONPATH so vLLM subprocess can find cosyvoice module
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    new_paths = f"{repo_path}:{repo_path / 'third_party' / 'Matcha-TTS'}"
    os.environ["PYTHONPATH"] = f"{new_paths}:{existing_pythonpath}" if existing_pythonpath else new_paths

    # Ensure conda env lib path is in LD_LIBRARY_PATH for ffmpeg/torchcodec
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, "lib")
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if conda_lib not in existing_ld:
            os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{existing_ld}" if existing_ld else conda_lib

    from cosyvoice.cli.cosyvoice import AutoModel

    model_path = config.get("model_path", "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")
    fp16 = config.get("fp16", False)
    load_trt = config.get("load_trt", False)

    print(f"Loading CosyVoice with vLLM acceleration...", file=sys.stderr)
    model = AutoModel(
        model_dir=str(model_path),
        load_vllm=True,
        load_trt=load_trt,
        fp16=fp16,
    )
    return model, model.sample_rate


MODEL_LOADERS = {
    "glm_tts": load_glm_tts,
    "glm_tts_rl": load_glm_tts_rl,
    "qwen_tts": load_qwen_tts,
    "qwen_tts_vc": load_qwen_tts_vc,
    "cosyvoice": load_cosyvoice,
    "cosyvoice_rl": load_cosyvoice_rl,
    "cosyvoice_vllm": load_cosyvoice_vllm,
}


def synthesize_glm(model, text: str, config: dict) -> tuple[np.ndarray, float]:
    """Synthesize with GLM-TTS.

    Returns:
        tuple: (audio_array, first_token_latency)
               For streaming: time to first audio chunk
               For non-streaming: time to complete audio (same as generation_time)
    """
    start_time = time.perf_counter()

    # Check if model supports streaming
    if hasattr(model, 'synthesize_stream'):
        first_token_latency = None
        audio_chunks = []
        for chunk in model.synthesize_stream(text):
            if first_token_latency is None:
                first_token_latency = time.perf_counter() - start_time
            if hasattr(chunk, 'cpu'):
                chunk = chunk.cpu().numpy()
            audio_chunks.append(chunk.flatten())
        if audio_chunks:
            audio = np.concatenate(audio_chunks)
        else:
            audio = np.array([], dtype=np.float32)
        return audio, first_token_latency
    else:
        # Non-streaming: first accessible audio = complete audio
        audio = model.synthesize(text)
        first_token_latency = time.perf_counter() - start_time
        if hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy()
        return audio.flatten(), first_token_latency


def synthesize_qwen(model, text: str, config: dict) -> tuple[np.ndarray, float]:
    """Synthesize with Qwen3-TTS (CustomVoice model).

    Returns:
        tuple: (audio_array, first_token_latency)
               qwen-tts library does NOT support true streaming, so first_token_latency
               is returned as None to distinguish from streaming models like CosyVoice.
    """
    speaker = config.get("speaker", "Vivian")
    language = config.get("language", "English")

    # Use generate_custom_voice for CustomVoice model
    # Returns: (wavs, sample_rate)
    wavs, sr = model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
    )

    # wavs is a list of tensors, concatenate them
    if isinstance(wavs, list):
        audio = wavs[0] if len(wavs) > 0 else np.array([], dtype=np.float32)
    else:
        audio = wavs

    if hasattr(audio, 'cpu'):
        audio = audio.cpu().numpy()

    # Return None for first_token_latency since qwen-tts doesn't support true streaming
    return audio.flatten(), None


def synthesize_qwen_vc(model, text: str, config: dict) -> tuple[np.ndarray, float]:
    """Synthesize with Qwen3-TTS voice cloning (Base model only).

    Returns:
        tuple: (audio_array, first_token_latency)
               qwen-tts library does NOT support true streaming, so first_token_latency
               is returned as None to distinguish from streaming models like CosyVoice.
    """
    language = config.get("language", "auto")
    ref_audio = config.get("ref_audio")
    ref_text = config.get("ref_text")
    x_vector_only = config.get("x_vector_only", False)

    if not ref_audio:
        raise ValueError("ref_audio is required for voice cloning")

    # qwen-tts does not support streaming - generate complete audio
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only,
    )
    audio = wavs[0] if isinstance(wavs, list) else wavs
    if hasattr(audio, 'cpu'):
        audio = audio.cpu().numpy()

    # Return None for first_token_latency since qwen-tts doesn't support true streaming
    return audio.flatten(), None


def synthesize_cosyvoice(model, text: str, config: dict) -> tuple[np.ndarray, float]:
    """Synthesize with CosyVoice (Fun-CosyVoice3-0.5B).

    Returns:
        tuple: (audio_array, first_token_latency)
               CosyVoice supports native streaming, so this is time to first chunk.
    """
    import torch
    mode = config.get("mode", "zero_shot")

    # Default prompt settings
    model_path = Path(config.get("model_path", "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"))
    prompt_wav = config.get("prompt_wav") or str(model_path / "asset" / "zero_shot_prompt.wav")
    prompt_text = config.get("prompt_text") or "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

    start_time = time.perf_counter()

    # Use streaming mode to measure first token latency
    if mode == "instruct":
        instruct = config.get("instruct", "请用普通话朗读。")
        output = model.inference_instruct2(
            tts_text=text,
            instruct_text=instruct,
            prompt_wav=prompt_wav,
            stream=True,
        )
    elif mode == "cross_lingual":
        output = model.inference_cross_lingual(
            tts_text=text,
            prompt_wav=prompt_wav,
            stream=True,
        )
    else:  # zero_shot (default)
        output = model.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            stream=True,
        )

    # Collect audio chunks and measure first token latency
    audio_chunks = []
    first_token_latency = None
    for chunk in output:
        if first_token_latency is None:
            first_token_latency = time.perf_counter() - start_time
        audio_chunks.append(chunk['tts_speech'])

    if audio_chunks:
        audio = torch.cat(audio_chunks, dim=1).squeeze().cpu().numpy()
        return audio, first_token_latency
    return np.array([], dtype=np.float32), first_token_latency


def synthesize_cosyvoice_vllm(model, text: str, config: dict) -> tuple[np.ndarray, float]:
    """Synthesize with CosyVoice vLLM backend.

    Supports single_batch mode for per-request performance measurement.

    Returns:
        tuple: (audio_array, first_token_latency)
    """
    import torch

    single_batch = config.get("single_batch", False)
    mode = config.get("mode", "zero_shot")

    # Default prompt settings
    model_path = Path(config.get("model_path", "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"))
    prompt_wav = config.get("prompt_wav") or str(model_path / "asset" / "zero_shot_prompt.wav")
    prompt_text = config.get("prompt_text") or "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

    start_time = time.perf_counter()

    # For single_batch mode, use stream=False to get complete audio at once
    # This avoids streaming overhead and gives cleaner per-request metrics
    use_stream = not single_batch

    if mode == "instruct":
        instruct = config.get("instruct", "请用普通话朗读。")
        output = model.inference_instruct2(
            tts_text=text,
            instruct_text=instruct,
            prompt_wav=prompt_wav,
            stream=use_stream,
        )
    elif mode == "cross_lingual":
        output = model.inference_cross_lingual(
            tts_text=text,
            prompt_wav=prompt_wav,
            stream=use_stream,
        )
    else:  # zero_shot (default)
        output = model.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            stream=use_stream,
        )

    # Collect audio
    audio_chunks = []
    first_token_latency = None

    for chunk in output:
        if first_token_latency is None:
            first_token_latency = time.perf_counter() - start_time
        audio_chunks.append(chunk['tts_speech'])

    if audio_chunks:
        audio = torch.cat(audio_chunks, dim=1).squeeze().cpu().numpy()
        return audio, first_token_latency
    return np.array([], dtype=np.float32), first_token_latency


SYNTHESIZERS = {
    "glm_tts": synthesize_glm,
    "glm_tts_rl": synthesize_glm,
    "qwen_tts": synthesize_qwen,
    "qwen_tts_vc": synthesize_qwen_vc,
    "cosyvoice": synthesize_cosyvoice,
    "cosyvoice_rl": synthesize_cosyvoice,
    "cosyvoice_vllm": synthesize_cosyvoice_vllm,  # vLLM backend with single_batch support
}


def run_batch(model_type: str, config: dict, tasks: list, output_dir: Path) -> list:
    """Run batch synthesis tasks."""
    results = []

    # Load model
    print(f"Loading model: {model_type}", file=sys.stderr)
    load_start = time.perf_counter()

    loader = MODEL_LOADERS.get(model_type)
    if not loader:
        raise ValueError(f"Unknown model type: {model_type}")

    model, sample_rate = loader(config)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s", file=sys.stderr)

    synthesizer = SYNTHESIZERS.get(model_type)

    # Process tasks
    for task in tasks:
        task_id = task["id"]
        text = task["text"]

        print(f"  Processing: {task_id}", file=sys.stderr)

        try:
            start_time = time.perf_counter()
            audio, first_token_latency = synthesizer(model, text, config)
            generation_time = time.perf_counter() - start_time

            duration = len(audio) / sample_rate
            rtf = generation_time / duration if duration > 0 else float('inf')

            # Save audio
            audio_path = output_dir / f"{task_id}.wav"
            sf.write(str(audio_path), audio, sample_rate)

            result_entry = {
                "id": task_id,
                "success": True,
                "audio_path": str(audio_path),
                "sample_rate": sample_rate,
                "duration": duration,
                "generation_time": generation_time,
                "rtf": rtf,
            }
            if first_token_latency is not None:
                result_entry["first_token_latency"] = first_token_latency

            results.append(result_entry)

        except Exception as e:
            print(f"    Error: {e}", file=sys.stderr)
            results.append({
                "id": task_id,
                "success": False,
                "error": str(e),
            })

    # Cleanup
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return results


def main():
    parser = argparse.ArgumentParser(description="TTS Model Runner")
    parser.add_argument("--model", "-m", required=True, help="Model type")
    parser.add_argument("--config", "-c", required=True, help="Config JSON file")
    parser.add_argument("--tasks", "-t", required=True, help="Tasks JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load tasks
    with open(args.tasks, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run batch
    results = run_batch(args.model, config, tasks, output_dir)

    # Output results as JSON to stdout
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
