#!/usr/bin/env python3
"""
Single Model Runner - Runs one TTS model and saves results to file.

Usage:
    conda run -n tts-glm python run_single_model.py \
        --model glm_tts \
        --config config.yaml \
        --scripts scripts/tts_evaluation_dataset.json \
        --output outputs/run_001/glm_tts

Output:
    - {output}/results.json - Metrics and results
    - {output}/*.wav - Generated audio files
"""

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import yaml
import numpy as np
import soundfile as sf

# Resource monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


class ResourceMonitor:
    """Monitor CPU and GPU usage during synthesis."""

    def __init__(self, interval: float = 0.5, gpu_index: int = 0):
        self.interval = interval
        self.gpu_index = gpu_index
        self.running = False
        self.thread = None

        # Collected samples
        self.cpu_samples = []
        self.gpu_samples = []
        self.gpu_mem_samples = []

        # Initialize NVML for GPU monitoring
        self.nvml_initialized = False
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception as e:
                print(f"Warning: Could not initialize NVML: {e}", file=sys.stderr)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            # CPU usage
            if HAS_PSUTIL:
                self.cpu_samples.append(psutil.cpu_percent(interval=None))

            # GPU usage
            if self.nvml_initialized:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.gpu_samples.append(util.gpu)
                    self.gpu_mem_samples.append(mem_info.used / 1024**3)  # GB
                except Exception:
                    pass

            time.sleep(self.interval)

    def start(self):
        """Start monitoring."""
        self.running = True
        self.cpu_samples = []
        self.gpu_samples = []
        self.gpu_mem_samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        """Stop monitoring and return statistics."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        stats = {}

        if self.cpu_samples:
            stats["cpu_avg"] = np.mean(self.cpu_samples)
            stats["cpu_max"] = np.max(self.cpu_samples)

        if self.gpu_samples:
            stats["gpu_util_avg"] = np.mean(self.gpu_samples)
            stats["gpu_util_max"] = np.max(self.gpu_samples)

        if self.gpu_mem_samples:
            stats["gpu_mem_avg_gb"] = np.mean(self.gpu_mem_samples)
            stats["gpu_mem_max_gb"] = np.max(self.gpu_mem_samples)

        return stats

    def cleanup(self):
        """Cleanup NVML."""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ============================================================
# Model Loading Functions
# ============================================================

def load_glm_tts(config: dict):
    """Load GLM-TTS model."""
    import os
    import torch
    import soundfile as sf_local
    import torchaudio.functional as F

    # Get script directory for relative paths
    script_dir = Path(__file__).parent.resolve()
    glm_path = script_dir / "GLM-TTS"

    # Change to GLM-TTS directory and add to path
    original_dir = os.getcwd()
    os.chdir(glm_path)
    sys.path.insert(0, str(glm_path))

    from glmtts_inference import load_models, generate_long, DEVICE

    device = DEVICE  # Use the device from glmtts_inference
    use_phoneme = config.get("use_phoneme", False)
    use_cache = config.get("use_cache", True)
    prompt_wav = config.get("prompt_wav")
    prompt_text = config.get("prompt_text", "")

    # Load models using the inference script (uses hardcoded ckpt paths)
    frontend, text_frontend, speech_tokenizer, llm, token2wav_model = load_models(
        use_phoneme=use_phoneme, sample_rate=24000
    )

    # Load prompt audio
    flow_prompt_token = None
    speech_feat = None
    embedding = None
    cache_speech_token = [[0]]  # Default placeholder

    if prompt_wav:
        # Resolve prompt path relative to original directory
        prompt_path = Path(original_dir) / prompt_wav if not Path(prompt_wav).is_absolute() else Path(prompt_wav)
        if prompt_path.exists():
            audio_np, sr = sf_local.read(str(prompt_path))
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            audio_tensor = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)

            # Resample for different components
            audio_24k = F.resample(audio_tensor, sr, 24000) if sr != 24000 else audio_tensor
            audio_16k = F.resample(audio_tensor, sr, 16000) if sr != 16000 else audio_tensor

            audio_24k = audio_24k.to(device)
            audio_16k = audio_16k.to(device)

            # Extract features
            speech_token = frontend._extract_speech_token([(audio_24k, 24000)])
            speech_feat = frontend._extract_speech_feat(audio_24k, 24000)
            embedding = frontend._extract_spk_embedding(audio_16k)

            # Prepare cache_speech_token and flow_prompt_token
            cache_speech_token = [speech_token.squeeze().tolist()]
            flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(device)

            print(f"  Loaded prompt audio: {prompt_path}", file=sys.stderr)

    # Change back to original directory
    os.chdir(original_dir)

    return {
        "frontend": frontend,
        "text_frontend": text_frontend,
        "llm": llm,
        "flow": token2wav_model,
        "device": device,
        "use_phoneme": use_phoneme,
        "use_cache": use_cache,
        "prompt_text": prompt_text,
        "flow_prompt_token": flow_prompt_token,
        "speech_feat": speech_feat,
        "embedding": embedding,
        "cache_speech_token": cache_speech_token,
        "generate_long": generate_long,
        "glm_path": glm_path,
    }, 24000


def load_glm_tts_rl(config: dict):
    """Load GLM-TTS with RL weights."""
    import torch

    model, sample_rate = load_glm_tts(config)

    # Load RL weights if available
    rl_path = config.get("rl_ckpt_path")
    if rl_path is None:
        rl_path = model["glm_path"] / "ckpt" / "glm" / "rl_weights.pt"

    if Path(rl_path).exists():
        print(f"  Loading RL weights from: {rl_path}", file=sys.stderr)
        state_dict = torch.load(rl_path, map_location=model["device"])
        model["llm"].model.load_state_dict(state_dict, strict=False)
    else:
        print(f"  Warning: RL weights not found at {rl_path}", file=sys.stderr)

    return model, sample_rate


def load_cosyvoice(config: dict):
    """Load CosyVoice model."""
    import torch

    repo_path = Path(config.get("repo_path", "./CosyVoice")).resolve()
    model_path = Path(config.get("model_path", "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")).resolve()

    sys.path.insert(0, str(repo_path))
    sys.path.insert(0, str(repo_path / "third_party" / "Matcha-TTS"))

    from cosyvoice.cli.cosyvoice import CosyVoice3

    model = CosyVoice3(str(model_path))

    return {
        "model": model,
        "mode": config.get("mode", "zero_shot"),
        "prompt_wav": config.get("prompt_wav"),
        "prompt_text": config.get("prompt_text", ""),
    }, 24000


def load_cosyvoice_rl(config: dict):
    """Load CosyVoice with RL weights."""
    import torch

    model_dict, sample_rate = load_cosyvoice(config)

    # Load RL weights
    model_path = Path(config.get("model_path", "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"))
    rl_path = model_path / "llm.rl.pt"

    if rl_path.exists():
        print(f"  Loading RL weights from: {rl_path}", file=sys.stderr)
        state_dict = torch.load(rl_path, map_location="cuda")
        model_dict["model"].model.llm.load_state_dict(state_dict, strict=False)
    else:
        print(f"  Warning: RL weights not found at {rl_path}", file=sys.stderr)

    return model_dict, sample_rate


def load_qwen_tts(config: dict):
    """Load Qwen3-TTS model (CustomVoice)."""
    from qwen_tts import Qwen3TTSModel

    model_id = config.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    device = config.get("device", "cuda")

    model = Qwen3TTSModel.from_pretrained(model_id, device=device)
    return {
        "model": model,
        "speaker": config.get("speaker", "Vivian"),
        "language": config.get("language", "English"),
    }, 24000


def load_qwen_tts_vc(config: dict):
    """Load Qwen3-TTS model for voice cloning (Base model)."""
    from qwen_tts import Qwen3TTSModel

    model_id = config.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    ref_audio = config.get("ref_audio")
    ref_text = config.get("ref_text", "")
    x_vector_only = config.get("x_vector_only", False)

    # Load model (device is auto-detected by qwen_tts)
    model = Qwen3TTSModel.from_pretrained(model_id)

    # Load reference audio
    ref_audio_data = None
    ref_sr = None
    if ref_audio and Path(ref_audio).exists():
        import soundfile as sf_local
        ref_audio_data, ref_sr = sf_local.read(ref_audio)
        if ref_audio_data.ndim > 1:
            ref_audio_data = ref_audio_data.mean(axis=1)
        print(f"  Loaded reference audio: {ref_audio}", file=sys.stderr)

    return {
        "model": model,
        "ref_audio_data": ref_audio_data,
        "ref_sr": ref_sr,
        "ref_text": ref_text,
        "x_vector_only": x_vector_only,
        "language": config.get("language", "auto"),
    }, 24000


def load_qwen_tts_vllm(config: dict):
    """Load Qwen3-TTS with vLLM-Omni acceleration."""
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    from vllm import SamplingParams
    from vllm_omni import Omni

    model_id = config.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    ref_audio = config.get("ref_audio")
    ref_text = config.get("ref_text", "")
    mode = config.get("mode", "icl")

    # Load reference audio
    ref_audio_data = None
    ref_sr = None
    if ref_audio and Path(ref_audio).exists():
        import soundfile as sf_local
        ref_audio_data, ref_sr = sf_local.read(ref_audio)
        if ref_audio_data.ndim > 1:
            ref_audio_data = ref_audio_data.mean(axis=1)
        print(f"  Loaded reference audio: {ref_audio}", file=sys.stderr)

    # Initialize vLLM-Omni
    model = Omni(model=model_id)

    sampling_params = SamplingParams(
        temperature=config.get("temperature", 0.9),
        top_p=config.get("top_p", 1.0),
        top_k=config.get("top_k", 50),
        max_tokens=config.get("max_tokens", 2048),
        repetition_penalty=config.get("repetition_penalty", 1.05),
    )

    return {
        "model": model,
        "sampling_params": sampling_params,
        "model_id": model_id,
        "ref_audio_data": ref_audio_data,
        "ref_sr": ref_sr,
        "ref_text": ref_text,
        "mode": mode,
        "language": config.get("language", "English"),
    }, 24000


MODEL_LOADERS = {
    "glm_tts": load_glm_tts,
    "glm_tts_rl": load_glm_tts_rl,
    "cosyvoice": load_cosyvoice,
    "cosyvoice_rl": load_cosyvoice_rl,
    "qwen_tts": load_qwen_tts,
    "qwen_tts_vc": load_qwen_tts_vc,
    "qwen_tts_vllm": load_qwen_tts_vllm,
    "qwen_tts_vllm_vc": load_qwen_tts_vllm,
}


# ============================================================
# Synthesis Functions
# ============================================================

def synthesize_glm(model_dict: dict, text: str, config: dict) -> tuple:
    """Synthesize with GLM-TTS. Returns (audio, first_token_latency)."""
    import torch

    start_time = time.perf_counter()

    # Normalize text
    synth_text = model_dict["text_frontend"].text_normalize(text)

    # Initialize cache - use frontend._extract_text_token instead of text_frontend.tokenize
    prompt_text_tn = model_dict["text_frontend"].text_normalize(model_dict["prompt_text"])
    prompt_text_token = model_dict["frontend"]._extract_text_token(prompt_text_tn + " ")

    cache = {
        "cache_text": [prompt_text_tn],
        "cache_text_token": [prompt_text_token],
        "cache_speech_token": model_dict["cache_speech_token"],
        "use_cache": model_dict["use_cache"],
    }

    # Generate audio
    tts_speech, _, _, _ = model_dict["generate_long"](
        frontend=model_dict["frontend"],
        text_frontend=model_dict["text_frontend"],
        llm=model_dict["llm"],
        flow=model_dict["flow"],
        text_info=["synth", synth_text],
        cache=cache,
        embedding=model_dict["embedding"],
        seed=0,
        flow_prompt_token=model_dict["flow_prompt_token"],
        speech_feat=model_dict["speech_feat"],
        device=model_dict["device"],
        use_phoneme=model_dict["use_phoneme"],
    )

    first_token_latency = time.perf_counter() - start_time
    audio = tts_speech.squeeze().cpu().numpy()

    return audio, first_token_latency


def synthesize_cosyvoice(model_dict: dict, text: str, config: dict) -> tuple:
    """Synthesize with CosyVoice. Returns (audio, first_token_latency)."""
    import torch

    model = model_dict["model"]
    mode = model_dict["mode"]
    prompt_wav = model_dict["prompt_wav"]
    prompt_text = model_dict["prompt_text"]

    start_time = time.perf_counter()
    first_token_latency = None
    audio_chunks = []

    if mode == "zero_shot" and prompt_wav:
        generator = model.inference_zero_shot(text, prompt_text, prompt_wav, stream=True)
    else:
        # Fallback to instruct mode
        generator = model.inference_instruct2(text, "default", stream=True)

    for chunk in generator:
        if first_token_latency is None:
            first_token_latency = time.perf_counter() - start_time
        audio_data = chunk["tts_speech"].numpy().flatten()
        audio_chunks.append(audio_data)

    if audio_chunks:
        audio = np.concatenate(audio_chunks)
    else:
        audio = np.array([], dtype=np.float32)

    if first_token_latency is None:
        first_token_latency = time.perf_counter() - start_time

    return audio, first_token_latency


def synthesize_qwen(model_dict: dict, text: str, config: dict) -> tuple:
    """Synthesize with Qwen3-TTS (CustomVoice). Returns (audio, first_token_latency)."""
    model = model_dict["model"]
    speaker = model_dict["speaker"]
    language = model_dict["language"]

    start_time = time.perf_counter()

    # Use generate_custom_voice for CustomVoice model
    audio = model.generate_custom_voice(text, speaker=speaker, language=language)
    first_token_latency = time.perf_counter() - start_time

    if hasattr(audio, 'cpu'):
        audio = audio.cpu().numpy()

    return audio.flatten(), first_token_latency


def synthesize_qwen_vc(model_dict: dict, text: str, config: dict) -> tuple:
    """Synthesize with Qwen3-TTS voice cloning. Returns (audio, first_token_latency)."""
    model = model_dict["model"]
    ref_audio_data = model_dict["ref_audio_data"]
    ref_sr = model_dict["ref_sr"]
    ref_text = model_dict["ref_text"]
    x_vector_only = model_dict["x_vector_only"]

    start_time = time.perf_counter()

    # Use voice cloning synthesis with generate_voice_clone
    # Returns: Tuple[List[numpy.ndarray], int] -> (audio_list, sample_rate)
    if ref_audio_data is not None:
        audio_list, sr = model.generate_voice_clone(
            text,
            ref_audio=(ref_audio_data, ref_sr),
            ref_text=ref_text if not x_vector_only else None,
            x_vector_only_mode=x_vector_only,
        )
        audio = audio_list[0]  # Get first audio from list
    else:
        # Fallback to custom voice without reference
        audio_list, sr = model.generate_custom_voice(text)
        audio = audio_list[0]

    first_token_latency = time.perf_counter() - start_time

    if hasattr(audio, 'cpu'):
        audio = audio.cpu().numpy()

    return audio.flatten(), first_token_latency


def synthesize_qwen_vllm(model_dict: dict, text: str, config: dict) -> tuple:
    """Synthesize with Qwen3-TTS via vLLM-Omni. Returns (audio, first_token_latency)."""
    model = model_dict["model"]
    sampling_params = model_dict["sampling_params"]
    ref_audio_data = model_dict["ref_audio_data"]
    ref_sr = model_dict["ref_sr"]
    ref_text = model_dict["ref_text"]
    mode = model_dict["mode"]
    language = model_dict["language"]

    start_time = time.perf_counter()

    # Build prompt
    prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    # Build input
    if ref_audio_data is not None:
        # Voice cloning mode
        inputs = {
            "prompt": prompt,
            "additional_information": {
                "task_type": ["VoiceClone"],
                "ref_audio": [ref_audio_data.tolist()],
                "ref_sr": [ref_sr],
                "ref_text": [ref_text],
                "mode": [mode],
            },
        }
    else:
        # Default mode
        inputs = {
            "prompt": prompt,
            "additional_information": {
                "task_type": ["CustomVoice"],
                "text": [text],
                "language": [language],
                "speaker": ["Vivian"],
            },
        }

    # Generate - returns a generator
    import torch
    omni_generator = model.generate(inputs, [sampling_params])
    first_token_latency = None

    # Extract audio from generator output
    audio = np.array([], dtype=np.float32)
    for stage_outputs in omni_generator:
        if first_token_latency is None:
            first_token_latency = time.perf_counter() - start_time
        for output in stage_outputs.request_output:
            if output.outputs and len(output.outputs) > 0:
                mm_output = output.outputs[0].multimodal_output
                if mm_output and "audio" in mm_output:
                    audio_tensor = mm_output["audio"]
                    audio = audio_tensor.float().detach().cpu().numpy()
                    if audio.ndim > 1:
                        audio = audio.flatten()
                    break
        if len(audio) > 0:
            break

    if first_token_latency is None:
        first_token_latency = time.perf_counter() - start_time

    return audio, first_token_latency


SYNTHESIZERS = {
    "glm_tts": synthesize_glm,
    "glm_tts_rl": synthesize_glm,
    "cosyvoice": synthesize_cosyvoice,
    "cosyvoice_rl": synthesize_cosyvoice,
    "qwen_tts": synthesize_qwen,
    "qwen_tts_vc": synthesize_qwen_vc,
    "qwen_tts_vllm": synthesize_qwen_vllm,
    "qwen_tts_vllm_vc": synthesize_qwen_vllm,
}


# ============================================================
# Main Runner
# ============================================================

def run_evaluation(model_type: str, model_config: dict, scripts: list, output_dir: Path) -> dict:
    """Run evaluation for a single model."""

    results = {
        "model_name": model_config.get("name", model_type),
        "model_type": model_type,
        "started_at": datetime.now().isoformat(),
        "samples": [],
        "summary": {},
    }

    # Initialize resource monitor
    monitor = ResourceMonitor(
        interval=0.5,
        gpu_index=model_config.get("gpu_index", 0)
    )

    try:
        # Load model
        print(f"Loading model: {model_type}", file=sys.stderr)
        load_start = time.perf_counter()

        loader = MODEL_LOADERS.get(model_type)
        if not loader:
            raise ValueError(f"Unknown model type: {model_type}")

        model, sample_rate = loader(model_config)
        load_time = time.perf_counter() - load_start
        print(f"Model loaded in {load_time:.2f}s", file=sys.stderr)

        results["load_time"] = load_time
        results["sample_rate"] = sample_rate

        synthesizer = SYNTHESIZERS.get(model_type)

        # Process each script
        total_generation_time = 0
        total_duration = 0
        first_token_latencies = []

        for i, script in enumerate(scripts):
            task_id = script["id"]
            text = script["text"]

            print(f"  [{i+1}/{len(scripts)}] Processing: {task_id}", file=sys.stderr)

            # Start monitoring
            monitor.start()

            try:
                gen_start = time.perf_counter()
                audio, first_token_latency = synthesizer(model, text, model_config)
                generation_time = time.perf_counter() - gen_start

                # Stop monitoring and get stats
                resource_stats = monitor.stop()

                duration = len(audio) / sample_rate if len(audio) > 0 else 0
                rtf = generation_time / duration if duration > 0 else float('inf')

                # Save audio
                audio_path = output_dir / f"{task_id}.wav"
                sf.write(str(audio_path), audio, sample_rate)

                sample_result = {
                    "id": task_id,
                    "success": True,
                    "audio_path": str(audio_path),
                    "duration": duration,
                    "generation_time": generation_time,
                    "rtf": rtf,
                    "first_token_latency": first_token_latency,
                    **resource_stats,
                }

                total_generation_time += generation_time
                total_duration += duration
                if first_token_latency:
                    first_token_latencies.append(first_token_latency)

                print(f"      RTF: {rtf:.3f}, First token: {first_token_latency:.3f}s", file=sys.stderr)

            except Exception as e:
                monitor.stop()
                print(f"      Error: {e}", file=sys.stderr)
                sample_result = {
                    "id": task_id,
                    "success": False,
                    "error": str(e),
                }

            results["samples"].append(sample_result)

        # Calculate summary statistics
        successful = [s for s in results["samples"] if s.get("success")]

        if successful:
            results["summary"] = {
                "total_samples": len(scripts),
                "successful_samples": len(successful),
                "failed_samples": len(scripts) - len(successful),
                "total_generation_time": total_generation_time,
                "total_audio_duration": total_duration,
                "avg_rtf": total_generation_time / total_duration if total_duration > 0 else None,
                "avg_first_token_latency": np.mean(first_token_latencies) if first_token_latencies else None,
                "min_first_token_latency": np.min(first_token_latencies) if first_token_latencies else None,
                "max_first_token_latency": np.max(first_token_latencies) if first_token_latencies else None,
                "avg_cpu": np.mean([s.get("cpu_avg", 0) for s in successful if s.get("cpu_avg")]) or None,
                "avg_gpu_util": np.mean([s.get("gpu_util_avg", 0) for s in successful if s.get("gpu_util_avg")]) or None,
                "avg_gpu_mem_gb": np.mean([s.get("gpu_mem_avg_gb", 0) for s in successful if s.get("gpu_mem_avg_gb")]) or None,
            }

    except Exception as e:
        results["error"] = str(e)
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    finally:
        monitor.cleanup()

        # Cleanup GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    results["finished_at"] = datetime.now().isoformat()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run single TTS model evaluation")
    parser.add_argument("--model", "-m", required=True, help="Model type (glm_tts, cosyvoice, etc.)")
    parser.add_argument("--config", "-c", required=True, help="Config YAML file")
    parser.add_argument("--scripts", "-s", required=True, help="Scripts JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--name", "-n", help="Model name (defaults to model type)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    # Find model config
    model_config = None
    model_name = args.name or args.model

    for m in full_config.get("models", []):
        if m.get("name") == model_name or m.get("type") == args.model:
            model_config = m
            break

    if not model_config:
        # Use defaults from full_config
        model_config = {
            "name": model_name,
            "type": args.model,
            "device": full_config.get("device", "cuda"),
            "gpu_index": full_config.get("gpu_index", 0),
        }

    # Merge global settings
    model_config["device"] = model_config.get("device", full_config.get("device", "cuda"))
    model_config["gpu_index"] = model_config.get("gpu_index", full_config.get("gpu_index", 0))

    # Load scripts
    with open(args.scripts, "r", encoding="utf-8") as f:
        scripts = json.load(f)

    print(f"Loaded {len(scripts)} test scripts", file=sys.stderr)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    results = run_evaluation(args.model, model_config, scripts, output_dir)

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\nResults saved to: {results_path}", file=sys.stderr)

    # Print summary
    if results.get("summary"):
        s = results["summary"]
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Summary for {results['model_name']}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"  Samples: {s.get('successful_samples', 0)}/{s.get('total_samples', 0)} successful", file=sys.stderr)
        print(f"  Avg RTF: {s.get('avg_rtf', 'N/A'):.3f}" if s.get('avg_rtf') else "  Avg RTF: N/A", file=sys.stderr)
        print(f"  Avg First Token Latency: {s.get('avg_first_token_latency', 'N/A'):.3f}s" if s.get('avg_first_token_latency') else "  Avg First Token Latency: N/A", file=sys.stderr)
        if s.get('avg_gpu_util'):
            print(f"  Avg GPU Utilization: {s.get('avg_gpu_util'):.1f}%", file=sys.stderr)
        if s.get('avg_gpu_mem_gb'):
            print(f"  Avg GPU Memory: {s.get('avg_gpu_mem_gb'):.2f} GB", file=sys.stderr)


if __name__ == "__main__":
    main()
