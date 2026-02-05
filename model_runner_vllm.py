"""
vLLM-Omni Model Runner for Qwen3-TTS.

This script provides high-performance TTS inference using vLLM-Omni.
NOTE: vLLM only supports Linux! Use WSL2 on Windows.

Usage:
    python model_runner_vllm.py --model qwen_tts_vllm --config config.json --tasks tasks.json --output ./output
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Set multiprocessing method before importing vLLM
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def check_vllm_available():
    """Check if vLLM-Omni is available."""
    try:
        from vllm import SamplingParams
        from vllm_omni import Omni
        return True
    except ImportError as e:
        print(f"vLLM-Omni not available: {e}", file=sys.stderr)
        print("Please install vLLM-Omni following the instructions in CLAUDE.md", file=sys.stderr)
        return False


class Qwen3TTSQuery:
    """Query builder for Qwen3-TTS with vLLM-Omni.

    Uses the correct input format for vLLM-Omni:
    {
        "prompt": "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n",
        "additional_information": {
            "task_type": [task_type],
            "text": [text],
            ...
        }
    }
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._determine_task_type()

    def _determine_task_type(self):
        """Determine task type based on model name."""
        if "CustomVoice" in self.model_name:
            self.task_type = "CustomVoice"
        elif "VoiceDesign" in self.model_name:
            self.task_type = "VoiceDesign"
        elif "Base" in self.model_name:
            self.task_type = "Base"
        else:
            self.task_type = "CustomVoice"  # Default

    def build_custom_voice_input(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "English",
        instruct: str = "",
        max_new_tokens: int = 2048,
    ) -> dict:
        """Build input for CustomVoice model."""
        prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

        return {
            "prompt": prompt,
            "additional_information": {
                "task_type": [self.task_type],
                "text": [text],
                "language": [language],
                "speaker": [speaker],
                "instruct": [instruct],
                "max_new_tokens": [max_new_tokens],
            },
        }

    def build_voice_clone_input(
        self,
        text: str,
        ref_audio: str,
        ref_text: str = "",
        language: str = "Auto",
        mode: str = "icl",
        max_new_tokens: int = 2048,
    ) -> dict:
        """Build input for Base model (voice cloning)."""
        prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        x_vector_only_mode = (mode == "xvec_only")

        return {
            "prompt": prompt,
            "additional_information": {
                "task_type": [self.task_type],
                "ref_audio": [ref_audio],
                "ref_text": [ref_text],
                "text": [text],
                "language": [language],
                "x_vector_only_mode": [x_vector_only_mode],
                "max_new_tokens": [max_new_tokens],
            },
        }


def load_vllm_model(config: dict):
    """Load Qwen3-TTS model with vLLM-Omni."""
    from vllm_omni import Omni

    model_id = config.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    stage_configs_path = config.get("stage_configs_path")
    enable_stats = config.get("enable_stats", False)
    stage_init_timeout = config.get("stage_init_timeout", 300)

    print(f"Loading vLLM-Omni model: {model_id}", file=sys.stderr)

    omni = Omni(
        model=model_id,
        stage_configs_path=stage_configs_path,
        log_stats=enable_stats,
        stage_init_timeout=stage_init_timeout,
    )

    return omni, model_id


def create_sampling_params(config: dict):
    """Create sampling parameters for vLLM."""
    from vllm import SamplingParams

    return SamplingParams(
        temperature=config.get("temperature", 0.9),
        top_p=config.get("top_p", 1.0),
        top_k=config.get("top_k", 50),
        max_tokens=config.get("max_tokens", 2048),
        seed=config.get("seed", 42),
        detokenize=False,
        repetition_penalty=config.get("repetition_penalty", 1.05),
    )


def run_batch_vllm(config: dict, tasks: list, output_dir: Path) -> list:
    """Run batch synthesis with vLLM-Omni."""
    results = []

    if not check_vllm_available():
        return [{"id": t["id"], "success": False, "error": "vLLM-Omni not available"} for t in tasks]

    # Load model
    print("Loading vLLM-Omni model...", file=sys.stderr)
    load_start = time.perf_counter()
    omni, model_id = load_vllm_model(config)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s", file=sys.stderr)

    # Create query builder and sampling params
    query_builder = Qwen3TTSQuery(model_id)
    sampling_params = create_sampling_params(config)

    # Prepare inputs
    inputs_list = []
    for task in tasks:
        text = task["text"]

        if query_builder.task_type == "Base":
            # Voice cloning
            ref_audio = config.get("ref_audio")
            ref_text = config.get("ref_text", "")
            if not ref_audio:
                results.append({
                    "id": task["id"],
                    "success": False,
                    "error": "ref_audio required for voice cloning",
                })
                continue

            inp = query_builder.build_voice_clone_input(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                language=config.get("language", "Auto"),
                mode=config.get("mode", "icl"),
            )
        else:
            # CustomVoice or VoiceDesign
            inp = query_builder.build_custom_voice_input(
                text=text,
                speaker=config.get("speaker", "Vivian"),
                language=config.get("language", "English"),
                instruct=config.get("instruct", ""),
            )

        inputs_list.append((task["id"], inp))

    if not inputs_list:
        return results

    # Run batch inference
    print(f"Running batch inference for {len(inputs_list)} samples...", file=sys.stderr)
    batch_start = time.perf_counter()

    # Prepare for vLLM generate
    all_inputs = [inp for _, inp in inputs_list]

    # Generate - vLLM-Omni expects a single sampling_params for the batch
    omni_generator = omni.generate(all_inputs, [sampling_params])

    # Map request index to task_id (request_id format is "idx_uuid")
    task_id_map = {i: task_id for i, (task_id, _) in enumerate(inputs_list)}
    task_start_times = {task_id: batch_start for task_id, _ in inputs_list}

    for stage_outputs in omni_generator:
        for output in stage_outputs.request_output:
            try:
                # request_id format: "idx_uuid", extract idx
                request_id_str = output.request_id
                request_idx = int(request_id_str.split("_")[0])
                task_id = task_id_map.get(request_idx)
                if task_id is None:
                    continue

                generation_time = time.perf_counter() - task_start_times[task_id]

                # Extract audio
                audio_tensor = output.outputs[0].multimodal_output["audio"]
                sample_rate = output.outputs[0].multimodal_output["sr"].item()
                audio = audio_tensor.float().detach().cpu().numpy()

                if audio.ndim > 1:
                    audio = audio.flatten()

                duration = len(audio) / sample_rate
                rtf = generation_time / duration if duration > 0 else float('inf')

                # Save audio
                audio_path = output_dir / f"{task_id}.wav"
                sf.write(str(audio_path), audio, sample_rate, format="WAV")

                results.append({
                    "id": task_id,
                    "success": True,
                    "audio_path": str(audio_path),
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "generation_time": generation_time,
                    "rtf": rtf,
                    "backend": "vllm-omni",
                })

                print(f"  {task_id}: RTF={rtf:.3f}, duration={duration:.2f}s", file=sys.stderr)

            except Exception as e:
                task_id = task_id_map.get(int(output.request_id), "unknown")
                print(f"  {task_id}: Error - {e}", file=sys.stderr)
                results.append({
                    "id": task_id,
                    "success": False,
                    "error": str(e),
                })

    batch_time = time.perf_counter() - batch_start
    print(f"Batch completed in {batch_time:.2f}s", file=sys.stderr)

    return results


def main():
    parser = argparse.ArgumentParser(description="vLLM-Omni TTS Model Runner")
    parser.add_argument("--model", "-m", required=True, help="Model type (qwen_tts_vllm)")
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
    results = run_batch_vllm(config, tasks, output_dir)

    # Output results as JSON to stdout
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
