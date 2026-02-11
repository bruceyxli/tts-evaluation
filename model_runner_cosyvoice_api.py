"""API-based Model Runner for CosyVoice vLLM server evaluation.

Sends HTTP requests to a running CosyVoice vLLM server's
/inference_zero_shot endpoint.

Usage:
    python model_runner_cosyvoice_api.py --model cosyvoice_vllm_api \
        --config config.json --tasks tasks.json --output ./output

Prerequisites:
    Start the CosyVoice vLLM server first:
    conda run -n tts-cosyvoice-vllm python cosyvoice_server_vllm.py \
        --model_dir ./CosyVoice/pretrained_models/CosyVoice2-0.5B \
        --port 50000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import soundfile as sf


def run_cosyvoice_api(config: dict, tasks: list, output_dir: Path) -> list:
    """Run TTS synthesis via CosyVoice API endpoint."""
    results = []

    api_base = config.get("api_base", "http://localhost:50000")
    timeout = config.get("api_timeout", 300.0)
    mode = config.get("mode", "zero_shot")

    prompt_wav = config.get("prompt_wav")
    prompt_text = config.get("prompt_text", "")
    instruct_text = config.get("instruct", "")

    if mode == "zero_shot":
        api_url = f"{api_base}/inference_zero_shot"
    elif mode == "cross_lingual":
        api_url = f"{api_base}/inference_cross_lingual"
    elif mode == "instruct":
        api_url = f"{api_base}/inference_instruct2"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Validate prompt_wav exists
    if prompt_wav and not os.path.exists(prompt_wav):
        raise FileNotFoundError(f"Prompt audio not found: {prompt_wav}")

    print(f"API endpoint: {api_url}", file=sys.stderr)
    print(f"Mode: {mode}", file=sys.stderr)
    print(f"Prompt WAV: {prompt_wav}", file=sys.stderr)
    print(f"Processing {len(tasks)} tasks...", file=sys.stderr)

    with httpx.Client(timeout=timeout) as client:
        for task in tasks:
            task_id = task["id"]
            text = task["text"]

            try:
                # Build multipart form data
                data = {"tts_text": text}
                files = {}

                if mode == "zero_shot":
                    data["prompt_text"] = prompt_text
                    files["prompt_wav"] = (
                        os.path.basename(prompt_wav),
                        open(prompt_wav, "rb"),
                        "audio/wav",
                    )
                elif mode == "cross_lingual":
                    files["prompt_wav"] = (
                        os.path.basename(prompt_wav),
                        open(prompt_wav, "rb"),
                        "audio/wav",
                    )
                elif mode == "instruct":
                    data["instruct_text"] = instruct_text
                    files["prompt_wav"] = (
                        os.path.basename(prompt_wav),
                        open(prompt_wav, "rb"),
                        "audio/wav",
                    )

                start_time = time.perf_counter()
                response = client.post(api_url, data=data, files=files)
                generation_time = time.perf_counter() - start_time

                # Close file handles
                for file_tuple in files.values():
                    fobj = file_tuple[1]
                    if hasattr(fobj, "close"):
                        fobj.close()

                if response.status_code != 200:
                    error_msg = response.text[:500]
                    print(
                        f"  {task_id}: API error {response.status_code}: {error_msg}",
                        file=sys.stderr,
                    )
                    results.append(
                        {
                            "id": task_id,
                            "success": False,
                            "error": f"API error {response.status_code}: {error_msg}",
                        }
                    )
                    continue

                # Check content type
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    error_data = response.json()
                    print(
                        f"  {task_id}: API returned error: {error_data}",
                        file=sys.stderr,
                    )
                    results.append(
                        {
                            "id": task_id,
                            "success": False,
                            "error": str(error_data),
                        }
                    )
                    continue

                # Save audio to file
                audio_path = output_dir / f"{task_id}.wav"
                with open(audio_path, "wb") as f:
                    f.write(response.content)

                # Read back to get duration and sample rate
                audio_data, sample_rate = sf.read(str(audio_path))
                duration = len(audio_data) / sample_rate
                rtf = generation_time / duration if duration > 0 else float("inf")

                results.append(
                    {
                        "id": task_id,
                        "success": True,
                        "audio_path": str(audio_path),
                        "sample_rate": sample_rate,
                        "duration": duration,
                        "generation_time": generation_time,
                        "rtf": rtf,
                        "backend": "cosyvoice-vllm-api",
                    }
                )
                print(
                    f"  {task_id}: RTF={rtf:.3f}, duration={duration:.2f}s, gen_time={generation_time:.2f}s",
                    file=sys.stderr,
                )

            except httpx.ConnectError as e:
                print(
                    f"  {task_id}: Connection failed - is the server running at {api_base}? {e}",
                    file=sys.stderr,
                )
                results.append(
                    {
                        "id": task_id,
                        "success": False,
                        "error": f"Connection failed: {e}",
                    }
                )
            except Exception as e:
                print(f"  {task_id}: Error - {e}", file=sys.stderr)
                results.append(
                    {"id": task_id, "success": False, "error": str(e)}
                )

    return results


def main():
    parser = argparse.ArgumentParser(description="CosyVoice API Model Runner")
    parser.add_argument("--model", "-m", required=True, help="Model type")
    parser.add_argument("--config", "-c", required=True, help="Config JSON file")
    parser.add_argument("--tasks", "-t", required=True, help="Tasks JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(args.tasks, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_cosyvoice_api(config, tasks, output_dir)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
