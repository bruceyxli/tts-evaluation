"""
API-based Model Runner for TTS evaluation.

Sends HTTP requests to a running vLLM-Omni server's OpenAI-compatible
/v1/audio/speech endpoint instead of loading models locally.

Usage:
    python model_runner_api.py --model qwen_tts_api --config config.json --tasks tasks.json --output ./output

Prerequisites:
    Start the vLLM-Omni server first:
    vllm-omni serve "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" \
        --stage-configs-path <path>/qwen3_tts.yaml \
        --host 0.0.0.0 --port 8000 \
        --trust-remote-code --enforce-eager --omni
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import httpx
import soundfile as sf


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = audio_path.lower()
    if ext.endswith(".wav"):
        mime_type = "audio/wav"
    elif ext.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif ext.endswith(".flac"):
        mime_type = "audio/flac"
    else:
        mime_type = "audio/wav"

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def build_payload(model_type: str, text: str, config: dict) -> dict:
    """Build the /v1/audio/speech request payload."""
    payload = {
        "input": text,
        "voice": config.get("speaker", "Vivian"),
        "response_format": config.get("response_format", "wav"),
    }

    if config.get("model_id"):
        payload["model"] = config["model_id"]
    if config.get("language"):
        payload["language"] = config["language"]
    if config.get("max_new_tokens"):
        payload["max_new_tokens"] = config["max_new_tokens"]
    if config.get("instructions"):
        payload["instructions"] = config["instructions"]

    if model_type == "qwen_tts_api_vc":
        payload["task_type"] = "Base"

        ref_audio = config.get("ref_audio")
        if not ref_audio:
            raise ValueError("ref_audio is required for qwen_tts_api_vc")

        # Encode local file to base64 data URL if it's a local path
        if ref_audio.startswith(("http://", "https://", "data:")):
            payload["ref_audio"] = ref_audio
        else:
            payload["ref_audio"] = encode_audio_to_base64(ref_audio)

        if config.get("ref_text"):
            payload["ref_text"] = config["ref_text"]

        mode = config.get("mode", "icl")
        payload["x_vector_only_mode"] = mode == "xvec_only"
    else:
        payload["task_type"] = "CustomVoice"

    return payload


def run_api(config: dict, tasks: list, output_dir: Path, model_type: str) -> list:
    """Run TTS synthesis via API endpoint."""
    results = []

    api_base = config.get("api_base", "http://localhost:8000")
    api_key = config.get("api_key", "EMPTY")
    timeout = config.get("api_timeout", 300.0)

    api_url = f"{api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Pre-encode ref_audio once for voice cloning
    ref_audio_encoded = None
    if model_type == "qwen_tts_api_vc" and config.get("ref_audio"):
        ref_path = config["ref_audio"]
        if not ref_path.startswith(("http://", "https://", "data:")):
            print(f"Pre-encoding reference audio: {ref_path}", file=sys.stderr)
            ref_audio_encoded = encode_audio_to_base64(ref_path)

    print(f"API endpoint: {api_url}", file=sys.stderr)
    print(f"Processing {len(tasks)} tasks...", file=sys.stderr)

    with httpx.Client(timeout=timeout) as client:
        for task in tasks:
            task_id = task["id"]
            text = task["text"]

            try:
                payload = build_payload(model_type, text, config)

                # Use pre-encoded ref_audio to avoid re-encoding per request
                if ref_audio_encoded and "ref_audio" in payload:
                    payload["ref_audio"] = ref_audio_encoded

                start_time = time.perf_counter()
                response = client.post(api_url, json=payload, headers=headers)
                generation_time = time.perf_counter() - start_time

                if response.status_code != 200:
                    error_msg = response.text[:500]
                    print(f"  {task_id}: API error {response.status_code}: {error_msg}", file=sys.stderr)
                    results.append({
                        "id": task_id, "success": False,
                        "error": f"API error {response.status_code}: {error_msg}",
                    })
                    continue

                # Check for JSON error in response body
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    error_data = response.json()
                    print(f"  {task_id}: API returned error: {error_data}", file=sys.stderr)
                    results.append({
                        "id": task_id, "success": False,
                        "error": str(error_data),
                    })
                    continue

                # Save audio to file
                audio_path = output_dir / f"{task_id}.wav"
                with open(audio_path, "wb") as f:
                    f.write(response.content)

                # Read back to get duration and sample rate
                audio_data, sample_rate = sf.read(str(audio_path))
                duration = len(audio_data) / sample_rate
                rtf = generation_time / duration if duration > 0 else float("inf")

                results.append({
                    "id": task_id,
                    "success": True,
                    "audio_path": str(audio_path),
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "generation_time": generation_time,
                    "rtf": rtf,
                    "backend": "vllm-omni-api",
                })
                print(f"  {task_id}: RTF={rtf:.3f}, duration={duration:.2f}s, gen_time={generation_time:.2f}s", file=sys.stderr)

            except httpx.ConnectError as e:
                print(f"  {task_id}: Connection failed - is the server running at {api_base}? {e}", file=sys.stderr)
                results.append({
                    "id": task_id, "success": False,
                    "error": f"Connection failed: {e}",
                })
            except Exception as e:
                print(f"  {task_id}: Error - {e}", file=sys.stderr)
                results.append({
                    "id": task_id, "success": False,
                    "error": str(e),
                })

    return results


async def _process_task(
    client: httpx.AsyncClient,
    api_url: str,
    headers: dict,
    task: dict,
    model_type: str,
    config: dict,
    ref_audio_encoded: str | None,
    output_dir: Path,
) -> dict:
    """Process a single TTS task asynchronously."""
    task_id = task["id"]
    text = task["text"]

    try:
        payload = build_payload(model_type, text, config)
        if ref_audio_encoded and "ref_audio" in payload:
            payload["ref_audio"] = ref_audio_encoded

        start_time = time.perf_counter()
        response = await client.post(api_url, json=payload, headers=headers)
        generation_time = time.perf_counter() - start_time

        if response.status_code != 200:
            error_msg = response.text[:500]
            print(f"  {task_id}: API error {response.status_code}: {error_msg}", file=sys.stderr)
            return {"id": task_id, "success": False, "error": f"API error {response.status_code}: {error_msg}"}

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            error_data = response.json()
            print(f"  {task_id}: API returned error: {error_data}", file=sys.stderr)
            return {"id": task_id, "success": False, "error": str(error_data)}

        audio_path = output_dir / f"{task_id}.wav"
        with open(audio_path, "wb") as f:
            f.write(response.content)

        audio_data, sample_rate = sf.read(str(audio_path))
        duration = len(audio_data) / sample_rate
        rtf = generation_time / duration if duration > 0 else float("inf")

        print(f"  {task_id}: RTF={rtf:.3f}, duration={duration:.2f}s, gen_time={generation_time:.2f}s", file=sys.stderr)
        return {
            "id": task_id, "success": True, "audio_path": str(audio_path),
            "sample_rate": sample_rate, "duration": duration,
            "generation_time": generation_time, "rtf": rtf, "backend": "vllm-omni-api",
        }

    except httpx.ConnectError as e:
        api_base = config.get("api_base", "http://localhost:8000")
        print(f"  {task_id}: Connection failed - is the server running at {api_base}? {e}", file=sys.stderr)
        return {"id": task_id, "success": False, "error": f"Connection failed: {e}"}
    except Exception as e:
        print(f"  {task_id}: Error - {e}", file=sys.stderr)
        return {"id": task_id, "success": False, "error": str(e)}


async def run_api_concurrent(config: dict, tasks: list, output_dir: Path, model_type: str) -> list:
    """Run TTS synthesis via API with concurrent requests."""
    concurrency = config.get("concurrency", 1)
    api_base = config.get("api_base", "http://localhost:8000")
    api_key = config.get("api_key", "EMPTY")
    timeout = config.get("api_timeout", 300.0)

    api_url = f"{api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Pre-encode ref_audio once
    ref_audio_encoded = None
    if model_type == "qwen_tts_api_vc" and config.get("ref_audio"):
        ref_path = config["ref_audio"]
        if not ref_path.startswith(("http://", "https://", "data:")):
            print(f"Pre-encoding reference audio: {ref_path}", file=sys.stderr)
            ref_audio_encoded = encode_audio_to_base64(ref_path)

    print(f"API endpoint: {api_url}", file=sys.stderr)
    print(f"Concurrency: {concurrency}", file=sys.stderr)
    print(f"Processing {len(tasks)} tasks...", file=sys.stderr)

    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def bounded_task(task):
            async with semaphore:
                return await _process_task(
                    client, api_url, headers, task, model_type,
                    config, ref_audio_encoded, output_dir,
                )

        coros = [bounded_task(task) for task in tasks]
        results = await asyncio.gather(*coros)

    return list(results)


def main():
    parser = argparse.ArgumentParser(description="API-based TTS Model Runner")
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

    concurrency = config.get("concurrency", 1)
    if concurrency > 1:
        results = asyncio.run(run_api_concurrent(config, tasks, output_dir, args.model))
    else:
        results = run_api(config, tasks, output_dir, args.model)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
