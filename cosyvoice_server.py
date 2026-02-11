"""
CosyVoice OpenAI-compatible TTS Server.

Wraps CosyVoice model and exposes an OpenAI-compatible /v1/audio/speech endpoint.
Voice prompts are pre-loaded at startup from voice_config.json.

Usage:
    conda run -n tts-cosyvoice python cosyvoice_server.py \
        --port 8005 \
        --model-dir ./CosyVoice/pretrained_models/CosyVoice2-0.5B \
        --repo-path ./CosyVoice \
        --voice-config voice_config.json
"""

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="CosyVoice TTS Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (set during startup)
cosyvoice_model = None
sample_rate = 24000
voice_prompts = {}  # name -> {"prompt_wav": str (file path), "prompt_text": str}


class SpeechRequest(BaseModel):
    model: str = "cosyvoice"
    input: str
    voice: str = "Professor Allen Yang"
    response_format: str = "pcm"  # "pcm" or "wav"
    stream: bool = True
    speed: float = 1.0


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "cosyvoice", "object": "model", "owned_by": "cosyvoice"}],
    }


@app.get("/v1/voices")
async def list_voices():
    return {"voices": list(voice_prompts.keys())}


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    if not request.input.strip():
        raise HTTPException(400, "input text is empty")

    voice = voice_prompts.get(request.voice)
    if voice is None:
        available = list(voice_prompts.keys())
        raise HTTPException(404, f"Voice '{request.voice}' not found. Available: {available}")

    prompt_wav = voice["prompt_wav"]
    prompt_text = voice["prompt_text"]
    tts_text = request.input

    def generate_pcm():
        # Pass file path directly — CosyVoice frontend loads wav at different
        # sample rates internally (_extract_speech_feat@24k, _extract_spk_embedding@16k)
        output = cosyvoice_model.inference_zero_shot(
            tts_text=tts_text,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            stream=request.stream,
            speed=request.speed,
        )
        for chunk in output:
            tts_speech = chunk["tts_speech"]  # torch.Tensor [1, N]
            pcm_bytes = (tts_speech.numpy() * (2**15)).astype(np.int16).tobytes()
            yield pcm_bytes

    if request.response_format == "wav":
        import io
        import soundfile as sf

        all_pcm = b"".join(generate_pcm())
        audio_array = np.frombuffer(all_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        buf = io.BytesIO()
        sf.write(buf, audio_array, sample_rate, format="WAV")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav")

    # Default: stream raw PCM s16le
    return StreamingResponse(
        generate_pcm(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Audio-Format": "pcm_s16le",
            "X-Channels": "1",
        },
    )


def load_voice_prompts(config_path: str):
    """Load voice prompts from config JSON, store file paths for lazy loading."""
    if not os.path.exists(config_path):
        logger.warning("Voice config %s not found, no voices loaded", config_path)
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    prompts = {}
    for name, info in config.items():
        wav_path = info["prompt_wav"]
        if not os.path.exists(wav_path):
            logger.warning("Prompt audio %s not found for voice '%s', skipping", wav_path, name)
            continue
        prompts[name] = {
            "prompt_wav": os.path.abspath(wav_path),
            "prompt_text": info["prompt_text"],
        }
        logger.info("Loaded voice '%s' from %s", name, wav_path)

    return prompts


def main():
    parser = argparse.ArgumentParser(description="CosyVoice OpenAI-compatible TTS Server")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./CosyVoice/pretrained_models/CosyVoice2-0.5B",
    )
    parser.add_argument("--repo-path", type=str, default="./CosyVoice")
    parser.add_argument("--voice-config", type=str, default="voice_config.json")
    args = parser.parse_args()

    # Setup CosyVoice sys.path
    repo = Path(args.repo_path).resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "third_party" / "Matcha-TTS"))

    from cosyvoice.cli.cosyvoice import AutoModel

    global cosyvoice_model, sample_rate, voice_prompts

    logger.info("Loading CosyVoice model from %s ...", args.model_dir)
    cosyvoice_model = AutoModel(model_dir=args.model_dir)
    sample_rate = cosyvoice_model.sample_rate
    logger.info("Model loaded. Sample rate: %d", sample_rate)

    voice_prompts = load_voice_prompts(args.voice_config)
    logger.info("Loaded %d voice(s): %s", len(voice_prompts), list(voice_prompts.keys()))

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
