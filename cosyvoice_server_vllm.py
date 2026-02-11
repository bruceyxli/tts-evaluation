"""CosyVoice vLLM Server - FastAPI server for CosyVoice with vLLM acceleration.

Wraps CosyVoice's inference with vLLM backend and serves via HTTP.
Returns WAV audio (not raw PCM) for easy client consumption.

Usage:
    conda run -n tts-cosyvoice-vllm python cosyvoice_server_vllm.py \
        --model_dir ./CosyVoice/pretrained_models/CosyVoice2-0.5B \
        --port 50000
"""

import argparse
import io
import logging
import os
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Monkey-patch torchaudio.load to use soundfile backend directly,
# bypassing broken torchcodec in torchaudio 2.9.0+
_original_torchaudio_load = torchaudio.load

def _patched_torchaudio_load(filepath, *args, **kwargs):
    """Use soundfile backend directly, bypassing torchcodec."""
    try:
        if hasattr(filepath, 'read'):
            audio_np, sr = sf.read(filepath, dtype="float32")
        else:
            audio_np, sr = sf.read(str(filepath), dtype="float32")
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]  # (1, T)
        else:
            audio_np = audio_np.T  # (channels, T)
        return torch.from_numpy(audio_np), sr
    except Exception:
        return _original_torchaudio_load(filepath, *args, **kwargs)

torchaudio.load = _patched_torchaudio_load

logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Setup CosyVoice paths before importing
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
COSYVOICE_DIR = os.path.join(ROOT_DIR, "CosyVoice")
sys.path.insert(0, COSYVOICE_DIR)
sys.path.insert(0, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS"))

# Set PYTHONPATH for vLLM subprocess
existing = os.environ.get("PYTHONPATH", "")
new_paths = f"{COSYVOICE_DIR}:{os.path.join(COSYVOICE_DIR, 'third_party', 'Matcha-TTS')}"
os.environ["PYTHONPATH"] = f"{new_paths}:{existing}" if existing else new_paths

# Set LD_LIBRARY_PATH for conda libs
conda_prefix = os.environ.get("CONDA_PREFIX", "")
if conda_prefix:
    conda_lib = os.path.join(conda_prefix, "lib")
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if conda_lib not in existing_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{existing_ld}" if existing_ld else conda_lib

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cosyvoice = None
cosyvoice_sample_rate = None


@app.get("/health")
async def health():
    return {"status": "ok", "model": "cosyvoice_vllm"}


def _save_upload_to_temp(upload_file) -> str:
    """Save an UploadFile to a temp WAV file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(upload_file.read())
    tmp.close()
    return tmp.name


@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(),
    prompt_text: str = Form(),
    prompt_wav: UploadFile = File(),
):
    """Voice cloning (zero-shot) endpoint. Returns WAV audio."""
    start = time.perf_counter()

    # Save uploaded file to temp path (CosyVoice expects a file path)
    prompt_path = _save_upload_to_temp(prompt_wav.file)
    try:
        model_output = cosyvoice.inference_zero_shot(
            tts_text, prompt_text, prompt_path, stream=False
        )

        # Collect all audio chunks
        audio_chunks = []
        for chunk in model_output:
            audio_chunks.append(chunk["tts_speech"])

        if not audio_chunks:
            return Response(content=b"", media_type="audio/wav", status_code=500)

        audio = torch.cat(audio_chunks, dim=1).squeeze().cpu().numpy()
        gen_time = time.perf_counter() - start

        # Convert to WAV
        buf = io.BytesIO()
        sf.write(buf, audio, cosyvoice_sample_rate, format="WAV")
        wav_bytes = buf.getvalue()

        duration = len(audio) / cosyvoice_sample_rate
        logging.info(
            f"Generated {duration:.2f}s audio in {gen_time:.2f}s (RTF={gen_time/duration:.3f})"
        )

        return Response(content=wav_bytes, media_type="audio/wav")
    finally:
        os.unlink(prompt_path)


@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Form(),
    prompt_wav: UploadFile = File(),
):
    """Cross-lingual synthesis endpoint. Returns WAV audio."""
    prompt_path = _save_upload_to_temp(prompt_wav.file)
    try:
        model_output = cosyvoice.inference_cross_lingual(
            tts_text, prompt_path, stream=False
        )

        audio_chunks = []
        for chunk in model_output:
            audio_chunks.append(chunk["tts_speech"])

        if not audio_chunks:
            return Response(content=b"", media_type="audio/wav", status_code=500)

        audio = torch.cat(audio_chunks, dim=1).squeeze().cpu().numpy()

        buf = io.BytesIO()
        sf.write(buf, audio, cosyvoice_sample_rate, format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav")
    finally:
        os.unlink(prompt_path)


@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Form(),
    instruct_text: str = Form(),
    prompt_wav: UploadFile = File(),
):
    """Instruction-guided synthesis endpoint. Returns WAV audio."""
    prompt_path = _save_upload_to_temp(prompt_wav.file)
    try:
        model_output = cosyvoice.inference_instruct2(
            tts_text, instruct_text, prompt_path, stream=False
        )

        audio_chunks = []
        for chunk in model_output:
            audio_chunks.append(chunk["tts_speech"])

        if not audio_chunks:
            return Response(content=b"", media_type="audio/wav", status_code=500)

        audio = torch.cat(audio_chunks, dim=1).squeeze().cpu().numpy()

        buf = io.BytesIO()
        sf.write(buf, audio, cosyvoice_sample_rate, format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav")
    finally:
        os.unlink(prompt_path)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="CosyVoice vLLM Server")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./CosyVoice/pretrained_models/CosyVoice2-0.5B",
    )
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--load_trt", action="store_true", default=False)
    parser.add_argument(
        "--no_vllm",
        action="store_true",
        default=False,
        help="Disable vLLM (use standard inference)",
    )
    args = parser.parse_args()

    from cosyvoice.cli.cosyvoice import AutoModel

    load_vllm = not args.no_vllm
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Loading CosyVoice from {args.model_dir} (vLLM={load_vllm})")

    cosyvoice = AutoModel(
        model_dir=args.model_dir,
        load_vllm=load_vllm,
        load_trt=args.load_trt,
        fp16=args.fp16,
    )
    cosyvoice_sample_rate = cosyvoice.sample_rate
    logging.info(f"Model loaded. Sample rate: {cosyvoice_sample_rate}")

    uvicorn.run(app, host=args.host, port=args.port)
