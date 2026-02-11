"""
Compare audio from:
1. Direct inference (working approach from debug_cosyvoice_all.py)
2. Server-style PCM conversion (cosyvoice_server.py approach)
3. Server-style WAV conversion

This isolates whether the issue is in the audio conversion code.
"""
import sys
import os

sys.path.insert(0, "./CosyVoice")
sys.path.insert(0, "./CosyVoice/third_party/Matcha-TTS")

import torch
import numpy as np
import soundfile as sf

PROMPT_WAV = "./processed_audio/normalized_Yang.wav"
PROMPT_TEXT = "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision."
MODEL_DIR = "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
TEST_TEXT = "Hello, this is a test of the text to speech system."
OUTPUT_DIR = "outputs/debug_server_audio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading CosyVoice model...")
from cosyvoice.cli.cosyvoice import AutoModel

model = AutoModel(model_dir=MODEL_DIR)
print(f"Model loaded. SR={model.sample_rate}")

# ============================================================
# Method 1: Direct approach (known-working from debug_cosyvoice_all.py)
# ============================================================
print("\n=== Method 1: Direct (working) approach ===")
output = model.inference_zero_shot(TEST_TEXT, PROMPT_TEXT, PROMPT_WAV, stream=True)
chunks = []
for chunk in output:
    chunks.append(chunk["tts_speech"])

audio_direct = torch.cat(chunks, dim=1).squeeze().cpu().numpy()
out1 = os.path.join(OUTPUT_DIR, "method1_direct.wav")
sf.write(out1, audio_direct, model.sample_rate)
print(f"  Saved: {out1}")
print(f"  Shape: {audio_direct.shape}, dtype: {audio_direct.dtype}")
print(f"  Stats: min={audio_direct.min():.6f}, max={audio_direct.max():.6f}, std={audio_direct.std():.6f}")
print(f"  Duration: {len(audio_direct)/model.sample_rate:.2f}s")

# ============================================================
# Method 2: Server PCM approach (from cosyvoice_server.py)
# ============================================================
print("\n=== Method 2: Server PCM approach ===")
output = model.inference_zero_shot(TEST_TEXT, PROMPT_TEXT, PROMPT_WAV, stream=True)
pcm_chunks = []
chunk_info = []
for i, chunk in enumerate(output):
    tts_speech = chunk["tts_speech"]  # torch.Tensor [1, N]
    chunk_info.append({
        "shape": tts_speech.shape,
        "device": str(tts_speech.device),
        "dtype": str(tts_speech.dtype),
        "min": tts_speech.min().item(),
        "max": tts_speech.max().item(),
    })
    # Exact same conversion as cosyvoice_server.py line 97
    pcm_bytes = (tts_speech.numpy() * (2**15)).astype(np.int16).tobytes()
    pcm_chunks.append(pcm_bytes)

all_pcm = b"".join(pcm_chunks)

# Convert back to float (same as server WAV path, lines 104-106)
audio_pcm = np.frombuffer(all_pcm, dtype=np.int16).astype(np.float32) / 32768.0
out2 = os.path.join(OUTPUT_DIR, "method2_server_pcm.wav")
sf.write(out2, audio_pcm, model.sample_rate)
print(f"  Saved: {out2}")
print(f"  Shape: {audio_pcm.shape}, dtype: {audio_pcm.dtype}")
print(f"  Stats: min={audio_pcm.min():.6f}, max={audio_pcm.max():.6f}, std={audio_pcm.std():.6f}")
print(f"  Duration: {len(audio_pcm)/model.sample_rate:.2f}s")
print(f"  Chunks: {len(pcm_chunks)}")
for i, ci in enumerate(chunk_info):
    print(f"    Chunk {i}: shape={ci['shape']}, device={ci['device']}, "
          f"dtype={ci['dtype']}, min={ci['min']:.6f}, max={ci['max']:.6f}")

# ============================================================
# Method 3: Non-streaming (stream=False)
# ============================================================
print("\n=== Method 3: Non-streaming ===")
output = model.inference_zero_shot(TEST_TEXT, PROMPT_TEXT, PROMPT_WAV, stream=False)
chunks_ns = []
for chunk in output:
    chunks_ns.append(chunk["tts_speech"])

audio_ns = torch.cat(chunks_ns, dim=1).squeeze().cpu().numpy()
out3 = os.path.join(OUTPUT_DIR, "method3_non_streaming.wav")
sf.write(out3, audio_ns, model.sample_rate)
print(f"  Saved: {out3}")
print(f"  Shape: {audio_ns.shape}, dtype: {audio_ns.dtype}")
print(f"  Stats: min={audio_ns.min():.6f}, max={audio_ns.max():.6f}, std={audio_ns.std():.6f}")
print(f"  Duration: {len(audio_ns)/model.sample_rate:.2f}s")

# ============================================================
# Compare: difference between methods
# ============================================================
print("\n=== Comparison ===")
# Method 1 vs 2 (direct vs server PCM conversion)
# Note: they use different inference runs, so tokens differ.
# But we can compare audio statistics to see if conversion distorts.
print(f"Method 1 (direct):        len={len(audio_direct)}, std={audio_direct.std():.6f}")
print(f"Method 2 (server PCM):    len={len(audio_pcm)}, std={audio_pcm.std():.6f}")
print(f"Method 3 (non-streaming): len={len(audio_ns)}, std={audio_ns.std():.6f}")

# Also test: what if we do the server conversion on the direct audio?
print("\n=== Method 4: Direct audio -> server PCM conversion -> back ===")
# Simulate: take the direct float audio, convert like server does
pcm_int16 = (audio_direct * (2**15)).astype(np.int16)
audio_roundtrip = pcm_int16.astype(np.float32) / 32768.0
out4 = os.path.join(OUTPUT_DIR, "method4_roundtrip.wav")
sf.write(out4, audio_roundtrip, model.sample_rate)
max_diff = np.max(np.abs(audio_direct - audio_roundtrip))
print(f"  Max difference from roundtrip: {max_diff:.8f}")
print(f"  (This should be very small - just int16 quantization)")

print(f"\nAll files saved to {OUTPUT_DIR}/")
print("Listen to all 4 files to compare audio quality.")
