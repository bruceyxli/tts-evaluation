"""Generate audio with exact Feb 4 evaluation texts for direct comparison."""
import sys
import os

sys.path.insert(0, "./CosyVoice")
sys.path.insert(0, "./CosyVoice/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel
import torch
import soundfile as sf
import numpy as np

model = AutoModel(model_dir="./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")

prompt_wav = "./processed_audio/normalized_Yang.wav"
prompt_text = "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision."

# Use exact texts from Feb 4 evaluation dataset
test_cases = [
    {
        "id": "math_integral_001",
        "text": "The integral of x squared dx from 0 to 1 equals one third.",
    },
    {
        "id": "tutor_simple",
        "text": "Hello, this is a test of the text to speech system.",
    },
    {
        "id": "code_sql_001",
        "text": "SELECT users.name, COUNT(orders.id) AS order_count FROM users LEFT JOIN orders ON users.id = orders.user_id GROUP BY users.name HAVING COUNT(orders.id) > 5 ORDER BY order_count DESC;",
    },
]

os.makedirs("outputs/debug_comparison", exist_ok=True)

for tc in test_cases:
    print(f"\n=== {tc['id']} ===")
    print(f"Text: {tc['text'][:80]}...")

    output = model.inference_zero_shot(
        tc["text"], prompt_text, prompt_wav, stream=True
    )

    chunks = []
    first_chunk_time = None
    import time

    start = time.perf_counter()
    for chunk in output:
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - start
        chunks.append(chunk["tts_speech"])

    gen_time = time.perf_counter() - start
    audio = torch.cat(chunks, dim=1).squeeze().cpu().numpy()
    duration = len(audio) / model.sample_rate
    rtf = gen_time / duration if duration > 0 else 0

    out_path = f"outputs/debug_comparison/{tc['id']}_current.wav"
    sf.write(out_path, audio, model.sample_rate)

    print(f"Duration: {duration:.2f}s (gen_time={gen_time:.2f}s, RTF={rtf:.3f})")
    print(f"First chunk latency: {first_chunk_time:.2f}s")
    print(f"Audio stats: std={audio.std():.4f}, min={audio.min():.4f}, max={audio.max():.4f}")
    print(f"Saved: {out_path}")

print("\n\nDone. Compare these with Feb 4 outputs:")
print("  Feb 4 dir: outputs/20260204_191134_cosyvoice-cosyvoice_rl-glm_tts-glm_tts_rl/cosyvoice/")
print("  Current dir: outputs/debug_comparison/")
