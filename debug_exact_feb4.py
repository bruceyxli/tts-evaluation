"""Generate audio with EXACT Feb 4 evaluation texts for comparison."""
import sys
import os
import time

sys.path.insert(0, "./CosyVoice")
sys.path.insert(0, "./CosyVoice/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel
import torch
import soundfile as sf

model = AutoModel(model_dir="./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")

prompt_wav = "./processed_audio/normalized_Yang.wav"
prompt_text = "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision."

# Exact texts from the Feb 4 evaluation dataset
test_cases = [
    # code_sql_001: Feb 4 duration was 7.76s
    {
        "id": "code_sql_001",
        "text": "SELECT name, age FROM Person WHERE age > 18 ORDER BY age DESC;",
    },
    # tutor_001: Feb 4 duration was 20.24s
    {
        "id": "tutor_001",
        "text": "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision.",
    },
    # tutor_002: Feb 4 duration was 13.84s
    {
        "id": "tutor_002",
        "text": "In this course, we've discussed trees many times. Today, let's explore how binary search trees maintain their balance.",
    },
]

os.makedirs("outputs/debug_exact_feb4", exist_ok=True)

feb4_durations = {
    "code_sql_001": 7.76,
    "tutor_001": 20.24,
    "tutor_002": 13.84,
}

for tc in test_cases:
    tid = tc["id"]
    print(f"\n=== {tid} ===")
    print(f"Text: {tc['text']}")

    start = time.perf_counter()
    output = model.inference_zero_shot(
        tc["text"], prompt_text, prompt_wav, stream=True
    )

    chunks = []
    first_chunk_time = None
    for chunk in output:
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - start
        chunks.append(chunk["tts_speech"])

    gen_time = time.perf_counter() - start
    audio = torch.cat(chunks, dim=1).squeeze().cpu().numpy()
    duration = len(audio) / model.sample_rate

    out_path = f"outputs/debug_exact_feb4/{tid}.wav"
    sf.write(out_path, audio, model.sample_rate)

    feb4_dur = feb4_durations.get(tid, "?")
    ratio = duration / feb4_dur if isinstance(feb4_dur, float) else "?"
    print(f"Current duration:  {duration:.2f}s")
    print(f"Feb 4 duration:    {feb4_dur}s")
    print(f"Duration ratio:    {ratio if isinstance(ratio, str) else f'{ratio:.2f}x'}")
    print(f"Gen time: {gen_time:.2f}s, RTF: {gen_time/duration:.3f}")
    print(f"Saved: {out_path}")

print("\n\nFeb 4 audio for comparison:")
print("  outputs/20260204_191134_cosyvoice-cosyvoice_rl-glm_tts-glm_tts_rl/cosyvoice/")
