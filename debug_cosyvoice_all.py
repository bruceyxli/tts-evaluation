"""Test all CosyVoice variants: standard, RL, and optionally vLLM."""
import sys
import os
import time
import argparse

sys.path.insert(0, "./CosyVoice")
sys.path.insert(0, "./CosyVoice/third_party/Matcha-TTS")

import torch
import soundfile as sf
import numpy as np

PROMPT_WAV = "./processed_audio/normalized_Yang.wav"
PROMPT_TEXT = "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision."
MODEL_DIR = "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"

TEST_TEXTS = [
    ("tutor_001", "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision."),
    ("tutor_002", "In this course, we've discussed trees many times. Today, let's explore how binary search trees maintain their balance."),
    ("simple_en", "Hello, this is a test of the text to speech system."),
]


def test_model(model, model_name, output_dir, stream=True):
    """Run inference for all test texts."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Stream: {stream}")
    print(f"{'='*60}")

    for tid, text in TEST_TEXTS:
        start = time.perf_counter()
        output = model.inference_zero_shot(
            text, PROMPT_TEXT, PROMPT_WAV, stream=stream
        )

        chunks = []
        first_chunk_time = None
        for chunk in output:
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - start
            chunks.append(chunk["tts_speech"])

        gen_time = time.perf_counter() - start

        if not chunks:
            print(f"  {tid}: ERROR - no audio generated")
            continue

        audio = torch.cat(chunks, dim=1).squeeze().cpu().numpy()
        duration = len(audio) / model.sample_rate
        rtf = gen_time / duration if duration > 0 else float("inf")

        out_path = os.path.join(output_dir, f"{tid}.wav")
        sf.write(out_path, audio, model.sample_rate)

        ftl = f"{first_chunk_time:.2f}s" if first_chunk_time else "N/A"
        print(f"  {tid}: dur={duration:.2f}s, gen={gen_time:.2f}s, RTF={rtf:.3f}, FTL={ftl}")
        print(f"    audio: std={audio.std():.4f}, min={audio.min():.4f}, max={audio.max():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standard", "rl", "vllm", "all"], default="all")
    args = parser.parse_args()

    from cosyvoice.cli.cosyvoice import AutoModel

    run_standard = args.mode in ("standard", "all")
    run_rl = args.mode in ("rl", "all")
    run_vllm = args.mode in ("vllm",)  # vllm needs separate env, don't include in "all"

    if run_standard:
        print("\n[1/3] Loading CosyVoice standard...")
        model = AutoModel(model_dir=MODEL_DIR)
        print(f"Model type: {type(model).__name__}, SR: {model.sample_rate}")
        test_model(model, "cosyvoice_standard", "outputs/debug_cosyvoice_standard")
        del model
        torch.cuda.empty_cache()

    if run_rl:
        print("\n[2/3] Loading CosyVoice RL...")
        model_rl = AutoModel(model_dir=MODEL_DIR)
        # Load RL weights
        rl_path = os.path.join(MODEL_DIR, "llm.rl.pt")
        if os.path.exists(rl_path):
            print(f"Loading RL weights from {rl_path}")
            state_dict = torch.load(rl_path, map_location=model_rl.model.device)
            model_rl.model.llm.load_state_dict(state_dict, strict=False)
            print("RL weights loaded successfully")
        else:
            print(f"WARNING: RL weights not found at {rl_path}")
        test_model(model_rl, "cosyvoice_rl", "outputs/debug_cosyvoice_rl")
        del model_rl
        torch.cuda.empty_cache()

    if run_vllm:
        print("\n[3/3] Loading CosyVoice vLLM...")
        model_vllm = AutoModel(model_dir=MODEL_DIR, load_vllm=True)
        print(f"Model type: {type(model_vllm).__name__}, SR: {model_vllm.sample_rate}")
        test_model(model_vllm, "cosyvoice_vllm", "outputs/debug_cosyvoice_vllm")
        del model_vllm
        torch.cuda.empty_cache()

    print("\n\nDone! Compare outputs in:")
    print("  outputs/debug_cosyvoice_standard/")
    print("  outputs/debug_cosyvoice_rl/")
    if run_vllm:
        print("  outputs/debug_cosyvoice_vllm/")


if __name__ == "__main__":
    main()
