#!/usr/bin/env python
"""
Debug script for testing a single TTS model directly.
Run this in the appropriate conda environment.

Usage:
    # GLM-TTS
    conda run -n tts-glm python debug_single_model.py --model glm_tts

    # CosyVoice
    conda run -n tts-cosyvoice python debug_single_model.py --model cosyvoice
"""

import argparse
import sys
import traceback
from pathlib import Path

def test_glm_tts():
    """Test GLM-TTS directly."""
    print("=" * 60)
    print("Testing GLM-TTS")
    print("=" * 60)

    import os
    os.chdir(Path("./GLM-TTS").resolve())
    print(f"Working directory: {os.getcwd()}")

    # Check CUDA
    print("\n[Step 1] Checking CUDA...")
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import GLM-TTS
    print("\n[Step 2] Importing GLM-TTS...")
    sys.path.insert(0, str(Path("./").resolve()))
    from glmtts_inference import load_models, generate_long, DEVICE
    print(f"  Import successful! Device: {DEVICE}")

    # Load models
    print("\n[Step 3] Loading models...")
    frontend, text_frontend, speech_tokenizer, llm, token2wav = load_models(
        use_phoneme=False, sample_rate=24000
    )
    print("  Models loaded successfully!")

    # Load prompt audio for zero-shot (Yang's voice - male)
    print("\n[Step 4] Loading prompt audio...")
    prompt_wav_path = "../processed_audio/normalized_Yang.wav"
    prompt_text = "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision."

    print(f"  Prompt audio: {prompt_wav_path}")
    print(f"  Prompt text: {prompt_text}")

    # Preload audio using soundfile to avoid torchaudio/torchcodec issues
    import soundfile as sf
    import torchaudio.functional as F
    audio_np, sr = sf.read(prompt_wav_path)
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]  # mono
    audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)  # [1, samples]
    print(f"  Loaded audio: shape={audio_tensor.shape}, sr={sr}")

    # Prepare different sample rates for different extractors
    # - speech_token extractor expects original sample rate (will resample internally)
    # - speech_feat extractor expects 24000
    # - spk_embedding extractor expects 16000
    audio_for_token = (audio_tensor, sr)  # tuple for token extractor

    # Resample to 24000 for feat extractor
    if sr != 24000:
        audio_24k = F.resample(audio_tensor, sr, 24000)
    else:
        audio_24k = audio_tensor
    print(f"  Audio 24k: shape={audio_24k.shape}")

    # Resample to 16000 for embedding extractor
    if sr != 16000:
        audio_16k = F.resample(audio_tensor, sr, 16000)
    else:
        audio_16k = audio_tensor
    print(f"  Audio 16k: shape={audio_16k.shape}")

    # Extract speech features from prompt
    prompt_speech_token = frontend._extract_speech_token([audio_for_token])
    speech_feat = frontend._extract_speech_feat(audio_24k, sample_rate=24000)
    embedding = frontend._extract_spk_embedding(audio_16k)

    # Prepare flow prompt token
    cache_speech_token = [prompt_speech_token.squeeze().tolist()]
    flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(device)

    print(f"  Speech token shape: {prompt_speech_token.shape}")
    print(f"  Speech feat shape: {speech_feat.shape}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Flow prompt token shape: {flow_prompt_token.shape}")

    # Test synthesis
    print("\n[Step 5] Testing synthesis...")
    test_text = "Hello, this is a test of the text to speech system."
    synth_text = text_frontend.text_normalize(test_text)
    print(f"  Text: {test_text}")
    print(f"  Normalized: {synth_text}")

    # Prepare cache with proper prompt
    prompt_text_tn = text_frontend.text_normalize(prompt_text)
    prompt_text_token = frontend._extract_text_token(prompt_text_tn + " ")
    cache = {
        "cache_text": [prompt_text_tn],
        "cache_text_token": [prompt_text_token],
        "cache_speech_token": cache_speech_token,
        "use_cache": True,
    }

    # Generate
    print("\n[Step 6] Generating audio...")
    tts_speech, _, _, _ = generate_long(
        frontend=frontend,
        text_frontend=text_frontend,
        llm=llm,
        flow=token2wav,
        text_info=["synth", synth_text],
        cache=cache,
        embedding=embedding,
        seed=0,
        flow_prompt_token=flow_prompt_token,
        speech_feat=speech_feat,
        device=device,
        use_phoneme=False,
    )

    audio = tts_speech.squeeze().cpu().numpy()
    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio duration: {len(audio) / 24000:.2f}s")

    # Save audio
    print("\n[Step 7] Saving audio...")
    import soundfile as sf
    output_path = Path("../outputs/debug_glm_tts.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, 24000)
    print(f"  Saved to: {output_path.resolve()}")

    print("\n" + "=" * 60)
    print("GLM-TTS TEST PASSED!")
    print("=" * 60)
    return True


def test_cosyvoice():
    """Test CosyVoice directly."""
    print("=" * 60)
    print("Testing CosyVoice")
    print("=" * 60)

    import os

    # Check CUDA
    print("\n[Step 1] Checking CUDA...")
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")

    # Setup paths
    print("\n[Step 2] Setting up paths...")
    repo_path = Path("./CosyVoice").resolve()
    print(f"  CosyVoice repo: {repo_path}")
    sys.path.insert(0, str(repo_path))
    sys.path.insert(0, str(repo_path / "third_party" / "Matcha-TTS"))

    # Import
    print("\n[Step 3] Importing CosyVoice...")
    from cosyvoice.cli.cosyvoice import AutoModel
    print("  Import successful!")

    # Load model
    print("\n[Step 4] Loading model...")
    model_path = "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
    model = AutoModel(model_dir=model_path)
    print(f"  Model loaded! Sample rate: {model.sample_rate}")

    # Check prompt audio
    print("\n[Step 5] Checking prompt audio...")
    prompt_wav = Path("./processed_audio/normalized_Yang.wav")
    if prompt_wav.exists():
        print(f"  Using: {prompt_wav}")
    else:
        prompt_wav = Path(model_path) / "asset" / "zero_shot_prompt.wav"
        print(f"  Fallback: {prompt_wav}")

    prompt_text = "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision."

    # Test synthesis
    print("\n[Step 6] Testing synthesis...")
    test_text = "Hello, this is a test of the text to speech system."
    print(f"  Text: {test_text}")
    print(f"  Prompt: {prompt_text[:50]}...")

    audio_chunks = []
    for chunk in model.inference_zero_shot(
        tts_text=test_text,
        prompt_text=prompt_text,
        prompt_wav=str(prompt_wav),
        stream=True,
    ):
        audio_chunks.append(chunk['tts_speech'])
        print(f"    Got chunk: {chunk['tts_speech'].shape}")

    if audio_chunks:
        audio = torch.cat(audio_chunks, dim=1).squeeze().cpu().numpy()
        print(f"  Total audio shape: {audio.shape}")
        print(f"  Audio duration: {len(audio) / model.sample_rate:.2f}s")

        # Save audio
        print("\n[Step 7] Saving audio...")
        import soundfile as sf
        output_path = Path("./outputs/debug_cosyvoice.wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, model.sample_rate)
        print(f"  Saved to: {output_path.resolve()}")

        print("\n" + "=" * 60)
        print("COSYVOICE TEST PASSED!")
        print("=" * 60)
        return True
    else:
        print("  ERROR: No audio chunks generated!")
        return False


def test_qwen_vllm():
    """Test Qwen-TTS with vLLM using voice cloning (Yang's voice)."""
    print("=" * 60)
    print("Testing Qwen-TTS with vLLM (Voice Cloning - Yang)")
    print("=" * 60)

    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Check CUDA
    print("\n[Step 1] Checking CUDA...")
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")

    # Import vLLM
    print("\n[Step 2] Importing vLLM-Omni...")
    try:
        from vllm import SamplingParams
        from vllm_omni import Omni
        import soundfile as sf
        print("  Import successful!")
    except ImportError as e:
        print(f"  ERROR: {e}")
        return False

    # Load model - Use Base model for voice cloning
    print("\n[Step 3] Loading model...")
    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  # Base model for voice cloning
    print(f"  Model: {model_id}")

    omni = Omni(
        model=model_id,
        stage_init_timeout=300,
    )
    print("  Model loaded!")

    # Build input using voice cloning with Yang's voice
    print("\n[Step 4] Building input with voice cloning...")
    test_text = "Hello, this is a test of the text to speech system."
    ref_audio_path = "./processed_audio/normalized_Yang.wav"
    ref_text = "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision."
    language = "English"
    mode = "icl"  # in-context learning mode

    # Load reference audio
    print(f"  Loading reference audio: {ref_audio_path}")
    audio_data_ref, sr_ref = sf.read(ref_audio_path)
    if audio_data_ref.ndim > 1:
        audio_data_ref = audio_data_ref[:, 0]  # Take first channel
    print(f"  Reference audio: {len(audio_data_ref)} samples, {sr_ref}Hz")

    # Build prompt for Base model voice cloning
    prompt = f"<|im_start|>assistant\n{test_text}<|im_end|>\n<|im_start|>assistant\n"

    inputs = {
        "prompt": prompt,
        "additional_information": {
            "task_type": ["VoiceClone"],
            "text": [test_text],
            "language": [language],
            "ref_audio": [audio_data_ref.tolist()],
            "ref_sr": [sr_ref],
            "ref_text": [ref_text],
            "mode": [mode],
            "max_new_tokens": [2048],
        },
    }

    print(f"  Text: {test_text}")
    print(f"  Reference text: {ref_text[:50]}...")
    print(f"  Language: {language}")
    print(f"  Mode: {mode}")

    # Generate
    print("\n[Step 5] Generating audio...")
    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        max_tokens=2048,
        seed=42,
        detokenize=False,
        repetition_penalty=1.05,
    )

    # generate() returns a generator
    omni_generator = omni.generate(inputs, [sampling_params])

    # Extract audio from generator output
    print("\n[Step 6] Extracting audio...")
    audio_data = None
    sample_rate = 24000  # Default, will be overwritten from output

    for stage_outputs in omni_generator:
        for output in stage_outputs.request_output:
            request_id = output.request_id
            print(f"  Processing request: {request_id}")

            # Get audio from multimodal_output
            if output.outputs and len(output.outputs) > 0:
                mm_output = output.outputs[0].multimodal_output
                if mm_output and "audio" in mm_output:
                    audio_tensor = mm_output["audio"]
                    sample_rate = mm_output.get("sr", torch.tensor(24000)).item()

                    # Convert to numpy
                    audio_data = audio_tensor.float().detach().cpu().numpy()
                    if audio_data.ndim > 1:
                        audio_data = audio_data.flatten()
                    break
        if audio_data is not None:
            break

    if audio_data is not None:
        print(f"  Audio shape: {audio_data.shape}")
        print(f"  Sample rate: {sample_rate}")
        print(f"  Audio duration: {len(audio_data) / sample_rate:.2f}s")

        # Save audio
        print("\n[Step 7] Saving audio...")
        import soundfile as sf
        output_path = Path("./outputs/debug_qwen_vllm.wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_data, sample_rate, format="WAV")
        print(f"  Saved to: {output_path.resolve()}")

        print("\n" + "=" * 60)
        print("QWEN-VLLM TEST PASSED!")
        print("=" * 60)
        return True
    else:
        print("  ERROR: No audio generated!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Debug single TTS model")
    parser.add_argument("--model", "-m", required=True,
                        choices=["glm_tts", "cosyvoice", "qwen_vllm"],
                        help="Model to test")
    args = parser.parse_args()

    try:
        if args.model == "glm_tts":
            success = test_glm_tts()
        elif args.model == "cosyvoice":
            success = test_cosyvoice()
        elif args.model == "qwen_vllm":
            success = test_qwen_vllm()
        else:
            print(f"Unknown model: {args.model}")
            success = False

        sys.exit(0 if success else 1)

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED WITH ERROR!")
        print("=" * 60)
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
