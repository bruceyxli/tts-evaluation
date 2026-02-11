"""Diagnostic: inspect CosyVoice3 LLM token output."""
import sys
import os

sys.path.insert(0, "./CosyVoice")
sys.path.insert(0, "./CosyVoice/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel
import torch
import soundfile as sf
import numpy as np
import uuid as uuid_mod

model = AutoModel(model_dir="./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")

text = "Hello, this is a test of the text to speech system."
prompt_wav = "./processed_audio/normalized_Yang.wav"
prompt_text = "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision."

# Normalize text
prompt_text_normalized = model.frontend.text_normalize(prompt_text, split=False)
tts_texts = model.frontend.text_normalize(text, split=True)
print(f"Normalized prompt: {prompt_text_normalized}")
print(f"Normalized tts texts: {tts_texts}")

# Get model input
model_input = model.frontend.frontend_zero_shot(
    tts_texts[0], prompt_text_normalized, prompt_wav, model.sample_rate, ""
)
print(f"Text token shape: {model_input['text'].shape}")
print(f"Prompt text token shape: {model_input['prompt_text'].shape}")
print(f"LLM prompt speech token shape: {model_input['llm_prompt_speech_token'].shape}")
print(f"Flow prompt speech token shape: {model_input['flow_prompt_speech_token'].shape}")
print(f"Prompt speech feat shape: {model_input['prompt_speech_feat'].shape}")
print(f"LLM embedding shape: {model_input['llm_embedding'].shape}")

# Run LLM to get tokens
this_uuid = str(uuid_mod.uuid1())
model.model.tts_speech_token_dict[this_uuid] = []
model.model.llm_end_dict[this_uuid] = False
model.model.hift_cache_dict[this_uuid] = None

# Run LLM job directly
print("\nRunning LLM inference...")
model.model.llm_job(
    model_input["text"],
    model_input["prompt_text"],
    model_input["llm_prompt_speech_token"],
    model_input["llm_embedding"],
    this_uuid,
)

tokens = model.model.tts_speech_token_dict[this_uuid]
print(f"LLM generated {len(tokens)} tokens")
if tokens:
    print(f"Token distribution: min={min(tokens)}, max={max(tokens)}, unique={len(set(tokens))}")
    print(f"Expected audio duration at 25 tokens/s: {len(tokens)/25:.2f}s")
    print(f"First 50 tokens: {tokens[:50]}")
    print(f"Silent tokens in model: {model.model.silent_tokens}")
    silent_count = sum(1 for t in tokens if t in model.model.silent_tokens)
    print(f"Silent tokens in output: {silent_count}/{len(tokens)}")

    # Now run token2wav to check the flow and vocoder
    print("\nRunning token2wav (flow + vocoder)...")
    this_tts_speech_token = torch.tensor(tokens).unsqueeze(dim=0)
    this_tts_speech = model.model.token2wav(
        token=this_tts_speech_token,
        prompt_token=model_input["flow_prompt_speech_token"],
        prompt_feat=model_input["prompt_speech_feat"],
        embedding=model_input["flow_embedding"],
        token_offset=0,
        uuid=this_uuid,
        finalize=True,
    )
    audio = this_tts_speech.squeeze().cpu().numpy()
    duration = len(audio) / model.sample_rate
    print(f"Audio: shape={audio.shape}, duration={duration:.2f}s")
    print(f"Audio stats: min={audio.min():.4f}, max={audio.max():.4f}, std={audio.std():.4f}")
    sf.write("outputs/debug_llm_tokens.wav", audio, model.sample_rate)
    print("Saved: outputs/debug_llm_tokens.wav")
else:
    print("ERROR: No tokens generated!")

# Cleanup
model.model.tts_speech_token_dict.pop(this_uuid, None)
model.model.llm_end_dict.pop(this_uuid, None)
model.model.hift_cache_dict.pop(this_uuid, None)
