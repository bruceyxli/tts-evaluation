"""Compare full pipeline intermediate outputs between environments."""
import sys
import os

sys.path.insert(0, "./CosyVoice")
sys.path.insert(0, "./CosyVoice/third_party/Matcha-TTS")

import torch
import numpy as np
import soundfile as sf

print(f"=== Environment Info ===")
print(f"torch: {torch.__version__}")
print(f"numpy: {np.__version__}")

try:
    import x_transformers
    print(f"x-transformers: {x_transformers.__version__}")
except:
    print("x-transformers: not found")

from cosyvoice.cli.cosyvoice import AutoModel

model = AutoModel(model_dir="./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")
print(f"Model type: {type(model).__name__}")

prompt_wav = "./processed_audio/normalized_Yang.wav"
prompt_text = "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision."
tts_text = "Hello, this is a test of the text to speech system."

# Step 1: Text normalize
prompt_text_norm = model.frontend.text_normalize(prompt_text, split=False)
tts_texts = model.frontend.text_normalize(tts_text, split=True)
print(f"\n=== Text Normalize ===")
print(f"prompt_text_norm: {prompt_text_norm}")
print(f"tts_texts: {tts_texts}")

# Step 2: Frontend zero_shot - get all intermediate values
print(f"\n=== Frontend Zero Shot ===")
model_input = model.frontend.frontend_zero_shot(
    tts_texts[0], prompt_text_norm, prompt_wav, model.sample_rate, ""
)
for k, v in model_input.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, sum={v.float().sum().item():.6f}")
    else:
        print(f"  {k}: {v}")

# Step 3: Run LLM and check token output
import uuid as uuid_mod
print(f"\n=== LLM Token Generation ===")
this_uuid = str(uuid_mod.uuid1())
model.model.tts_speech_token_dict[this_uuid] = []
model.model.llm_end_dict[this_uuid] = False
model.model.hift_cache_dict[this_uuid] = None

# Set seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model.model.llm_job(
    model_input["text"],
    model_input["prompt_text"],
    model_input["llm_prompt_speech_token"],
    model_input["llm_embedding"],
    this_uuid,
)

tokens = model.model.tts_speech_token_dict[this_uuid]
print(f"Token count: {len(tokens)}")
print(f"Token stats: min={min(tokens)}, max={max(tokens)}, unique={len(set(tokens))}")
print(f"First 30: {tokens[:30]}")
print(f"Token sum: {sum(tokens)}")

# Step 4: Token2wav
print(f"\n=== Token2Wav ===")
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
print(f"Audio shape: {audio.shape}")
print(f"Audio stats: min={audio.min():.6f}, max={audio.max():.6f}, std={audio.std():.6f}")
print(f"Audio duration: {len(audio)/model.sample_rate:.2f}s")

# Cleanup
model.model.tts_speech_token_dict.pop(this_uuid, None)
model.model.llm_end_dict.pop(this_uuid, None)
model.model.hift_cache_dict.pop(this_uuid, None)
