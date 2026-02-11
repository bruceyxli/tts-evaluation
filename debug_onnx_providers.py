"""Check which ONNX providers are actually used for speech tokenizer."""
import sys
sys.path.insert(0, "./CosyVoice")
sys.path.insert(0, "./CosyVoice/third_party/Matcha-TTS")

import onnxruntime
import torch
import numpy as np
import whisper
import soundfile as sf
import torchaudio.functional as F

print(f"onnxruntime: {onnxruntime.__version__}")
print(f"Available providers: {onnxruntime.get_available_providers()}")

MODEL_DIR = "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
tokenizer_path = f"{MODEL_DIR}/speech_tokenizer_v3.onnx"

# Load audio and compute mel (same as CosyVoice frontend)
audio_np, sr = sf.read("./processed_audio/normalized_Yang.wav")
if audio_np.ndim > 1:
    audio_np = audio_np.mean(axis=1)
speech = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)
if sr != 16000:
    speech = F.resample(speech, sr, 16000)
feat = whisper.log_mel_spectrogram(speech, n_mels=128)
print(f"\nMel shape: {feat.shape}, sum: {feat.sum().item():.6f}")

# Test with different providers
providers_to_test = [
    ("CUDA only", ["CUDAExecutionProvider"]),
    ("CPU only", ["CPUExecutionProvider"]),
]

# Check if TensorRT is available
if "TensorrtExecutionProvider" in onnxruntime.get_available_providers():
    providers_to_test.insert(0, ("TensorRT+CUDA", ["TensorrtExecutionProvider", "CUDAExecutionProvider"]))

for name, providers in providers_to_test:
    print(f"\n=== Provider: {name} ===")
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    try:
        session = onnxruntime.InferenceSession(tokenizer_path, sess_options=option, providers=providers)
        actual_providers = session.get_providers()
        print(f"  Actual providers: {actual_providers}")

        speech_token = session.run(
            None,
            {
                session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
            },
        )[0].flatten().tolist()

        token_tensor = torch.tensor(speech_token, dtype=torch.int32)
        print(f"  Token count: {len(speech_token)}")
        print(f"  Token sum: {sum(speech_token)}")
        print(f"  First 10: {speech_token[:10]}")
    except Exception as e:
        print(f"  ERROR: {e}")
