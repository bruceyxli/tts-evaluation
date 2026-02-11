"""Compare whisper mel spectrogram output between environments."""
import whisper
import torch
import numpy as np
import soundfile as sf

print(f"whisper version: {whisper.__version__}")
print(f"numpy version: {np.__version__}")

# Load the prompt audio
audio_np, sr = sf.read("./processed_audio/normalized_Yang.wav")
if audio_np.ndim > 1:
    audio_np = audio_np.mean(axis=1)
speech = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)

# Resample to 16kHz (same as CosyVoice frontend)
import torchaudio.functional as F
if sr != 16000:
    speech = F.resample(speech, sr, 16000)

print(f"Speech shape: {speech.shape}, sr: 16000")

# Extract mel spectrogram (same as CosyVoice _extract_speech_token)
feat = whisper.log_mel_spectrogram(speech, n_mels=128)
print(f"Mel shape: {feat.shape}")
print(f"Mel stats: min={feat.min():.4f}, max={feat.max():.4f}, mean={feat.mean():.4f}, std={feat.std():.4f}")
print(f"Mel first 10 values: {feat[0, 0, :10].tolist()}")
print(f"Mel checksum: {feat.sum().item():.6f}")

# Save for comparison
np.save("/tmp/whisper_mel.npy", feat.numpy())
print("Saved mel to /tmp/whisper_mel.npy")
