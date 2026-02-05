"""Audio utilities for loading and saving audio files."""

from pathlib import Path
from typing import Tuple, Union
import numpy as np


def save_audio(
    audio: np.ndarray,
    path: Union[str, Path],
    sample_rate: int = 24000,
    normalize: bool = True,
) -> None:
    """
    Save audio to file.

    Args:
        audio: Audio samples as numpy array
        path: Output file path
        sample_rate: Sample rate in Hz
        normalize: Whether to normalize audio to [-1, 1]
    """
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure 1D array
    if audio.ndim > 1:
        audio = audio.squeeze()

    # Normalize if requested
    if normalize:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95

    # Ensure float32
    audio = audio.astype(np.float32)

    sf.write(str(path), audio, sample_rate)


def load_audio(
    path: Union[str, Path],
    target_sr: int = None,
) -> Tuple[np.ndarray, int]:
    """
    Load audio from file.

    Args:
        path: Input file path
        target_sr: Target sample rate (resample if different)

    Returns:
        Tuple of (audio_samples, sample_rate)
    """
    import soundfile as sf

    audio, sr = sf.read(str(path))

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr
