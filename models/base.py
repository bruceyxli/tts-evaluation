"""Base adapter interface for TTS models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
import time


@dataclass
class TTSOutput:
    """Output from TTS synthesis."""
    audio: np.ndarray           # Waveform samples (1D array)
    sample_rate: int            # Sample rate in Hz
    duration: float             # Audio duration in seconds
    generation_time: float      # Time taken to generate in seconds
    rtf: float                  # Real-Time Factor (generation_time / duration)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TTSAdapter(ABC):
    """Abstract base class for TTS model adapters."""

    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def synthesize(self, text: str, **kwargs) -> TTSOutput:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            **kwargs: Model-specific parameters

        Returns:
            TTSOutput with audio and metrics
        """
        pass

    def unload(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False

        # Force CUDA memory cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _calculate_rtf(self, generation_time: float, duration: float) -> float:
        """Calculate Real-Time Factor."""
        if duration <= 0:
            return float('inf')
        return generation_time / duration

    def _wrap_synthesis(self, audio: np.ndarray, sample_rate: int,
                        start_time: float, **metadata) -> TTSOutput:
        """Wrap synthesis result into TTSOutput."""
        generation_time = time.perf_counter() - start_time
        duration = len(audio) / sample_rate
        rtf = self._calculate_rtf(generation_time, duration)

        return TTSOutput(
            audio=audio,
            sample_rate=sample_rate,
            duration=duration,
            generation_time=generation_time,
            rtf=rtf,
            metadata=metadata
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, loaded={self._loaded})"
