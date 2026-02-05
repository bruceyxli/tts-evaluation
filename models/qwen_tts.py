"""Qwen3-TTS model adapter."""

import time
from typing import Optional
import numpy as np

from .base import TTSAdapter, TTSOutput


class QwenTTSAdapter(TTSAdapter):
    """Adapter for Qwen3-TTS model."""

    def __init__(
        self,
        name: str = "qwen_tts",
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device: str = "cuda:0",
        speaker: str = "Vivian",
        language: str = "English",
        use_flash_attn: bool = True,
    ):
        super().__init__(name, device)
        self.model_id = model_id
        self.speaker = speaker
        self.language = language
        self.use_flash_attn = use_flash_attn
        self.sample_rate = 24000  # Qwen3-TTS default

    def load(self) -> None:
        """Load Qwen3-TTS model."""
        if self._loaded:
            return

        import torch
        from qwen_tts import Qwen3TTSModel

        attn_impl = "flash_attention_2" if self.use_flash_attn else "sdpa"

        self.model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        self._loaded = True

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        **kwargs
    ) -> TTSOutput:
        """
        Synthesize speech using Qwen3-TTS.

        Args:
            text: Input text to synthesize
            speaker: Speaker name (overrides default)
            language: Language (overrides default)
            instruct: Optional instruction for voice modification
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Use defaults if not specified
        speaker = speaker or self.speaker
        language = language or self.language

        # Generate audio
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )

        # Convert to numpy array
        if hasattr(wavs, 'cpu'):
            audio = wavs.squeeze().cpu().numpy()
        else:
            audio = np.array(wavs).squeeze()

        return self._wrap_synthesis(
            audio=audio,
            sample_rate=sr,
            start_time=start_time,
            text=text,
            model="qwen_tts",
            speaker=speaker,
            language=language,
        )

    def synthesize_with_reference(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        language: Optional[str] = None,
        **kwargs
    ) -> TTSOutput:
        """
        Synthesize with voice cloning.

        Args:
            text: Input text to synthesize
            ref_audio: Path to reference audio file
            ref_text: Transcript of reference audio
            language: Target language
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()
        language = language or self.language

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

        if hasattr(wavs, 'cpu'):
            audio = wavs.squeeze().cpu().numpy()
        else:
            audio = np.array(wavs).squeeze()

        return self._wrap_synthesis(
            audio=audio,
            sample_rate=sr,
            start_time=start_time,
            text=text,
            model="qwen_tts_clone",
            ref_audio=ref_audio,
        )
