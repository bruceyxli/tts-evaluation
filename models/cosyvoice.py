"""CosyVoice model adapter (Fun-CosyVoice3-0.5B)."""

import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np

from .base import TTSAdapter, TTSOutput


class CosyVoiceAdapter(TTSAdapter):
    """Adapter for Fun-CosyVoice3-0.5B model."""

    def __init__(
        self,
        name: str = "cosyvoice",
        model_path: str = "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B",
        repo_path: str = "./CosyVoice",
        device: str = "cuda",
        use_rl: bool = False,
    ):
        super().__init__(name, device)
        self.model_path = Path(model_path)
        self.repo_path = Path(repo_path)
        self.use_rl = use_rl
        self.sample_rate = 24000  # CosyVoice default

    def load(self) -> None:
        """Load CosyVoice model."""
        if self._loaded:
            return

        # Add CosyVoice paths to sys.path
        cosyvoice_path = self.repo_path.resolve()
        matcha_path = cosyvoice_path / "third_party" / "Matcha-TTS"

        for p in [str(cosyvoice_path), str(matcha_path)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from cosyvoice.cli.cosyvoice import AutoModel

        self.model = AutoModel(model_dir=str(self.model_path))
        self.sample_rate = self.model.sample_rate
        self._loaded = True

    def synthesize(
        self,
        text: str,
        mode: str = "zero_shot",
        prompt_text: Optional[str] = None,
        prompt_audio: Optional[str] = None,
        instruct: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> TTSOutput:
        """
        Synthesize speech using CosyVoice.

        Args:
            text: Input text to synthesize
            mode: Synthesis mode - "zero_shot", "cross_lingual", "instruct"
            prompt_text: Text prompt for voice cloning
            prompt_audio: Path to reference audio for voice cloning
            instruct: Instruction text for instruct mode
            stream: Whether to use streaming (not supported in this adapter)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Default prompt if not provided
        if prompt_audio is None:
            prompt_audio = str(self.model_path / "asset" / "zero_shot_prompt.wav")
        if prompt_text is None:
            prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

        # Select inference method based on mode
        if mode == "instruct" and instruct:
            generator = self.model.inference_instruct2(
                tts_text=text,
                instruct_text=instruct,
                prompt_wav=prompt_audio,
                stream=False,
            )
        elif mode == "cross_lingual":
            generator = self.model.inference_cross_lingual(
                tts_text=text,
                prompt_wav=prompt_audio,
                stream=False,
            )
        else:  # zero_shot (default)
            generator = self.model.inference_zero_shot(
                tts_text=text,
                prompt_text=prompt_text,
                prompt_wav=prompt_audio,
                stream=False,
            )

        # Collect audio chunks
        audio_chunks = []
        for chunk in generator:
            audio_chunks.append(chunk['tts_speech'])

        # Concatenate audio
        import torch
        if audio_chunks:
            audio_tensor = torch.cat(audio_chunks, dim=1)
            audio = audio_tensor.squeeze().cpu().numpy()
        else:
            audio = np.array([], dtype=np.float32)

        return self._wrap_synthesis(
            audio=audio,
            sample_rate=self.sample_rate,
            start_time=start_time,
            text=text,
            model="cosyvoice",
            mode=mode,
        )

    def synthesize_with_reference(
        self,
        text: str,
        ref_audio: str,
        ref_text: Optional[str] = None,
        **kwargs
    ) -> TTSOutput:
        """
        Synthesize with voice cloning using reference audio.

        Args:
            text: Input text to synthesize
            ref_audio: Path to reference audio file
            ref_text: Transcript of reference audio (optional)
        """
        prompt_text = ref_text or "You are a helpful assistant.<|endofprompt|>"
        return self.synthesize(
            text=text,
            mode="zero_shot",
            prompt_text=prompt_text,
            prompt_audio=ref_audio,
            **kwargs
        )

    def unload(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        super().unload()


class CosyVoiceRLAdapter(CosyVoiceAdapter):
    """Adapter for CosyVoice with RL-enhanced model (llm.rl.pt)."""

    def __init__(
        self,
        name: str = "cosyvoice_rl",
        model_path: str = "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B",
        repo_path: str = "./CosyVoice",
        device: str = "cuda",
    ):
        super().__init__(name, model_path, repo_path, device, use_rl=True)

    def load(self) -> None:
        """Load CosyVoice with RL model weights."""
        if self._loaded:
            return

        # Add CosyVoice paths
        cosyvoice_path = self.repo_path.resolve()
        matcha_path = cosyvoice_path / "third_party" / "Matcha-TTS"

        for p in [str(cosyvoice_path), str(matcha_path)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        import torch
        from cosyvoice.cli.cosyvoice import AutoModel

        # Load base model first
        self.model = AutoModel(model_dir=str(self.model_path))

        # Swap to RL weights if available
        rl_weights = self.model_path / "llm.rl.pt"
        if rl_weights.exists():
            print(f"Loading RL weights from {rl_weights}")
            state_dict = torch.load(rl_weights, map_location=self.device)
            self.model.model.llm.load_state_dict(state_dict, strict=False)

        self.sample_rate = self.model.sample_rate
        self._loaded = True

    def synthesize(self, text: str, **kwargs) -> TTSOutput:
        """Synthesize with RL-enhanced model."""
        output = super().synthesize(text, **kwargs)
        output.metadata["model"] = "cosyvoice_rl"
        return output
