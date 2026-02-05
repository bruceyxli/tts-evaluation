"""GLM-TTS and GLM-TTS-RL model adapters."""

import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np

from .base import TTSAdapter, TTSOutput


class GLMTTSAdapter(TTSAdapter):
    """Adapter for GLM-TTS model."""

    def __init__(
        self,
        name: str = "glm_tts",
        model_path: str = "./GLM-TTS/ckpt",
        device: str = "cuda",
        use_phoneme: bool = False,
        use_cache: bool = True,
    ):
        super().__init__(name, device)
        self.model_path = Path(model_path)
        self.use_phoneme = use_phoneme
        self.use_cache = use_cache
        self.frontend = None
        self.llm_model = None
        self.flow_model = None
        self.sample_rate = 22050  # GLM-TTS default

    def load(self) -> None:
        """Load GLM-TTS model components."""
        if self._loaded:
            return

        # Add GLM-TTS to path
        glm_tts_path = self.model_path.parent
        if str(glm_tts_path) not in sys.path:
            sys.path.insert(0, str(glm_tts_path))

        import torch
        from omegaconf import OmegaConf

        # Load configs
        llm_config_path = glm_tts_path / "configs" / "llm.yaml"
        flow_config_path = glm_tts_path / "configs" / "flow.yaml"

        llm_cfg = OmegaConf.load(llm_config_path) if llm_config_path.exists() else {}
        flow_cfg = OmegaConf.load(flow_config_path) if flow_config_path.exists() else {}

        # Load frontend
        from cosyvoice.cli.frontend import CosyVoiceFrontEnd
        self.frontend = CosyVoiceFrontEnd(
            speech_tokenizer_model=str(self.model_path / "speech_tokenizer_v1.onnx"),
            allowed_special="all",
        )

        # Load LLM model
        from llm.glmtts import GLMTTS
        self.llm_model = GLMTTS(
            text_dim=llm_cfg.get("text_dim", 4096),
            audio_dim=llm_cfg.get("audio_dim", 4096),
            llm_path=str(self.model_path / "language_model"),
        )
        llm_ckpt = torch.load(
            self.model_path / "llm.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.llm_model.load_state_dict(llm_ckpt, strict=False)
        self.llm_model.to(self.device).eval()

        # Load Flow model
        from flow.flow import ConsistencyFlowMatching
        from flow.dit import DiT
        dit_model = DiT(
            in_channels=flow_cfg.get("in_channels", 80),
            depth=flow_cfg.get("depth", 12),
        )
        self.flow_model = ConsistencyFlowMatching(
            estimator=dit_model,
            in_channels=flow_cfg.get("in_channels", 80),
        )
        flow_ckpt = torch.load(
            self.model_path / "flow.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.flow_model.load_state_dict(flow_ckpt, strict=False)
        self.flow_model.to(self.device).eval()

        self._loaded = True

    def synthesize(self, text: str, **kwargs) -> TTSOutput:
        """Synthesize speech using GLM-TTS."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        start_time = time.perf_counter()

        # Preprocess text
        with torch.no_grad():
            # Text to tokens via LLM
            text_tokens = self.frontend.text_to_ids(text)
            text_tensor = torch.tensor([text_tokens], device=self.device)

            # Generate speech tokens
            speech_tokens = self.llm_model.generate(
                text_tensor,
                max_new_tokens=2048,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
            )

            # Tokens to mel spectrogram via Flow
            mel = self.flow_model.inference(speech_tokens)

            # Mel to waveform via vocoder (HiFi-GAN in Flow)
            audio = self.flow_model.mel_to_wav(mel)
            audio = audio.squeeze().cpu().numpy()

        return self._wrap_synthesis(
            audio=audio,
            sample_rate=self.sample_rate,
            start_time=start_time,
            text=text,
            model="glm_tts",
        )

    def unload(self) -> None:
        """Unload model components."""
        if self.frontend is not None:
            del self.frontend
            self.frontend = None
        if self.llm_model is not None:
            del self.llm_model
            self.llm_model = None
        if self.flow_model is not None:
            del self.flow_model
            self.flow_model = None
        super().unload()


class GLMTTSRLAdapter(GLMTTSAdapter):
    """Adapter for GLM-TTS-RL (Reinforcement Learning enhanced) model."""

    def __init__(
        self,
        name: str = "glm_tts_rl",
        model_path: str = "./GLM-TTS/ckpt",
        rl_ckpt_path: Optional[str] = None,
        device: str = "cuda",
        use_phoneme: bool = False,
    ):
        super().__init__(name, model_path, device, use_phoneme)
        self.rl_ckpt_path = rl_ckpt_path or str(Path(model_path) / "rl_model.pt")

    def load(self) -> None:
        """Load GLM-TTS-RL model with RL checkpoint."""
        # First load base model
        super().load()

        import torch

        # Load RL-enhanced weights if available
        rl_ckpt = Path(self.rl_ckpt_path)
        if rl_ckpt.exists():
            rl_state = torch.load(rl_ckpt, map_location=self.device, weights_only=True)
            self.llm_model.load_state_dict(rl_state, strict=False)

    def synthesize(self, text: str, **kwargs) -> TTSOutput:
        """Synthesize with RL-enhanced model."""
        output = super().synthesize(text, **kwargs)
        output.metadata["model"] = "glm_tts_rl"
        return output
