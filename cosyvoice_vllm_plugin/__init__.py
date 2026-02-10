"""vLLM plugin to register CosyVoice2ForCausalLM model class."""


def register():
    """Register CosyVoice2ForCausalLM in all vLLM processes (including engine subprocess)."""
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "CosyVoice2ForCausalLM",
        "cosyvoice.vllm.cosyvoice2:CosyVoice2ForCausalLM",
    )
