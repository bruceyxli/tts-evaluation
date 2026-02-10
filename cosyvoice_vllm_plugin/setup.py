from setuptools import setup

setup(
    name="cosyvoice-vllm-plugin",
    version="0.1.0",
    py_modules=["cosyvoice_vllm_plugin"],
    entry_points={
        "vllm.general_plugins": [
            "cosyvoice2 = cosyvoice_vllm_plugin:register",
        ],
    },
)
