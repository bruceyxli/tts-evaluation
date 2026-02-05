# TTS Evaluation Pipeline - Architecture Design

## Project Structure
```
tts-evaluation/
├── main.py                 # Unified entry point
├── config.yaml             # Configuration
├── models/                 # Model adapters
│   ├── __init__.py
│   ├── base.py             # Abstract base adapter
│   ├── glm_tts.py          # GLM-TTS & GLM-TTS-RL
│   └── qwen_tts.py         # Qwen3-TTS
├── metrics/                # Metrics collection
│   ├── __init__.py
│   ├── rtf.py              # Real-Time Factor
│   └── resource.py         # CPU/GPU monitoring
├── utils/                  # Utilities
│   ├── __init__.py
│   └── audio.py            # Audio I/O
└── outputs/                # Results (auto-generated)
    └── {timestamp}_{models}/
        ├── scripts.json    # Test scripts used
        ├── metrics.json    # RTF & resource usage
        ├── glm_tts/        # GLM-TTS outputs
        │   └── *.wav
        ├── glm_tts_rl/     # GLM-TTS-RL outputs
        │   └── *.wav
        └── qwen_tts/       # Qwen3-TTS outputs
            └── *.wav
```

## ERD (Entity Relationship Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TTS Evaluation Pipeline                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    Config        │────▶│    Pipeline      │────▶│    OutputDir     │
│                  │     │                  │     │                  │
│ - models[]       │     │ - run()          │     │ - timestamp      │
│ - scripts[]      │     │ - load_models()  │     │ - model_names    │
│ - output_dir     │     │ - evaluate()     │     │ - create()       │
│ - device         │     │ - save_results() │     │ - save_audio()   │
└──────────────────┘     └────────┬─────────┘     └──────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   TTSAdapter     │  │  MetricsCollector│  │   ResourceMonitor│
│   (Abstract)     │  │                  │  │                  │
│                  │  │ - start()        │  │ - start()        │
│ - load()         │  │ - record_rtf()   │  │ - sample()       │
│ - synthesize()   │  │ - get_summary()  │  │ - stop()         │
│ - unload()       │  │                  │  │ - get_stats()    │
└────────┬─────────┘  └──────────────────┘  └──────────────────┘
         │
         ├────────────────────────┬────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  GLMTTSAdapter   │   │  GLMTTSRLAdapter │   │  QwenTTSAdapter  │
│                  │   │                  │   │                  │
│ - model_path     │   │ - model_path     │   │ - model_id       │
│ - use_phoneme    │   │ - rl_ckpt        │   │ - speaker        │
│ - load()         │   │ - load()         │   │ - load()         │
│ - synthesize()   │   │ - synthesize()   │   │ - synthesize()   │
└──────────────────┘   └──────────────────┘   └──────────────────┘
```

## Data Flow

```
Input Scripts (JSON/YAML)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Pipeline.run()                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  1. Create output directory                             │ │
│  │  2. Copy scripts to output                              │ │
│  │  3. For each model:                                     │ │
│  │     ├─ Start resource monitor (background thread)       │ │
│  │     ├─ Load model                                       │ │
│  │     ├─ For each script:                                 │ │
│  │     │   ├─ Start RTF timer                              │ │
│  │     │   ├─ Synthesize audio                             │ │
│  │     │   ├─ Stop RTF timer                               │ │
│  │     │   └─ Save audio to model subfolder                │ │
│  │     ├─ Stop resource monitor                            │ │
│  │     └─ Unload model                                     │ │
│  │  4. Save metrics.json                                   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
Output Directory: {timestamp}_{glm-qwen}/
├── scripts.json          # Input scripts backup
├── metrics.json          # RTF + CPU/GPU stats per model
├── glm_tts/              # Audio outputs
├── glm_tts_rl/
└── qwen_tts/
```

## Key Classes

### TTSAdapter (base.py)
```python
@dataclass
class TTSOutput:
    audio: np.ndarray       # Waveform samples
    sample_rate: int        # Hz
    duration: float         # seconds
    generation_time: float  # seconds
    rtf: float              # Real-Time Factor

class TTSAdapter(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def synthesize(self, text: str, **kwargs) -> TTSOutput: ...

    @abstractmethod
    def unload(self) -> None: ...
```

### MetricsCollector (metrics/rtf.py)
```python
class MetricsCollector:
    def start(self) -> None: ...
    def record(self, model: str, text: str, output: TTSOutput) -> None: ...
    def get_summary(self) -> Dict[str, ModelMetrics]: ...
```

### ResourceMonitor (metrics/resource.py)
```python
class ResourceMonitor:
    def start(self, interval: float = 0.5) -> None: ...
    def stop(self) -> ResourceStats: ...
    # Returns: cpu_percent, gpu_percent, gpu_memory_mb
```

## Output Format

### metrics.json
```json
{
  "timestamp": "2024-02-04T15:30:00",
  "models": ["glm_tts", "glm_tts_rl", "qwen_tts"],
  "total_scripts": 10,
  "results": {
    "glm_tts": {
      "rtf": {"mean": 0.35, "min": 0.28, "max": 0.45},
      "resource": {
        "cpu_percent": {"mean": 45.2, "max": 78.1},
        "gpu_percent": {"mean": 82.3, "max": 95.0},
        "gpu_memory_mb": {"mean": 4200, "max": 5100}
      },
      "audio_files": 10
    },
    "glm_tts_rl": { ... },
    "qwen_tts": { ... }
  }
}
```

## Usage

```bash
# Run evaluation with all models
python main.py --config config.yaml

# Run with specific models
python main.py --models glm_tts qwen_tts

# Run with custom scripts
python main.py --scripts my_scripts.json
```
