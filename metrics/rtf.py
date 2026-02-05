"""RTF (Real-Time Factor) and synthesis metrics collection."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import statistics

from models.base import TTSOutput


@dataclass
class SampleMetrics:
    """Metrics for a single synthesis sample."""
    text: str
    text_length: int
    audio_duration: float
    generation_time: float
    rtf: float
    sample_rate: int


@dataclass
class ModelMetrics:
    """Aggregated metrics for a model."""
    model_name: str
    samples: List[SampleMetrics] = field(default_factory=list)

    @property
    def rtf_mean(self) -> float:
        if not self.samples:
            return 0.0
        return statistics.mean(s.rtf for s in self.samples)

    @property
    def rtf_std(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return statistics.stdev(s.rtf for s in self.samples)

    @property
    def rtf_min(self) -> float:
        if not self.samples:
            return 0.0
        return min(s.rtf for s in self.samples)

    @property
    def rtf_max(self) -> float:
        if not self.samples:
            return 0.0
        return max(s.rtf for s in self.samples)

    @property
    def total_audio_duration(self) -> float:
        return sum(s.audio_duration for s in self.samples)

    @property
    def total_generation_time(self) -> float:
        return sum(s.generation_time for s in self.samples)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "sample_count": len(self.samples),
            "rtf": {
                "mean": round(self.rtf_mean, 4),
                "std": round(self.rtf_std, 4),
                "min": round(self.rtf_min, 4),
                "max": round(self.rtf_max, 4),
            },
            "total_audio_duration_sec": round(self.total_audio_duration, 2),
            "total_generation_time_sec": round(self.total_generation_time, 2),
        }


class MetricsCollector:
    """Collects and aggregates synthesis metrics."""

    def __init__(self):
        self._metrics: Dict[str, ModelMetrics] = {}

    def record(self, model_name: str, text: str, output: TTSOutput) -> None:
        """Record metrics from a synthesis output."""
        if model_name not in self._metrics:
            self._metrics[model_name] = ModelMetrics(model_name=model_name)

        sample = SampleMetrics(
            text=text,
            text_length=len(text),
            audio_duration=output.duration,
            generation_time=output.generation_time,
            rtf=output.rtf,
            sample_rate=output.sample_rate,
        )
        self._metrics[model_name].samples.append(sample)

    def get_model_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """Get metrics for a specific model."""
        return self._metrics.get(model_name)

    def get_all_metrics(self) -> Dict[str, ModelMetrics]:
        """Get all collected metrics."""
        return self._metrics

    def get_summary(self) -> Dict[str, dict]:
        """Get summary of all metrics as dict."""
        return {name: metrics.to_dict() for name, metrics in self._metrics.items()}

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()
