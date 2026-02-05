"""CPU and GPU resource monitoring."""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional
import statistics


@dataclass
class ResourceSample:
    """Single resource measurement sample."""
    timestamp: float
    cpu_percent: float
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None


@dataclass
class ResourceStats:
    """Aggregated resource statistics."""
    samples: List[ResourceSample] = field(default_factory=list)
    duration_sec: float = 0.0

    @property
    def cpu_mean(self) -> float:
        if not self.samples:
            return 0.0
        return statistics.mean(s.cpu_percent for s in self.samples)

    @property
    def cpu_max(self) -> float:
        if not self.samples:
            return 0.0
        return max(s.cpu_percent for s in self.samples)

    @property
    def gpu_mean(self) -> Optional[float]:
        gpu_vals = [s.gpu_percent for s in self.samples if s.gpu_percent is not None]
        if not gpu_vals:
            return None
        return statistics.mean(gpu_vals)

    @property
    def gpu_max(self) -> Optional[float]:
        gpu_vals = [s.gpu_percent for s in self.samples if s.gpu_percent is not None]
        if not gpu_vals:
            return None
        return max(gpu_vals)

    @property
    def gpu_memory_mean(self) -> Optional[float]:
        mem_vals = [s.gpu_memory_mb for s in self.samples if s.gpu_memory_mb is not None]
        if not mem_vals:
            return None
        return statistics.mean(mem_vals)

    @property
    def gpu_memory_max(self) -> Optional[float]:
        mem_vals = [s.gpu_memory_mb for s in self.samples if s.gpu_memory_mb is not None]
        if not mem_vals:
            return None
        return max(mem_vals)

    def to_dict(self) -> dict:
        result = {
            "duration_sec": round(self.duration_sec, 2),
            "sample_count": len(self.samples),
            "cpu_percent": {
                "mean": round(self.cpu_mean, 1),
                "max": round(self.cpu_max, 1),
            },
        }

        if self.gpu_mean is not None:
            result["gpu_percent"] = {
                "mean": round(self.gpu_mean, 1),
                "max": round(self.gpu_max, 1),
            }

        if self.gpu_memory_mean is not None:
            result["gpu_memory_mb"] = {
                "mean": round(self.gpu_memory_mean, 0),
                "max": round(self.gpu_memory_max, 0),
            }

        return result


class ResourceMonitor:
    """Monitor CPU and GPU resource usage in background thread."""

    def __init__(self, interval: float = 0.5, gpu_index: int = 0):
        """
        Initialize resource monitor.

        Args:
            interval: Sampling interval in seconds
            gpu_index: GPU index to monitor
        """
        self.interval = interval
        self.gpu_index = gpu_index
        self._samples: List[ResourceSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0
        self._lock = threading.Lock()

        # Check if GPU monitoring is available
        self._has_gpu = self._check_gpu_available()

    def _check_gpu_available(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True
        except Exception:
            return False

    def _get_gpu_stats(self) -> tuple:
        """Get GPU utilization and memory usage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return util.gpu, mem.used / (1024 * 1024)  # Convert to MB
        except Exception:
            return None, None

    def _sample_loop(self) -> None:
        """Background sampling loop."""
        import psutil

        while self._running:
            cpu_percent = psutil.cpu_percent(interval=None)

            gpu_percent, gpu_memory = None, None
            if self._has_gpu:
                gpu_percent, gpu_memory = self._get_gpu_stats()

            sample = ResourceSample(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                gpu_percent=gpu_percent,
                gpu_memory_mb=gpu_memory,
            )

            with self._lock:
                self._samples.append(sample)

            time.sleep(self.interval)

    def start(self) -> None:
        """Start monitoring in background."""
        if self._running:
            return

        self._samples = []
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceStats:
        """Stop monitoring and return stats."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            samples = self._samples.copy()

        duration = time.time() - self._start_time

        return ResourceStats(
            samples=samples,
            duration_sec=duration,
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
