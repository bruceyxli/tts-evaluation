"""TTS Evaluation Pipeline - Unified entry point with isolated environments."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from metrics.resource import ResourceMonitor

# Model type to conda environment mapping
MODEL_ENV_MAP = {
    "glm_tts": "tts-glm",
    "glm_tts_rl": "tts-glm",
    "qwen_tts": "tts-qwen",
    "qwen_tts_vc": "tts-qwen",  # Voice cloning with Base model
    "qwen_tts_vllm": "tts-qwen-vllm",  # vLLM-Omni accelerated (Linux only)
    "qwen_tts_vllm_vc": "tts-qwen-vllm",  # vLLM voice cloning
    "cosyvoice": "tts-cosyvoice",
    "cosyvoice_rl": "tts-cosyvoice",
    "cosyvoice_vllm": "tts-cosyvoice-vllm",  # CosyVoice with vLLM acceleration (Linux only)
    "qwen_tts_api": None,      # API-based, no conda env needed
    "qwen_tts_api_vc": None,   # API-based, no conda env needed
    "cosyvoice_vllm_api": None,  # API-based, CosyVoice vLLM server
}

# Model types that use vLLM runner
VLLM_MODEL_TYPES = {"qwen_tts_vllm", "qwen_tts_vllm_vc"}

# Model types that use API runner (no conda environment needed)
API_MODEL_TYPES = {"qwen_tts_api", "qwen_tts_api_vc", "cosyvoice_vllm_api"}


class TTSPipeline:
    """Unified TTS evaluation pipeline with automatic environment switching."""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cuda")
        self.output_base = Path(config.get("output_dir", "./outputs"))
        self.project_root = Path(__file__).parent.resolve()

        # Resource monitoring config
        rm_config = config.get("resource_monitor", {})
        self.monitor_interval = rm_config.get("interval", 0.5)
        self.monitor_gpu_index = rm_config.get("gpu_index", 0)

    def _get_conda_exe(self) -> str:
        """Get conda executable path."""
        return os.environ.get("CONDA_EXE", "conda")

    def _archive_model_outputs(self, model_names: List[str]):
        """Archive only the specified models' outputs to history/."""
        import shutil

        latest_dir = self.output_base / "latest"
        history_dir = self.output_base / "history"

        if not latest_dir.exists():
            return  # Nothing to archive

        # Check which models need archiving
        models_to_archive = []
        for name in model_names:
            model_dir = latest_dir / name
            if model_dir.exists() and model_dir.is_dir():
                models_to_archive.append(name)

        if not models_to_archive:
            return  # Nothing to archive

        # Create timestamped archive folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = history_dir / timestamp
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Move only the specified model folders to archive
        for name in models_to_archive:
            src = latest_dir / name
            dst = archive_dir / name
            shutil.move(str(src), str(dst))
            print(f"  Archived {name}/ -> history/{timestamp}/{name}/")

    def _setup_output_dir(self, model_names: List[str]) -> Path:
        """Setup output directory structure. Returns the latest/ path."""
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Create latest/ directory if not exists
        latest_dir = self.output_base / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)

        # Archive only the models we're about to run
        self._archive_model_outputs(model_names)

        return latest_dir

    def _generate_report(self, all_results: Dict, output_dir: Path):
        """Generate a comparison report in markdown table format."""
        report_lines = [
            "# TTS Evaluation Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Comparison",
            "",
        ]

        # Build table data for alignment
        headers = ["Model", "RTF (mean)", "RTF (min)", "RTF (max)", "First Token (mean)", "First Token (min)", "First Token (max)", "Success Rate"]
        rows = []

        for name, data in all_results.items():
            rtf_mean = f"{data['rtf']['mean']:.4f}" if data['rtf']['mean'] else "N/A"
            rtf_min = f"{data['rtf']['min']:.4f}" if data['rtf']['min'] else "N/A"
            rtf_max = f"{data['rtf']['max']:.4f}" if data['rtf']['max'] else "N/A"

            ftl_mean = f"{data['first_token_latency']['mean']:.4f}s" if data['first_token_latency']['mean'] else "N/A"
            ftl_min = f"{data['first_token_latency']['min']:.4f}s" if data['first_token_latency']['min'] else "N/A"
            ftl_max = f"{data['first_token_latency']['max']:.4f}s" if data['first_token_latency']['max'] else "N/A"

            success_rate = f"{data['success']}/{data['total']} ({100*data['success']/data['total']:.1f}%)" if data['total'] > 0 else "N/A"

            rows.append([name, rtf_mean, rtf_min, rtf_max, ftl_mean, ftl_min, ftl_max, success_rate])

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        # Generate aligned table
        def format_row(cells):
            return "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells)) + " |"

        def format_separator():
            return "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

        report_lines.append(format_row(headers))
        report_lines.append(format_separator())
        for row in rows:
            report_lines.append(format_row(row))

        # Add summary section
        report_lines.extend([
            "",
            "## Metrics Explanation",
            "",
            "- **RTF (Real-Time Factor)**: Generation time / Audio duration. RTF < 1.0 means faster than real-time.",
            "- **First Token Latency**: Time until the first audio chunk is generated (streaming latency).",
            "- **Success Rate**: Number of successful synthesis / Total tests.",
            "",
            "## Detailed Statistics",
            "",
        ])

        # Add per-model details
        for name, data in all_results.items():
            report_lines.extend([
                f"### {name}",
                "",
                f"- Total samples: {data['total']}",
                f"- Successful: {data['success']}",
                f"- Failed: {data['failed']}",
                f"- Audio files generated: {data['audio_files']}",
            ])

            # Add resource usage if available
            if "resource_usage" in data:
                rs = data["resource_usage"]
                report_lines.append(f"- CPU usage: mean {rs['cpu_percent']['mean']:.1f}%, max {rs['cpu_percent']['max']:.1f}%")
                if "gpu_percent" in rs:
                    report_lines.append(f"- GPU usage: mean {rs['gpu_percent']['mean']:.1f}%, max {rs['gpu_percent']['max']:.1f}%")
                if "gpu_memory_mb" in rs:
                    report_lines.append(f"- GPU memory: mean {rs['gpu_memory_mb']['mean']:.0f}MB, max {rs['gpu_memory_mb']['max']:.0f}MB")

            report_lines.append("")

        report_content = "\n".join(report_lines)

        # Save report
        report_path = output_dir / "report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\nReport saved to: {report_path}")

        # Also print a compact table to console
        print(f"\n{'=' * 100}")
        print("COMPARISON TABLE")
        print(f"{'=' * 100}")
        print(f"{'Model':<20} {'RTF (mean)':<12} {'First Token Latency (mean)':<28} {'Success Rate':<15}")
        print("-" * 100)
        for name, data in all_results.items():
            rtf_mean = f"{data['rtf']['mean']:.4f}" if data['rtf']['mean'] else "N/A"
            ftl_mean = f"{data['first_token_latency']['mean']:.4f}s" if data['first_token_latency']['mean'] else "N/A"
            success_rate = f"{data['success']}/{data['total']}" if data['total'] > 0 else "N/A"
            print(f"{name:<20} {rtf_mean:<12} {ftl_mean:<28} {success_rate:<15}")
        print(f"{'=' * 100}")

    def load_scripts(self, scripts_path: Optional[str] = None) -> List[dict]:
        """Load test scripts from file or config."""
        # Priority: 1) command line arg, 2) config scripts_file, 3) inline scripts
        if not scripts_path:
            scripts_path = self.config.get("scripts_file")

        if scripts_path:
            path = Path(scripts_path)
            if path.exists():
                if path.suffix == ".json":
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return data.get("samples", data) if isinstance(data, dict) else data
                elif path.suffix in (".yaml", ".yml"):
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        return data.get("samples", data) if isinstance(data, dict) else data

        # Use inline scripts from config
        return self.config.get("scripts", [
            {"id": "test_001", "text": "Hello, this is a test."},
            {"id": "test_002", "text": "The quick brown fox jumps over the lazy dog."},
        ])

    def _run_api_model(
        self,
        model_type: str,
        model_config: dict,
        tasks: List[dict],
        output_dir: Path,
    ) -> List[dict]:
        """Run model synthesis via API (no conda environment needed)."""
        if model_type == "cosyvoice_vllm_api":
            runner_script = self.project_root / "model_runner_cosyvoice_api.py"
        else:
            runner_script = self.project_root / "model_runner_api.py"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as config_file:
            full_config = {**model_config, "device": self.device}
            json.dump(full_config, config_file, ensure_ascii=False)
            config_path = config_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tasks_file:
            json.dump(tasks, tasks_file, ensure_ascii=False)
            tasks_path = tasks_file.name

        try:
            cmd = [
                sys.executable,
                str(runner_script),
                "--model", model_type,
                "--config", config_path,
                "--tasks", tasks_path,
                "--output", str(output_dir.resolve()),
            ]

            print(f"  Using API backend (no conda environment)")
            print(f"  Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )

            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    print(f"    {line}")

            if result.returncode != 0:
                print(f"  ERROR: Process exited with code {result.returncode}")
                print(f"  STDERR: {result.stderr}")
                return []

            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        try:
                            return json.loads(line)
                        except json.JSONDecodeError:
                            continue
                try:
                    return json.loads(result.stdout.strip())
                except json.JSONDecodeError:
                    print(f"  WARNING: Could not parse JSON from stdout")
                    print(f"  STDOUT (first 500 chars): {result.stdout[:500]}")
                    return []
            return []
        finally:
            os.unlink(config_path)
            os.unlink(tasks_path)

    def _run_model_in_env(
        self,
        model_type: str,
        model_config: dict,
        tasks: List[dict],
        output_dir: Path,
    ) -> List[dict]:
        """Run model synthesis in its dedicated conda environment."""
        # API model types: run directly without conda wrapping
        if model_type in API_MODEL_TYPES:
            return self._run_api_model(model_type, model_config, tasks, output_dir)

        env_name = MODEL_ENV_MAP.get(model_type)
        if not env_name:
            raise ValueError(f"No environment mapping for model type: {model_type}")

        conda = self._get_conda_exe()

        # Create temp files for config and tasks
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as config_file:
            # Merge model config with device setting
            full_config = {**model_config, "device": self.device}
            json.dump(full_config, config_file, ensure_ascii=False)
            config_path = config_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tasks_file:
            json.dump(tasks, tasks_file, ensure_ascii=False)
            tasks_path = tasks_file.name

        try:
            # Choose runner script based on model type
            if model_type in VLLM_MODEL_TYPES:
                runner_script = self.project_root / "model_runner_vllm.py"
                print(f"  Using vLLM-Omni backend for accelerated inference")
            else:
                runner_script = self.project_root / "model_runner.py"

            # Build command
            cmd = [
                conda, "run", "-n", env_name,
                "python", str(runner_script),
                "--model", model_type,
                "--config", config_path,
                "--tasks", tasks_path,
                "--output", str(output_dir.resolve()),
            ]

            print(f"  Running in environment: {env_name}")
            print(f"  Command: {' '.join(cmd)}")

            # Prepare environment variables
            env = os.environ.copy()
            if model_type == "cosyvoice_vllm":
                # Set compiler paths for triton compilation in vLLM
                env["CC"] = "x86_64-conda-linux-gnu-gcc"
                env["CXX"] = "x86_64-conda-linux-gnu-g++"

            # Run subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                env=env,
            )

            # Print stderr (progress info)
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    print(f"    {line}")

            if result.returncode != 0:
                print(f"  ERROR: Process exited with code {result.returncode}")
                print(f"  STDERR: {result.stderr}")
                return []

            # Parse results from stdout
            # Some libraries print warnings to stdout, so extract only the JSON line
            if result.stdout.strip():
                # Find the JSON array in stdout (starts with '[' and ends with ']')
                lines = result.stdout.strip().split('\n')
                for line in reversed(lines):  # JSON is usually at the end
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        try:
                            return json.loads(line)
                        except json.JSONDecodeError:
                            continue
                # Fallback: try the whole output
                try:
                    return json.loads(result.stdout.strip())
                except json.JSONDecodeError:
                    print(f"  WARNING: Could not parse JSON from stdout")
                    print(f"  STDOUT (first 500 chars): {result.stdout[:500]}")
                    return []
            return []

        finally:
            # Cleanup temp files
            os.unlink(config_path)
            os.unlink(tasks_path)

    def run(
        self,
        models: Optional[List[str]] = None,
        scripts_path: Optional[str] = None,
    ) -> Path:
        """
        Run the evaluation pipeline.

        Args:
            models: List of model names to evaluate (None = all from config)
            scripts_path: Path to scripts file (None = use config)

        Returns:
            Path to output directory
        """
        # Get models to evaluate
        model_configs = self.config.get("models", [])
        if models:
            model_configs = [m for m in model_configs if m["name"] in models]

        if not model_configs:
            print("No models to evaluate!")
            return None

        # Load scripts
        scripts = self.load_scripts(scripts_path)
        print(f"Loaded {len(scripts)} test scripts")

        # Normalize scripts to list of dicts
        tasks = []
        for i, script in enumerate(scripts):
            if isinstance(script, str):
                tasks.append({"id": f"sample_{i:03d}", "text": script})
            elif isinstance(script, dict):
                task_id = script.get("id", f"sample_{i:03d}")
                tasks.append({"id": task_id, "text": script.get("text", "")})

        # Setup output directory (archives existing outputs)
        model_names = [m["name"] for m in model_configs]
        output_dir = self._setup_output_dir(model_names)
        print(f"Output directory: {output_dir}")

        # Save scripts to output
        scripts_output = output_dir / "scripts.json"
        with open(scripts_output, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        # Results storage
        all_results = {}

        # Evaluate each model
        for mc in model_configs:
            name = mc["name"]
            model_type = mc["type"]

            print(f"\n{'=' * 60}")
            print(f"Evaluating: {name} (type: {model_type})")
            print(f"{'=' * 60}")

            # Create model output directory
            model_dir = output_dir / name
            model_dir.mkdir(exist_ok=True)

            # Start resource monitoring
            monitor = ResourceMonitor(
                interval=self.monitor_interval,
                gpu_index=self.monitor_gpu_index,
            )
            monitor.start()

            # Run in isolated environment
            results = self._run_model_in_env(model_type, mc, tasks, model_dir)

            # Stop monitoring and get stats
            resource_stats = monitor.stop()

            # Aggregate results
            successful = [r for r in results if r.get("success", False)]
            failed = [r for r in results if not r.get("success", False)]

            rtf_values = [r["rtf"] for r in successful if "rtf" in r]
            ftl_values = [r["first_token_latency"] for r in successful if "first_token_latency" in r]

            all_results[name] = {
                "total": len(tasks),
                "success": len(successful),
                "failed": len(failed),
                "rtf": {
                    "mean": sum(rtf_values) / len(rtf_values) if rtf_values else None,
                    "min": min(rtf_values) if rtf_values else None,
                    "max": max(rtf_values) if rtf_values else None,
                },
                "first_token_latency": {
                    "mean": sum(ftl_values) / len(ftl_values) if ftl_values else None,
                    "min": min(ftl_values) if ftl_values else None,
                    "max": max(ftl_values) if ftl_values else None,
                },
                "resource_usage": resource_stats.to_dict(),
                "audio_files": len(list(model_dir.glob("*.wav"))),
                "details": results,
            }

            print(f"  Completed: {len(successful)}/{len(tasks)} successful")
            if rtf_values:
                print(f"  RTF: mean={all_results[name]['rtf']['mean']:.3f}, "
                      f"min={all_results[name]['rtf']['min']:.3f}, "
                      f"max={all_results[name]['rtf']['max']:.3f}")
            if ftl_values:
                print(f"  First Token Latency: mean={all_results[name]['first_token_latency']['mean']:.3f}s, "
                      f"min={all_results[name]['first_token_latency']['min']:.3f}s, "
                      f"max={all_results[name]['first_token_latency']['max']:.3f}s")

            # Print resource usage
            rs = all_results[name]["resource_usage"]
            print(f"  Resource: CPU={rs['cpu_percent']['mean']:.1f}% (max {rs['cpu_percent']['max']:.1f}%)", end="")
            if "gpu_percent" in rs:
                print(f", GPU={rs['gpu_percent']['mean']:.1f}% (max {rs['gpu_percent']['max']:.1f}%)", end="")
            if "gpu_memory_mb" in rs:
                print(f", VRAM={rs['gpu_memory_mb']['mean']:.0f}MB (max {rs['gpu_memory_mb']['max']:.0f}MB)", end="")
            print()

        # Save final metrics
        metrics_output = output_dir / "metrics.json"
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "models": model_names,
            "total_scripts": len(tasks),
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "details"}
                        for k, v in all_results.items()},
        }

        with open(metrics_output, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        # Save detailed results
        detailed_output = output_dir / "detailed_results.json"
        with open(detailed_output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Print summary
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        for name, data in all_results.items():
            print(f"\n{name}:")
            print(f"  Success: {data['success']}/{data['total']}")
            if data["rtf"]["mean"]:
                print(f"  RTF: mean={data['rtf']['mean']:.3f}, "
                      f"min={data['rtf']['min']:.3f}, max={data['rtf']['max']:.3f}")
            if data["first_token_latency"]["mean"]:
                print(f"  First Token Latency: mean={data['first_token_latency']['mean']:.3f}s, "
                      f"min={data['first_token_latency']['min']:.3f}s, "
                      f"max={data['first_token_latency']['max']:.3f}s")

        # Generate comparison report
        self._generate_report(all_results, output_dir)

        print(f"\nResults saved to: {output_dir}")
        return output_dir


def create_default_config(path: Path):
    """Create a default configuration file."""
    default_config = {
        "device": "cuda",
        "output_dir": "./outputs",
        "models": [
            {
                "name": "glm_tts",
                "type": "glm_tts",
                "model_path": "./GLM-TTS/ckpt",
            },
            {
                "name": "qwen_tts",
                "type": "qwen_tts",
                "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                "speaker": "Vivian",
                "language": "English",
            },
            {
                "name": "cosyvoice",
                "type": "cosyvoice",
                "model_dir": "pretrained_models/CosyVoice-300M",
            },
        ],
        "scripts": [
            {"id": "test_001", "text": "Hello, this is a test of the text to speech system."},
            {"id": "test_002", "text": "The quick brown fox jumps over the lazy dog."},
            {"id": "test_003", "text": "Machine learning has transformed how we interact with technology."},
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

    print(f"Created default config at: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="TTS Evaluation Pipeline (with isolated environments)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run all models from config
  python main.py -m glm_tts qwen_tts       # Run specific models
  python main.py -s test_scripts.json      # Use custom scripts
  python main.py -c my_config.yaml         # Use custom config
        """
    )
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--models", "-m", nargs="+", help="Models to evaluate (default: all)")
    parser.add_argument("--scripts", "-s", help="Scripts file path")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Creating default config...")
        create_default_config(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Run pipeline
    pipeline = TTSPipeline(config)
    pipeline.run(models=args.models, scripts_path=args.scripts)


if __name__ == "__main__":
    main()
