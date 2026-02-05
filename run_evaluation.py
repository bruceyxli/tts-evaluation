#!/usr/bin/env python3
"""
TTS Evaluation Pipeline - Orchestrates running multiple models and generates report.

Usage:
    python run_evaluation.py -c config.yaml -s scripts/tts_evaluation_dataset.json

Each model runs in its own conda environment via subprocess.
Results are saved to individual JSON files and a final report is generated.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


# Model type to conda environment mapping
MODEL_ENV_MAP = {
    "glm_tts": "tts-glm",
    "glm_tts_rl": "tts-glm",
    "qwen_tts": "tts-qwen",
    "qwen_tts_vc": "tts-qwen",
    "qwen_tts_vllm": "tts-qwen-vllm",
    "qwen_tts_vllm_vc": "tts-qwen-vllm",
    "cosyvoice": "tts-cosyvoice",
    "cosyvoice_rl": "tts-cosyvoice",
}


def get_conda_executable():
    """Find conda executable."""
    import shutil

    # Try common locations
    for name in ["conda", "mamba"]:
        path = shutil.which(name)
        if path:
            return path

    # Try WSL paths
    wsl_paths = [
        "/home/adminlinux/miniconda3/bin/conda",
        "/opt/conda/bin/conda",
    ]
    for path in wsl_paths:
        if Path(path).exists():
            return path

    return "conda"


def run_model(
    model_name: str,
    model_type: str,
    config_path: Path,
    scripts_path: Path,
    output_dir: Path,
    conda_exec: str,
) -> dict:
    """Run a single model evaluation in its conda environment."""

    env_name = MODEL_ENV_MAP.get(model_type)
    if not env_name:
        print(f"  ERROR: Unknown model type: {model_type}")
        return {"error": f"Unknown model type: {model_type}"}

    model_output = output_dir / model_name
    model_output.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        conda_exec, "run", "-n", env_name,
        "python", "-u", "run_single_model.py",
        "--model", model_type,
        "--config", str(config_path),
        "--scripts", str(scripts_path),
        "--output", str(model_output),
        "--name", model_name,
    ]

    print(f"\n{'='*60}")
    print(f"Running: {model_name} (type: {model_type}, env: {env_name})")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    try:
        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            cwd=str(Path.cwd()),
        )

        if result.returncode != 0:
            print(f"  ERROR: Process exited with code {result.returncode}")
            return {"error": f"Process exited with code {result.returncode}"}

        # Load results
        results_path = model_output / "results.json"
        if results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print(f"  ERROR: Results file not found: {results_path}")
            return {"error": "Results file not found"}

    except Exception as e:
        print(f"  ERROR: {e}")
        return {"error": str(e)}


def generate_report(all_results: dict, output_dir: Path) -> Path:
    """Generate comparison report from all model results."""

    report_lines = []
    report_lines.append("# TTS Evaluation Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\nOutput Directory: `{output_dir}`\n")

    # Summary table - collect data first for alignment
    report_lines.append("## Summary\n")

    # Build table data
    headers = ["Model", "Samples", "Avg RTF", "Avg First Token (s)", "Avg GPU Util (%)", "Avg GPU Mem (GB)", "Load Time (s)"]
    rows = []

    for model_name, result in all_results.items():
        if "error" in result:
            rows.append([model_name, "ERROR", "-", "-", "-", "-", "-"])
            continue

        summary = result.get("summary", {})
        samples = f"{summary.get('successful_samples', 0)}/{summary.get('total_samples', 0)}"
        rtf = f"{summary.get('avg_rtf', 0):.3f}" if summary.get('avg_rtf') else "-"
        ftl = f"{summary.get('avg_first_token_latency', 0):.2f}" if summary.get('avg_first_token_latency') else "-"
        gpu_util = f"{summary.get('avg_gpu_util', 0):.1f}" if summary.get('avg_gpu_util') else "-"
        gpu_mem = f"{summary.get('avg_gpu_mem_gb', 0):.1f}" if summary.get('avg_gpu_mem_gb') else "-"
        load_time = f"{result.get('load_time', 0):.1f}" if result.get('load_time') else "-"

        rows.append([model_name, samples, rtf, ftl, gpu_util, gpu_mem, load_time])

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

    report_lines.append("")

    # Detailed results per model
    report_lines.append("## Detailed Results\n")

    for model_name, result in all_results.items():
        report_lines.append(f"### {model_name}\n")

        if "error" in result:
            report_lines.append(f"**Error:** {result['error']}\n")
            continue

        summary = result.get("summary", {})
        report_lines.append(f"- **Type:** {result.get('model_type', 'unknown')}")
        report_lines.append(f"- **Load Time:** {result.get('load_time', 0):.2f}s")
        report_lines.append(f"- **Sample Rate:** {result.get('sample_rate', 0)} Hz")
        report_lines.append(f"- **Total Samples:** {summary.get('total_samples', 0)}")
        report_lines.append(f"- **Successful:** {summary.get('successful_samples', 0)}")
        report_lines.append(f"- **Failed:** {summary.get('failed_samples', 0)}")
        report_lines.append(f"- **Total Generation Time:** {summary.get('total_generation_time', 0):.2f}s")
        report_lines.append(f"- **Total Audio Duration:** {summary.get('total_audio_duration', 0):.2f}s")

        if summary.get('avg_rtf'):
            report_lines.append(f"- **Average RTF:** {summary['avg_rtf']:.3f}")

        if summary.get('avg_first_token_latency'):
            report_lines.append(f"- **First Token Latency:**")
            report_lines.append(f"  - Average: {summary['avg_first_token_latency']:.3f}s")
            report_lines.append(f"  - Min: {summary.get('min_first_token_latency', 0):.3f}s")
            report_lines.append(f"  - Max: {summary.get('max_first_token_latency', 0):.3f}s")

        if summary.get('avg_gpu_util'):
            report_lines.append(f"- **GPU Utilization:** {summary['avg_gpu_util']:.1f}%")

        if summary.get('avg_gpu_mem_gb'):
            report_lines.append(f"- **GPU Memory:** {summary['avg_gpu_mem_gb']:.2f} GB")

        report_lines.append("")

    # RTF Comparison
    report_lines.append("## RTF Comparison (Lower is Better)\n")
    report_lines.append("```")

    # Sort by RTF
    rtf_data = []
    for model_name, result in all_results.items():
        if "error" not in result and result.get("summary", {}).get("avg_rtf"):
            rtf_data.append((model_name, result["summary"]["avg_rtf"]))

    rtf_data.sort(key=lambda x: x[1])

    max_name_len = max(len(name) for name, _ in rtf_data) if rtf_data else 10
    for name, rtf in rtf_data:
        bar_len = int(rtf * 20)
        bar = "█" * bar_len
        report_lines.append(f"{name:<{max_name_len}} | {rtf:.3f} | {bar}")

    report_lines.append("```\n")

    # First Token Latency Comparison
    report_lines.append("## First Token Latency Comparison (Lower is Better)\n")
    report_lines.append("```")

    ftl_data = []
    for model_name, result in all_results.items():
        if "error" not in result and result.get("summary", {}).get("avg_first_token_latency"):
            ftl_data.append((model_name, result["summary"]["avg_first_token_latency"]))

    ftl_data.sort(key=lambda x: x[1])

    for name, ftl in ftl_data:
        bar_len = int(ftl * 10)
        bar = "█" * bar_len
        report_lines.append(f"{name:<{max_name_len}} | {ftl:.3f}s | {bar}")

    report_lines.append("```\n")

    # Write report
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return report_path


def main():
    parser = argparse.ArgumentParser(description="TTS Evaluation Pipeline")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config YAML file")
    parser.add_argument("--scripts", "-s", help="Scripts JSON file (overrides config)")
    parser.add_argument("--models", "-m", nargs="+", help="Specific models to run (default: all enabled)")
    parser.add_argument("--output", "-o", help="Output directory (default: auto-generated)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Determine scripts file
    scripts_path = Path(args.scripts) if args.scripts else Path(config.get("scripts_file", "scripts/tts_evaluation_dataset.json"))

    if not scripts_path.exists():
        print(f"ERROR: Scripts file not found: {scripts_path}")
        sys.exit(1)

    # Load scripts to get count
    with open(scripts_path, "r", encoding="utf-8") as f:
        scripts = json.load(f)
    print(f"Loaded {len(scripts)} test scripts from {scripts_path}")

    # Get models to evaluate
    model_configs = config.get("models", [])

    # Filter by enabled and command line args
    if args.models:
        model_configs = [m for m in model_configs if m.get("name") in args.models]
    else:
        model_configs = [m for m in model_configs if m.get("enabled", True)]

    if not model_configs:
        print("ERROR: No models to evaluate")
        sys.exit(1)

    model_names = [m.get("name") for m in model_configs]
    print(f"Models to evaluate: {', '.join(model_names)}")

    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_str = "-".join(sorted(model_names))[:50]  # Truncate if too long
        output_dir = Path(config.get("output_dir", "./outputs")) / f"{timestamp}_{models_str}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config and scripts to output
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    with open(output_dir / "scripts.json", "w", encoding="utf-8") as f:
        json.dump(scripts, f, indent=2, ensure_ascii=False)

    # Find conda
    conda_exec = get_conda_executable()
    print(f"Using conda: {conda_exec}")

    # Run each model
    all_results = {}

    for model_config in model_configs:
        model_name = model_config.get("name")
        model_type = model_config.get("type", model_name)

        result = run_model(
            model_name=model_name,
            model_type=model_type,
            config_path=config_path,
            scripts_path=scripts_path,
            output_dir=output_dir,
            conda_exec=conda_exec,
        )

        all_results[model_name] = result

    # Save combined results
    combined_path = output_dir / "all_results.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nCombined results saved to: {combined_path}")

    # Generate report
    report_path = generate_report(all_results, output_dir)
    print(f"Report generated: {report_path}")

    # Print final summary
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Report: {report_path}")

    # Quick summary
    print("\nQuick Summary:")
    for model_name, result in all_results.items():
        if "error" in result:
            print(f"  {model_name}: ERROR - {result['error']}")
        else:
            summary = result.get("summary", {})
            rtf = summary.get("avg_rtf")
            ftl = summary.get("avg_first_token_latency")
            if rtf is not None:
                print(f"  {model_name}: RTF={rtf:.3f}" + (f", FTL={ftl:.3f}s" if ftl else ""))
            else:
                failed = summary.get("failed_samples", 0)
                print(f"  {model_name}: {failed} samples failed")


if __name__ == "__main__":
    main()
