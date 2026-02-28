"""Result output: JSON and Markdown summary writers."""

from __future__ import annotations

import dataclasses as dc
import datetime
import json
import os
from pathlib import Path
from typing import Any

from .types import BenchResult, EnvInfo, RunRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dc_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances to dicts."""
    if dc.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _dc_to_dict(v) for k, v in dc.asdict(obj).items()}
    if isinstance(obj, list):
        return [_dc_to_dict(v) for v in obj]
    return obj


def make_run_id(kernel_filter: str, device: str) -> str:
    """Generate a filesystem-safe run ID."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    dev = device.replace(":", "")
    return f"{ts}__{kernel_filter}__{dev}"


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------

def write_json(record: RunRecord, output_dir: str | Path) -> Path:
    """Write run record as ``run.json`` under ``output_dir/<run_id>/``."""
    run_dir = Path(output_dir) / record.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_dc_to_dict(record), f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Markdown summary writer
# ---------------------------------------------------------------------------

def write_summary_md(record: RunRecord, output_dir: str | Path) -> Path:
    """Write a human-readable Markdown summary of the run."""
    run_dir = Path(output_dir) / record.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "summary.md"

    lines: list[str] = []
    lines.append(f"# Benchmark Run: `{record.run_id}`\n")

    # Environment
    e = record.env
    lines.append("## Environment\n")
    lines.append(f"| Key | Value |")
    lines.append(f"|-----|-------|")
    lines.append(f"| Timestamp | {e.timestamp_utc} |")
    lines.append(f"| GPU | {e.gpu_name} |")
    lines.append(f"| CUDA | {e.cuda_version} |")
    lines.append(f"| PyTorch | {e.torch_version} |")
    lines.append(f"| Triton | {e.triton_version} |")
    lines.append(f"| Git Commit | `{e.git_commit[:12]}` {'(dirty)' if e.git_dirty else ''} |")
    lines.append(f"| Seed | {record.seed} |")
    lines.append(f"| Timer | {record.timer_backend} |")
    lines.append(f"| Warmup / Rep | {record.warmup_ms}ms / {record.rep_ms}ms |")
    lines.append("")

    # Results table
    lines.append("## Results\n")
    lines.append("| Kernel | Case | Mode | dtype | p50 (ms) | p95 (ms) | TFLOPS | GB/s | Correct |")
    lines.append("|--------|------|------|-------|----------|----------|--------|------|---------|")
    for r in record.results:
        tflops = f"{r.tflops:.2f}" if r.tflops is not None else "-"
        gbps = f"{r.gbps:.1f}" if r.gbps is not None else "-"
        correct = "-"
        if r.correctness is not None:
            correct = "passed" if r.correctness.passed else "failed"
        lines.append(
            f"| {r.kernel} | {r.case_name} | {r.pass_type} | {r.dtype} | "
            f"{r.latency_ms_p50:.4f} | {r.latency_ms_p95:.4f} | "
            f"{tflops} | {gbps} | {correct} |"
        )
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
