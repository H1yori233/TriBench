from __future__ import annotations

import dataclasses as dc
from typing import Any, Callable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Meta / Case
# ---------------------------------------------------------------------------

@dc.dataclass
class CaseDef:
    name: str
    params: dict[str, Any]


@dc.dataclass
class SupportedSpec:
    dtypes: list[str]
    layouts: list[str] = dc.field(default_factory=lambda: ["contiguous"])
    backends: list[str] = dc.field(default_factory=lambda: ["cuda"])


@dc.dataclass
class EntrypointsSpec:
    make_inputs: str
    reference: str
    triton: str
    backward: str | None = None
    estimate: str | None = None
    variants: dict[str, str] = dc.field(default_factory=dict)


@dc.dataclass
class CorrectnessSpec:
    """Default correctness thresholds."""
    atol: float = 1e-2
    rtol: float = 1e-2
    per_dtype: dict[str, dict[str, float]] = dc.field(default_factory=dict)


@dc.dataclass
class KernelMeta:
    """Parsed representation of a kernel's meta.json."""
    schema_version: str
    name: str
    family: str
    description: str
    tags: list[str]
    entrypoints: EntrypointsSpec
    supported: SupportedSpec
    cases: list[CaseDef]
    correctness: CorrectnessSpec
    metrics: str           # e.g. "elementwise" or "matmul"
    notes: str = ""
    # runtime - set by registry
    kernel_dir: str = ""


# ---------------------------------------------------------------------------
# Kernel entrypoint protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class KernelEntrypoints(Protocol):
    """Protocol that a loaded kernel module must satisfy."""
    def make_inputs(self, case: dict, device: str, seed: int, dtype: Any) -> dict: ...
    def reference(self, **inputs: Any) -> Any: ...
    def triton(self, **inputs: Any) -> Any: ...
    def backward(self, **inputs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dc.dataclass
class CorrectnessResult:
    """Result of a single correctness check."""
    passed: bool
    atol: float
    rtol: float
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    error_msg: str = ""


@dc.dataclass
class BenchResult:
    """Result of a single benchmark measurement."""
    kernel: str
    case_name: str
    dtype: str
    layout: str
    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p95: float
    latency_ms_p99: float
    latency_ms_min: float
    latency_ms_max: float
    latency_ms_mean: float
    latency_ms_std: float
    pass_type: str = "forward"
    variant: str | None = None
    tflops: float | None = None
    gbps: float | None = None
    timer_backend: str = "triton_do_bench"
    warmup_ms: float = 200.0
    rep_ms: float = 2000.0
    correctness: CorrectnessResult | None = None
    compile_time_ms: float | None = None


@dc.dataclass
class EnvInfo:
    """Captured environment information."""
    torch_version: str = ""
    triton_version: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_driver: str = ""
    git_commit: str = ""
    git_dirty: bool = False
    cmdline: str = ""
    timestamp_utc: str = ""


@dc.dataclass
class RunRecord:
    """A complete benchmark run."""
    schema_version: str = "1.0"
    run_id: str = ""
    env: EnvInfo = dc.field(default_factory=EnvInfo)
    seed: int = 0
    timer_backend: str = "triton_do_bench"
    warmup_ms: float = 200.0
    rep_ms: float = 2000.0
    quantiles: list[float] = dc.field(default_factory=lambda: [0.5, 0.9, 0.95, 0.99])
    results: list[BenchResult] = dc.field(default_factory=list)


# ---------------------------------------------------------------------------
# Dtype mapping helpers
# ---------------------------------------------------------------------------

import torch  # noqa: E402

DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp64": torch.float64,
}

DTYPE_STR_MAP: dict[torch.dtype, str] = {v: k for k, v in DTYPE_MAP.items()}


def resolve_dtype(s: str) -> torch.dtype:
    """Convert a string dtype name to a torch.dtype."""
    if s not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{s}'. Supported: {list(DTYPE_MAP.keys())}")
    return DTYPE_MAP[s]
