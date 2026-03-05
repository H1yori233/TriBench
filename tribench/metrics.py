from __future__ import annotations

from typing import Any, Callable


def compute_tflops(flops: float, latency_ms: float) -> float:
    """Compute TFLOPS from FLOPs count and latency in milliseconds."""
    if latency_ms <= 0:
        return 0.0
    return flops / (latency_ms * 1e-3) / 1e12


def compute_gbps(bytes_: float, latency_ms: float) -> float:
    """Compute GB/s from byte count and latency in milliseconds."""
    if latency_ms <= 0:
        return 0.0
    return bytes_ / (latency_ms * 1e-3) / 1e9


def get_estimates(
    estimate_fn: Callable | None,
    case_params: dict[str, Any],
) -> tuple[float | None, float | None]:
    """Call the kernel's estimate function and return (flops, bytes).

    Returns (None, None) if no estimate function is provided.
    """
    if estimate_fn is None:
        return None, None
    result = estimate_fn(case_params)
    flops = result.get("flops")
    bytes_ = result.get("bytes")
    return flops, bytes_
