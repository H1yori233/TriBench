from __future__ import annotations

import time
from typing import Any, Callable

import torch

from .correctness import check_correctness
from .eval_metrics import (
    compute_domain_throughputs,
    compute_jitter_cv,
    compute_tail_ratio,
    read_peak_memory_mb,
    reset_peak_memory_stats,
)
from .gen import materialize_inputs
from .metrics import compute_gbps, compute_tflops, get_estimates
from .types import BenchResult, CorrectnessResult, KernelMeta


# ---------------------------------------------------------------------------
# Timer backends
# ---------------------------------------------------------------------------

def _do_bench_triton(
    fn: Callable,
    warmup_ms: float,
    rep_ms: float,
    quantiles: list[float],
) -> list[float]:
    """Use ``triton.testing.do_bench``: returns all latencies."""
    import triton
    # We use return_mode="all" to get all raw measurements
    results = triton.testing.do_bench(
        fn, warmup=int(warmup_ms), rep=int(rep_ms), return_mode="all",
    )
    if not isinstance(results, (list, tuple)):
        results = [results]
    return [float(r) for r in results]


def _do_bench_cudagraph(
    fn: Callable,
    warmup_ms: float,
    rep_ms: float,
    quantiles: list[float],
) -> list[float]:
    """Use ``triton.testing.do_bench_cudagraph``."""
    import triton
    if not hasattr(triton.testing, "do_bench_cudagraph"):
        raise RuntimeError("triton.testing.do_bench_cudagraph not available in this Triton version")
    # Note: do_bench_cudagraph might not support return_mode="all" in all versions.
    # If it doesn't, we fallback to quantiles but we try to get a broad set.
    try:
        results = triton.testing.do_bench_cudagraph(
            fn, warmup=int(warmup_ms), rep=int(rep_ms), return_mode="all",
        )
    except TypeError:
        # Fallback if return_mode is not supported
        results = triton.testing.do_bench_cudagraph(
            fn, warmup=int(warmup_ms), rep=int(rep_ms), quantiles=quantiles,
        )
    
    if not isinstance(results, (list, tuple)):
        results = [results]
    return [float(r) for r in results]


def _do_bench_cuda_event(
    fn: Callable,
    warmup_ms: float,
    rep_ms: float,
    quantiles: list[float],
) -> list[float]:
    """Manual CUDA event timing with warmup, returns all latencies."""
    # Warmup
    warmup_iters = max(1, int(warmup_ms / 5))
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    # Measure
    rep_iters = max(1, int(rep_ms / 5))
    times: list[float] = []
    for _ in range(rep_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return times


TIMER_BACKENDS: dict[str, Callable] = {
    "triton_do_bench": _do_bench_triton,
    "triton_cudagraph": _do_bench_cudagraph,
    "cuda_event": _do_bench_cuda_event,
}


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    *,
    kernel_name: str,
    meta: KernelMeta,
    make_inputs_fn: Callable,
    reference_fn: Callable,
    triton_fn: Callable,
    estimate_fn: Callable | None,
    case: dict[str, Any],
    device: str,
    seed: int,
    warmup_ms: float = 200.0,
    rep_ms: float = 2000.0,
    quantiles: list[float] | None = None,
    timer_backend: str = "triton_do_bench",
    run_correctness: bool = True,
    pass_type: str = "forward",
) -> BenchResult:
    """Run a two-stage (compile/warmup → benchmark) measurement for one case."""

    if quantiles is None:
        quantiles = [0.5, 0.9, 0.95, 0.99]
    if timer_backend not in TIMER_BACKENDS:
        raise ValueError(
            f"Unknown timer backend '{timer_backend}'. "
            f"Available: {list(TIMER_BACKENDS.keys())}"
        )

    dtype_str = case["dtype"]
    layout = case["layout"]

    reset_peak_memory_stats(device)

    # ---- Stage 0: materialise inputs ----
    inputs = materialize_inputs(make_inputs_fn, case, device, seed)

    # ---- Stage 1: compile / warmup (timed but not counted in benchmark) ----
    t0 = time.perf_counter()
    _ = triton_fn(**inputs)
    torch.cuda.synchronize()
    compile_time_ms = (time.perf_counter() - t0) * 1000.0

    corr_result: CorrectnessResult | None = None
    if run_correctness:
        ref_out = reference_fn(**inputs)
        tri_out = triton_fn(**inputs)
        torch.cuda.synchronize()
        corr_result = check_correctness(ref_out, tri_out, dtype_str, meta.correctness)

    timer_fn = TIMER_BACKENDS[timer_backend]
    all_latencies = timer_fn(lambda: triton_fn(**inputs), warmup_ms, rep_ms, quantiles)
    
    import numpy as np
    lats = np.array(all_latencies)
    
    p50 = float(np.percentile(lats, 50))
    p90 = float(np.percentile(lats, 90))
    p95 = float(np.percentile(lats, 95))
    p99 = float(np.percentile(lats, 99))
    
    l_min = float(np.min(lats))
    l_max = float(np.max(lats))
    l_mean = float(np.mean(lats))
    l_std = float(np.std(lats))
    tail_ratio = compute_tail_ratio(p99, p50)
    jitter_cv = compute_jitter_cv(l_std, l_mean)
    domain_tp = compute_domain_throughputs(case["params"], p50)
    peak_mem_alloc_mb, peak_mem_reserved_mb = read_peak_memory_mb(device)

    # ---- Metrics ----
    flops, bytes_ = get_estimates(estimate_fn, case["params"])
    tflops = compute_tflops(flops, p50) if flops else None
    gbps = compute_gbps(bytes_, p50) if bytes_ else None

    return BenchResult(
        kernel=kernel_name,
        case_name=case["case_name"],
        dtype=dtype_str,
        layout=layout,
        latency_ms_p50=p50,
        latency_ms_p90=p90,
        latency_ms_p95=p95,
        latency_ms_p99=p99,
        latency_ms_min=l_min,
        latency_ms_max=l_max,
        latency_ms_mean=l_mean,
        latency_ms_std=l_std,
        pass_type=pass_type,
        tflops=tflops,
        gbps=gbps,
        tail_ratio_p99_p50=tail_ratio,
        jitter_cv=jitter_cv,
        tokens_per_s=domain_tp.tokens_per_s,
        elements_per_s=domain_tp.elements_per_s,
        sequences_per_s=domain_tp.sequences_per_s,
        peak_mem_alloc_mb=peak_mem_alloc_mb,
        peak_mem_reserved_mb=peak_mem_reserved_mb,
        timer_backend=timer_backend,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
        correctness=corr_result,
        compile_time_ms=compile_time_ms,
    )
