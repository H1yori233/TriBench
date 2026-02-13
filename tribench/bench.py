from __future__ import annotations

import time
from typing import Any, Callable

import torch

from .correctness import check_correctness
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
    """Use ``triton.testing.do_bench``: returns latencies at requested quantiles."""
    import triton
    results = triton.testing.do_bench(
        fn, warmup=int(warmup_ms), rep=int(rep_ms), quantiles=quantiles,
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
    """Manual CUDA event timing with warmup, returns sorted quantile latencies."""
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

    times.sort()
    results: list[float] = []
    for q in quantiles:
        idx = min(int(q * len(times)), len(times) - 1)
        results.append(times[idx])
    return results


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
) -> BenchResult:
    """Run a two-stage (compile/warmup → benchmark) measurement for one case."""

    if quantiles is None:
        quantiles = [0.5, 0.95]
    if timer_backend not in TIMER_BACKENDS:
        raise ValueError(
            f"Unknown timer backend '{timer_backend}'. "
            f"Available: {list(TIMER_BACKENDS.keys())}"
        )

    dtype_str = case["dtype"]
    layout = case["layout"]

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
    latencies = timer_fn(lambda: triton_fn(**inputs), warmup_ms, rep_ms, quantiles)

    p50 = latencies[quantiles.index(0.5)] if 0.5 in quantiles else latencies[0]
    p95 = latencies[quantiles.index(0.95)] if 0.95 in quantiles else (latencies[1] if len(latencies) > 1 else latencies[0])

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
        latency_ms_p95=p95,
        tflops=tflops,
        gbps=gbps,
        timer_backend=timer_backend,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
        correctness=corr_result,
        compile_time_ms=compile_time_ms,
    )
