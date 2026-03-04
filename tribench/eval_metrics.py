from __future__ import annotations

import dataclasses as dc
from typing import Any

import torch


@dc.dataclass
class DomainThroughput:
    tokens_per_s: float | None = None
    elements_per_s: float | None = None
    sequences_per_s: float | None = None


def compute_tail_ratio(p99_ms: float, p50_ms: float) -> float | None:
    if p50_ms <= 0:
        return None
    return p99_ms / p50_ms


def compute_jitter_cv(std_ms: float, mean_ms: float) -> float | None:
    if mean_ms <= 0:
        return None
    return std_ms / mean_ms


def _positive_int(v: Any) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        return None
    return v


def _infer_domain_counts(case_params: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    b = _positive_int(case_params.get("B"))
    if b is None:
        b = _positive_int(case_params.get("batch_size"))

    t = _positive_int(case_params.get("T"))
    if t is None:
        t = _positive_int(case_params.get("seq_len"))

    # Token count: prefer B*T or batch_size*seq_len
    tokens = None
    if b is not None and t is not None:
        tokens = b * t
    elif t is not None:
        tokens = t

    # Sequence count
    sequences = b

    # Element count: simple, transparent heuristics over common kernel shapes.
    n = _positive_int(case_params.get("N"))
    m = _positive_int(case_params.get("M"))
    h = _positive_int(case_params.get("H"))
    s = _positive_int(case_params.get("S"))
    num_heads = _positive_int(case_params.get("num_heads"))
    head_dim = _positive_int(case_params.get("head_dim"))

    elements = None
    if b is not None and m is not None and n is not None:
        elements = b * m * n
    elif m is not None and n is not None:
        elements = m * n
    elif b is not None and t is not None and h is not None:
        elements = b * t * h
    elif b is not None and s is not None and h is not None:
        elements = b * s * h
    elif b is not None and num_heads is not None and t is not None and head_dim is not None:
        elements = b * num_heads * t * head_dim
    elif n is not None and b is not None:
        elements = b * n
    elif n is not None:
        elements = n

    return tokens, elements, sequences


def compute_domain_throughputs(case_params: dict[str, Any], latency_ms: float) -> DomainThroughput:
    if latency_ms <= 0:
        return DomainThroughput()

    sec = latency_ms * 1e-3
    tokens, elements, sequences = _infer_domain_counts(case_params)

    return DomainThroughput(
        tokens_per_s=(tokens / sec) if tokens is not None else None,
        elements_per_s=(elements / sec) if elements is not None else None,
        sequences_per_s=(sequences / sec) if sequences is not None else None,
    )


def reset_peak_memory_stats(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device)
    except Exception:
        pass


def read_peak_memory_mb(device: str) -> tuple[float | None, float | None]:
    if not device.startswith("cuda"):
        return None, None
    try:
        if not torch.cuda.is_available():
            return None, None
        alloc = float(torch.cuda.max_memory_allocated(device=device)) / (1024.0 * 1024.0)
        reserved = float(torch.cuda.max_memory_reserved(device=device)) / (1024.0 * 1024.0)
        return alloc, reserved
    except Exception:
        return None, None
