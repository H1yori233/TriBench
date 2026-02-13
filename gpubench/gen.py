"""Case enumeration and input generation."""

from __future__ import annotations

from typing import Any, Callable

import torch

from .types import CaseDef, KernelMeta, resolve_dtype


def generate_cases(
    meta: KernelMeta,
    *,
    case_filter: list[str] | None = None,
    dtype_filter: list[str] | None = None,
    layout_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate cases from meta."""
    cases = meta.cases
    if case_filter:
        cases = [c for c in cases if c.name in case_filter]

    dtypes = meta.supported.dtypes
    if dtype_filter:
        for dt in dtype_filter:
            if dt not in meta.supported.dtypes:
                raise ValueError(
                    f"Kernel '{meta.name}' does not support dtype '{dt}'. "
                    f"Supported: {meta.supported.dtypes}"
                )
        dtypes = dtype_filter

    layouts = meta.supported.layouts
    if layout_filter:
        for ly in layout_filter:
            if ly not in meta.supported.layouts:
                raise ValueError(
                    f"Kernel '{meta.name}' does not support layout '{ly}'. "
                    f"Supported: {meta.supported.layouts}"
                )
        layouts = layout_filter

    grid: list[dict[str, Any]] = []
    for case in cases:
        for dtype in dtypes:
            for layout in layouts:
                grid.append({
                    "case_name": case.name,
                    "params": dict(case.params),
                    "dtype": dtype,
                    "layout": layout,
                })
    return grid


def materialize_inputs(
    make_inputs_fn: Callable,
    case: dict[str, Any],
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Call the kernel's ``make_inputs`` with deterministic seeding."""
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    dtype = resolve_dtype(case["dtype"])
    return make_inputs_fn(case["params"], device, seed, dtype)
