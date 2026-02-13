from __future__ import annotations

from typing import Any, Callable

import torch

from .types import CorrectnessResult, CorrectnessSpec, resolve_dtype


# ---------------------------------------------------------------------------
# Default tolerances per dtype
# ---------------------------------------------------------------------------

DEFAULT_TOLERANCES: dict[str, dict[str, float]] = {
    "fp32": {"atol": 1e-5, "rtol": 1e-4},
    "fp64": {"atol": 1e-7, "rtol": 1e-5},
    "fp16": {"atol": 1e-2, "rtol": 1e-2},
    "bf16": {"atol": 1e-2, "rtol": 1e-2},
}


def _get_tolerance(
    dtype_str: str,
    correctness_spec: CorrectnessSpec,
) -> tuple[float, float]:
    if dtype_str in correctness_spec.per_dtype:
        d = correctness_spec.per_dtype[dtype_str]
        return d.get("atol", correctness_spec.atol), d.get("rtol", correctness_spec.rtol)

    if correctness_spec.atol != 1e-2 or correctness_spec.rtol != 1e-2:
        return correctness_spec.atol, correctness_spec.rtol

    # framework default
    if dtype_str in DEFAULT_TOLERANCES:
        d = DEFAULT_TOLERANCES[dtype_str]
        return d["atol"], d["rtol"]

    return correctness_spec.atol, correctness_spec.rtol


def check_correctness(
    ref_output: Any,
    tri_output: Any,
    dtype_str: str,
    correctness_spec: CorrectnessSpec,
) -> CorrectnessResult:
    """Compare reference and Triton outputs, returning a CorrectnessResult."""
    atol, rtol = _get_tolerance(dtype_str, correctness_spec)

    # Normalise to tensors for comparison
    if not isinstance(ref_output, torch.Tensor):
        ref_output = ref_output[0] if isinstance(ref_output, (tuple, list)) else ref_output
    if not isinstance(tri_output, torch.Tensor):
        tri_output = tri_output[0] if isinstance(tri_output, (tuple, list)) else tri_output

    try:
        diff = (ref_output.float() - tri_output.float()).abs()
        max_abs_err = diff.max().item()

        ref_abs = ref_output.float().abs().clamp(min=1e-20)
        max_rel_err = (diff / ref_abs).max().item()

        passed = bool(
            torch.allclose(ref_output.float(), tri_output.float(), atol=atol, rtol=rtol)
        )
        error_msg = "" if passed else (
            f"max_abs_err={max_abs_err:.6e}, max_rel_err={max_rel_err:.6e}, "
            f"atol={atol}, rtol={rtol}"
        )
    except Exception as exc:
        return CorrectnessResult(
            passed=False, atol=atol, rtol=rtol, error_msg=str(exc)
        )

    return CorrectnessResult(
        passed=passed,
        atol=atol,
        rtol=rtol,
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        error_msg=error_msg,
    )
