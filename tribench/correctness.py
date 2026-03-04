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

    # Helper to normalise and compare
    def _compare(ref: Any, tri: Any) -> tuple[bool, float, float, str]:
        if isinstance(ref, (list, tuple)) and isinstance(tri, (list, tuple)):
            if len(ref) != len(tri):
                return False, 0.0, 0.0, f"Length mismatch: ref={len(ref)}, tri={len(tri)}"
            
            all_passed = True
            max_abs = 0.0
            max_rel = 0.0
            msgs = []
            for i, (r, t) in enumerate(zip(ref, tri)):
                p, ma, mr, msg = _compare(r, t)
                all_passed = all_passed and p
                max_abs = max(max_abs, ma)
                max_rel = max(max_rel, mr)
                if not p:
                    msgs.append(f"item[{i}]: {msg}")
            return all_passed, max_abs, max_rel, "; ".join(msgs)

        if not isinstance(ref, torch.Tensor):
            return True, 0.0, 0.0, ""  # Not a tensor, skip
        if not isinstance(tri, torch.Tensor):
            return False, 0.0, 0.0, f"Expected tensor, got {type(tri)}"

        diff = (ref.float() - tri.float()).abs()
        max_abs_err = diff.max().item()

        ref_abs = ref.float().abs().clamp(min=1e-20)
        max_rel_err = (diff / ref_abs).max().item()

        passed = bool(
            torch.allclose(ref.float(), tri.float(), atol=atol, rtol=rtol)
        )
        error_msg = "" if passed else (
            f"max_abs_err={max_abs_err:.6e}, max_rel_err={max_rel_err:.6e}"
        )
        return passed, max_abs_err, max_rel_err, error_msg

    try:
        passed, max_abs_err, max_rel_err, error_msg = _compare(ref_output, tri_output)
        if not passed and not error_msg:
             error_msg = f"Comparison failed (atol={atol}, rtol={rtol})"
        elif not passed:
             error_msg += f" (atol={atol}, rtol={rtol})"
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
