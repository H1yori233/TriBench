from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .types import (
    CaseDef,
    CorrectnessSpec,
    EntrypointsSpec,
    KernelMeta,
    SupportedSpec,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_DTYPES = {"fp16", "bf16", "fp32", "fp64"}
VALID_LAYOUTS = {"contiguous", "lastdim_contig"}
VALID_BACKENDS = {"cuda", "hip"}

REQUIRED_TOP_FIELDS = {
    "schema_version", "name", "family", "description",
    "entrypoints", "supported", "cases",
}

REQUIRED_ENTRYPOINT_FIELDS = {"make_inputs", "reference", "triton"}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_meta(path: str | Path) -> KernelMeta:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    ep_raw = raw.get("entrypoints", {})
    entrypoints = EntrypointsSpec(
        make_inputs=ep_raw.get("make_inputs"),
        reference=ep_raw.get("reference"),
        triton=ep_raw.get("triton"),
        estimate=ep_raw.get("estimate"),
        variants=ep_raw.get("variants", {}),
    )

    sup_raw = raw.get("supported", {})
    supported = SupportedSpec(
        dtypes=sup_raw.get("dtypes", ["fp32"]),
        layouts=sup_raw.get("layouts", ["contiguous"]),
        backends=sup_raw.get("backends", ["cuda"]),
    )

    cases_raw = raw.get("cases", [])
    cases: list[CaseDef] = []
    for c in cases_raw:
        name = c.pop("name")
        cases.append(CaseDef(name=name, params=dict(c)))
        c["name"] = name

    corr_raw = raw.get("correctness", {})
    correctness = CorrectnessSpec(
        atol=corr_raw.get("atol", 1e-2),
        rtol=corr_raw.get("rtol", 1e-2),
        per_dtype=corr_raw.get("per_dtype", {}),
    )

    return KernelMeta(
        schema_version=raw["schema_version"],
        name=raw["name"],
        family=raw.get("family", ""),
        description=raw.get("description", ""),
        tags=raw.get("tags", []),
        entrypoints=entrypoints,
        supported=supported,
        cases=cases,
        correctness=correctness,
        metrics=raw.get("metrics", ""),
        notes=raw.get("notes", ""),
        kernel_dir=str(path.parent),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_meta(raw: dict[str, Any]) -> list[str]:
    """Validate a raw meta.json dict. Returns a list of error strings (empty = valid)."""
    errors: list[str] = []

    # Required top-level fields
    for field in REQUIRED_TOP_FIELDS:
        if field not in raw:
            errors.append(f"Missing required field: '{field}'")

    # schema_version
    sv = raw.get("schema_version")
    if sv is not None and not isinstance(sv, str):
        errors.append(f"'schema_version' must be a string, got {type(sv).__name__}")

    # entrypoints
    ep = raw.get("entrypoints")
    if isinstance(ep, dict):
        for field in REQUIRED_ENTRYPOINT_FIELDS:
            if field not in ep:
                errors.append(f"Missing entrypoint: '{field}'")
        variants = ep.get("variants")
        if variants is not None:
            if not isinstance(variants, dict):
                errors.append("'entrypoints.variants' must be a dict")
            else:
                for k, v in variants.items():
                    if not isinstance(k, str) or not isinstance(v, str):
                        errors.append("'entrypoints.variants' must be a dict of string keys/values")
    elif ep is not None:
        errors.append("'entrypoints' must be a dict")

    # supported
    sup = raw.get("supported")
    if isinstance(sup, dict):
        for dt in sup.get("dtypes", []):
            if dt not in VALID_DTYPES:
                errors.append(f"Invalid dtype '{dt}'. Valid: {VALID_DTYPES}")
        for ly in sup.get("layouts", []):
            if ly not in VALID_LAYOUTS:
                errors.append(f"Invalid layout '{ly}'. Valid: {VALID_LAYOUTS}")
        for be in sup.get("backends", []):
            if be not in VALID_BACKENDS:
                errors.append(f"Invalid backend '{be}'. Valid: {VALID_BACKENDS}")
    elif sup is not None:
        errors.append("'supported' must be a dict")

    # cases
    cases = raw.get("cases")
    if isinstance(cases, list):
        for i, c in enumerate(cases):
            if not isinstance(c, dict):
                errors.append(f"cases[{i}] must be a dict")
            elif "name" not in c:
                errors.append(f"cases[{i}] missing 'name' field")
    elif cases is not None:
        errors.append("'cases' must be a list")

    return errors


def validate_entrypoints(meta: KernelMeta) -> list[str]:
    """Check that entrypoint files and symbols exist on disk."""
    errors: list[str] = []
    kernel_dir = Path(meta.kernel_dir)

    for ep_name in ("make_inputs", "reference", "triton", "estimate"):
        spec: str | None = getattr(meta.entrypoints, ep_name)
        if spec is None:
            continue

        if ":" not in spec:
            errors.append(f"Entrypoint '{ep_name}' must be 'file.py:symbol', got '{spec}'")
            continue

        file_part, symbol = spec.rsplit(":", 1)
        file_path = kernel_dir / file_part

        if not file_path.exists():
            errors.append(f"Entrypoint file not found: {file_path}")

    # check variants
    for var_name, spec in meta.entrypoints.variants.items():
        if ":" not in spec:
            errors.append(f"Variant '{var_name}' must be 'file.py:symbol', got '{spec}'")
            continue
        file_part, symbol = spec.rsplit(":", 1)
        file_path = kernel_dir / file_part
        if not file_path.exists():
            errors.append(f"Variant '{var_name}' file not found: {file_path}")

    return errors
