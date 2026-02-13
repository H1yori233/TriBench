import json
import pytest
import sys
from pathlib import Path
from gpubench.meta import validate_meta, load_meta, validate_entrypoints


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_META = {
    "schema_version": "1.0",
    "name": "test_kernel",
    "family": "test",
    "description": "A test kernel",
    "tags": ["test"],
    "entrypoints": {
        "make_inputs": "reference.py:make_inputs",
        "reference": "reference.py:ref",
        "triton": "triton_impl.py:run",
    },
    "supported": {
        "dtypes": ["fp16", "fp32"],
        "layouts": ["contiguous"],
        "backends": ["cuda"],
    },
    "cases": [
        {"name": "small", "N": 1024},
    ],
    "correctness": {
        "atol": 0.01,
        "rtol": 0.01,
    },
    "metrics": "elementwise",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidateMeta:
    """Test the meta.json schema validation logic."""

    def test_valid_meta_passes(self):
        errors = validate_meta(VALID_META)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_required_field(self):
        for field in ("schema_version", "name", "entrypoints", "supported", "cases"):
            broken = {k: v for k, v in VALID_META.items() if k != field}
            errors = validate_meta(broken)
            assert any(field in e for e in errors), (
                f"Expected error about missing '{field}', got: {errors}"
            )

    def test_missing_entrypoint_field(self):
        for ep_field in ("make_inputs", "reference", "triton"):
            broken = dict(VALID_META)
            broken["entrypoints"] = {
                k: v for k, v in VALID_META["entrypoints"].items() if k != ep_field
            }
            errors = validate_meta(broken)
            assert any(ep_field in e for e in errors), (
                f"Expected error about missing entrypoint '{ep_field}', got: {errors}"
            )

    def test_invalid_dtype(self):
        broken = dict(VALID_META)
        broken["supported"] = dict(VALID_META["supported"])
        broken["supported"]["dtypes"] = ["fp16", "float128"]
        errors = validate_meta(broken)
        assert any("float128" in e for e in errors)

    def test_invalid_layout(self):
        broken = dict(VALID_META)
        broken["supported"] = dict(VALID_META["supported"])
        broken["supported"]["layouts"] = ["nonexistent_layout"]
        errors = validate_meta(broken)
        assert any("nonexistent_layout" in e for e in errors)

    def test_invalid_backend(self):
        broken = dict(VALID_META)
        broken["supported"] = dict(VALID_META["supported"])
        broken["supported"]["backends"] = ["tpu"]
        errors = validate_meta(broken)
        assert any("tpu" in e for e in errors)

    def test_case_missing_name(self):
        broken = dict(VALID_META)
        broken["cases"] = [{"N": 1024}]
        errors = validate_meta(broken)
        assert any("name" in e for e in errors)

    def test_schema_version_not_string(self):
        broken = dict(VALID_META)
        broken["schema_version"] = 1.0
        errors = validate_meta(broken)
        assert any("schema_version" in e for e in errors)


class TestLoadMeta:
    """Test loading actual kernel meta.json files."""

    @pytest.fixture(params=["vector_add", "fused_softmax"])
    def kernel_name(self, request):
        return request.param

    def test_load_example_kernel(self, kernel_name):
        kernels_dir = Path(__file__).resolve().parent.parent / "kernels"
        meta_path = kernels_dir / kernel_name / "meta.json"
        assert meta_path.exists(), f"meta.json not found: {meta_path}"

        meta = load_meta(meta_path)
        assert meta.name == kernel_name
        assert meta.schema_version == "1.0"
        assert len(meta.cases) > 0
        assert len(meta.supported.dtypes) > 0

    def test_validate_example_kernel(self, kernel_name):
        kernels_dir = Path(__file__).resolve().parent.parent / "kernels"
        meta_path = kernels_dir / kernel_name / "meta.json"
        with open(meta_path) as f:
            raw = json.load(f)
        errors = validate_meta(raw)
        assert errors == [], f"Validation errors for {kernel_name}: {errors}"

    def test_entrypoints_exist(self, kernel_name):
        kernels_dir = Path(__file__).resolve().parent.parent / "kernels"
        meta_path = kernels_dir / kernel_name / "meta.json"
        meta = load_meta(meta_path)
        errors = validate_entrypoints(meta)
        assert errors == [], f"Entrypoint errors for {kernel_name}: {errors}"
