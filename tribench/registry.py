from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .meta import load_meta, validate_meta, validate_entrypoints
from .types import KernelMeta

# ---------------------------------------------------------------------------
# Default kernels directory (relative to repo root)
# ---------------------------------------------------------------------------

_DEFAULT_KERNELS_DIR = Path(__file__).resolve().parent.parent / "kernels"


class KernelRegistry:
    """Discover and lazily load kernel packs from ``kernels/*/meta.json``."""

    def __init__(self, kernels_dir: str | Path | None = None) -> None:
        self._kernels_dir = Path(kernels_dir) if kernels_dir else _DEFAULT_KERNELS_DIR
        self._metas: dict[str, KernelMeta] = {}
        self._modules: dict[str, dict[str, Any]] = {}  # name -> {symbol: callable}
        self._scan()

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan(self) -> None:
        if not self._kernels_dir.is_dir():
            return
        for meta_path in sorted(self._kernels_dir.glob("*/meta.json")):
            try:
                meta = load_meta(meta_path)
                self._metas[meta.name] = meta
            except Exception as exc:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_kernels(self) -> list[KernelMeta]:
        """Return all discovered kernel metadata (no code import)."""
        return list(self._metas.values())

    def get_meta(self, name: str) -> KernelMeta:
        """Get metadata for a specific kernel by name."""
        if name not in self._metas:
            available = ", ".join(sorted(self._metas.keys())) or "(none)"
            raise KeyError(f"Kernel '{name}' not found. Available: {available}")
        return self._metas[name]

    def kernel_names(self) -> list[str]:
        """Return sorted list of registered kernel names."""
        return sorted(self._metas.keys())

    def load_symbol(self, name: str, ep_spec: str) -> Callable:
        """Lazily import a symbol from a kernel entrypoint spec ``file.py:symbol``."""
        meta = self.get_meta(name)
        cache_key = f"{name}::{ep_spec}"

        if cache_key in self._modules:
            return self._modules[cache_key]

        file_part, symbol = ep_spec.rsplit(":", 1)
        file_path = Path(meta.kernel_dir) / file_part
        module = self._import_file(file_path, f"tribench.kernels.{name}.{file_path.stem}")
        func = getattr(module, symbol)
        self._modules[cache_key] = func
        return func

    def load_make_inputs(self, name: str) -> Callable:
        return self.load_symbol(name, self.get_meta(name).entrypoints.make_inputs)

    def load_reference(self, name: str) -> Callable:
        return self.load_symbol(name, self.get_meta(name).entrypoints.reference)

    def load_triton(self, name: str) -> Callable:
        return self.load_symbol(name, self.get_meta(name).entrypoints.triton)

    def load_variant(self, name: str, variant_name: str) -> Callable:
        spec = self.get_meta(name).entrypoints.variants.get(variant_name)
        if spec is None:
            raise KeyError(f"Variant '{variant_name}' not found in kernel '{name}'")
        return self.load_symbol(name, spec)

    def load_estimate(self, name: str) -> Callable | None:
        spec = self.get_meta(name).entrypoints.estimate
        if spec is None:
            return None
        return self.load_symbol(name, spec)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_all(self) -> dict[str, list[str]]:
        """Validate all discovered kernels. Returns {kernel_name: [errors]}."""
        report: dict[str, list[str]] = {}
        for name, meta in self._metas.items():
            raw_path = Path(meta.kernel_dir) / "meta.json"
            with open(raw_path, "r") as f:
                raw = json.load(f)
            errors = validate_meta(raw)
            errors.extend(validate_entrypoints(meta))
            if errors:
                report[name] = errors
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_file(file_path: Path, module_name: str) -> ModuleType:
        """Import a Python file as a module without polluting sys.modules permanently."""
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
