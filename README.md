<div align="center">
  <img src="assets/logo.png" width="60" alt="TriBench Logo" style="vertical-align: middle; margin-right: 15px;" />
  <h1 style="display: inline-block; vertical-align: middle; margin: 0;">TriBench</h1>
  <p><strong>Reproducible and extensible benchmark suite for Triton kernels.</strong></p>
</div>


## Overview

```text
tribench/          - core framework
kernels/           - kernel implementations
scripts/           - helper scripts
results/           - measurement results
tests/             - framework tests
```

Each kernel in `kernels/` is a self-contained directory with `meta.json`, `reference.py`, `triton_impl.py`, `test_correctness.py`, and `bench_entry.py`.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### List kernels

```bash
tribench list
```

### Validate metadata

```bash
tribench validate-meta
```

### Run correctness tests

```bash
tribench test --kernel all
```

### Run benchmarks

```bash
tribench run --kernel all --dtype fp16 --warmup-ms 200 --rep-ms 2000
```

## Methodology

- **Metrics**: Reports `p50` (median) and `p95` latency, TFLOPS, and GB/s.
- **Timers**: Supports `triton.testing.do_bench`, `triton_cudagraph`, and `cuda_event`.
- **Reproducibility**: Explicit seed control and environment capture.
- **Isolation**: Compile/warmup phase is separated from measurement to avoid JIT/autotune overhead.

## Kernel Composition

Compare fused vs. sequential implementations using `variants` in `meta.json`:

```json
"entrypoints": {
    "triton": "fused.py:run",
    "variants": { "sequential": "baseline.py:run" }
}
```

Both targets are automatically verified and benchmarked under identical conditions.


## Adding Kernels

1. Create a directory in `kernels/`.
2. Add `meta.json` to define entrypoints and test cases.
3. Implement reference (PyTorch) and Triton kernels.
4. Verify with `tribench test`.

No framework modifications are required to add new kernels.


## Open Source Credits

TriBench incorporates high-performance Triton kernels from the following open-source projects:

- **[Liger-Kernel](https://github.com/linkedin/Liger-Kernel)**.
- **[FlagGems](https://github.com/flagos-ai/FlagGems)**.
- **[SageAttention](https://github.com/thu-ml/SageAttention)**.
- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)**.
- **[Triton Tutorials](https://github.com/triton-lang/triton)**.