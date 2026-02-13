<div align="center">
  <img src="assets/logo.png" width="60" alt="TriBench Logo" style="vertical-align: middle; margin-right: 15px;" />
  <h1 style="display: inline-block; vertical-align: middle; margin: 0;">TriBench</h1>
  <p><strong>Reproducible and extensible benchmark suite for Triton kernels.</strong></p>
</div>


## Overview

```text
gpubench/          - core framework
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
gpubench list
```

### Validate metadata

```bash
gpubench validate-meta
```

### Run correctness tests

```bash
gpubench test --kernel all
```

### Run benchmarks

```bash
gpubench run --kernel all --dtype fp16 --warmup-ms 200 --rep-ms 2000
```

## Methodology

- **Metrics**: Reports `p50` (median) and `p95` latency, TFLOPS, and GB/s.
- **Timers**: Supports `triton.testing.do_bench`, `triton_cudagraph`, and `cuda_event`.
- **Reproducibility**: Explicit seed control and environment capture.
- **Isolation**: Compile/warmup phase is separated from measurement to avoid JIT/autotune overhead.

## Adding Kernels

1. Create a directory in `kernels/`.
2. Add `meta.json` to define entrypoints and test cases.
3. Implement reference (PyTorch) and Triton kernels.
4. Verify with `gpubench test`.

No framework modifications are required to add new kernels.