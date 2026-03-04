<div align="center">
  <img src="../assets/logo.png" width="80" alt="TriBench Logo" style="vertical-align: middle; margin-bottom: 10px;" />
  <h1>TriBench</h1>
  <p><strong>A Reproducible and Extensible Benchmark Suite for Triton Kernels.</strong></p>

  <p>
    <a href="#"><img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python Version"></a>
    <a href="#"><img src="https://img.shields.io/badge/Triton-Performance-orange" alt="Triton"></a>
    <a href="#"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  </p>
</div>

---

TriBench is a lightweight framework designed to simplify the evaluation of Triton kernels. It provides a standardized environment for testing correctness, profiling latency, and comparing different kernel optimizations (e.g., fused vs. sequential) with high reproducibility.

## ✨ Features

- **Standardized Evaluation**: Unified CLI to list, test, and benchmark all registered kernels.
- **Fair Comparison**: Compile and warmup phases are strictly isolated from the measurement phase to eliminate JIT and autotuning overhead.
- **Kernel Composition**: Easily bench and compare multiple kernel variants alongside your main implementation under identical conditions.
- **Reproducibility**: Environment capture and strict random seed control ensure stable benchmark numbers across different runs.
- **Rich Metrics**: Reports median (`p50`) and tail (`p95`) latency, hardware utilization (TFLOPS), and memory bandwidth (GB/s).

## 🧠 Design Philosophy

TriBench adopts a **decoupled, data-driven architecture**. The core framework logic is separated from the kernel implementations, ensuring minimal overhead and maximum extensibility.

- **Zero Framework Modification**: Adding a new kernel requires zero changes to the core framework code. Simply drop a self-contained kernel directory into `kernels/`.
- **Data-Driven Configuration**: Each kernel defines its entrypoints, test shapes, and benchmark parameters locally via a `meta.json` file. 

```text
tribench/          # Core framework logic and CLI
kernels/           # Self-contained kernel implementations
  ├── rope/
  │   ├── meta.json         # Kernel configuration & test cases
  │   ├── reference.py      # PyTorch baseline
  │   ├── triton_impl.py    # Triton implementation
  │   └── bench_entry.py    # Custom benchmark harnesses (optional)
scripts/           # Automations and helper scripts
tests/             # Unit tests for the framework itself
```

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/your-org/TriBench.git
cd TriBench
pip install -e ".[dev]"
```

### Quick Commands

| Command | Description |
|---------|-------------|
| `tribench list` | List all available kernels in the registry. |
| `tribench validate-meta` | Verify the structure and syntax of all `meta.json` files. |
| `tribench test --kernel all` | Run correctness checks against PyTorch references. |
| `tribench run --kernel all` | Execute benchmarks and report performance metrics. |

*Example Benchmark Run:*
```bash
tribench run --kernel rope --dtype fp16 --warmup-ms 200 --rep-ms 2000
```

## 🛠 Adding a Custom Kernel

Extending TriBench is straightforward. To add a new kernel, follow these steps:

1. Create a new directory under `kernels/<your_kernel>/`.
2. Define the exact shapes and entrypoints in `meta.json`.
3. Implement the PyTorch reference inside `reference.py`.
4. Implement the Triton logic inside `triton_impl.py`.
5. Run `tribench test --kernel <your_kernel>` to verify correctness.

To evaluate against another variant (like a sequential baseline), simply declare it under the `variants` block in `meta.json`:
```json
"entrypoints": {
    "triton": "fused.py:run",
    "variants": { 
        "sequential": "baseline.py:run" 
    }
}
```

## 🙏 Acknowledgements

TriBench incorporates high-performance Triton kernels and ideas from several outstanding open-source projects:

- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
- [FlagGems](https://github.com/flagos-ai/FlagGems)
- [SageAttention](https://github.com/thu-ml/SageAttention)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [Triton Tutorials](https://github.com/triton-lang/triton)

---
<div align="center">
  <sub>Built with ❤️ by the TriBench Team</sub>
</div>