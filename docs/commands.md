# All Commands

TriBench provides a set of CLI commands through the `tribench` tool.

## `tribench new`

Scaffold a new benchmark kernel directory.

```bash
tribench new <kernel_name>
```

**What it creates:**
- `meta.json`: Kernel configuration & test cases.
- `reference.py`: PyTorch baseline template.
- `triton_impl.py`: Triton implementation template.

## `tribench list`

List all available kernels in the registry and their supported data types/cases.

```bash
tribench list
```

## `tribench validate-meta`

Verify the structure and syntax of all `meta.json` files to ensure they conform to the TriBench schema.

```bash
tribench validate-meta
```

## `tribench test`

Run correctness checks against PyTorch references.

```bash
tribench test --kernel all
```

**Options:**
- `--kernel <name>`: Specific kernel to test.
- `--dtype <type>`: Limit to specific data type.

## `tribench run`

Execute benchmarks and report performance metrics (Latency, TFLOPS, Memory Bandwidth).

```bash
tribench run --kernel all
```

**Options:**
- `--warmup-ms <ms>`: Time spent on warmup (default: 200).
- `--rep-ms <ms>`: Time spent on measurement (default: 1000).
- `--dtype <type>`: Data type to benchmark.
