from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpubench",
        description="GPU Kernel Benchmark Suite: reproducible Triton kernel benchmarking",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- new ----
    p_new = sub.add_parser("new", help="Scaffold a new benchmark kernel directory")
    p_new.add_argument("kernel_name", help="Name of the new kernel")

    # ---- list ----
    sub.add_parser("list", help="List all registered kernels and their supported dtypes/cases")

    # ---- validate-meta ----
    sub.add_parser("validate-meta", help="Validate all kernels' meta.json files")

    # ---- test ----
    p_test = sub.add_parser("test", help="Run correctness tests")
    p_test.add_argument("--kernel", default="all", help="Kernel name or 'all'")
    p_test.add_argument("--cases", default=None, help="Comma-separated case names")
    p_test.add_argument("--dtype", default=None, help="Comma-separated dtypes (fp16,bf16,fp32)")
    p_test.add_argument("--pass-type", default="forward", choices=["forward", "backward", "both"], help="Forward or backward pass")
    p_test.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    p_test.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    # ---- run ----
    p_run = sub.add_parser("run", help="Run benchmarks and save results")
    p_run.add_argument("--kernel", default="all", help="Kernel name or 'all'")
    p_run.add_argument("--cases", default=None, help="Comma-separated case names")
    p_run.add_argument("--dtype", default=None, help="Comma-separated dtypes")
    p_run.add_argument("--layout", default=None, help="Comma-separated layouts")
    p_run.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    p_run.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p_run.add_argument("--warmup-ms", type=float, default=200.0, help="Warmup time in ms")
    p_run.add_argument("--rep-ms", type=float, default=2000.0, help="Repetition time in ms")
    p_run.add_argument("--quantiles", default="0.5,0.9,0.95,0.99", help="Comma-separated quantiles")
    p_run.add_argument(
        "--timer",
        default="triton_do_bench",
        choices=["triton_do_bench", "triton_cudagraph", "cuda_event"],
        help="Timer backend",
    )
    p_run.add_argument("--output-dir", default="results", help="Output directory")
    p_run.add_argument("--pass-type", default="forward", choices=["forward", "backward", "both"], help="Forward or backward pass")
    p_run.add_argument("--no-correctness", action="store_true", help="Skip correctness checks")
    p_run.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_new(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    kernel_name = args.kernel_name
    kernel_dir = Path("kernels") / kernel_name
    if kernel_dir.exists():
        print(f"Error: Directory '{kernel_dir}' already exists.")
        return 1

    kernel_dir.mkdir(parents=True)

    # 1. meta.json
    meta_json = {
        "schema_version": "1.0",
        "name": kernel_name,
        "family": "custom",
        "description": f"Custom {kernel_name} kernel",
        "tags": ["custom"],
        "supported": {
            "dtypes": ["fp16", "bf16", "fp32"],
            "backends": ["cuda"]
        },
        "entrypoints": {
            "make_inputs": "reference.py:make_inputs",
            "reference": "reference.py:reference",
            "triton": "triton_impl.py:triton_impl",
            "estimate": "reference.py:estimate"
        },
        "correctness": {
            "atol": 1e-2,
            "rtol": 1e-2
        },
        "metrics": "elementwise",
        "cases": [
            {
                "name": "case_1",
                "N": 1024
            }
        ]
    }
    (kernel_dir / "meta.json").write_text(json.dumps(meta_json, indent=4) + "\n")

    # 2. reference.py
    (kernel_dir / "reference.py").write_text(f'''\
import torch

def make_inputs(case: dict, device: str, seed: int, dtype: torch.dtype) -> dict:
    # TODO: Generate dummy inputs
    torch.manual_seed(seed)
    N = case["N"]
    x = torch.randn(N, dtype=dtype, device=device)
    return {{"x": x}}

def reference(x: torch.Tensor) -> torch.Tensor:
    # TODO: Implement reference logic
    return x.clone()

def estimate(**kwargs) -> dict:
    # TODO: Estimate theoretical FLOPS and memory traffic
    return {{"flops": 0.0, "bytes": 0.0}}
''')

    # 3. triton_impl.py
    (kernel_dir / "triton_impl.py").write_text(f'''\
import torch
import triton
import triton.language as tl

@triton.jit
def _{kernel_name}_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # TODO: Implement Triton kernel logic
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

def triton_impl(x: torch.Tensor) -> torch.Tensor:
    # TODO: Implement Triton wrapper
    out = torch.empty_like(x)
    n_elements = x.numel()
    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _{kernel_name}_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out
''')

    print(f"Scaffolded new kernel '{kernel_name}' in '{kernel_dir}'")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from .registry import KernelRegistry

    reg = KernelRegistry()
    kernels = reg.list_kernels()

    if not kernels:
        print("No kernels found.")
        return 0

    for meta in kernels:
        print(f"\n{'='*60}")
        print(f"  Kernel:  {meta.name}")
        print(f"  Family:  {meta.family}")
        print(f"  Desc:    {meta.description}")
        print(f"  Tags:    {', '.join(meta.tags)}")
        print(f"  Dtypes:  {', '.join(meta.supported.dtypes)}")
        print(f"  Layouts: {', '.join(meta.supported.layouts)}")
        print(f"  Cases:")
        for c in meta.cases:
            params_str = ", ".join(f"{k}={v}" for k, v in c.params.items())
            print(f"    - {c.name}: {params_str}")
    print(f"\n{'='*60}")
    print(f"Total: {len(kernels)} kernel(s)")
    return 0


def _cmd_validate_meta(args: argparse.Namespace) -> int:
    from .registry import KernelRegistry

    reg = KernelRegistry()
    report = reg.validate_all()

    if not report:
        print(f"All {len(reg.kernel_names())} kernel meta.json files are valid.")
        return 0

    for name, errors in report.items():
        print(f"\nFAILED {name}:")
        for err in errors:
            print(f"   - {err}")
    return 1


def _cmd_test(args: argparse.Namespace) -> int:
    from .gen import generate_cases, materialize_inputs
    from .correctness import check_correctness
    from .registry import KernelRegistry
    from .errors import get_triton_hint
    import torch

    reg = KernelRegistry()
    if args.kernel == "all":
        names = reg.kernel_names()
    else:
        names = [n.strip() for n in args.kernel.split(",")]

    dtype_filter = args.dtype.split(",") if args.dtype else None
    case_filter = args.cases.split(",") if args.cases else None

    def _clear_cuda_cache() -> None:
        # CUDA may be left in an error state after OOM; cleanup itself must never crash test flow.
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception as cleanup_err:
                    print(
                        f"     Note: CUDA cache cleanup skipped "
                        f"({cleanup_err.__class__.__name__}: {str(cleanup_err).splitlines()[0] if str(cleanup_err) else ''})"
                    )
        except Exception:
            pass

    def _print_skip(name: str, reason: str, e: Exception) -> None:
        print(f"\n--- {name} ---")
        print(f"  SKIP  {reason}")
        print(f"   Error: {e.__class__.__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        hint = get_triton_hint(e)
        if hint:
            print(f"   Hint: {hint}")

    all_passed = True
    for name in names:
        skip_kernel = False
        try:
            meta = reg.get_meta(name)
            cases = generate_cases(meta, case_filter=case_filter, dtype_filter=dtype_filter)
            make_inputs_fn = reg.load_make_inputs(name)
            reference_fn = reg.load_reference(name)
            triton_fn = reg.load_triton(name)
        except Exception as e:
            all_passed = False
            _print_skip(name, "kernel initialization failed", e)
            _clear_cuda_cache()
            continue

        try:
            # Collect all implementations to test (main + variants)
            impls = [("triton", triton_fn)]
            for v_name in meta.entrypoints.variants.keys():
                impls.append((f"variant:{v_name}", reg.load_variant(name, v_name)))

            print(f"\n--- {name} ({len(cases)} case(s)) ---")
            pass_types = ["forward", "backward"] if args.pass_type == "both" else [args.pass_type]

            for pass_type in pass_types:
                if pass_type == "backward" and meta.entrypoints.backward is None:
                    continue

                if skip_kernel:
                    break

                print(f"  Mode: {pass_type}")
                make_inputs_fn = reg.load_make_inputs(name)
                
                if pass_type == "forward":
                    reference_fn = reg.load_reference(name)
                    triton_fn = reg.load_triton(name)
                    impls = [("triton", triton_fn)]
                    for v_name in meta.entrypoints.variants.keys():
                        impls.append((f"variant:{v_name}", reg.load_variant(name, v_name)))
                else:
                    reference_fn = reg.load_backward(name) # In backward mode, 'reference' is the backward ref
                    # Note: Currently we don't support variants for backward in meta.json beyond 'backward' entrypoint
                    # we could add variants: {"backward:variant": "..."} but let's keep it simple for now.
                    triton_fn = reg.load_backward(name)
                    impls = [("triton", triton_fn)]

                for case in cases:
                    if skip_kernel:
                        break
                    try:
                        inputs = materialize_inputs(make_inputs_fn, case, args.device, args.seed)
                        ref_out = reference_fn(**inputs)
                    except Exception as e:
                        all_passed = False
                        print(f"    SKIP  {case['case_name']} dtype={case['dtype']}  (setup/reference exception)")
                        print(f"     Error: {e.__class__.__name__}: {str(e).splitlines()[0] if str(e) else ''}")
                        hint = get_triton_hint(e)
                        if hint:
                            print(f"     Hint: {hint}")
                        _clear_cuda_cache()
                        if args.kernel == "all":
                            print(f"    SKIP  remaining cases for kernel '{name}'")
                            skip_kernel = True
                        continue
                    
                    for impl_name, impl_fn in impls:
                        try:
                            tri_out = impl_fn(**inputs)
                            torch.cuda.synchronize()

                            result = check_correctness(ref_out, tri_out, case["dtype"], meta.correctness)
                            status = "PASS" if result.passed else "FAIL"
                            print(
                                f"    {status}  {case['case_name']} [{impl_name}] dtype={case['dtype']}  "
                                f"max_abs={result.max_abs_err:.2e}  max_rel={result.max_rel_err:.2e}"
                            )
                            if not result.passed:
                                all_passed = False
                                if result.error_msg:
                                    print(f"     {result.error_msg}")
                        except Exception as e:
                            all_passed = False
                            print(f"    FAIL  {case['case_name']} [{impl_name}] dtype={case['dtype']}  (Exception)")
                            print(f"     Error: {e.__class__.__name__}: {str(e).splitlines()[0] if str(e) else ''}")
                            hint = get_triton_hint(e)
                            if hint:
                                print(f"     Hint: {hint}")
                            _clear_cuda_cache()
                            if args.kernel == "all":
                                print(f"    SKIP  remaining cases for kernel '{name}'")
                                skip_kernel = True
                                break
        except Exception as e:
            all_passed = False
            _print_skip(name, "kernel aborted by unexpected exception", e)
            _clear_cuda_cache()
            if args.kernel != "all":
                raise

    return 0 if all_passed else 1


def _cmd_run(args: argparse.Namespace) -> int:
    from .bench import run_benchmark
    from .env import capture_env
    from .gen import generate_cases
    from .io import make_run_id, write_json, write_summary_md
    from .registry import KernelRegistry
    from .types import RunRecord
    from .errors import get_triton_hint
    import torch

    reg = KernelRegistry()
    if args.kernel == "all":
        names = reg.kernel_names()
    else:
        names = [n.strip() for n in args.kernel.split(",")]

    dtype_filter = args.dtype.split(",") if args.dtype else None
    case_filter = args.cases.split(",") if args.cases else None
    layout_filter = args.layout.split(",") if args.layout else None
    quantiles = [float(q) for q in args.quantiles.split(",")]

    def _clear_cuda_cache() -> None:
        # CUDA may be left in an error state after OOM; cleanup itself must never crash benchmark flow.
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception as cleanup_err:
                    print(
                        f"     Note: CUDA cache cleanup skipped "
                        f"({cleanup_err.__class__.__name__}: {str(cleanup_err).splitlines()[0] if str(cleanup_err) else ''})"
                    )
        except Exception:
            pass

    def _print_skip(name: str, reason: str, e: Exception) -> None:
        print(f"\n--- Benchmarking {name} ---")
        print(f"  SKIP  {reason}")
        print(f"   Error: {e.__class__.__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        hint = get_triton_hint(e)
        if hint:
            print(f"   Hint: {hint}")

    def _format_run_metrics(result: "BenchResult") -> str:
        extras: list[str] = []
        if result.tail_ratio_p99_p50 is not None:
            extras.append(f"tail={result.tail_ratio_p99_p50:.3f}")
        if result.jitter_cv is not None:
            extras.append(f"jitter={result.jitter_cv:.3f}")
        if result.triton_vs_variant_p50_ratio is not None:
            extras.append(f"tri/var={result.triton_vs_variant_p50_ratio:.3f}")
        if result.tokens_per_s is not None:
            extras.append(f"tok/s={result.tokens_per_s:.1f}")
        if result.elements_per_s is not None:
            extras.append(f"elem/s={result.elements_per_s:.1f}")
        if result.sequences_per_s is not None:
            extras.append(f"seq/s={result.sequences_per_s:.1f}")
        if result.peak_mem_alloc_mb is not None or result.peak_mem_reserved_mb is not None:
            alloc = "-" if result.peak_mem_alloc_mb is None else f"{result.peak_mem_alloc_mb:.1f}"
            reserved = "-" if result.peak_mem_reserved_mb is None else f"{result.peak_mem_reserved_mb:.1f}"
            extras.append(f"peakMB={alloc}/{reserved}")
        return f"  {'  '.join(extras)}" if extras else ""

    # Build run record
    env = capture_env(cmdline=" ".join(sys.argv))
    run_id = make_run_id(args.kernel, args.device)
    record = RunRecord(
        run_id=run_id,
        env=env,
        seed=args.seed,
        timer_backend=args.timer,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        quantiles=quantiles,
    )
    has_failures = False
    triton_p50_by_case: dict[tuple[str, str, str, str, str], float] = {}

    for name in names:
        skip_kernel = False
        try:
            meta = reg.get_meta(name)
            cases = generate_cases(
                meta,
                case_filter=case_filter,
                dtype_filter=dtype_filter,
                layout_filter=layout_filter,
            )
            make_inputs_fn = reg.load_make_inputs(name)
            reference_fn = reg.load_reference(name)
            triton_fn = reg.load_triton(name)
            estimate_fn = reg.load_estimate(name)
        except Exception as e:
            has_failures = True
            _print_skip(name, "kernel initialization failed", e)
            _clear_cuda_cache()
            continue

        try:
            # Collect all implementations to test (main + variants)
            impls = [("triton", triton_fn)]
            for v_name in meta.entrypoints.variants.keys():
                impls.append((v_name, reg.load_variant(name, v_name)))

            print(f"\n--- Benchmarking {name} ({len(cases)} case(s)) ---")
            pass_types = ["forward", "backward"] if args.pass_type == "both" else [args.pass_type]

            for pass_type in pass_types:
                if pass_type == "backward" and meta.entrypoints.backward is None:
                    continue

                if skip_kernel:
                    break
                
                print(f"  Mode: {pass_type}")
                make_inputs_fn = reg.load_make_inputs(name)
                estimate_fn = reg.load_estimate(name) # TODO: backward estimate?

                if pass_type == "forward":
                    reference_fn = reg.load_reference(name)
                    main_triton_fn = reg.load_triton(name)
                    impls = [("triton", main_triton_fn)]
                    for v_name in meta.entrypoints.variants.keys():
                        impls.append((v_name, reg.load_variant(name, v_name)))
                else:
                    reference_fn = reg.load_backward(name)
                    main_triton_fn = reg.load_backward(name)
                    impls = [("triton", main_triton_fn)]

                for case in cases:
                    if skip_kernel:
                        break
                    for impl_name, impl_fn in impls:
                        try:
                            result = run_benchmark(
                                kernel_name=name,
                                meta=meta,
                                make_inputs_fn=make_inputs_fn,
                                reference_fn=reference_fn,
                                triton_fn=impl_fn,
                                estimate_fn=estimate_fn,
                                case=case,
                                device=args.device,
                                seed=args.seed,
                                warmup_ms=args.warmup_ms,
                                rep_ms=args.rep_ms,
                                quantiles=quantiles,
                                timer_backend=args.timer,
                                run_correctness=not args.no_correctness,
                                pass_type=pass_type,
                            )
                        except Exception as e:
                            has_failures = True
                            print(
                                f"    SKIP  {case['case_name']} [{impl_name}] dtype={case['dtype']}  "
                                f"(benchmark exception)"
                            )
                            print(f"     Error: {e.__class__.__name__}: {str(e).splitlines()[0] if str(e) else ''}")
                            hint = get_triton_hint(e)
                            if hint:
                                print(f"     Hint: {hint}")
                            _clear_cuda_cache()
                            if args.kernel == "all":
                                print(f"    SKIP  remaining cases for kernel '{name}'")
                                skip_kernel = True
                                break
                            raise
                        
                        # set variant name (if not default triton)
                        if impl_name != "triton":
                            result.variant = impl_name

                        case_key = (
                            result.kernel,
                            result.case_name,
                            result.dtype,
                            result.layout,
                            result.pass_type,
                        )
                        if impl_name == "triton":
                            triton_p50_by_case[case_key] = result.latency_ms_p50
                            result.triton_vs_variant_p50_ratio = 1.0
                        else:
                            tri_p50 = triton_p50_by_case.get(case_key)
                            if tri_p50 is not None and result.latency_ms_p50 > 0:
                                result.triton_vs_variant_p50_ratio = tri_p50 / result.latency_ms_p50
                        
                        record.results.append(result)

                        corr = ""
                        if result.correctness:
                            corr = " ✅" if result.correctness.passed else " ❌"
                        tflops = f"  {result.tflops:.2f} TFLOPS" if result.tflops is not None else ""
                        gbps = f"  {result.gbps:.1f} GB/s" if result.gbps is not None else ""
                        extras = _format_run_metrics(result)
                        v_label = f" [{impl_name}]" if impl_name != "triton" else ""
                        print(
                            f"    {result.case_name}{v_label}  {result.dtype}  "
                            f"p50={result.latency_ms_p50:.4f}ms  p95={result.latency_ms_p95:.4f}ms"
                            f"{tflops}{gbps}{extras}{corr}"
                        )
        except Exception as e:
            has_failures = True
            _print_skip(name, "kernel aborted by unexpected exception", e)
            _clear_cuda_cache()
            if args.kernel != "all":
                raise

    # Write outputs
    json_path = write_json(record, args.output_dir)
    md_path = write_summary_md(record, args.output_dir)
    print(f"\n📁 Results saved to: {json_path.parent}")
    print(f"   JSON:     {json_path}")
    print(f"   Summary:  {md_path}")
    if not args.no_plots:
        try:
            from .viz import generate_run_plots

            plot_paths = generate_run_plots(record, json_path.parent)
            if plot_paths:
                print("   Plots:")
                for p in plot_paths:
                    print(f"     - {p}")
            else:
                print("   Plots: no plottable data")
        except Exception as e:
            print(f"   Plots: skipped ({e})")
            print("   Hint: install plotting deps with `pip install -e \".[viz]\"`")

    return 0 if not has_failures else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "new": _cmd_new,
        "list": _cmd_list,
        "validate-meta": _cmd_validate_meta,
        "test": _cmd_test,
        "run": _cmd_run,
    }
    try:
        exit_code = handlers[args.command](args)
    except Exception as e:
        from .errors import get_triton_hint
        hint = get_triton_hint(e)
        if hint:
            print(f"\n[TriBench] Triton Error Context", file=sys.stderr)
            print(f"Error: {e.__class__.__name__}: {str(e).splitlines()[0] if str(e) else ''}", file=sys.stderr)
            print(f"Hint: {hint}\n", file=sys.stderr)
        raise

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
