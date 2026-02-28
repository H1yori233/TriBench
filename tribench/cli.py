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

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

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

    reg = KernelRegistry()
    if args.kernel == "all":
        names = reg.kernel_names()
    else:
        names = [n.strip() for n in args.kernel.split(",")]

    dtype_filter = args.dtype.split(",") if args.dtype else None
    case_filter = args.cases.split(",") if args.cases else None

    all_passed = True
    for name in names:
        meta = reg.get_meta(name)
        cases = generate_cases(meta, case_filter=case_filter, dtype_filter=dtype_filter)
        make_inputs_fn = reg.load_make_inputs(name)
        reference_fn = reg.load_reference(name)
        triton_fn = reg.load_triton(name)

        # Collect all implementations to test (main + variants)
        impls = [("triton", triton_fn)]
        for v_name in meta.entrypoints.variants.keys():
            impls.append((f"variant:{v_name}", reg.load_variant(name, v_name)))

        print(f"\n--- {name} ({len(cases)} case(s)) ---")
        pass_types = ["forward", "backward"] if args.pass_type == "both" else [args.pass_type]

        for pass_type in pass_types:
            if pass_type == "backward" and meta.entrypoints.backward is None:
                continue

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
                inputs = materialize_inputs(make_inputs_fn, case, args.device, args.seed)
                ref_out = reference_fn(**inputs)
                
                for impl_name, impl_fn in impls:
                    tri_out = impl_fn(**inputs)
                    import torch
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

    return 0 if all_passed else 1


def _cmd_run(args: argparse.Namespace) -> int:
    from .bench import run_benchmark
    from .env import capture_env
    from .gen import generate_cases
    from .io import make_run_id, write_json, write_summary_md
    from .registry import KernelRegistry
    from .types import RunRecord

    reg = KernelRegistry()
    if args.kernel == "all":
        names = reg.kernel_names()
    else:
        names = [n.strip() for n in args.kernel.split(",")]

    dtype_filter = args.dtype.split(",") if args.dtype else None
    case_filter = args.cases.split(",") if args.cases else None
    layout_filter = args.layout.split(",") if args.layout else None
    quantiles = [float(q) for q in args.quantiles.split(",")]

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

    for name in names:
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

        # Collect all implementations to test (main + variants)
        impls = [("triton", triton_fn)]
        for v_name in meta.entrypoints.variants.keys():
            impls.append((v_name, reg.load_variant(name, v_name)))

        print(f"\n--- Benchmarking {name} ({len(cases)} case(s)) ---")
        pass_types = ["forward", "backward"] if args.pass_type == "both" else [args.pass_type]

        for pass_type in pass_types:
            if pass_type == "backward" and meta.entrypoints.backward is None:
                continue
            
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
                for impl_name, impl_fn in impls:
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
                    
                    # set variant name (if not default triton)
                    if impl_name != "triton":
                        result.variant = impl_name
                    
                    record.results.append(result)

                    corr = ""
                    if result.correctness:
                        corr = " ✅" if result.correctness.passed else " ❌"
                    tflops = f"  {result.tflops:.2f} TFLOPS" if result.tflops else ""
                    gbps = f"  {result.gbps:.1f} GB/s" if result.gbps else ""
                    v_label = f" [{impl_name}]" if impl_name != "triton" else ""
                    print(
                        f"    {result.case_name}{v_label}  {result.dtype}  "
                        f"p50={result.latency_ms_p50:.4f}ms  p95={result.latency_ms_p95:.4f}ms"
                        f"{tflops}{gbps}{corr}"
                    )

    # Write outputs
    json_path = write_json(record, args.output_dir)
    md_path = write_summary_md(record, args.output_dir)
    print(f"\n📁 Results saved to: {json_path.parent}")
    print(f"   JSON:     {json_path}")
    print(f"   Summary:  {md_path}")

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "list": _cmd_list,
        "validate-meta": _cmd_validate_meta,
        "test": _cmd_test,
        "run": _cmd_run,
    }
    exit_code = handlers[args.command](args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
