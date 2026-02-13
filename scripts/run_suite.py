import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpubench.cli import main as gpubench_main


def run():
    parser = argparse.ArgumentParser(
        description="Run the full GPU kernel benchmark suite"
    )
    parser.add_argument("--kernel", default="all", help="Kernel name or 'all'")
    parser.add_argument("--dtype", default=None, help="Comma-separated dtypes")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-ms", type=float, default=200)
    parser.add_argument("--rep-ms", type=float, default=2000)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    # Build gpubench CLI args
    cli_args = ["run", "--kernel", args.kernel, "--device", args.device,
                "--seed", str(args.seed), "--warmup-ms", str(args.warmup_ms),
                "--rep-ms", str(args.rep_ms), "--output-dir", args.output_dir]
    if args.dtype:
        cli_args.extend(["--dtype", args.dtype])

    gpubench_main(cli_args)


if __name__ == "__main__":
    run()
