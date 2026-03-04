from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

from .types import BenchResult, RunRecord


def _primary_results(record: RunRecord) -> list[BenchResult]:
    # "variant" is set only for non-triton implementations.
    return [r for r in record.results if r.variant in (None, "triton")]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return path


def _result_label(r: BenchResult) -> str:
    base = f"{r.kernel}/{r.case_name}/{r.dtype}"
    if r.variant:
        return f"{base}/{r.variant}"
    return base


def _annotate_bars(ax, orientation: str = "v", fmt: str = "{:.3f}") -> None:
    if orientation == "h":
        for p in ax.patches:
            width = p.get_width()
            y = p.get_y() + p.get_height() / 2.0
            ax.text(width, y, f" {fmt.format(width)}", va="center", ha="left", fontsize=8)
    else:
        for p in ax.patches:
            height = p.get_height()
            x = p.get_x() + p.get_width() / 2.0
            ax.text(x, height, f"{fmt.format(height)}", va="bottom", ha="center", fontsize=8)


def _set_visual_theme(plt, sns) -> None:
    if sns is not None:
        sns.set_theme(
            style="whitegrid",
            context="talk",
            palette="deep",
            rc={
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.facecolor": "#f8fafc",
                "axes.facecolor": "#ffffff",
                "axes.titleweight": "bold",
            },
        )
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def _plot_kernel_latency_p50(record: RunRecord, out_dir: Path, plt, sns) -> Path | None:
    data: dict[str, list[float]] = defaultdict(list)
    for r in _primary_results(record):
        data[r.kernel].append(r.latency_ms_p50)
    if not data:
        return None

    ranked = sorted(((k, _mean(v)) for k, v in data.items()), key=lambda kv: kv[1])
    labels = [k for k, _ in ranked]
    values = [v for _, v in ranked]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(labels) + 3.0), 5.4))
    if sns is not None:
        sns.barplot(x=labels, y=values, hue=labels, legend=False, ax=ax, palette="Blues_d")
    else:
        ax.bar(labels, values, color="#2563eb")
    ax.set_title("Kernel Latency (p50, Triton)")
    ax.set_ylabel("Latency (ms)")
    ax.grid(axis="y", alpha=0.30)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    _annotate_bars(ax, orientation="v", fmt="{:.3f}")
    ax.text(
        0.99,
        0.96,
        f"kernels={len(labels)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#334155",
    )

    return _save(fig, out_dir / "kernel_latency_p50.png")


def _plot_speedup_vs_variant(record: RunRecord, out_dir: Path, plt, sns) -> Path | None:
    data: dict[str, list[float]] = defaultdict(list)
    for r in record.results:
        if r.variant is None or r.triton_vs_variant_p50_ratio is None:
            continue
        key = f"{r.kernel}:{r.variant}"
        data[key].append(r.triton_vs_variant_p50_ratio)
    if not data:
        return None

    ranked = sorted(((k, _mean(v)) for k, v in data.items()), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in ranked]
    values = [v for _, v in ranked]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.52 * len(labels) + 3.0), 5.8))
    y = list(range(len(labels)))
    if sns is not None:
        sns.barplot(x=values, y=labels, hue=labels, legend=False, ax=ax, palette="Greens_d", orient="h")
    else:
        ax.barh(y, values, color="#059669")
        ax.set_yticks(y, labels)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(1.0, linestyle="--", color="black", linewidth=1.0, alpha=0.6)
    ax.set_title("Triton vs Variant Speedup (p50 ratio)")
    ax.set_xlabel("Ratio (Triton / Variant), >1 means Triton faster")
    ax.grid(axis="x", alpha=0.30)
    _annotate_bars(ax, orientation="h", fmt="{:.2f}")

    return _save(fig, out_dir / "speedup_vs_variant_p50.png")


def _plot_tail_vs_jitter(record: RunRecord, out_dir: Path, plt, sns) -> Path | None:
    xs: list[float] = []
    ys: list[float] = []
    kernels: list[str] = []
    for r in _primary_results(record):
        if r.tail_ratio_p99_p50 is None or r.jitter_cv is None:
            continue
        xs.append(r.tail_ratio_p99_p50)
        ys.append(r.jitter_cv)
        kernels.append(r.kernel)
    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    if sns is not None:
        sns.scatterplot(
            x=xs,
            y=ys,
            hue=kernels if len(set(kernels)) <= 12 else None,
            ax=ax,
            alpha=0.85,
            legend=(len(set(kernels)) <= 12),
            s=60,
        )
        if len(set(kernels)) <= 12:
            ax.legend(title="kernel", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    else:
        scatter = ax.scatter(xs, ys, alpha=0.8, c=xs, cmap="viridis")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Tail ratio")
    ax.set_title("Tail Ratio vs Jitter")
    ax.set_xlabel("Tail ratio (p99 / p50)")
    ax.set_ylabel("Jitter (std / mean)")
    ax.grid(alpha=0.30)

    return _save(fig, out_dir / "tail_vs_jitter.png")


def _plot_memory_vs_latency(record: RunRecord, out_dir: Path, plt, sns) -> Path | None:
    xs: list[float] = []
    ys: list[float] = []
    for r in _primary_results(record):
        if r.peak_mem_alloc_mb is None:
            continue
        xs.append(r.latency_ms_p50)
        ys.append(r.peak_mem_alloc_mb)
    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    if sns is not None:
        sns.regplot(x=xs, y=ys, scatter_kws={"s": 52, "alpha": 0.8}, line_kws={"alpha": 0.65}, ax=ax, color="#d97706")
    else:
        ax.scatter(xs, ys, alpha=0.8, color="#d97706")
    ax.set_title("Peak Memory vs Latency")
    ax.set_xlabel("p50 latency (ms)")
    ax.set_ylabel("Peak allocated memory (MB)")
    ax.grid(alpha=0.30)

    return _save(fig, out_dir / "memory_vs_latency.png")


def _plot_throughput(record: RunRecord, out_dir: Path, plt, sns) -> Path | None:
    token_points = [r for r in _primary_results(record) if r.tokens_per_s is not None]
    elem_points = [r for r in _primary_results(record) if r.elements_per_s is not None]
    seq_points = [r for r in _primary_results(record) if r.sequences_per_s is not None]
    if not token_points and not elem_points and not seq_points:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.6))
    groups = [
        ("Tokens/s", token_points, "tokens_per_s", "#7c3aed"),
        ("Elements/s", elem_points, "elements_per_s", "#0ea5e9"),
        ("Sequences/s", seq_points, "sequences_per_s", "#ef4444"),
    ]

    for ax, (title, points, field, color) in zip(axes, groups):
        if not points:
            ax.set_title(title)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        data: dict[str, list[float]] = defaultdict(list)
        for r in points:
            v = getattr(r, field)
            if v is not None:
                data[r.kernel].append(v)
        ranked = sorted(((k, _mean(vs)) for k, vs in data.items()), key=lambda kv: kv[1], reverse=True)
        top = ranked[:12]
        labels = [k for k, _ in top]
        values = [v for _, v in top]

        if sns is not None:
            sns.barplot(x=labels, y=values, ax=ax, color=color)
        else:
            ax.bar(labels, values, color=color, alpha=0.9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=50, labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.grid(axis="y", alpha=0.30)
        _annotate_bars(ax, orientation="v", fmt="{:.1f}")

    return _save(fig, out_dir / "throughput_overview.png")


def _plot_latency_detail_all(record: RunRecord, out_dir: Path, plt, sns) -> Path | None:
    rows = []
    for r in record.results:
        rows.append(
            {
                "label": _result_label(r),
                "p50": r.latency_ms_p50,
                "kernel": r.kernel,
            }
        )
    if not rows:
        return None

    rows.sort(key=lambda x: x["p50"])
    labels = [x["label"] for x in rows]
    values = [x["p50"] for x in rows]
    kernels = [x["kernel"] for x in rows]

    height = max(7.0, min(0.24 * len(rows) + 2.0, 44.0))
    fig, ax = plt.subplots(figsize=(14.0, height))
    if sns is not None:
        sns.barplot(x=values, y=labels, hue=kernels if len(set(kernels)) <= 12 else None, dodge=False, ax=ax)
        if len(set(kernels)) <= 12:
            ax.legend(title="kernel", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    else:
        y = list(range(len(labels)))
        ax.barh(y, values, color="#2563eb", alpha=0.9)
        ax.set_yticks(y, labels)
    ax.set_title("Detailed Latency (p50) by Kernel/Case/Dtype[/Variant]")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Benchmark entry")
    ax.grid(axis="x", alpha=0.30)
    _annotate_bars(ax, orientation="h", fmt="{:.3f}")

    return _save(fig, out_dir / "latency_detail_all.png")


def generate_run_plots(record: RunRecord, run_dir: str | Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns  # type: ignore
        except Exception:
            sns = None
    except Exception as e:
        raise RuntimeError(f"matplotlib is unavailable: {e}")

    _set_visual_theme(plt, sns)

    out_dir = Path(run_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    plotters: list[Callable[[RunRecord, Path, object, object], Path | None]] = [
        _plot_kernel_latency_p50,
        _plot_latency_detail_all,
        _plot_speedup_vs_variant,
        _plot_tail_vs_jitter,
        _plot_memory_vs_latency,
        _plot_throughput,
    ]

    for plotter in plotters:
        try:
            path = plotter(record, out_dir, plt, sns)
            if path is not None:
                generated.append(path)
        finally:
            plt.close("all")

    return generated
