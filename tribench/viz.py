"""HTML report generator for TriBench benchmark runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .types import BenchResult, RunRecord


# ---------------------------------------------------------------------------
# Kernel colour palette (ECharts-inspired, consistent per run)
# ---------------------------------------------------------------------------

_COLOR_CYCLE = [
    "#5470c6",
    "#ee6666",
    "#91cc75",
    "#8675a9",
    "#fac858",
    "#73c0de",
    "#3ba272",
    "#fc8452",
    "#9a60b4",
    "#ea7ccc",
]


def _kernel_colors(kernels: list[str]) -> dict[str, str]:
    seen: dict[str, str] = {}
    for k in kernels:
        if k not in seen:
            seen[k] = _COLOR_CYCLE[len(seen) % len(_COLOR_CYCLE)]
    return seen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _passed(r: BenchResult) -> bool:
    """Return True if the result has no correctness failure."""
    return r.correctness is None or r.correctness.passed


def _primary_results(record: RunRecord) -> list[BenchResult]:
    return [r for r in record.results if r.variant in (None, "triton") and _passed(r)]


def _fmt(v: float | None, decimals: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{decimals}f}"


def _esc(s: str) -> str:
    """Minimal HTML escaping."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# Chart data builders
# ---------------------------------------------------------------------------

def _build_latency_chart_data(record: RunRecord, colors: dict[str, str]) -> dict:
    """Build Chart.js dataset for the latency bar chart (passing results only)."""
    all_results = [r for r in record.results if _passed(r)]

    # Preserve insertion order: (kernel, case_name, dtype, variant)
    kernels_ordered = list(dict.fromkeys(r.kernel for r in all_results))

    # Build case-key list in order
    def _case_key(r: BenchResult) -> str:
        v = f"__{r.variant}" if r.variant and r.variant != "triton" else ""
        return f"{r.kernel}__{r.case_name}__{r.dtype}{v}"

    case_keys: list[str] = list(dict.fromkeys(_case_key(r) for r in all_results))

    # Multi-line labels for Chart.js: list of arrays
    def _case_label(key: str) -> list[str]:
        parts = key.split("__")
        # parts: [kernel, case_name, dtype] or [kernel, case_name, dtype, variant]
        label = [parts[1], parts[2]]
        if len(parts) == 4:
            label.append(f"[{parts[3]}]")
        return label

    x_labels = [_case_label(k) for k in case_keys]

    # One dataset per kernel (sparse)
    key_index = {k: i for i, k in enumerate(case_keys)}
    datasets = []
    for kernel in kernels_ordered:
        color = colors.get(kernel, "#888888")
        data: list[float | None] = [None] * len(case_keys)
        error_min: list[float | None] = [None] * len(case_keys)
        error_max: list[float | None] = [None] * len(case_keys)
        for r in all_results:
            if r.kernel != kernel:
                continue
            k = _case_key(r)
            idx = key_index.get(k)
            if idx is None:
                continue
            data[idx] = round(r.latency_ms_p50, 6)
            error_min[idx] = round(r.latency_ms_min, 6)
            error_max[idx] = round(r.latency_ms_p99, 6)
        datasets.append({
            "label": kernel,
            "data": data,
            "errorMin": error_min,
            "errorMax": error_max,
            "backgroundColor": color,
            "borderColor": color,
            "stack": "latency",
            "barPercentage": 0.9,
            "categoryPercentage": 0.85,
        })

    return {"labels": x_labels, "datasets": datasets}


def _build_stability_data(primary: list[BenchResult]) -> dict:
    labels: list[list[str]] = []
    cv_values: list[float] = []
    cv_colors: list[str] = []
    jitter_values: list[float] = []
    jitter_colors: list[str] = []

    def _cv_color(v: float) -> str:
        if v < 0.05:
            return "#4caf50"
        elif v < 0.10:
            return "#ffc107"
        return "#f44336"

    def _jitter_color(v: float) -> str:
        if v < 10.0:
            return "#4caf50"
        elif v < 25.0:
            return "#ffc107"
        return "#f44336"

    for r in primary:
        lbl = [r.kernel, r.case_name, r.dtype]
        if r.jitter_cv is not None:
            labels.append(lbl)
            cv = round(r.jitter_cv, 6)
            cv_values.append(cv)
            cv_colors.append(_cv_color(cv))
            j_pct = round((r.tail_ratio_p99_p50 - 1.0) * 100.0, 4) if r.tail_ratio_p99_p50 is not None else 0.0
            jitter_values.append(j_pct)
            jitter_colors.append(_jitter_color(j_pct))

    return {
        "labels": labels,
        "cv_values": cv_values,
        "cv_colors": cv_colors,
        "jitter_values": jitter_values,
        "jitter_colors": jitter_colors,
    }


def _build_tflops_data(primary: list[BenchResult], colors: dict[str, str]) -> dict:
    entries = [r for r in primary if r.tflops is not None]
    labels = [[r.case_name, r.dtype] for r in entries]
    values = [round(r.tflops, 4) for r in entries]
    bar_colors = [colors.get(r.kernel, "#ee6666") for r in entries]
    return {"labels": labels, "values": values, "colors": bar_colors}


# ---------------------------------------------------------------------------
# Table row builder
# ---------------------------------------------------------------------------

def _table_rows_html(record: RunRecord) -> str:
    rows: list[str] = []
    for r in record.results:
        if not _passed(r):
            continue
        jitter_pct = (
            f"{(r.tail_ratio_p99_p50 - 1.0) * 100:.1f}%"
            if r.tail_ratio_p99_p50 is not None
            else "-"
        )
        cv_str = _fmt(r.jitter_cv, 3)
        tflops_str = _fmt(r.tflops, 2)
        gbps_str = _fmt(r.gbps, 1)
        compile_str = _fmt(r.compile_time_ms, 1)
        iqr_str = _fmt(r.latency_ms_std * 1.35 if r.latency_ms_std is not None else None, 4)
        variant_str = _esc(r.variant or "triton")
        rows.append(
            f"<tr>"
            f"<td>{_esc(r.kernel)}</td>"
            f"<td>{_esc(r.case_name)}</td>"
            f"<td>{_esc(r.pass_type)}</td>"
            f"<td>{_esc(r.dtype)}</td>"
            f"<td>{variant_str}</td>"
            f"<td>{r.latency_ms_min:.4f}</td>"
            f"<td>{r.latency_ms_p50:.4f}</td>"
            f"<td>{r.latency_ms_p90:.4f}</td>"
            f"<td>{r.latency_ms_p95:.4f}</td>"
            f"<td>{r.latency_ms_p99:.4f}</td>"
            f"<td>{iqr_str}</td>"
            f"<td>{cv_str}</td>"
            f"<td>{jitter_pct}</td>"
            f"<td>{tflops_str}</td>"
            f"<td>{gbps_str}</td>"
            f"<td>-</td>"
            f"<td>{compile_str}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TriBench Report \u2013 {run_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:2rem 2.5rem;min-width:900px}}
h1{{font-size:1.9rem;font-weight:700;color:#f0f6fc;display:flex;align-items:center;gap:.45rem;margin-bottom:.3rem}}
.bolt{{color:#e3b341;font-style:normal}}
.run-id{{font-size:.82rem;color:#8b949e;margin-bottom:1.8rem}}
/* cards */
.cards-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:.65rem;margin-bottom:2rem}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:.7rem 1rem}}
.card-label{{font-size:.65rem;text-transform:uppercase;letter-spacing:.06em;color:#8b949e;margin-bottom:.2rem}}
.card-value{{font-size:.95rem;font-weight:600;color:#f0f6fc;word-break:break-all}}
/* section headings */
h2{{font-size:1.1rem;font-weight:600;color:#f0f6fc;margin:1.8rem 0 .9rem}}
/* chart wrappers */
.chart-box{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1.2rem 1.4rem;margin-bottom:1.4rem}}
.chart-title{{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#8b949e;margin-bottom:.9rem}}
.charts-row{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.4rem}}
.chart-canvas-wrap{{position:relative}}
/* table */
h2.table-heading{{margin-top:2rem}}
.table-wrap{{overflow-x:auto}}
table.results{{width:100%;border-collapse:collapse;font-size:.78rem}}
table.results th{{background:#161b22;color:#8b949e;font-weight:700;text-transform:uppercase;letter-spacing:.05em;font-size:.67rem;padding:.55rem .7rem;text-align:left;border-bottom:1px solid #30363d;white-space:nowrap}}
table.results td{{padding:.45rem .7rem;border-bottom:1px solid #21262d;color:#c9d1d9;white-space:nowrap}}
table.results tr:hover td{{background:#1c2128}}
</style>
</head>
<body>

<h1><i class="bolt">\u26a1</i> TriBench Report</h1>
<p class="run-id">Run ID: {run_id}</p>

<div class="cards-grid">
  <div class="card"><div class="card-label">GPU</div><div class="card-value">{gpu}</div></div>
  <div class="card"><div class="card-label">CUDA</div><div class="card-value">{cuda}</div></div>
  <div class="card"><div class="card-label">PyTorch</div><div class="card-value">{pytorch}</div></div>
  <div class="card"><div class="card-label">Triton</div><div class="card-value">{triton}</div></div>
  <div class="card"><div class="card-label">Git Commit</div><div class="card-value">{git_commit}</div></div>
  <div class="card"><div class="card-label">Timer</div><div class="card-value">{timer}</div></div>
  <div class="card"><div class="card-label">Warmup / Rep</div><div class="card-value">{warmup_rep}</div></div>
  <div class="card"><div class="card-label">Timestamp</div><div class="card-value">{timestamp}</div></div>
</div>

<h2>Latency (p50 \u00b1 error bars: min / p99, log scale)</h2>
<div class="chart-box">
  <div class="chart-canvas-wrap" style="height:400px">
    <canvas id="latencyChart"></canvas>
  </div>
</div>

<div class="charts-row">
  <div class="chart-box">
    <div class="chart-title">Stability: Coefficient of Variation (lower = more stable)</div>
    <div class="chart-canvas-wrap" style="height:280px">
      <canvas id="cvChart"></canvas>
    </div>
  </div>
  <div class="chart-box">
    <div class="chart-title">Tail Jitter: (P99 \u2013 P50) / P50 \u00d7 100% (lower = less tail spike)</div>
    <div class="chart-canvas-wrap" style="height:280px">
      <canvas id="jitterChart"></canvas>
    </div>
  </div>
</div>

<h2>Performance (TFLOPS @ p50)</h2>
<div class="chart-box">
  <div class="chart-canvas-wrap" style="height:300px">
    <canvas id="tflopsChart"></canvas>
  </div>
</div>

<h2 class="table-heading">Full Results Table</h2>
<div class="table-wrap">
<table class="results">
<thead><tr>
  <th>Kernel</th><th>Case</th><th>Mode</th><th>Dtype</th><th>Variant</th>
  <th>Min (ms)</th><th>P50 (ms)</th><th>P90 (ms)</th><th>P95 (ms)</th><th>P99 (ms)</th>
  <th>IQR (ms)</th><th>CV</th><th>Jitter%</th><th>TFLOPS</th><th>GB/s</th>
  <th>AI (F/B)</th><th>Compile (ms)</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>

<script>
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#21262d';

// ── Error-bar plugin ──────────────────────────────────────────────────────
const errorBarPlugin = {{
  id: 'errorBars',
  afterDraw(chart) {{
    const ctx = chart.ctx;
    chart.data.datasets.forEach((ds, di) => {{
      if (!ds.errorMin || !ds.errorMax) return;
      const meta = chart.getDatasetMeta(di);
      meta.data.forEach((bar, i) => {{
        const rawVal = ds.data[i];
        if (rawVal == null) return;
        const eMin = ds.errorMin[i];
        const eMax = ds.errorMax[i];
        if (eMin == null || eMax == null) return;
        const x  = bar.x;
        const yT = chart.scales.y.getPixelForValue(eMax);
        const yB = chart.scales.y.getPixelForValue(eMin);
        const hw = 4;
        ctx.save();
        ctx.strokeStyle = 'rgba(201,209,217,0.8)';
        ctx.lineWidth = 1.5;
        // vertical line
        ctx.beginPath(); ctx.moveTo(x, yT); ctx.lineTo(x, yB); ctx.stroke();
        // top cap
        ctx.beginPath(); ctx.moveTo(x-hw, yT); ctx.lineTo(x+hw, yT); ctx.stroke();
        // bottom cap
        ctx.beginPath(); ctx.moveTo(x-hw, yB); ctx.lineTo(x+hw, yB); ctx.stroke();
        ctx.restore();
      }});
    }});
  }},
}};

// ── Latency chart ─────────────────────────────────────────────────────────
const latData = {latency_data_json};
new Chart(document.getElementById('latencyChart'), {{
  type: 'bar',
  data: latData,
  plugins: [errorBarPlugin],
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'right', labels: {{ boxWidth: 14, font: {{ size: 11 }} }} }},
      tooltip: {{
        callbacks: {{
          label(ctx) {{
            const ds = ctx.dataset;
            const i  = ctx.dataIndex;
            const v  = ctx.parsed.y;
            if (v == null) return null;
            const lo = ds.errorMin[i] != null ? ds.errorMin[i].toFixed(4) : '?';
            const hi = ds.errorMax[i] != null ? ds.errorMax[i].toFixed(4) : '?';
            return ` ${{ds.label}}: ${{v.toFixed(4)}} ms  [min=${{lo}}, p99=${{hi}}]`;
          }},
        }},
      }},
    }},
    scales: {{
      x: {{
        ticks: {{
          maxRotation: 45,
          minRotation: 30,
          font: {{ size: 9 }},
          autoSkip: false,
        }},
        grid: {{ color: '#21262d' }},
      }},
      y: {{
        type: 'logarithmic',
        title: {{ display: true, text: 'p50 Latency (ms)', color: '#8b949e' }},
        grid: {{ color: '#21262d' }},
      }},
    }},
  }},
}});

// ── CV chart ──────────────────────────────────────────────────────────────
const stabData = {stab_data_json};
new Chart(document.getElementById('cvChart'), {{
  type: 'bar',
  data: {{
    labels: stabData.labels,
    datasets: [{{
      data: stabData.cv_values,
      backgroundColor: stabData.cv_colors,
      borderWidth: 0,
    }}],
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{
        ticks: {{ maxRotation: 65, font: {{ size: 8 }}, autoSkip: false }},
        grid: {{ color: '#21262d' }},
      }},
      y: {{
        title: {{ display: true, text: 'CV (std/mean)', color: '#8b949e' }},
        grid: {{ color: '#21262d' }},
        beginAtZero: true,
      }},
    }},
  }},
}});

// ── Jitter chart ──────────────────────────────────────────────────────────
new Chart(document.getElementById('jitterChart'), {{
  type: 'bar',
  data: {{
    labels: stabData.labels,
    datasets: [{{
      data: stabData.jitter_values,
      backgroundColor: stabData.jitter_colors,
      borderWidth: 0,
    }}],
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{
        ticks: {{ maxRotation: 65, font: {{ size: 8 }}, autoSkip: false }},
        grid: {{ color: '#21262d' }},
      }},
      y: {{
        title: {{ display: true, text: 'Jitter (%)', color: '#8b949e' }},
        grid: {{ color: '#21262d' }},
        beginAtZero: true,
      }},
    }},
  }},
}});

// ── TFLOPS chart ──────────────────────────────────────────────────────────
const tflopsData = {tflops_data_json};
new Chart(document.getElementById('tflopsChart'), {{
  type: 'bar',
  data: {{
    labels: tflopsData.labels,
    datasets: [{{
      data: tflopsData.values,
      backgroundColor: tflopsData.colors,
      borderWidth: 0,
    }}],
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{
        ticks: {{ maxRotation: 45, font: {{ size: 9 }}, autoSkip: false }},
        grid: {{ color: '#21262d' }},
      }},
      y: {{
        title: {{ display: true, text: 'TFLOPS', color: '#8b949e' }},
        grid: {{ color: '#21262d' }},
        beginAtZero: true,
      }},
    }},
  }},
}});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_run_plots(record: RunRecord, run_dir: str | Path) -> list[Path]:
    """Generate a self-contained HTML report and return its path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    primary = _primary_results(record)
    passing = [r for r in record.results if _passed(r)]
    all_kernels = list(dict.fromkeys(r.kernel for r in passing))
    colors = _kernel_colors(all_kernels)

    latency_chart = _build_latency_chart_data(record, colors)
    stab_data = _build_stability_data(primary)
    tflops_data = _build_tflops_data(primary, colors)
    table_rows = _table_rows_html(record)

    e = record.env
    git = (_esc(e.git_commit[:12]) if e.git_commit else "N/A") + (
        " (dirty)" if e.git_dirty else ""
    )

    html = _HTML_TEMPLATE.format(
        run_id=_esc(record.run_id),
        gpu=_esc(e.gpu_name or "N/A"),
        cuda=_esc(e.cuda_version or "N/A"),
        pytorch=_esc(e.torch_version or "N/A"),
        triton=_esc(e.triton_version or "N/A"),
        git_commit=git,
        timer=_esc(record.timer_backend),
        warmup_rep=f"{record.warmup_ms}ms / {record.rep_ms}ms",
        timestamp=_esc(e.timestamp_utc or "N/A"),
        latency_data_json=json.dumps(latency_chart, ensure_ascii=False),
        stab_data_json=json.dumps(stab_data, ensure_ascii=False),
        tflops_data_json=json.dumps(tflops_data, ensure_ascii=False),
        table_rows=table_rows,
    )

    out_path = run_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return [out_path]
