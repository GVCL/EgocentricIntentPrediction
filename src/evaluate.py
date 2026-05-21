"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      EPIC-KITCHENS Benchmark — Evaluation & Report Generator                 ║
║                                                                              ║
║  Loads results saved by benchmark.py, computes per-model & per-video         ║
║  statistics, produces:                                                       ║
║    • PNG comparison charts  (saved alongside the JSON)                       ║
║    • Detailed HTML report   (open in any browser)                            ║
║    • Printed leaderboard    (terminal)                                       ║
║                                                                              ║
║  Prerequisites:                                                              ║
║    pip install matplotlib pandas numpy                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage

  python evaluate.py                             # uses latest.json manifest
  python evaluate.py --results path/to/raw.json  # specific run
  python evaluate.py --top 3                     # show only top-3 models in charts
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

RESULTS_DIR = Path("D:/jayant_gathik/benchmark_results")

# Metric display config: (column_suffix, display_name, higher_is_better, y_scale)
METRIC_CFG = [
    ("bleu1",             "BLEU-1",             True,  (0, 1)),
    ("bleu2",             "BLEU-2",             True,  (0, 1)),
    ("rouge1",            "ROUGE-1",            True,  (0, 1)),
    ("rouge2",            "ROUGE-2",            True,  (0, 1)),
    ("rougeL",            "ROUGE-L",            True,  (0, 1)),
    ("verb_match",        "Verb Match %",       True,  (0, 1)),
    ("noun_match",        "Noun Match %",       True,  (0, 1)),
    ("hallucination_rate","Hallucination ↓",    False, (0, 1)),
    ("latency_s",         "Latency (s)",        False, None),
]

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

def load_raw(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def extract_models(df: pd.DataFrame) -> list[str]:
    """Infer model list from column names (columns end with __bleu1)."""
    models = []
    for col in df.columns:
        if col.endswith("__bleu1"):
            prefix = col[: -len("__bleu1")]
            # Convert prefix back to model name heuristic
            models.append(prefix)
    return models

def prefix_of(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace(".", "_")

def model_df(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Extract per-row metrics for one model as a clean DataFrame."""
    px = prefix_of(model)
    cols = {
        col.replace(f"{px}__", ""): col
        for col in df.columns
        if col.startswith(f"{px}__") and not col.endswith("__error")
    }
    sub = df[[v for v in cols.values() if v in df.columns]].copy()
    sub.columns = list(cols.keys())
    sub["video_id"]     = df["video_id"]
    sub["ground_truth"] = df["ground_truth"]
    sub["gt_verb"]      = df["gt_verb"]
    sub["gt_noun"]      = df["gt_noun"]
    sub["error"]        = df.get(f"{px}__error", 0)
    return sub[sub["error"] == 0].copy()

def per_model_stats(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Returns a DataFrame with one row per model, mean of every metric."""
    rows = []
    for model in models:
        mdf = model_df(df, model)
        row = {"model": model, "n_samples": len(mdf)}
        for metric, _, _, _ in METRIC_CFG:
            if metric in mdf.columns:
                row[metric]          = mdf[metric].mean()
                row[f"{metric}_std"] = mdf[metric].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")

def per_video_stats(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Returns mean ROUGE-L per (video_id, model)."""
    rows = []
    for model in models:
        mdf = model_df(df, model)
        mdf["model"] = model
        if "rougeL" in mdf.columns:
            rows.append(mdf[["video_id", "model", "rougeL", "verb_match", "noun_match"]])
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows)
    return combined.groupby(["video_id", "model"])[["rougeL", "verb_match", "noun_match"]].mean()

#  Chart helpers 
def _bar_compare(ax, stats: pd.DataFrame, metric: str, title: str,
                 palette: list[str], higher_better: bool = True,
                 ylim: tuple | None = None):
    models = list(stats.index)
    vals   = stats[metric].values
    stds   = stats.get(f"{metric}_std", pd.Series([0]*len(models))).values

    colors = [palette[i % len(palette)] for i in range(len(models))]
    bars   = ax.bar(range(len(models)), vals, color=colors, edgecolor="white",
                    linewidth=0.8, zorder=3)
    ax.errorbar(range(len(models)), vals, yerr=stds, fmt="none",
                capsize=4, color="black", linewidth=1.2, zorder=4)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.split(":")[0] for m in models], rotation=30,
                       ha="right", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylabel(title, fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate best bar
    best_idx = int(np.argmax(vals) if higher_better else np.argmin(vals))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)

def make_overview_chart(stats: pd.DataFrame, out_path: Path) -> None:
    """6-panel comparison bar chart for key metrics."""
    KEY_METRICS = [
        ("bleu1",             "BLEU-1",         True,  (0, 1)),
        ("rougeL",            "ROUGE-L",        True,  (0, 1)),
        ("verb_match",        "Verb Match",     True,  (0, 1)),
        ("noun_match",        "Noun Match",     True,  (0, 1)),
        ("hallucination_rate","Hallucination↓", False, (0, 1)),
        ("latency_s",         "Latency (s)",    False, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Intent Prediction Benchmark — Model Comparison",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, (metric, name, hb, ylim) in zip(axes.flat, KEY_METRICS):
        if metric in stats.columns:
            # _bar_compare(ax, stats, metric, name, PALETTE, hb, ylim)
            _bar_compare(ax, stats, metric, name, PALETTE, hb)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overview chart  →  {out_path}")

def make_radar_chart(stats: pd.DataFrame, out_path: Path) -> None:
    """Radar / spider chart — normalised metrics, one line per model."""
    RADAR_METRICS = ["bleu1", "rouge1", "rougeL", "verb_match", "noun_match"]
    available     = [m for m in RADAR_METRICS if m in stats.columns]
    if len(available) < 3:
        return

    N      = len(available)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    labels = ["BLEU-1", "ROUGE-1", "ROUGE-L", "Verb%", "Noun%"][:N]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    # ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_title("Model Radar Comparison\n(all metrics normalised to [0,1])",
                 fontsize=11, fontweight="bold", y=1.08)

    for i, (model, row) in enumerate(stats.iterrows()):
        vals = [row.get(m, 0.0) for m in available] + [row.get(available[0], 0.0)]
        ax.plot(angles, vals, linewidth=2, color=PALETTE[i % len(PALETTE)], label=model)
        ax.fill(angles, vals, alpha=0.08, color=PALETTE[i % len(PALETTE)])

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Radar chart      →  {out_path}")

def make_per_video_chart(video_stats: pd.DataFrame, out_path: Path) -> None:
    """Mean ROUGE-L per video, grouped by model."""
    if video_stats.empty or "rougeL" not in video_stats.columns:
        return

    pivot = video_stats["rougeL"].unstack("model")
    n_vids   = len(pivot)
    n_models = len(pivot.columns)

    fig_w  = max(10, n_vids * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    x   = np.arange(n_vids)
    w   = 0.8 / n_models
    off = -(n_models - 1) / 2 * w

    for i, model in enumerate(pivot.columns):
        ax.bar(x + off + i * w, pivot[model].fillna(0), width=w * 0.9,
               color=PALETTE[i % len(PALETTE)], label=model, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean ROUGE-L")
    ax.set_title("ROUGE-L per Video × Model", fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=max(1, n_models // 3))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-video chart  →  {out_path}")

def make_latency_box(df: pd.DataFrame, models: list[str], out_path: Path) -> None:
    """Box-plot of latency distribution per model."""
    data   = []
    labels = []
    for model in models:
        mdf = model_df(df, model)
        if "latency_s" in mdf.columns and len(mdf):
            data.append(mdf["latency_s"].values)
            labels.append(model.split(":")[0])

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.5), 4))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Latency (s)")
    ax.set_title("Response Latency Distribution", fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊  Latency box-plot →  {out_path}")

#  HTML report 
_HTML_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f7fa; color: #1a1a2e; line-height: 1.6; }
header { background: linear-gradient(135deg,#0f3460,#16213e);
         color: white; padding: 24px 40px; }
header h1 { font-size: 1.6rem; }
header p  { opacity: .75; font-size: .9rem; }
main { max-width: 1200px; margin: 32px auto; padding: 0 24px; }
h2 { color: #0f3460; border-left: 4px solid #e94560;
     padding-left: 12px; margin: 32px 0 16px; font-size: 1.2rem; }
.card { background: white; border-radius: 10px; padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,.08); margin-bottom: 24px; }
table { width: 100%; border-collapse: collapse; font-size: .88rem; }
th { background: #0f3460; color: white; padding: 10px 12px;
     text-align: right; white-space: nowrap; }
th:first-child { text-align: left; }
td { padding: 8px 12px; border-bottom: 1px solid #e8eaf0;
     text-align: right; }
td:first-child { text-align: left; font-weight: 600; }
tr:hover td { background: #f0f4ff; }
.best  { color: #2d6a4f; font-weight: 700; }
.worst { color: #c44e52; }
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(420px,1fr));
              gap: 20px; }
.chart-card img { width:100%; border-radius:6px; }
.pill { display:inline-block; padding:2px 10px; border-radius:20px;
        font-size:.78rem; font-weight:600; }
.pill-good { background:#d8f3dc; color:#2d6a4f; }
.pill-bad  { background:#ffe0e0; color:#c44e52; }
.sample-table td { font-size: .8rem; }
footer { text-align:center; padding:24px; color:#888; font-size:.8rem; }
"""

def _fmt(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}" if not np.isnan(val) else "—"


def make_html_report(
    stats:        pd.DataFrame,
    video_stats:  pd.DataFrame,
    df:           pd.DataFrame,
    models:       list[str],
    chart_paths:  dict,
    out_path:     Path,
    run_ts:       str,
) -> None:

    #  Best-per-column highlighting 
    def mark(col: str, higher_better: bool) -> dict:
        """Return {model: 'best'|'worst'|''} for a given metric column."""
        vals = stats[col].dropna()
        if vals.empty:
            return {}
        best  = vals.idxmax() if higher_better else vals.idxmin()
        worst = vals.idxmin() if higher_better else vals.idxmax()
        return {m: ("best" if m == best else ("worst" if m == worst else ""))
                for m in stats.index}

    marks = {}
    for metric, _, hb, _ in METRIC_CFG:
        if metric in stats.columns:
            marks[metric] = mark(metric, hb)

    #  Leaderboard rows 
    def leaderboard_row(model: str) -> str:
        row = stats.loc[model] if model in stats.index else {}
        def td(metric, fmt=4):
            v = row.get(metric, float("nan"))
            cls = marks.get(metric, {}).get(model, "")
            s   = _fmt(v, fmt)
            pct = f"{v*100:.1f}%" if metric in ("verb_match","noun_match") and not np.isnan(v) else ""
            display = pct or s
            tag_cls = f' class="{cls}"' if cls else ""
            return f"<td{tag_cls}>{display}</td>"
        ns = int(row.get("n_samples", 0))
        return (f"<tr><td>{model}</td>"
                + td("bleu1") + td("bleu2") + td("rouge1") + td("rouge2")
                + td("rougeL") + td("verb_match") + td("noun_match")
                + td("hallucination_rate") + td("latency_s", 3)
                + f"<td>{ns}</td></tr>")

    leaderboard = "\n".join(leaderboard_row(m) for m in sorted(
        stats.index,
        key=lambda m: -stats.loc[m].get("rougeL", 0)
    ))

    #  Sample predictions table (10 rows, first model) 
    sample_rows = ""
    if models:
        mdf = model_df(df, models[0]).head(10)
        for _, r in mdf.iterrows():
            pred  = str(r.get("prediction", ""))[:80]
            gt    = str(r.get("ground_truth", ""))
            rl    = r.get("rougeL", 0)
            color = "pill-good" if rl >= 0.3 else "pill-bad"
            sample_rows += (
                f"<tr>"
                f"<td>{r.get('video_id','')}</td>"
                f"<td>{gt}</td>"
                f"<td>{pred}</td>"
                f"<td><span class='pill {color}'>{rl:.3f}</span></td>"
                f"</tr>\n"
            )

    #  Chart img tags 
    def img_card(key: str, title: str) -> str:
        p = chart_paths.get(key)
        if not p or not Path(p).exists():
            return ""
        return (f"<div class='card chart-card'>"
                f"<h2>{title}</h2>"
                f"<img src='{Path(p).name}' alt='{title}'></div>")

    charts_html = (
        img_card("overview",   "Model Overview") +
        img_card("radar",      "Radar Comparison") +
        img_card("per_video",  "Per-Video ROUGE-L") +
        img_card("latency",    "Latency Distribution")
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EPIC-KITCHENS Benchmark Report</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<header>
  <h1>EPIC-KITCHENS Intent Prediction Benchmark</h1>
  <p>Run: {run_ts} &nbsp;|&nbsp; Participant: P01 &nbsp;|&nbsp;
     Models: {', '.join(models)} &nbsp;|&nbsp;
     Total samples: {len(df)}</p>
</header>
<main>

<h2>Leaderboard</h2>
<div class="card">
<table>
<thead><tr>
  <th>Model</th>
  <th>BLEU-1</th><th>BLEU-2</th><th>ROUGE-1</th><th>ROUGE-2</th><th>ROUGE-L</th>
  <th>Verb %</th><th>Noun %</th><th>Hall ↓</th><th>Lat (s)</th><th>N</th>
</tr></thead>
<tbody>{leaderboard}</tbody>
</table>
<p style="margin-top:12px;font-size:.8rem;color:#666">
  <span class="pill pill-good">best ↑</span> highest score per column &nbsp;
  <span class="pill pill-bad">worst ↓</span> lowest score per column
</p>
</div>

<h2>Charts</h2>
<div class="chart-grid">
{charts_html}
</div>

<h2>Sample Predictions ({models[0] if models else '—'})</h2>
<div class="card">
<table class="sample-table">
<thead><tr>
  <th style="text-align:left">Video</th>
  <th style="text-align:left">Ground Truth</th>
  <th style="text-align:left">Prediction</th>
  <th>ROUGE-L</th>
</tr></thead>
<tbody>{sample_rows}</tbody>
</table>
</div>

<h2>Metric Glossary</h2>
<div class="card">
<table>
<tr><td><b>BLEU-1 / BLEU-2</b></td><td>Unigram / bigram precision overlap between prediction and ground-truth narration.</td></tr>
<tr><td><b>ROUGE-1</b></td><td>Unigram F1 recall of ground-truth tokens in the prediction.</td></tr>
<tr><td><b>ROUGE-L</b></td><td>Longest-Common-Subsequence F1 — rewards in-order word matching.</td></tr>
<tr><td><b>Verb Match %</b></td><td>Fraction of samples where the GT action verb appeared in the prediction.</td></tr>
<tr><td><b>Noun Match %</b></td><td>Fraction of samples where at least one GT object noun appeared in the prediction.</td></tr>
<tr><td><b>Hallucination ↓</b></td><td>Fraction of content words in the prediction NOT present in YOLO detections (lower = better grounded).</td></tr>
<tr><td><b>Latency</b></td><td>Wall-clock time for the LLM to return a response (Ollama local inference).</td></tr>
</table>
</div>

</main>
<footer>Generated by evaluate.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — EPIC-KITCHENS Benchmark</footer>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML report     →  {out_path}")

#  CLI + main 
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate benchmark results")
    p.add_argument("--results", type=str, default=None,
                   help="Path to raw JSON results file (default: uses latest.json manifest)")
    p.add_argument("--top",    type=int, default=None,
                   help="Show only top-N models by ROUGE-L in charts")
    return p.parse_args()

def main():
    args = parse_args()

    #  Resolve results file 
    if args.results:
        raw_path = Path(args.results)
    else:
        manifest = RESULTS_DIR / "latest.json"
        if not manifest.exists():
            print(f"[!] No manifest found at {manifest}. Run benchmark.py first.")
            sys.exit(1)
        with open(manifest) as f:
            m = json.load(f)
        raw_path = Path(m["raw_json"])
        run_ts   = m.get("timestamp", "unknown")
        models_hint = m.get("models", [])

    if not raw_path.exists():
        print(f"[!] Results file not found: {raw_path}")
        sys.exit(1)

    print(f"📂  Loading: {raw_path}")
    df = load_raw(raw_path)
    print(f"    {len(df)} rows × {len(df.columns)} columns")

    models = models_hint if "models_hint" in dir() else extract_models(df)
    if not models:
        print("[!] Could not detect model columns. Check the raw JSON format.")
        sys.exit(1)
    print(f"    Models: {models}")

    #  Compute stats 
    stats       = per_model_stats(df, models)
    video_stats = per_video_stats(df, models)

    if args.top:
        top_models = (stats["rougeL"].nlargest(args.top).index.tolist()
                      if "rougeL" in stats.columns else models[:args.top])
        stats       = stats.loc[top_models]
        models      = top_models

    #  Printed leaderboard 
    print("\n" + "═" * 105)
    print(f"{'MODEL':<28} {'BLEU-1':>7} {'BLEU-2':>7} {'ROUGE-1':>8} {'ROUGE-L':>8} "
          f"{'VERB%':>7} {'NOUN%':>7} {'HALL↓':>7} {'LAT(s)':>8} {'N':>5}")
    print("═" * 105)
    for model in sorted(models, key=lambda m: -stats.loc[m].get("rougeL", 0)):
        row = stats.loc[model]
        print(
            f"{model:<28} "
            f"{row.get('bleu1',0):>7.4f} "
            f"{row.get('bleu2',0):>7.4f} "
            f"{row.get('rouge1',0):>8.4f} "
            f"{row.get('rougeL',0):>8.4f} "
            f"{row.get('verb_match',0)*100:>6.1f}% "
            f"{row.get('noun_match',0)*100:>6.1f}% "
            f"{row.get('hallucination_rate',0):>7.4f} "
            f"{row.get('latency_s',0):>8.3f} "
            f"{int(row.get('n_samples',0)):>5}"
        )
    print("═" * 105)

    #  Generate charts 
    out_dir = raw_path.parent
    ts      = run_ts if "run_ts" in dir() else datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n🖼️   Generating charts …")
    chart_paths: dict = {}

    ov = out_dir / f"chart_overview_{ts}.png"
    make_overview_chart(stats, ov)
    chart_paths["overview"] = str(ov)

    rd = out_dir / f"chart_radar_{ts}.png"
    make_radar_chart(stats, rd)
    chart_paths["radar"] = str(rd)

    if not video_stats.empty:
        pv = out_dir / f"chart_per_video_{ts}.png"
        make_per_video_chart(video_stats, pv)
        chart_paths["per_video"] = str(pv)

    lb = out_dir / f"chart_latency_{ts}.png"
    make_latency_box(df, models, lb)
    chart_paths["latency"] = str(lb)

    #  HTML report 
    print("\n Writing HTML report …")
    html_out = out_dir / f"report_{ts}.html"
    make_html_report(stats, video_stats, df, models, chart_paths, html_out, ts)

    #  Save aggregate CSV 
    agg_csv = out_dir / f"aggregate_{ts}.csv"
    stats.to_csv(agg_csv)
    print(f"  Aggregate CSV   →  {agg_csv}")

    print(f"\nAll outputs saved to:  {out_dir}")

if __name__ == "__main__":
    main()
