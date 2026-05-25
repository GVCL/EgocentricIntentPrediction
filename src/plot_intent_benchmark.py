import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    print("Error: numpy is required. Install it with `pip install numpy`")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Error: matplotlib is required. Install it with `pip install matplotlib`")
    sys.exit(1)

# NLP Tools for semantic matching and METEOR
try:
    import nltk
    from nltk.translate.meteor_score import meteor_score
    from nltk.corpus import wordnet

    # Download required dictionaries silently
    for pkg in ["punkt", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    NLTK_OK = True
except ImportError:
    print("Warning: NLTK not found. Install with `pip install nltk`. Falling back to exact matching.")
    NLTK_OK = False

# ROUGE-L Scoring Tool
try:
    from rouge_score import rouge_scorer
    ROUGE_OK = True
except ImportError:
    print("Warning: rouge-score not found. Install with `pip install rouge-score`. ROUGE-L will be 0.0.")
    ROUGE_OK = False

# --- Config ---
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
RESULTS_DIR = Path("benchmark_results")
CHARTS_DIR = RESULTS_DIR / "intent_prediction_graphs"

# --- NLP Utilities ---
def tokenize(text: str):
    return re.findall(r"\b\w+\b", str(text).lower())

def get_synonyms(word: str):
    if not NLTK_OK:
        return {word.lower()}
    synonyms = {word.lower()}
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace("_", " "))
    return synonyms

def semantic_match(prediction: str, ground_truth: str):
    pred_tokens = set(tokenize(prediction))
    gt_synonyms = get_synonyms(ground_truth)
    return bool(pred_tokens & gt_synonyms)

def compute_meteor(prediction: str, reference: str):
    if not NLTK_OK:
        return 0.0
    pred_tok = tokenize(prediction)
    ref_tok = tokenize(reference)
    if not pred_tok or not ref_tok:
        return 0.0
    return round(meteor_score([ref_tok], pred_tok), 4)

def compute_rouge_l(prediction: str, reference: str):
    if not ROUGE_OK:
        return 0.0
    if not prediction.strip() or not reference.strip():
        return 0.0
    # Initialize the scorer specifically for ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return round(scores['rougeL'].fmeasure, 4)

# --- Plotting ---
def save_bar_chart(title: str, labels: list[str], values: list[float], errors: list[float], ylabel: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.15), 4.5))
    
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(labels))]
    
    ax.bar(range(len(labels)), values, yerr=errors, capsize=5, ecolor='black', 
           color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    
    for i, (v, e) in enumerate(zip(values, errors)):
        ax.text(i, v + e + (max(values)*0.02), f"{v:.3f}", ha='center', va='bottom', fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# --- Main Logic ---
def main():
    if not RESULTS_DIR.exists():
        print(f"Directory {RESULTS_DIR} not found.")
        return

    csv_files = list(RESULTS_DIR.glob("raw_*.csv"))
    if not csv_files:
        print(f"No raw_*.csv files found in {RESULTS_DIR}.")
        return

    print(f"Found {len(csv_files)} CSV files. Calculating metrics from raw data...")

    model_scores = defaultdict(lambda: defaultdict(list))

    # Parse all CSVs
    for csv_file in csv_files:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            models_in_csv = []
            for col in reader.fieldnames:
                if col.endswith("__prediction"):
                    models_in_csv.append(col.replace("__prediction", ""))
            
            for row in reader:
                gt_verb = row.get("gt_verb", "")
                gt_noun = row.get("gt_noun", "")
                gt_full = row.get("ground_truth", "")

                for prefix in models_in_csv:
                    if int(row.get(f"{prefix}__error", 1)) == 1:
                        continue

                    pred = row.get(f"{prefix}__prediction", "")

                    # 1. Read static metrics from CSV
                    model_scores[prefix]["bleu1"].append(float(row.get(f"{prefix}__bleu1", 0)))
                    model_scores[prefix]["hallucination_rate"].append(float(row.get(f"{prefix}__hallucination_rate", 0)))
                    model_scores[prefix]["latency_s"].append(float(row.get(f"{prefix}__latency_s", 0)))

                    # 2. Calculate dynamic text metrics on the fly!
                    model_scores[prefix]["rougeL"].append(compute_rouge_l(pred, gt_full)) # <--- NEW DYNAMIC CALCULATION
                    model_scores[prefix]["meteor"].append(compute_meteor(pred, gt_full))
                    model_scores[prefix]["verb_match"].append(1.0 if semantic_match(pred, gt_verb) else 0.0)
                    model_scores[prefix]["noun_match"].append(1.0 if semantic_match(pred, gt_noun) else 0.0)

    if not model_scores:
        print("No valid predictions found in CSVs.")
        return

    models = list(model_scores.keys())
    clean_labels = [m.replace("_", ".") for m in models]
    
    print(f"Models processed: {', '.join(clean_labels)}")

    graphs_to_make = [
        {"metric_key": "bleu1", "title": "Intent Prediction: BLEU-1 Score", "ylabel": "BLEU-1", "multiplier": 1.0},
        {"metric_key": "rougeL", "title": "Intent Prediction: ROUGE-L Score", "ylabel": "ROUGE-L", "multiplier": 1.0},
        {"metric_key": "meteor", "title": "Intent Prediction: METEOR Score (Synonym Aware)", "ylabel": "METEOR", "multiplier": 1.0},
        {"metric_key": "verb_match", "title": "Intent Prediction: Verb Match (Semantic)", "ylabel": "Match %", "multiplier": 100.0},
        {"metric_key": "noun_match", "title": "Intent Prediction: Noun Match (Semantic)", "ylabel": "Match %", "multiplier": 100.0},
        {"metric_key": "hallucination_rate", "title": "Intent Prediction: Hallucination Rate", "ylabel": "Rate (Lower is Better)", "multiplier": 1.0},
        {"metric_key": "latency_s", "title": "Intent Prediction: Average Latency", "ylabel": "Seconds per clip", "multiplier": 1.0},
    ]

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    for graph in graphs_to_make:
        key = graph["metric_key"]
        values = []
        errors = []
        
        for model in models:
            raw_data = model_scores[model].get(key, [])
            if raw_data:
                mean_val = float(np.mean(raw_data))
                std_val = float(np.std(raw_data))
            else:
                mean_val, std_val = 0.0, 0.0
                
            values.append(mean_val * graph["multiplier"])
            errors.append(std_val * graph["multiplier"])

        output_filename = CHARTS_DIR / f"{key}_comparison.png"
        
        save_bar_chart(
            title=graph["title"],
            labels=clean_labels,
            values=values,
            errors=errors,
            ylabel=graph["ylabel"],
            output_path=output_filename
        )
        print(f"✅ Saved graph: {output_filename}")

    print(f"\nAll graphs successfully saved to {CHARTS_DIR}")
    
    # --- Generate LaTeX Table ---
    print("\n" + "="*50)
    print("📋 AUTOMATIC LATEX TABLE GENERATOR")
    print("="*50)
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Intent Prediction Results (mean). Best in \\textbf{bold}.}")
    print("\\label{tab:intent_results}")
    print("\\begin{tabular}{lccccccc}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{BLEU-1}$\\uparrow$ & \\textbf{ROUGE-L}$\\uparrow$ & \\textbf{METEOR}$\\uparrow$ & \\textbf{Verb\\%}$\\uparrow$ & \\textbf{Noun\\%}$\\uparrow$ & \\textbf{Halluc.\\,$\\downarrow$} & \\textbf{Latency (s)}$\\downarrow$ \\\\")
    print("\\hline")
    
    for model in models:
        lbl = clean_labels[models.index(model)]
        
        # Safely calculate means
        def get_mean(k):
            return float(np.mean(model_scores[model][k])) if model_scores[model][k] else 0.0

        b1 = get_mean("bleu1")
        rl = get_mean("rougeL")
        met = get_mean("meteor")
        v = get_mean("verb_match") * 100
        n = get_mean("noun_match") * 100
        h = get_mean("hallucination_rate")
        lat = get_mean("latency_s")
        
        # Print the formatted LaTeX row
        print(f"{lbl} & {b1:.3f} & {rl:.3f} & {met:.3f} & {v:.1f}\\% & {n:.1f}\\% & {h:.2f} & {lat:.2f} \\\\")
        
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}\n")

if __name__ == "__main__":
    main()