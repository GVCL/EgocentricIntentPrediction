"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         EPIC-KITCHENS Intent Prediction Benchmark — Ollama Edition           ║
║                                                                              ║
║  Streams P01 videos, samples annotation midpoints, runs YOLO object          ║
║  detection + multiple local Ollama LLMs, evaluates against ground-truth      ║
║  narrations and saves raw + aggregate results.                               ║
║                                                                              ║
║  Prerequisites:                                                              ║
║    pip install ultralytics requests tqdm rouge-score nltk                    ║
║    ollama pull llama3.2  # or whichever models you want to benchmark         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
  python benchmark.py                      # run all videos, all models
  python benchmark.py --video P01_01       # single video
  python benchmark.py --max-ann 20         # cap annotations per video (dev/test)
  python benchmark.py --models llama3.2 mistral
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from tqdm import tqdm

# Optional nltk
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    NLTK_OK = True
except ImportError:
    NLTK_OK = False

try:
    from rouge_score import rouge_scorer as rouge_mod
    ROUGE_OK = True
except ImportError:
    ROUGE_OK = False

# Config
DATASET_ROOT    = Path("D:/jayant_gathik/EPIC-KITCHENS/P01")
RESULTS_DIR     = Path("D:/jayant_gathik/benchmark_results")

# Annotation CSV
_ann_candidates = [
    DATASET_ROOT / "meta_data" / "EPIC_100_train.csv",
    DATASET_ROOT.parent / "EPIC_100_train.csv",
]
ANNOTATIONS_CSV = next((p for p in _ann_candidates if p.exists()), _ann_candidates[0])

# Yolo model
YOLO_MODEL_PATH = Path("models/yolo11n.pt")

CONFIDENCE      = 0.50          # YOLO detection confidence threshold

OLLAMA_URL      = "http://localhost:8080/api/generate"
OLLAMA_TAGS_URL = "http://localhost:8080/api/tags"
OLLAMA_TIMEOUT  = 90            # seconds per LLM request

# Models to benchmark 
DEFAULT_MODELS = [
    "llama3.2",
    "mistral",
    "gemma3:4b",
    "phi4-mini",
    "qwen2.5:3b",
]

# How many frames to skip between annotation samples inside a single video.
ANN_STRIDE = 0

# In-context learning (ICL) examples
ICL_EXAMPLES = """
Examples of Intent Prediction based on State Vectors [Emotion, Objects, Gaze]:
- Vector [Focused,       [knife, cutting board],    knife]          -> Intent: Preparing to slice vegetables on the board.
- Vector [Concentrated,  [pan, stove, oil],          stove]          -> Intent: About to heat oil in the pan before frying.
- Vector [Curious,       [spice jar, pot],            spice jar]      -> Intent: Reading a spice label before seasoning the dish.
- Vector [Neutral,       [tap, sponge, bowl],         tap]            -> Intent: Rinsing a bowl under the tap.
- Vector [Frustrated,    [jar, lid],                  lid]            -> Intent: Struggling to open a tightly sealed jar.
- Vector [Focused,       [egg, bowl, whisk],          bowl]           -> Intent: Cracking an egg into the bowl to begin whisking.
"""


_yolo_model = None

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(str(YOLO_MODEL_PATH))
    return _yolo_model


def detect_objects(frame: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Returns (detections, annotated_frame). Each detection: {label, box, conf}."""
    model   = get_yolo()
    results = model(frame, conf=CONFIDENCE, verbose=False)[0]
    dets    = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        dets.append({
            "label": results.names[int(box.cls)],
            "conf":  float(box.conf),
            "box":   (x1, y1, x2, y2),
        })
    return dets, results.plot()


# Annotation loader
def load_annotations(csv_path: Path, participant_id: str = "P01") -> dict:
    """
    Returns {video_id: [ {start_frame, stop_frame, narration, verb, noun}, ... ]}
    sorted by start_frame.
    """
    ann: dict = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("participant_id", "").strip() != participant_id:
                continue
            try:
                ann[row["video_id"].strip()].append({
                    "start_frame":  int(row["start_frame"]),
                    "stop_frame":   int(row["stop_frame"]),
                    "narration":    row["narration"].strip(),
                    "verb":         row["verb"].strip().lower(),
                    "noun":         row["noun"].strip().lower(),
                    "all_nouns":    row.get("all_nouns", row.get("noun", "")).strip().lower(),
                })
            except (KeyError, ValueError):
                continue
    # sort annotations by start_frame for each video
    for vid in ann:
        ann[vid].sort(key=lambda x: x["start_frame"])
    return dict(ann)


# Prompt builder
def build_prompt(emotion: str, objects: list, gaze: str, history: list) -> str:
    history_str = "\n".join(
        f"  [{i+1}] [{s['emotion']}, {s['objects']}, gaze={s['gaze']}]"
        for i, s in enumerate(history[-6:])
    ) or "  (no prior states)"

    return f"""You are an intent prediction engine embedded in a kitchen Mixed Reality assistant.
Your task: given the user's rolling state history and current state, output ONE short sentence
predicting their immediate next action. Start with a verb. No preamble or explanation.

{ICL_EXAMPLES}

Recent state history (oldest → newest):
{history_str}

Current state: [emotion={emotion}, objects_in_view={objects}, gaze_target={gaze}]

Intent:"""


# Ollama interface
def query_ollama(model: str, prompt: str) -> tuple[str, float]:
    """Returns (prediction_text, latency_s). On error returns ('ERROR: ...', latency)."""
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.1, "num_predict": 80, "top_p": 0.9},
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        # Strip any echoed "Intent:" prefix the model might produce
        text = re.sub(r"(?i)^intent\s*:\s*", "", text).strip()
        # Take only the first sentence if model is verbose
        text = re.split(r"(?<=[.!?])\s", text)[0].strip()
        return text, time.perf_counter() - t0
    except requests.exceptions.Timeout:
        return "ERROR: timeout", time.perf_counter() - t0
    except Exception as e:
        return f"ERROR: {e}", time.perf_counter() - t0


def list_ollama_models() -> list[str]:
    """Returns names of models currently pulled in Ollama."""
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# Metrics
_smooth = None

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def compute_bleu(prediction: str, reference: str) -> dict:
    """Returns BLEU-1 and BLEU-2 scores."""
    pred_tok = _tokenize(prediction)
    ref_tok  = _tokenize(reference)
    if not pred_tok or not ref_tok:
        return {"bleu1": 0.0, "bleu2": 0.0}

    if NLTK_OK:
        global _smooth
        if _smooth is None:
            _smooth = SmoothingFunction().method1
        b1 = sentence_bleu([ref_tok], pred_tok, weights=(1, 0, 0, 0),
                           smoothing_function=_smooth)
        b2 = sentence_bleu([ref_tok], pred_tok, weights=(0.5, 0.5, 0, 0),
                           smoothing_function=_smooth)
        return {"bleu1": round(b1, 4), "bleu2": round(b2, 4)}
    else:
        # Fallback: simple precision-based BLEU-1
        ref_set = set(ref_tok)
        b1 = sum(1 for t in pred_tok if t in ref_set) / len(pred_tok)
        return {"bleu1": round(b1, 4), "bleu2": 0.0}


def compute_rouge(prediction: str, reference: str) -> dict:
    """Returns ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1."""
    if ROUGE_OK:
        scorer = rouge_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    else:
        # Fallback: manual ROUGE-L (LCS F1)
        p = _tokenize(prediction)
        r = _tokenize(reference)
        if not p or not r:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        m, n = len(p), len(r)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if p[i-1] == r[j-1] else max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        prec = lcs / m; rec = lcs / n
        rl = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        # rough ROUGE-1
        ref_set = set(r)
        r1_prec = sum(1 for t in p if t in ref_set) / m
        r1_rec  = sum(1 for t in r if t in set(p)) / n
        r1 = (2 * r1_prec * r1_rec / (r1_prec + r1_rec)) if (r1_prec + r1_rec) > 0 else 0.0
        return {"rouge1": round(r1, 4), "rouge2": 0.0, "rougeL": round(rl, 4)}


def verb_match(prediction: str, gt_verb: str) -> bool:
    """True if the ground-truth verb appears anywhere in the prediction."""
    pred_tokens = set(_tokenize(prediction))
    return gt_verb.lower() in pred_tokens


def noun_match(prediction: str, all_nouns_str: str) -> bool:
    """
    True if ANY of the ground-truth nouns appears in the prediction.
    all_nouns_str may be comma-separated (EPIC-100 all_nouns field).
    """
    pred_tokens = set(_tokenize(prediction))
    for noun_phrase in re.split(r"[,;]", all_nouns_str):
        for token in _tokenize(noun_phrase):
            if token in pred_tokens:
                return True
    return False


def hallucination_rate(prediction: str, objects_detected: list[str]) -> float:
    """
    Fraction of content words in the prediction that don't appear in either
    the detected objects list or common kitchen stop-words.
    A proxy for how much the model is hallucinating objects not visible.
    """
    STOP = {"the", "a", "an", "to", "and", "or", "is", "are", "on", "in",
            "into", "from", "with", "for", "of", "it", "at", "by", "be",
            "about", "up", "before", "after", "then", "that", "this",
            "place", "put", "get", "take", "use", "using", "cut", "add",
            "wash", "pour", "pick", "grab", "open", "close", "move",
            "prepare", "start", "begin", "ready", "set", "begin"}
    obj_tokens = set(_tokenize(" ".join(objects_detected)))
    pred_content = [t for t in _tokenize(prediction) if t not in STOP and len(t) > 2]
    if not pred_content:
        return 0.0
    hallucinated = sum(1 for t in pred_content if t not in obj_tokens)
    return round(hallucinated / len(pred_content), 4)


# Per-video processor
def process_video(
    video_path: Path,
    annotations: list[dict],
    models: list[str],
    max_ann: int | None = None,
    show_preview: bool = False,
) -> list[dict]:
    """
    Streams one video file, seeks to the midpoint frame of each annotation,
    runs YOLO + all Ollama models, returns list of result dicts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open: {video_path}")
        return []

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results    = []
    history    = []          # rolling state history for the LLM prompt
    video_id   = video_path.stem

    anns = annotations[::ANN_STRIDE + 1]
    if max_ann:
        anns = anns[:max_ann]

    print(f"\n  📹  {video_id}  |  {len(anns)} annotations  |  "
          f"{total_fr} frames @ {fps:.1f} fps")

    for ann in tqdm(anns, desc=f"    {video_id}", leave=False, unit="ann"):
        mid_frame = (ann["start_frame"] + ann["stop_frame"]) // 2
        mid_frame = max(0, min(mid_frame, total_fr - 1))

        # Seek to the annotation midpoint
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        timestamp_s = mid_frame / fps

        # Yolo
        dets, annotated = detect_objects(frame)
        obj_labels = [d["label"] for d in dets]

        # Gaze: centre-point collision
        h, w = frame.shape[:2]
        gx, gy = w // 2, h // 2
        gaze = "Nothing"
        for obj in dets:
            x1, y1, x2, y2 = obj["box"]
            if x1 <= gx <= x2 and y1 <= gy <= y2:
                gaze = obj["label"]
                break

        # No real telemetry in offline mode → neutral emotion
        emotion = "Neutral"

        # Update rolling history (thread-safe in single-threaded bench)
        history.append({"emotion": emotion, "objects": obj_labels, "gaze": gaze})
        if len(history) > 10:
            history.pop(0)

        prompt = build_prompt(emotion, obj_labels, gaze, history)

        row: dict = {
            "video_id":          video_id,
            "frame":             mid_frame,
            "timestamp_s":       round(timestamp_s, 3),
            "start_frame":       ann["start_frame"],
            "stop_frame":        ann["stop_frame"],
            "ground_truth":      ann["narration"],
            "gt_verb":           ann["verb"],
            "gt_noun":           ann["noun"],
            "gt_all_nouns":      ann["all_nouns"],
            "objects_detected":  obj_labels,
            "gaze":              gaze,
        }

        # Query every model
        for model in models:
            pred, latency = query_ollama(model, prompt)
            is_error      = pred.startswith("ERROR:")
            safe_pred     = pred if not is_error else ""

            bleu   = compute_bleu(safe_pred, ann["narration"]) if not is_error else {"bleu1": 0.0, "bleu2": 0.0}
            rg     = compute_rouge(safe_pred, ann["narration"]) if not is_error else {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            vm     = verb_match(safe_pred, ann["verb"]) if not is_error else False
            nm     = noun_match(safe_pred, ann["all_nouns"]) if not is_error else False
            hall   = hallucination_rate(safe_pred, obj_labels) if not is_error else 1.0

            # Store under flat key names for easy CSV export
            prefix = model.replace(":", "_").replace("/", "_").replace(".", "_")
            row.update({
                f"{prefix}__prediction":   pred,
                f"{prefix}__latency_s":    round(latency, 3),
                f"{prefix}__bleu1":        bleu["bleu1"],
                f"{prefix}__bleu2":        bleu["bleu2"],
                f"{prefix}__rouge1":       rg["rouge1"],
                f"{prefix}__rouge2":       rg["rouge2"],
                f"{prefix}__rougeL":       rg["rougeL"],
                f"{prefix}__verb_match":   int(vm),
                f"{prefix}__noun_match":   int(nm),
                f"{prefix}__hallucination_rate": hall,
                f"{prefix}__error":        int(is_error),
            })

        results.append(row)

        if show_preview:
            cv2.imshow("Benchmark Preview", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    return results


# Aggregation
def aggregate(results: list[dict], models: list[str]) -> dict:
    """
    Compute mean ± std for every numeric metric, per model.
    Returns {model: {metric: {mean, std, n_samples}}}
    """
    agg: dict = {}
    METRICS = ["bleu1", "bleu2", "rouge1", "rouge2", "rougeL",
               "verb_match", "noun_match", "hallucination_rate", "latency_s"]

    for model in models:
        prefix = model.replace(":", "_").replace("/", "_").replace(".", "_")
        scores: dict = {m: [] for m in METRICS}

        for r in results:
            # Skip rows where this model errored
            if r.get(f"{prefix}__error", 1):
                continue
            for m in METRICS:
                key = f"{prefix}__{m}"
                if key in r:
                    scores[m].append(float(r[key]))

        n = len(scores["bleu1"]) or 1
        agg[model] = {
            m: {
                "mean": round(float(np.mean(v)), 4) if v else 0.0,
                "std":  round(float(np.std(v)),  4) if v else 0.0,
                "n":    len(v),
            }
            for m, v in scores.items()
        }
        agg[model]["n_total"]  = len([r for r in results if f"{prefix}__prediction" in r])
        agg[model]["n_errors"] = len([r for r in results if r.get(f"{prefix}__error", 0)])

    return agg


# Console summary table
def print_summary(agg: dict) -> None:
    SEP = "═" * 105
    HDR = f"{'MODEL':<28} {'BLEU-1':>7} {'BLEU-2':>7} {'ROUGE-1':>8} {'ROUGE-L':>8} " \
          f"{'VERB%':>7} {'NOUN%':>7} {'HALL↓':>7} {'LAT(s)':>8} {'N':>5}"
    print(f"\n{SEP}")
    print(HDR)
    print(SEP)
    for model, m in sorted(agg.items(), key=lambda x: -x[1]["rougeL"]["mean"]):
        print(
            f"{model:<28} "
            f"{m['bleu1']['mean']:>7.4f} "
            f"{m['bleu2']['mean']:>7.4f} "
            f"{m['rouge1']['mean']:>8.4f} "
            f"{m['rougeL']['mean']:>8.4f} "
            f"{m['verb_match']['mean']*100:>6.1f}% "
            f"{m['noun_match']['mean']*100:>6.1f}% "
            f"{m['hallucination_rate']['mean']:>7.4f} "
            f"{m['latency_s']['mean']:>8.3f} "
            f"{m['n_total']:>5}"
        )
    print(SEP)


# Main
def parse_args():
    p = argparse.ArgumentParser(description="EPIC-KITCHENS Intent Prediction Benchmark")
    p.add_argument("--video",    type=str, default=None,
                   help="Only benchmark this video ID, e.g. P01_01")
    p.add_argument("--max-ann",  type=int, default=None,
                   help="Max annotations to process per video (useful for quick tests)")
    p.add_argument("--models",   nargs="+", default=None,
                   help="Ollama model names to benchmark (overrides DEFAULT_MODELS)")
    p.add_argument("--preview",  action="store_true",
                   help="Show live OpenCV preview window while benchmarking")
    p.add_argument("--stride",   type=int, default=0,
                   help="Skip every N annotations between samples (0=use all)")
    return p.parse_args()


def main():
    args = parse_args()

    # Override stride if passed
    global ANN_STRIDE
    ANN_STRIDE = args.stride

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check Ollama
    print("Checking Ollama …")
    available_models = list_ollama_models()
    if not available_models:
        print(f"[!] Cannot reach Ollama at {OLLAMA_URL}. Is it running?  (ollama serve)")
        sys.exit(1)
    print(f"    Available: {available_models}")

    requested = args.models or DEFAULT_MODELS
    models_to_run = [m for m in requested if any(m in a for a in available_models)]
    skipped = [m for m in requested if m not in models_to_run]
    if skipped:
        print(f"    [!] Not found in Ollama (skipping): {skipped}")
    if not models_to_run:
        print("[!] No usable models. Pull them first:  ollama pull <model>")
        sys.exit(1)
    print(f"    Benchmarking: {models_to_run}\n")

    # Load annotations
    print(f"Loading annotations from:\n    {ANNOTATIONS_CSV}")
    if not ANNOTATIONS_CSV.exists():
        print(f"[!] Annotation file not found: {ANNOTATIONS_CSV}")
        sys.exit(1)
    all_annotations = load_annotations(ANNOTATIONS_CSV, participant_id="P01")
    print(f"    Loaded {sum(len(v) for v in all_annotations.values())} annotations "
          f"across {len(all_annotations)} videos.")

    # Load YOLO
    print(f"\nLoading YOLO from: {YOLO_MODEL_PATH}")
    get_yolo()   # warm up

    # Discover videos
    video_dir   = DATASET_ROOT / "videos"
    video_files = sorted(
        list(video_dir.glob("P01_*.MP4")) + list(video_dir.glob("P01_*.mp4"))
    )
    if args.video:
        video_files = [v for v in video_files if v.stem == args.video]
        if not video_files:
            print(f"[!] Video {args.video} not found in {video_dir}")
            sys.exit(1)
    print(f"\nFound {len(video_files)} video(s) to process.")

    # Run benchmark
    all_results: list[dict] = []
    t_start = time.time()

    for vf in video_files:
        vid_id = vf.stem
        anns   = all_annotations.get(vid_id, [])
        if not anns:
            print(f"  ↩  {vid_id}: no annotations in CSV — skipping.")
            continue
        rows = process_video(
            vf, anns, models_to_run,
            max_ann=args.max_ann,
            show_preview=args.preview,
        )
        all_results.extend(rows)

    if args.preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\nDone. Processed {len(all_results)} annotation samples in {elapsed:.1f}s.")

    if not all_results:
        print("[!] No results to save. Exiting.")
        return

    # Aggregate metrics
    agg = aggregate(all_results, models_to_run)
    print_summary(agg)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Raw results — JSON
    raw_json = RESULTS_DIR / f"raw_{ts}.json"
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw JSON  →  {raw_json}")

    # Raw results — CSV
    raw_csv = RESULTS_DIR / f"raw_{ts}.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(raw_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    print(f"Raw CSV   →  {raw_csv}")

    # Aggregate metrics — JSON
    agg_json = RESULTS_DIR / f"aggregate_{ts}.json"
    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "models": models_to_run, "metrics": agg}, f, indent=2)
    print(f"Aggregate →  {agg_json}")

    manifest = RESULTS_DIR / "latest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":    ts,
            "raw_json":     str(raw_json),
            "raw_csv":      str(raw_csv),
            "aggregate":    str(agg_json),
            "models":       models_to_run,
            "n_samples":    len(all_results),
            "elapsed_s":    round(elapsed, 1),
        }, f, indent=2)
    print(f"Manifest  →  {manifest}")
    print("\nRun  python evaluate.py  to generate charts and a detailed HTML report.")


if __name__ == "__main__":
    main()
