"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             MR Activity Benchmark — Oculus Capture Edition                   ║
║                                                                              ║
║  Benchmarks local MP4 recordings against manually-created ground-truth       ║
║  action JSON annotations. Samples temporal clip midpoints, runs YOLO         ║
║  detection + Ollama VLM/LLM inference, evaluates predictions, and saves      ║
║  raw + aggregate metrics.                                                    ║
║                                                                              ║
║  Designed for:                                                               ║
║    C:\\Users\\JayantGathik\\AppData\\Roaming\\odh\\captures                  ║
║                                                                              ║
║  Usage:                                                                      ║
║    python benchmark_oculus.py                                                ║
║    python benchmark_oculus.py --models llama3.2 mistral                      ║
║    python benchmark_oculus.py --preview                                      ║
║    python benchmark_oculus.py --max-clips 5                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
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
CAPTURE_DIR = Path(
    r"C:\Users\JayantGathik\AppData\Roaming\odh\captures"
)

RESULTS_DIR = Path("./benchmark_results")

YOLO_MODEL_PATH = Path("models/yolo11n.pt")

CONFIDENCE = 0.45

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_TIMEOUT = 120

DEFAULT_MODELS = [
    "llama3.2",
    "mistral",
    "gemma3:4b",
]

VIDEO_IDS = [
    "com.oculus.vrshell-20260430-221242-0",
    "com.oculus.vrshell-20260430-221558-0",
    "com.oculus.vrshell-20260430-221840-0",
    "com.oculus.vrshell-20260430-222034-0",
]

# Optional VLM prompt tuning
PROMPT_TEMPLATE = """
You are observing a first-person mixed reality recording.

Your task:
Predict the user's CURRENT action in one short sentence.

Rules:
- Start with a verb
- Be concise
- Mention the main object
- No explanation
- No extra formatting

Visible objects: {objects}
Center gaze target: {gaze}

Current action:
"""


# Yolo
_yolo_model = None


def get_yolo():
    global _yolo_model

    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(str(YOLO_MODEL_PATH))

    return _yolo_model


def detect_objects(frame):

    model = get_yolo()

    results = model(frame, conf=CONFIDENCE, verbose=False)[0]

    detections = []

    for box in results.boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "label": results.names[int(box.cls)],
            "conf": float(box.conf),
            "box": (x1, y1, x2, y2),
        })

    return detections, results.plot()


# Ollama
def query_ollama(model: str, prompt: str):

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 40,
        },
    }

    t0 = time.perf_counter()

    try:

        r = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )

        r.raise_for_status()

        text = r.json().get("response", "").strip()

        text = re.sub(r"(?i)^current action\s*:\s*", "", text).strip()

        text = re.split(r"(?<=[.!?])\s", text)[0].strip()

        latency = time.perf_counter() - t0

        return text, latency

    except Exception as e:

        latency = time.perf_counter() - t0

        return f"ERROR: {e}", latency


def list_ollama_models():

    try:

        r = requests.get(OLLAMA_TAGS_URL, timeout=5)

        r.raise_for_status()

        return [m["name"] for m in r.json().get("models", [])]

    except Exception:
        return []


# Utilities
def tokenize(text: str):

    return re.findall(r"\b\w+\b", text.lower())


def compute_bleu(prediction, reference):

    pred_tok = tokenize(prediction)
    ref_tok = tokenize(reference)

    if not pred_tok or not ref_tok:
        return {"bleu1": 0.0, "bleu2": 0.0}

    if NLTK_OK:

        smooth = SmoothingFunction().method1

        b1 = sentence_bleu(
            [ref_tok],
            pred_tok,
            weights=(1, 0, 0, 0),
            smoothing_function=smooth,
        )

        b2 = sentence_bleu(
            [ref_tok],
            pred_tok,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smooth,
        )

        return {
            "bleu1": round(b1, 4),
            "bleu2": round(b2, 4),
        }

    return {"bleu1": 0.0, "bleu2": 0.0}


def compute_rouge(prediction, reference):

    if not ROUGE_OK:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
        }

    scorer = rouge_mod.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    scores = scorer.score(reference, prediction)

    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def verb_match(prediction, gt_verb):

    return gt_verb.lower() in tokenize(prediction)


def noun_match(prediction, gt_noun):

    return gt_noun.lower() in tokenize(prediction)


# Ground truth
def load_truth_file(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Process Video
def process_video(
    video_path,
    truth_annotations,
    models,
    max_clips=None,
    preview=False,
):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print(f"[!] Failed to open {video_path}")

        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []

    current_time_s = 0.0

    if max_clips:
        truth_annotations = truth_annotations[:max_clips]

    print(
        f"\n📹 {video_path.name} "
        f"| {len(truth_annotations)} clips "
        f"| {fps:.2f} fps"
    )

    for idx, ann in enumerate(
        tqdm(truth_annotations, leave=False)
    ):

        duration_s = float(ann["duration_s"])

        midpoint_s = current_time_s + (duration_s / 2.0)

        midpoint_frame = int(midpoint_s * fps)

        midpoint_frame = max(
            0,
            min(midpoint_frame, total_frames - 1)
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, midpoint_frame)

        ret, frame = cap.read()

        if not ret or frame is None:

            current_time_s += duration_s
            continue

        detections, annotated = detect_objects(frame)

        object_labels = [d["label"] for d in detections]

        # center gaze heuristic
        h, w = frame.shape[:2]

        gx = w // 2
        gy = h // 2

        gaze = "nothing"

        for obj in detections:

            x1, y1, x2, y2 = obj["box"]

            if x1 <= gx <= x2 and y1 <= gy <= y2:

                gaze = obj["label"]
                break

        prompt = PROMPT_TEMPLATE.format(
            objects=object_labels,
            gaze=gaze,
        )

        row = {
            "video_id": ann["video_id"],
            "ground_truth": ann["ground_truth"],
            "gt_verb": ann["gt_verb"],
            "gt_noun": ann["gt_noun"],
            "duration_s": duration_s,
            "midpoint_frame": midpoint_frame,
            "timestamp_s": round(midpoint_s, 3),
            "objects_detected": object_labels,
            "gaze": gaze,
        }

        for model in models:

            pred, latency = query_ollama(model, prompt)

            is_error = pred.startswith("ERROR:")

            safe_pred = pred if not is_error else ""

            bleu = compute_bleu(
                safe_pred,
                ann["ground_truth"],
            )

            rouge = compute_rouge(
                safe_pred,
                ann["ground_truth"],
            )

            vm = verb_match(
                safe_pred,
                ann["gt_verb"],
            )

            nm = noun_match(
                safe_pred,
                ann["gt_noun"],
            )

            prefix = (
                model.replace(":", "_")
                .replace("/", "_")
                .replace(".", "_")
            )

            row.update({
                f"{prefix}__prediction": pred,
                f"{prefix}__latency_s": round(latency, 3),
                f"{prefix}__bleu1": bleu["bleu1"],
                f"{prefix}__bleu2": bleu["bleu2"],
                f"{prefix}__rouge1": rouge["rouge1"],
                f"{prefix}__rouge2": rouge["rouge2"],
                f"{prefix}__rougeL": rouge["rougeL"],
                f"{prefix}__verb_match": int(vm),
                f"{prefix}__noun_match": int(nm),
                f"{prefix}__error": int(is_error),
            })

        results.append(row)

        current_time_s += duration_s

        if preview:

            cv2.imshow("Preview", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()

    return results


# Aggregation
def aggregate(results, models):

    metrics = [
        "bleu1",
        "bleu2",
        "rouge1",
        "rouge2",
        "rougeL",
        "verb_match",
        "noun_match",
        "latency_s",
    ]

    agg = {}

    for model in models:

        prefix = (
            model.replace(":", "_")
            .replace("/", "_")
            .replace(".", "_")
        )

        model_scores = defaultdict(list)

        for row in results:

            if row.get(f"{prefix}__error", 1):
                continue

            for metric in metrics:

                key = f"{prefix}__{metric}"

                if key in row:
                    model_scores[metric].append(
                        float(row[key])
                    )

        agg[model] = {}

        for metric in metrics:

            vals = model_scores[metric]

            agg[model][metric] = {
                "mean": round(float(np.mean(vals)), 4) if vals else 0.0,
                "std": round(float(np.std(vals)), 4) if vals else 0.0,
                "n": len(vals),
            }

    return agg


# Summary
def print_summary(agg):

    print("\n" + "═" * 110)

    print(
        f"{'MODEL':<24}"
        f"{'BLEU1':>10}"
        f"{'ROUGE-L':>12}"
        f"{'VERB%':>10}"
        f"{'NOUN%':>10}"
        f"{'LATENCY':>12}"
    )

    print("═" * 110)

    for model, m in agg.items():

        print(
            f"{model:<24}"
            f"{m['bleu1']['mean']:>10.4f}"
            f"{m['rougeL']['mean']:>12.4f}"
            f"{m['verb_match']['mean'] * 100:>9.1f}%"
            f"{m['noun_match']['mean'] * 100:>9.1f}%"
            f"{m['latency_s']['mean']:>12.3f}"
        )

    print("═" * 110)


# Main
def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--models",
        nargs="+",
        default=None,
    )

    p.add_argument(
        "--preview",
        action="store_true",
    )

    p.add_argument(
        "--max-clips",
        type=int,
        default=None,
    )

    return p.parse_args()


def main():

    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("🔍 Checking Ollama...")

    available_models = list_ollama_models()

    if not available_models:

        print("[!] Ollama is not reachable.")
        sys.exit(1)

    requested_models = args.models or DEFAULT_MODELS

    models_to_run = [
        m for m in requested_models
        if any(m in a for a in available_models)
    ]

    if not models_to_run:

        print("[!] No valid models available.")
        sys.exit(1)

    print(f"✅ Models: {models_to_run}")

    print("\n🤖 Loading YOLO...")
    get_yolo()

    all_results = []

    t0 = time.time()

    for vid in VIDEO_IDS:

        video_path = CAPTURE_DIR / f"{vid}.mp4"

        truth_path = CAPTURE_DIR / f"truth_{vid}.json"

        if not video_path.exists():

            print(f"[!] Missing video: {video_path}")
            continue

        if not truth_path.exists():

            print(f"[!] Missing truth file: {truth_path}")
            continue

        annotations = load_truth_file(truth_path)

        rows = process_video(
            video_path,
            annotations,
            models_to_run,
            max_clips=args.max_clips,
            preview=args.preview,
        )

        all_results.extend(rows)

    elapsed = time.time() - t0

    print(
        f"\n✅ Finished benchmark "
        f"({len(all_results)} clips in {elapsed:.1f}s)"
    )

    if not all_results:
        return

    agg = aggregate(
        all_results,
        models_to_run,
    )

    print_summary(agg)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_json = RESULTS_DIR / f"raw_{ts}.json"

    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    raw_csv = RESULTS_DIR / f"raw_{ts}.csv"

    fieldnames = list(all_results[0].keys())

    with open(raw_csv, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_results)

    agg_json = RESULTS_DIR / f"aggregate_{ts}.json"

    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print(f"\n💾 Raw JSON  -> {raw_json}")
    print(f"💾 Raw CSV   -> {raw_csv}")
    print(f"💾 Aggregate -> {agg_json}")

    if args.preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()