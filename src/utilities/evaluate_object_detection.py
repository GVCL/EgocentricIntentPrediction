import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None


SCHEMA_VERSION = "object_detection_annotations_v1"
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_text: str | Path, base: Path | None = None) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base or repo_root()) / path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name).strip("_") or "chart"


def charts_dir_for_output(output_path: Path) -> Path:
    return output_path.parent / "object_detection_graphs" / output_path.stem


def require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required to generate charts. Install it with `pip install matplotlib`.")


def save_bar_chart(title: str, labels: list[str], values: list[float], ylabel: str, output_path: Path, *, rotation: int = 25) -> None:
    require_matplotlib()
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.15), 4))
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(labels))]
    ax.bar(range(len(labels)), values, color=colors,
           edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_grouped_metrics_chart(title: str, series_by_label: dict[str, dict[str, float]], metric_names: list[str], output_path: Path) -> None:
    require_matplotlib()
    labels = list(series_by_label.keys())
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4.5))
    width = 0.8 / max(1, len(metric_names))
    offsets = [(-0.4 + width / 2) + index *
               width for index in range(len(metric_names))]

    for metric_index, metric_name in enumerate(metric_names):
        values = [series_by_label[label].get(
            metric_name, 0.0) for label in labels]
        ax.bar(
            [index + offsets[metric_index] for index in range(len(labels))],
            values,
            width=width * 0.95,
            color=PALETTE[metric_index % len(PALETTE)],
            label=metric_name,
            zorder=3,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_graphs(results: dict, output_dir: Path) -> list[Path]:
    require_matplotlib()

    output_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []

    overall = results["overall"]
    label_metrics = overall["label_metrics"]
    metric_order = ["precision", "recall", "f1_score",
                    "accuracy", "exact_match_accuracy"]
    metric_labels = ["Precision", "Recall", "F1", "Accuracy", "Exact Match"]
    metric_values = [label_metrics[name] for name in metric_order]
    overall_path = output_dir / "overall_label_metrics.png"
    save_bar_chart("Overall Label Metrics", metric_labels,
                   metric_values, "Score", overall_path, rotation=0)
    created_paths.append(overall_path)

    count_names = ["tp", "fp", "fn", "tn"]
    count_labels = [name.upper() for name in count_names]
    count_values = [overall["label_counts"].get(
        name, 0) for name in count_names]
    counts_path = output_dir / "overall_label_counts.png"
    save_bar_chart("Overall Label Counts", count_labels,
                   count_values, "Count", counts_path, rotation=0)
    created_paths.append(counts_path)

    per_video = results.get("per_video", [])
    if per_video:
        series_by_label = {}
        video_labels = []
        for video in per_video:
            label = safe_name(Path(video["name"]).stem)
            video_labels.append(label)
            video_metrics = video["label_metrics"]
            series_by_label[label] = {
                "Precision": video_metrics["precision"],
                "Recall": video_metrics["recall"],
                "F1": video_metrics["f1_score"],
                "Accuracy": video_metrics["accuracy"],
            }

        per_video_path = output_dir / "per_video_label_metrics.png"
        save_grouped_metrics_chart(
            "Per-Video Label Metrics",
            series_by_label,
            ["Precision", "Recall", "F1", "Accuracy"],
            per_video_path,
        )
        created_paths.append(per_video_path)

        sample_counts = [video.get("evaluated_samples", 0)
                         for video in per_video]
        samples_path = output_dir / "per_video_samples.png"
        save_bar_chart("Evaluated Samples per Video", video_labels,
                       sample_counts, "Samples", samples_path)
        created_paths.append(samples_path)

    if "box_metrics" in overall:
        box_metrics = overall["box_metrics"]
        box_metric_order = ["precision", "recall", "f1_score", "accuracy"]
        box_labels = ["Precision", "Recall", "F1", "Accuracy"]
        box_values = [box_metrics[name] for name in box_metric_order]
        box_path = output_dir / "overall_box_metrics.png"
        save_bar_chart("Overall Box Metrics", box_labels,
                       box_values, "Score", box_path, rotation=0)
        created_paths.append(box_path)

    return created_paths


def video_metadata(video_path: Path) -> dict:
    fallback_duration = infer_duration_from_truth(video_path)
    if cv2 is None:
        return {
            "fps": None,
            "frame_count": None,
            "duration_s": fallback_duration,
            "width": None,
            "height": None,
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration_s = frame_count / fps if fps > 0 else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": round(duration_s or fallback_duration or 0.0, 3),
        "width": width,
        "height": height,
    }


def infer_duration_from_truth(video_path: Path) -> float | None:
    truth_path = video_path.with_name(f"truth_{video_path.stem}.json")
    if not truth_path.exists():
        return None
    try:
        truth = load_json(truth_path)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(truth, list):
        return None
    duration = sum(float(item.get("duration_s", 0.0))
                   for item in truth if isinstance(item, dict))
    return round(duration, 3) if duration > 0 else None


def make_template(dataset_dir: Path, output_path: Path, sample_every_s: float, max_samples: int | None) -> None:
    videos = sorted(dataset_dir.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No .mp4 files found in {dataset_dir}")

    template = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Fill objects for each sampled frame. Use labels from your detector when possible. "
            "bbox is optional and uses [x1, y1, x2, y2] pixel coordinates."
        ),
        "label_universe": [
            "monitor",
            "notebook",
            "pen",
            "keyboard",
            "mouse",
            "person",
            "chair",
            "table",
            "phone",
            "cup",
        ],
        "videos": [],
    }

    for video_path in videos:
        meta = video_metadata(video_path)
        if meta["frame_count"] is None:
            fps = 30.0
            if meta["duration_s"]:
                sample_count = int(meta["duration_s"] // sample_every_s) + 1
            else:
                sample_count = max_samples or 10
            frame_indices = [round(index * sample_every_s * fps)
                             for index in range(sample_count)]
            if max_samples is not None:
                frame_indices = frame_indices[:max_samples]
        else:
            fps = meta["fps"] or 30.0
            step_frames = max(1, round(sample_every_s * fps))
            frame_indices = list(range(0, meta["frame_count"], step_frames))
            if max_samples is not None:
                frame_indices = frame_indices[:max_samples]

        samples = []
        for frame_index in frame_indices:
            samples.append(
                {
                    "frame_index": frame_index,
                    "time_s": round(frame_index / fps, 3),
                    "objects": [
                        {
                            "label": "",
                            "bbox": [],
                            "notes": "",
                        }
                    ],
                }
            )

        template["videos"].append(
            {
                "video_path": str(video_path.relative_to(repo_root())).replace("\\", "/"),
                "metadata": meta,
                "samples": samples,
            }
        )

    write_json(output_path, template)


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("_", " ")


def is_filled_object(obj: dict) -> bool:
    return bool(normalize_label(obj.get("label", "")))


def object_labels(objects: list[dict]) -> set[str]:
    return {normalize_label(obj["label"]) for obj in objects if is_filled_object(obj)}


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


def has_bbox(obj: dict) -> bool:
    bbox = obj.get("bbox")
    return isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox)


def match_boxes(gt_objects: list[dict], pred_objects: list[dict], iou_threshold: float) -> tuple[int, int, int]:
    gt = [obj for obj in gt_objects if is_filled_object(obj) and has_bbox(obj)]
    pred = [obj for obj in pred_objects if is_filled_object(
        obj) and has_bbox(obj)]
    unmatched_pred = set(range(len(pred)))
    tp = 0

    for gt_obj in gt:
        best_index = None
        best_iou = 0.0
        gt_label = normalize_label(gt_obj["label"])
        for pred_index in unmatched_pred:
            pred_obj = pred[pred_index]
            if normalize_label(pred_obj["label"]) != gt_label:
                continue
            iou = bbox_iou(gt_obj["bbox"], pred_obj["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_index = pred_index
        if best_index is not None and best_iou >= iou_threshold:
            tp += 1
            unmatched_pred.remove(best_index)

    fp = len(unmatched_pred)
    fn = len(gt) - tp
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def read_frame(video_path: Path, frame_index: int):
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required to read video frames. Install opencv-python or pass --predictions.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(
            f"Could not read frame {frame_index} from {video_path}")
    return frame


def load_yolo(model_path: Path, confidence: float):
    from ultralytics import YOLO

    return YOLO(str(model_path)), confidence


def yolo_predict(model_pair, frame) -> list[dict]:
    model, confidence = model_pair
    results = model(frame, conf=confidence, verbose=False)
    predictions = []
    for result in results:
        for box in result.boxes:
            predictions.append(
                {
                    "label": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [float(x) for x in box.xyxy[0].tolist()],
                }
            )
    return predictions


def load_prediction_map(predictions_path: Path) -> dict:
    data = load_json(predictions_path)
    if isinstance(data, dict) and "videos" in data:
        videos = data["videos"]
    elif isinstance(data, list):
        videos = data
    else:
        raise ValueError(
            "Predictions JSON must be a list or an object with a 'videos' list.")

    prediction_map = {}
    for video in videos:
        video_path = video["video_path"]
        for sample in video.get("samples", []):
            key = (video_path, int(sample["frame_index"]))
            prediction_map[key] = sample.get("objects", [])
    return prediction_map


def evaluate(annotations_path: Path, predictions_path: Path | None, model_path: Path | None, confidence: float, iou_threshold: float) -> dict:
    annotations = load_json(annotations_path)
    videos = annotations.get("videos", [])
    label_universe = {normalize_label(label) for label in annotations.get(
        "label_universe", []) if label}

    prediction_map = load_prediction_map(
        predictions_path) if predictions_path else None
    model_pair = load_yolo(model_path, confidence) if model_path else None

    totals = Counter()
    per_video = []

    for video in videos:
        video_path_text = video["video_path"]
        video_path = resolve_path(video_path_text)
        video_totals = Counter()

        for sample in video.get("samples", []):
            gt_objects = [obj for obj in sample.get(
                "objects", []) if is_filled_object(obj)]
            if not gt_objects:
                continue

            frame_index = int(sample["frame_index"])
            if prediction_map is not None:
                pred_objects = prediction_map.get(
                    (video_path_text, frame_index), [])
            else:
                frame = read_frame(video_path, frame_index)
                pred_objects = yolo_predict(model_pair, frame)

            gt_labels = object_labels(gt_objects)
            pred_labels = object_labels(pred_objects)
            labels_for_accuracy = label_universe | gt_labels | pred_labels

            tp_labels = len(gt_labels & pred_labels)
            fp_labels = len(pred_labels - gt_labels)
            fn_labels = len(gt_labels - pred_labels)
            tn_labels = len(labels_for_accuracy - (gt_labels | pred_labels))

            for counter in (totals, video_totals):
                counter["samples"] += 1
                counter["label_tp"] += tp_labels
                counter["label_fp"] += fp_labels
                counter["label_fn"] += fn_labels
                counter["label_tn"] += tn_labels
                counter["exact_matches"] += int(gt_labels == pred_labels)

            if any(has_bbox(obj) for obj in gt_objects):
                box_tp, box_fp, box_fn = match_boxes(
                    gt_objects, pred_objects, iou_threshold)
                for counter in (totals, video_totals):
                    counter["box_samples"] += 1
                    counter["box_tp"] += box_tp
                    counter["box_fp"] += box_fp
                    counter["box_fn"] += box_fn

        per_video.append(metrics_from_totals(video_path_text, video_totals))

    return {
        "annotations": str(annotations_path),
        "prediction_source": str(predictions_path or model_path),
        "confidence_threshold": confidence if model_path else None,
        "iou_threshold": iou_threshold,
        "overall": metrics_from_totals("overall", totals),
        "per_video": per_video,
    }


def metrics_from_totals(name: str, totals: Counter) -> dict:
    label_metrics = prf(totals["label_tp"],
                        totals["label_fp"], totals["label_fn"])
    accuracy_denominator = totals["label_tp"] + \
        totals["label_fp"] + totals["label_fn"] + totals["label_tn"]
    label_metrics["accuracy"] = (
        (totals["label_tp"] + totals["label_tn"]) /
        accuracy_denominator if accuracy_denominator else 0.0
    )
    label_metrics["exact_match_accuracy"] = totals["exact_matches"] / \
        totals["samples"] if totals["samples"] else 0.0

    result = {
        "name": name,
        "evaluated_samples": totals["samples"],
        "label_metrics": label_metrics,
        "label_counts": {
            "tp": totals["label_tp"],
            "fp": totals["label_fp"],
            "fn": totals["label_fn"],
            "tn": totals["label_tn"],
        },
    }

    if totals["box_samples"]:
        box_metrics = prf(totals["box_tp"], totals["box_fp"], totals["box_fn"])
        denominator = totals["box_tp"] + totals["box_fp"] + totals["box_fn"]
        box_metrics["accuracy"] = totals["box_tp"] / \
            denominator if denominator else 0.0
        result["box_metrics"] = box_metrics
        result["box_counts"] = {
            "tp": totals["box_tp"],
            "fp": totals["box_fp"],
            "fn": totals["box_fn"],
        }

    return result


def print_summary(results: dict) -> None:
    overall = results["overall"]["label_metrics"]
    print("Object detection label metrics")
    print(f"  samples:   {results['overall']['evaluated_samples']}")
    print(f"  precision: {overall['precision']:.4f}")
    print(f"  recall:    {overall['recall']:.4f}")
    print(f"  f1-score:  {overall['f1_score']:.4f}")
    print(f"  accuracy:  {overall['accuracy']:.4f}")
    print(f"  exact-set: {overall['exact_match_accuracy']:.4f}")
    if "box_metrics" in results["overall"]:
        box = results["overall"]["box_metrics"]
        print("Object detection bbox metrics")
        print(f"  precision: {box['precision']:.4f}")
        print(f"  recall:    {box['recall']:.4f}")
        print(f"  f1-score:  {box['f1_score']:.4f}")
        print(f"  accuracy:  {box['accuracy']:.4f}")


def default_model_path() -> Path:
    for candidate in (repo_root() / "models" / "yolo11n.pt", repo_root() / "src" / "models" / "yolo11n.pt"):
        if candidate.exists():
            return candidate
    return repo_root() / "models" / "yolo11n.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create templates and evaluate object detection annotations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser(
        "template", help="Create a manual annotation template for videos.")
    template_parser.add_argument(
        "--dataset", default="dataset", help="Directory containing .mp4 videos.")
    template_parser.add_argument(
        "--output", default="dataset/object_detection_annotations_template.json")
    template_parser.add_argument(
        "--sample-every", type=float, default=1.0, help="Seconds between annotation samples.")
    template_parser.add_argument(
        "--max-samples", type=int, default=None, help="Optional cap per video.")

    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate detector predictions against annotations.")
    eval_parser.add_argument(
        "--annotations", required=True, help="Filled annotation JSON.")
    eval_parser.add_argument("--predictions", default=None,
                             help="Optional prediction JSON with the same schema.")
    eval_parser.add_argument("--model", default=str(default_model_path()),
                             help="YOLO model path used when --predictions is omitted.")
    eval_parser.add_argument("--confidence", type=float, default=0.25)
    eval_parser.add_argument("--iou-threshold", type=float, default=0.5)
    eval_parser.add_argument(
        "--output", default="benchmark_results/object_detection_metrics.json")
    eval_parser.add_argument(
        "--charts-dir",
        default=None,
        help="Directory for generated PNG charts. Defaults to a subfolder under benchmark_results.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "template":
            make_template(
                dataset_dir=resolve_path(args.dataset),
                output_path=resolve_path(args.output),
                sample_every_s=args.sample_every,
                max_samples=args.max_samples,
            )
            print(f"Wrote annotation template to {args.output}")
            return 0

        predictions_path = resolve_path(
            args.predictions) if args.predictions else None
        model_path = None if predictions_path else resolve_path(args.model)
        results = evaluate(
            annotations_path=resolve_path(args.annotations),
            predictions_path=predictions_path,
            model_path=model_path,
            confidence=args.confidence,
            iou_threshold=args.iou_threshold,
        )
        output_path = resolve_path(args.output)
        charts_dir = resolve_path(
            args.charts_dir) if args.charts_dir else charts_dir_for_output(output_path)
        chart_paths = create_graphs(results, charts_dir)
        results["charts"] = {
            "directory": str(charts_dir),
            "files": [str(path) for path in chart_paths],
        }
        write_json(output_path, results)
        print_summary(results)
        print(f"Wrote charts to {charts_dir}")
        print(f"Wrote full metrics to {args.output}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
