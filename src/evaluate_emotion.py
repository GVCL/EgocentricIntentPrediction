import json
import csv
import sys
from pathlib import Path
from collections import Counter

try:
    from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
except ImportError:
    print("Error: scikit-learn is required. Install it with `pip install scikit-learn`")
    sys.exit(1)

# Import your classifier
try:
    import emotion_classifier
except ImportError:
    print("Error: Could not import emotion_classifier.py. Ensure it is in the same folder.")
    sys.exit(1)

# --- Config ---
CAPTURE_DIR = Path(__file__).resolve().parent.parent / "dataset"

VIDEO_IDS = [
    "com.oculus.vrshell-20260430-221242-0",
    "com.oculus.vrshell-20260430-221558-0",
    "com.oculus.vrshell-20260430-221840-0",
    "com.oculus.vrshell-20260430-222034-0",
]

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv(path: Path):
    """Loads the FACS weights CSV and returns a list of rows (as floats)."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if not row or not row[0].strip():
                continue
            data.append([float(val) for val in row])
    return data

def main():
    print("🔍 Initializing Emotion Classifier Evaluation...")
    
    y_true_frame = []
    y_pred_frame = []
    
    y_true_clip = []
    y_pred_clip = []

    total_clips = 0
    missing_data_clips = 0

    for vid in VIDEO_IDS:
        json_path = CAPTURE_DIR / f"truth_{vid}.json"
        csv_path = CAPTURE_DIR / f"{vid}.csv"

        if not json_path.exists() or not csv_path.exists():
            print(f"[!] Missing data for {vid}, skipping...")
            continue

        annotations = load_json(json_path)
        csv_data = load_csv(csv_path)

        current_time_s = 0.0

        for ann in annotations:
            duration_s = float(ann["duration_s"])
            gt_emotion = ann.get("emotion", "Unknown").upper()
            
            # Calculate the time window for this action in milliseconds
            start_ms = current_time_s * 1000
            end_ms = (current_time_s + duration_s) * 1000
            
            current_time_s += duration_s  # Advance the timeline
            total_clips += 1

            # Extract all facial frames that fall within this action's time window
            clip_frames = [row for row in csv_data if start_ms <= row[0] < end_ms]

            if not clip_frames:
                missing_data_clips += 1
                continue

            clip_predictions = []

            for frame in clip_frames:
                # Extract the 63 face weights (skip the timestamp at index 0)
                face_weights = frame[1:64]
                
                # Run Inference
                pred_emotion, conf = emotion_classifier.classify_emotion(face_weights)
                
                clip_predictions.append(pred_emotion)
                
                # Log for frame-level evaluation
                y_true_frame.append(gt_emotion)
                y_pred_frame.append(pred_emotion)

            # --- Clip-Level Logic (Majority Vote) ---
            # What emotion was detected most often during this specific action?
            if clip_predictions:
                majority_emotion = Counter(clip_predictions).most_common(1)[0][0]
                y_true_clip.append(gt_emotion)
                y_pred_clip.append(majority_emotion)

    print("\n" + "="*60)
    print("🎬 CLIP-LEVEL EVALUATION (Majority Vote per Action)")
    print("="*60)
    print(f"Total Clips Evaluated: {len(y_true_clip)}")
    if missing_data_clips > 0:
        print(f"Clips skipped (no CSV data in time window): {missing_data_clips}")
    print("-" * 60)
    print(classification_report(y_true_clip, y_pred_clip, zero_division=0))
    
    print("\n" + "="*60)
    print("🎞️ FRAME-LEVEL EVALUATION (Every individual millisecond snapshot)")
    print("="*60)
    print(f"Total Frames Evaluated: {len(y_true_frame)}")
    print("-" * 60)
    print(classification_report(y_true_frame, y_pred_frame, zero_division=0))
    
    print("\n" + "="*60)
    print("🔍 CONFUSION MATRIX (What is it mistaking Neutral for?)")
    print("="*60)
    import pandas as pd
    labels = sorted(list(set(y_true_frame + y_pred_frame)))
    cm = confusion_matrix(y_true_frame, y_pred_frame, labels=labels)
    print(pd.DataFrame(cm, index=[f"True {x}" for x in labels], columns=[f"Pred {x}" for x in labels]))

if __name__ == "__main__":
    main()