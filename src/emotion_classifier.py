import pickle
import numpy as np
import os
import warnings

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

MODEL_PATH = os.path.join(MODELS_DIR, "xgb_custom_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
INDICES_PATH = os.path.join(MODELS_DIR, "top_feature_indices.pkl")

_model = None
_encoder = None
_top_indices = None

def _load_pickle(file_path, description):
    """Helper function to safely load a pickle file and handle logging."""
    if not os.path.exists(file_path):
        print(f"File not found: {description} at {file_path}")
        return None
        
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {description}: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading {description} from {file_path}: {e}")
        return None

def load_artifacts():
    global _model, _encoder, _top_indices
    
    _model = _load_pickle(MODEL_PATH, "Emotion Model")
    _encoder = _load_pickle(ENCODER_PATH, "Label Encoder")
    _top_indices = _load_pickle(INDICES_PATH, "Feature Indices")
    
    if _top_indices is not None:
        print(f"Feature Indices configured: Kept {len(_top_indices)} features.")

# Load immediately on import
load_artifacts()

def classify_emotion(face_data):
    """
    Input: Can be a dictionary or a numpy array of 63 weights from Unity
    Output: Detected Emotion String (e.g., 'HOSTILE') and Confidence Float
    """
    if _model is None or _encoder is None or _top_indices is None:
        return "SYSTEM UNINITIALIZED", 0.0

    # Handle Input (Extract values if dict, convert to 1D array)
    if isinstance(face_data, dict):
        input_matrix = np.array(list(face_data.values())).reshape(1, -1)
    else:
        input_matrix = np.array(face_data).reshape(1, -1)

    # Validate Incoming Unity Data
    if input_matrix.shape[1] != 63:
        return f"INVALID DATA: Expected 63, got {input_matrix.shape[1]}", 0.0

    # Apply the 22-Feature Filter
    # We slice out only the columns our model was trained on
    filtered_matrix = input_matrix[:, _top_indices]

    # Predict using XGBoost
    try:
        class_id = _model.predict(filtered_matrix)[0]
        
        # Get Confidence Score
        probs = _model.predict_proba(filtered_matrix)[0]
        confidence = float(np.max(probs))

        # Decode Label (e.g., turns class '2' into 'Hostile')
        emotion_label = _encoder.inverse_transform([class_id])[0]
        
        return str(emotion_label).upper(), confidence

    except Exception as e:
        print(f"Prediction Error: {e}")
        return "ERROR", 0.0

if __name__ == "__main__":
    print("\n--- Running Quick Inference Test ---")
    # Simulate a dummy packet from Unity (63 zeros)
    dummy_unity_data = np.zeros(63)
    emotion, conf = classify_emotion(dummy_unity_data)
    print(f"Test Prediction: {emotion} (Confidence: {conf:.2f})")