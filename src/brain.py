"""
brain.py — Modified for Ollama (local) + optional Gemini fallback.

Drop-in replacement for the original brain.py used in main.py.
Switch backend via BRAIN_BACKEND env var:
  BRAIN_BACKEND=ollama  (default)
  BRAIN_BACKEND=gemini
  BRAIN_BACKEND=both    (run both in parallel, print both predictions)
"""

import os
import threading
import time
import json
from pathlib import Path
from collections import deque

import requests

# CONFIG
HISTORY_SIZE    = 10
PREDICT_EVERY_S = 4.5          # seconds between predictions

OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")   # override in .env
OLLAMA_TIMEOUT  = 60

BRAIN_BACKEND   = os.getenv("BRAIN_BACKEND", "ollama").lower()

script_dir = Path(__file__).parent
file_path  = script_dir.parent / "data" / "history.json"

# Ego4D-style ICL examples
_ICL_EXAMPLES = """
Examples of Intent Prediction based on State Vectors [Emotion, Detected Objects, Gaze Focus]:
- Vector [Concentrated, [Screwdriver, Bolt], Screwdriver] -> Intent: Preparing to tighten a loose fastener.
- Vector [Frustrated, [Wrench, Bolt], Bolt] -> Intent: Attempting to loosen a stuck bolt.
- Vector [Curious, [Circuit Board, Soldering Iron], Circuit Board] -> Intent: Inspecting the soldering joints on a PCB.
- Vector [Focused, [Hammer, Nail], Nail] -> Intent: Aligning a nail before driving it into the surface.
- Vector [Surprised, [Multimeter, Battery], Multimeter Screen] -> Intent: Checking why a battery is reading low voltage.
"""


def _build_prompt(history_snapshot: list, icl: str) -> str:
    history_str = "\n".join([
        f"- Vector [{s['emotion']}, {s['objects_in_view']}, {s['gaze_target']}]"
        for s in history_snapshot
    ])
    return f"""You are the "Brain" Architecture, a multimodal fusion engine for a Mixed Reality system.
Your goal is to provide real-time intent prediction based on environmental and user data.

Core Reasoning Constraints:
1. Causal Mapping (Ego4D Grounding): Use explicit relationships derived from Ego4D metadata.
2. Sequential Dependency: Analyze the provided rolling window of previous state vectors.
3. Knowledge Graph Integration: Relate the user to specific tools in their immediate field of view.

{icl}

Recent State History (S_t-9 to S_t):
{history_str}

Output Format: Generate a single-sentence prediction of the user's immediate intent. Be natural and direct.
Intent:"""


# Ollama predictor 
def _predict_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str | None:
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  True,
        "options": {"temperature": 0.2, "num_predict": 80},
    }
    try:
        print(f"\n\n--- [PREDICTION (Ollama/{model}) @ {time.strftime('%H:%M:%S')}] ---")
        full_text = ""
        with requests.post(OLLAMA_URL, json=payload, stream=True,
                           timeout=OLLAMA_TIMEOUT) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                full_text += token
                if chunk.get("done", False):
                    break
        print("\n-------------------------------------")
        return full_text.strip()
    except requests.exceptions.ConnectionError:
        print(f"\n[!] Ollama not reachable at {OLLAMA_URL}. Is it running? (ollama serve)")
        return None
    except Exception as e:
        print(f"\n[!] Ollama error: {e}")
        return None


#  Gemini predictor
def _predict_gemini(prompt: str, client) -> str | None:
    try:
        response = client.models.generate_content_stream(
            model="gemma-4-26b-a4b-it",
            contents=prompt,
        )
        print(f"\n\n--- [PREDICTION (Gemini) @ {time.strftime('%H:%M:%S')}] ---")
        has_content = False
        for chunk in response:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                has_content = True
        if not has_content:
            res = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=prompt,
            )
            print(res.text)
        print("\n-------------------------------------")
        return None
    except Exception as e:
        print(f"\n[!] Gemini error: {e}")
        return None


#  Brain class 
class Brain:
    def __init__(self, api_key: str | None = None):
        os.makedirs(file_path.parent, exist_ok=True)

        # Gemini client
        self._gemini_client = None
        if BRAIN_BACKEND in ("gemini", "both"):
            if api_key is None:
                api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    from google import genai
                    self._gemini_client = genai.Client(api_key=api_key)
                except ImportError:
                    print("Warning: google-genai not installed; Gemini disabled.")
            else:
                print("Warning: GEMINI_API_KEY not set; Gemini disabled.")

        self.history      = deque(maxlen=HISTORY_SIZE)
        self.history_lock = threading.Lock()

        self.is_predicting     = False
        self.prediction_thread = None

    #  Public API
    def update_state(self, emotion: str, objects, gaze: str):
        """Thread-safe state update + JSON flush."""
        state_vector = {
            "timestamp":       time.strftime("%H:%M:%S"),
            "emotion":         emotion,
            "objects_in_view": objects,
            "gaze_target":     gaze,
        }
        with self.history_lock:
            self.history.append(state_vector)
            with open(file_path, "w") as fh:
                json.dump(list(self.history), fh, indent=4, default=str)

    def start_continuous_prediction(self):
        if not self.is_predicting:
            self.is_predicting    = True
            self.prediction_thread = threading.Thread(
                target=self._prediction_loop, daemon=True
            )
            self.prediction_thread.start()

    def stop_continuous_prediction(self):
        self.is_predicting = False
        if self.prediction_thread:
            self.prediction_thread.join(timeout=1.0)

    # Internal 
    def _predict_intent_stream(self):
        with self.history_lock:
            if not self.history:
                return
            snapshot = list(self.history)

        prompt = _build_prompt(snapshot, _ICL_EXAMPLES)

        if BRAIN_BACKEND == "ollama":
            _predict_ollama(prompt)

        elif BRAIN_BACKEND == "gemini":
            if self._gemini_client:
                _predict_gemini(prompt, self._gemini_client)

        elif BRAIN_BACKEND == "both":
            if self._gemini_client:
                t = threading.Thread(
                    target=_predict_gemini,
                    args=(prompt, self._gemini_client),
                    daemon=True,
                )
                t.start()
            _predict_ollama(prompt)

    def _prediction_loop(self):
        while self.is_predicting:
            self._predict_intent_stream()
            time.sleep(PREDICT_EVERY_S)
