import os
from google import genai
from collections import deque
from pathlib import Path

HISTORY_SIZE = 10 # Last 10 state vectors

script_dir = Path(__file__).parent

file_path = script_dir.parent / "data" / "history.txt"

os.makedirs(file_path.parent, exist_ok=True)

with open(file_path, 'w') as fh:
    string_to_write = f"f"
    fh.write(string_to_write)