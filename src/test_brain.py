import os
import time
from dotenv import load_dotenv
from brain import Brain

def run_diagnostics():
    print("--- BRAIN DIAGNOSTICS ---")
    
    # Check Environment Variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not loaded! Check your .env file.")
        return
    else:
        print(f"API Key found: {api_key[:5]}...{api_key[-4:]}")

    # Initialize Brain
    print("Initializing Brain module...")
    brain = Brain()
    
    if brain.client is None:
        print("ERROR: Brain failed to initialize the API client.")
        return
    else:
        print("API Client initialized successfully.")

    # Inject Dummy State Vectors
    print("\nInjecting dummy state vectors into history...")
    dummy_states = [
        ("Neutral", [{"label": "Meta Quest Pro", "box": [0,0,0,0]}], "Meta Quest Pro"),
        ("Concentrated", [{"label": "Keyboard", "box": [0,0,0,0]}], "Keyboard"),
        ("Frustrated", [{"label": "Unity Editor", "box": [0,0,0,0]}], "Console Error")
    ]
    
    for state in dummy_states:
        brain.update_state(*state)
        print(f"   -> Added state: {state}")
        time.sleep(0.1)

    # Force a Prediction
    print("\nForcing a synchronous API call (Ignoring background threads)...")
    print("Waiting for response...")
    
    try:
        # We call the internal method directly so we can see any hidden errors
        brain._predict_intent_stream()
        print("\nAPI Call completed.")
    except Exception as e:
        print(f"\nEXCEPTION CAUGHT: {e}")

if __name__ == "__main__":
    run_diagnostics()