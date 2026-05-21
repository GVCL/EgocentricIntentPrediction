import cv2
import threading
from udp_listener import UdpListener
from window_capture import WindowCapture
from yolo_detector import YoloDetector
from emotion_classifier import classify_emotion
from brain import Brain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# CONFIGURATION
TARGET_WINDOW_TITLE = "Meta Quest Casting"
MODEL_NAME = "models/yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.5
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# We use this to toggle the background LLM thread
continuous_llm_active = False

def main():
    global continuous_llm_active
    udp_listener = UdpListener(ip=UDP_IP, port=UDP_PORT)
    capture = WindowCapture(TARGET_WINDOW_TITLE)
    detector = YoloDetector(model_name=MODEL_NAME, confidence_threshold=CONFIDENCE_THRESHOLD)
    brain = Brain()

    udp_listener.start()

    cv2.namedWindow("Multimodal VR Analysis")
    cv2.moveWindow("Multimodal VR Analysis", 10, 10)

    print("System Ready: Emotion + Focus + YOLO Vision")
    print("PRESS 's' IN THE VIDEO WINDOW TO TOGGLE CONTINUOUS LLM PREDICTION")

    while True:
        frame = capture.grab_frame()
        if frame is None:
            break

        # YOLO Object Detection
        detections, annotated_frame = detector.detect(frame)

        # 2Focus Logic
        height, width, _ = frame.shape
        gaze_x, gaze_y = width // 2, height // 2

        # Collision Check
        focused_object = "Nothing"
        for obj in detections:
            x1, y1, x2, y2 = obj["box"]
            if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
                focused_object = obj["label"]
                cv2.rectangle(annotated_frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0, 255, 0), 4)
                break

        # Get Emotion Data
        telemetry = udp_listener.get_latest_data()
        
        # Check if the matrix is exactly 63 elements long
        if telemetry is None or len(telemetry['matrix']) != 63:
            emotion_label, confidence, rot = "NO DATA", 0.0, [0, 0, 0]
        else:
            emotion_label, confidence = classify_emotion(telemetry['matrix'])
            rot = telemetry['focus']

        # Constantly update the Brain's state
        brain.update_state(emotion_label, detections, focused_object)

        # VISUALIZATION
        cv2.rectangle(annotated_frame, (10, 10), (480, 200), (0, 0, 0), -1)

        color = (255, 255, 255)
        if "HAPPINESS" in emotion_label:
            color = (0, 255, 255)
        elif "ANGER" in emotion_label:
            color = (0, 0, 255)

        cv2.putText(annotated_frame, f"EMOTION: {emotion_label}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(annotated_frame, f"LOOKING AT: {focused_object.upper()}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"CONFIDENCE: {confidence:.1%}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(annotated_frame, f"HEAD FOCUS: P:{rot[0]:.1f} Y:{rot[1]:.1f}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.drawMarker(annotated_frame, (gaze_x, gaze_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # Display UI Indicator for LLM Status
        llm_status_text = "LLM: ACTIVE" if continuous_llm_active else "LLM: PAUSED (Press 's')"
        llm_status_color = (0, 255, 0) if continuous_llm_active else (0, 0, 255)
        cv2.putText(annotated_frame, llm_status_text, (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, llm_status_color, 2)

        cv2.imshow("Multimodal VR Analysis", annotated_frame)

        # Keyboard Input Handling
        key = cv2.waitKey(1)
        if key == ord('q'):
            brain.stop_continuous_prediction() # Cleanly stop the thread
            break
        elif key == ord('s'):
            continuous_llm_active = not continuous_llm_active
            if continuous_llm_active:
                print("\n\n>>> STARTING CONTINUOUS LLM STREAM 🟢 <<<")
                brain.start_continuous_prediction()
            else:
                print("\n\n>>> PAUSING CONTINUOUS LLM STREAM 🔴 <<<")
                brain.stop_continuous_prediction()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()