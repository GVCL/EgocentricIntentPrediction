from ultralytics import YOLO
import torch


class YoloDetector:
    def __init__(self, model_name="yolo11n.pt", confidence_threshold=0.5):
        self.device = self._get_device()
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def _get_device(self):
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def detect(self, frame):
        results = self.model(frame, device=self.device,
                             conf=self.confidence_threshold, verbose=False)

        detections = []
        # Extract bounding boxes and labels
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                label = self.model.names[int(box.cls[0])]
                detections.append({
                    "box": coords,
                    "label": label
                })

        # Return both the list of detections and the drawn-on frame
        return detections, results[0].plot()
