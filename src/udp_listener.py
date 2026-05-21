import socket
import json
import threading
import numpy as np
import time

class UdpListener:
    def __init__(self, ip="127.0.0.1", port=5005):
        self.ip = ip
        self.port = port
        # We now store a dictionary containing the matrix and focus data
        self.latest_data = {
            "matrix": np.zeros(63), 
            "focus": [0.0, 0.0, 0.0]
        }
        self.lock = threading.Lock() # Added lock for thread safety
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.setblocking(False)
        self.running = True

    def start(self):
        """Starts the listening loop in a background thread"""
        thread = threading.Thread(target=self.listen_for_unity_data, daemon=True)
        thread.start()

    def listen_for_unity_data(self):
        """Background thread to receive Face Data from Unity via UDP"""
        print(f"👂 Listening for Matrix Data on {self.ip}:{self.port}...")
        while self.running:
            try:
                # Buffer increased to 8192 to safely handle the 63-float JSON string
                data, _ = self.sock.recvfrom(8192)
                json_str = data.decode('utf-8')
                raw_dict = json.loads(json_str)
                
                # Extract the 63 weights and convert to NumPy Matrix
                # Inside listen_for_unity_data, after raw_dict = json.loads(json_str)
                face_weights = np.array(raw_dict.get("FaceWeights", []), dtype=np.float32)

                # DEBUG: Check the length
                if len(face_weights) != 63:
                    print(f"⚠️ Warning: Received {len(face_weights)} weights, but need 63!")
                
                # Extract Head Focus (Rotation)
                head_rot = [
                    raw_dict.get("HeadRotX", 0.0),
                    raw_dict.get("HeadRotY", 0.0),
                    raw_dict.get("HeadRotZ", 0.0)
                ]

                with self.lock:
                    self.latest_data["matrix"] = face_weights
                    self.latest_data["focus"] = head_rot

            except BlockingIOError:
                time.sleep(0.001)
            except Exception as e:
                print(f"UDP Error: {e}")

    def get_latest_data(self):
        """Returns the latest matrix and focus data"""
        with self.lock:
            return self.latest_data

    def stop(self):
        self.running = False
        self.sock.close()