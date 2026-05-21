# VR Headset Multimodal Analysis

This project is a Python-based real-time multimodal analysis tool. It captures the screen of a "Meta Quest Casting" window, performs object detection using a YOLOv11 nano model, and analyzes facial data received over UDP to determine the user's emotional state. The results are then displayed in an OpenCV window.

## How it Works

The project is composed of a Python application that receives facial tracking data from a VR headset (forwarded by a Unity application) and combines it with screen-captured video of the VR session.

- **Video Analysis**: It uses a YOLO object detection model to identify objects in the VR environment.
- **Emotion Analysis**: It uses a Facial Action Coding System (FACS) based model to interpret the facial blendshape data from the headset and classify the user's emotion.
- **UDP Communication**: A Unity application (not included in this repo) is expected to capture the facial tracking data and send it over UDP to this Python application.

## Building and Running

### 1. PyTorch Installation (with CUDA)

This project uses PyTorch for deep learning. For GPU acceleration, you need to install the version of PyTorch that matches your CUDA toolkit. The user has CUDA 12.8. A compatible PyTorch version can be installed with the following command:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Setup

Once PyTorch is installed, you can install the remaining Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Running the Application

To run the application, execute the `src/main.py` script:

```bash
python src/main.py
```

The script will search for a window with the title "Meta Quest Casting". Make sure this window is active. It will also start listening for UDP data on port 5005.

## Emotion Model Training Workflow

To train your own custom emotion classification model:

1.  **Collect Data:** Run `python src/collect_data.py`. Follow the prompts to make different facial expressions. This will create a `collected_blendshape_data.csv` file in the `data` directory.
2.  **Train Model:** Run `python src/train_model.py`. This will train a Random Forest classifier and save it as `emotion_random_forest_model.joblib` in the `models` directory.

Once the model is trained, `src/emotion_classifier.py` will automatically load and use it.

## Troubleshooting

*   **Window not found:** If the script prints "Window not found. Make sure you are casting!", ensure that the "Meta Quest Casting" window is open and not minimized.
*   **No blendshape data received:** If you see "No blendshape data received" when running `collect_data.py`, ensure that your Unity application is running and sending UDP data to `127.0.0.1:5005`.

## Project Structure

The project is organized into a `src` directory containing the following modules:

*   `main.py`: The main entry point of the application.
*   `emotion_classifier.py`: A module that takes a dictionary of facial blendshape values and returns an emotion classification.
*   `udp_listener.py`: Handles receiving facial data from a UDP socket.
*   `window_capture.py`: Responsible for capturing the screen of a specific window.
*   `yolo_detector.py`: Loads the YOLO model and performs object detection.

The YOLO model (`yolo11n.pt`) is located in the `models` directory.

## Unity Integration

The communication with the Unity application is done via UDP on `127.0.0.1:5005`. The data format is a JSON string of facial blendshape values.

The following C# code snippet shows how to extract the facial features and send them over UDP from Unity:

```csharp
void ExtractFeatures()
{
    // Helper to shorten the call
    float GetWeight(OVRFaceExpressions.FaceExpression expression)
    {
        return faceExpressions[expression];
    }

    // --- BROWS ---
    currentFeatures.BrowLowererL = GetWeight(OVRFaceExpressions.FaceExpression.BrowLowererL);
    currentFeatures.BrowLowererR = GetWeight(OVRFaceExpressions.FaceExpression.BrowLowererR);
    // OVR doesn't always split Inner/Outer perfectly for all headsets, but we map what we can
    // Note: Some models define 'BrowDown' implies Lowerer. 

    // --- EYES ---
    currentFeatures.EyesClosedL = GetWeight(OVRFaceExpressions.FaceExpression.EyesClosedL);
    currentFeatures.EyesClosedR = GetWeight(OVRFaceExpressions.FaceExpression.EyesClosedR);
    currentFeatures.LidTightenerL = GetWeight(OVRFaceExpressions.FaceExpression.LidTightenerL);
    currentFeatures.LidTightenerR = GetWeight(OVRFaceExpressions.FaceExpression.LidTightenerR);
    currentFeatures.UpperLidRaiserL = GetWeight(OVRFaceExpressions.FaceExpression.UpperLidRaiserL);
    currentFeatures.UpperLidRaiserR = GetWeight(OVRFaceExpressions.FaceExpression.UpperLidRaiserR);

    // --- CHEEKS & NOSE ---
    currentFeatures.CheekRaiserL = GetWeight(OVRFaceExpressions.FaceExpression.CheekRaiserL);
    currentFeatures.CheekRaiserR = GetWeight(OVRFaceExpressions.FaceExpression.CheekRaiserR);
    currentFeatures.CheekPuffL = GetWeight(OVRFaceExpressions.FaceExpression.CheekPuffL);
    currentFeatures.CheekPuffR = GetWeight(OVRFaceExpressions.FaceExpression.CheekPuffR);
    currentFeatures.NoseWrinklerL = GetWeight(OVRFaceExpressions.FaceExpression.NoseWrinklerL);
    currentFeatures.NoseWrinklerR = GetWeight(OVRFaceExpressions.FaceExpression.NoseWrinklerR);

    // --- MOUTH ---
    currentFeatures.LipCornerPullerL = GetWeight(OVRFaceExpressions.FaceExpression.LipCornerPullerL);
    currentFeatures.LipCornerPullerR = GetWeight(OVRFaceExpressions.FaceExpression.LipCornerPullerR);
    currentFeatures.LipCornerDepressorL = GetWeight(OVRFaceExpressions.FaceExpression.LipCornerDepressorL);
    currentFeatures.LipCornerDepressorR = GetWeight(OVRFaceExpressions.FaceExpression.LipCornerDepressorR);
    currentFeatures.LipStretcherL = GetWeight(OVRFaceExpressions.FaceExpression.LipStretcherL);
    currentFeatures.LipStretcherR = GetWeight(OVRFaceExpressions.FaceExpression.LipStretcherR);
    currentFeatures.LipPuckerL = GetWeight(OVRFaceExpressions.FaceExpression.LipPuckerL);
    currentFeatures.LipPuckerR = GetWeight(OVRFaceExpressions.FaceExpression.LipPuckerR);
    currentFeatures.LipPressorL = GetWeight(OVRFaceExpressions.FaceExpression.LipPressorL);
    currentFeatures.LipPressorR = GetWeight(OVRFaceExpressions.FaceExpression.LipPressorR);

    // --- JAW ---
    currentFeatures.JawDrop = GetWeight(OVRFaceExpressions.FaceExpression.JawDrop);
    currentFeatures.JawThrust = GetWeight(OVRFaceExpressions.FaceExpression.JawThrust);
}

void SendData()
{
    try
    {
        string json = JsonUtility.ToJson(currentFeatures);
        byte[] data = Encoding.UTF8.GetBytes(json);
        udpClient.Send(data, data.Length, remoteEndPoint);
    }
    catch (System.Exception e)
    {
        Debug.LogError($"UDP Send Error: {e.Message}");
    }
}
```
