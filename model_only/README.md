# Obstacle Detection Model for Raspberry Pi 4B

Plug-and-play obstacle detection model for blind navigation.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements_pi.txt
   ```

2. **Run detection:**
   ```bash
   python3 pi_inference.py .
   ```

3. **Use in your code:**
   ```python
   from pi_inference import ObstacleDetector
   import cv2
   
   detector = ObstacleDetector(".")
   camera = cv2.VideoCapture(0)
   ret, frame = camera.read()
   detections = detector.detect(frame)
   detector.announce(detections)
   ```

## Model Info

- **Classes**: person, vehicle
- **Input**: 640x640 images
- **Performance**: 10-20 FPS on Pi 4B
- **Memory**: ~200-300 MB

## Files

- `pi_inference.py` - Standalone inference (no dependencies on training code)
- `model_weights.pt` - Trained model (4.3 MB)
- `model_metadata.json` - Model configuration
- `requirements_pi.txt` - Dependencies
- `example_usage.py` - Usage example

## Camera Setup

**Pi Camera:**
```python
from picamera2 import Picamera2
camera = Picamera2()
camera.start()
frame = camera.capture_array()
```

**USB Camera:**
```python
import cv2
camera = cv2.VideoCapture(0)
ret, frame = camera.read()
```

That's it! Copy this folder to your Pi and run.

