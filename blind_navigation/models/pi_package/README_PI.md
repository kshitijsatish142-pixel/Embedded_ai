# Obstacle Detection Model for Raspberry Pi 4B

## Quick Start

1. **Copy this entire folder to your Raspberry Pi**

2. **Install dependencies:**
   ```bash
   pip install -r requirements_pi.txt
   ```

3. **Test the model:**
   ```bash
   python3 pi_inference.py .
   ```

4. **Use in your code:**
   ```python
   from pi_inference import ObstacleDetector
   
   detector = ObstacleDetector(".")
   detections = detector.detect(frame)
   ```

## Model Information

- Classes: person, vehicle
- Input size: [640, 640]
- Device: CPU (optimized for Pi 4B)

## Files Included

- `pi_inference.py` - Standalone inference module
- `model_weights.pt` - Trained model weights
- `model_metadata.json` - Model configuration
- `requirements_pi.txt` - Python dependencies
- `example_usage.py` - Usage example

## Camera Setup

**Pi Camera v2 (5MP):**
```python
import picamera2
camera = picamera2.Picamera2()
camera.start()
frame = camera.capture_array()
```

**USB Camera:**
```python
import cv2
camera = cv2.VideoCapture(0)
ret, frame = camera.read()
```

## Performance

- Inference: ~50-100ms per frame
- FPS: 10-20 FPS on Pi 4B
- Memory: ~200-300 MB

## Troubleshooting

**"Model weights not found"**
- Make sure model_weights.pt is in the same directory

**"Module not found: torch"**
- Install: `pip install -r requirements_pi.txt`

**"CUDA not available" warning**
- Normal on Pi - model uses CPU automatically

## Notes

- Model is self-contained - no training code needed
- Works offline once loaded
- Minimal dependencies
- CPU-only (no GPU needed)
