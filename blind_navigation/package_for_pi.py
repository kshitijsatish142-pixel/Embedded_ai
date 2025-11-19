"""
Package trained obstacle detection model for Raspberry Pi 4B deployment.
Creates a self-contained, plug-and-play package.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict


def package_model(
    model_dir: Path,
    output_dir: Path,
) -> None:
    """
    Package model for Pi deployment.
    
    Creates:
    - pi_inference.py (standalone inference)
    - model weights
    - metadata
    - requirements
    - README
    - example usage
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find model files
    model_weights = None
    for f in ["best_model.pt", "final_model.pt"]:
        if (model_dir / f).exists():
            model_weights = model_dir / f
            break
    
    metadata_file = model_dir / "model_metadata.json"
    
    if not model_weights or not metadata_file.exists():
        raise FileNotFoundError(
            f"Model files not found in {model_dir}\n"
            f"Expected: best_model.pt or final_model.pt, and model_metadata.json"
        )
    
    print(f"📦 Packaging model from {model_dir}...")
    
    # Copy model files
    shutil.copy2(model_weights, output_dir / "model_weights.pt")
    shutil.copy2(metadata_file, output_dir / "model_metadata.json")
    print(f"✅ Copied model files")
    
    # Copy inference module
    inference_file = Path(__file__).parent / "pi_inference.py"
    if inference_file.exists():
        shutil.copy2(inference_file, output_dir / "pi_inference.py")
        print(f"✅ Copied inference module")
    
    # Create requirements
    requirements = """# Requirements for Raspberry Pi 4B
# Install with: pip install -r requirements_pi.txt

torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
numpy>=1.20.0
Pillow>=8.0.0
"""
    (output_dir / "requirements_pi.txt").write_text(requirements)
    print(f"✅ Created requirements_pi.txt")
    
    # Create example
    example = '''"""
Example: Use obstacle detector on Raspberry Pi
"""

from pi_inference import ObstacleDetector
import cv2

# Initialize detector
detector = ObstacleDetector(".")

# Open camera (0 for Pi Camera, 1+ for USB)
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    # Detect obstacles
    detections = detector.detect(frame)
    
    # Announce critical obstacles
    detector.announce(detections, audio_enabled=False)
    
    # Draw on frame (optional)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 0, 255) if det["priority"] == 1 else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['distance']:.1f}m"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow("Obstacle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
'''
    (output_dir / "example_usage.py").write_text(example)
    print(f"✅ Created example_usage.py")
    
    # Create README
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    readme = f"""# Obstacle Detection Model for Raspberry Pi 4B

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

- Classes: {', '.join(metadata.get('classes', []))}
- Input size: {metadata.get('input_size', 'Unknown')}
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
"""
    
    (output_dir / "README_PI.md").write_text(readme)
    print(f"✅ Created README_PI.md")
    
    print(f"\n✅ Package created at: {output_dir}")
    print(f"   Copy this entire folder to your Raspberry Pi")
    print(f"   Then run: python3 pi_inference.py {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Package model for Raspberry Pi")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("blind_navigation/models/obstacle_detector"),
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("blind_navigation/models/pi_package"),
        help="Output directory for Pi package",
    )
    
    args = parser.parse_args()
    package_model(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()

