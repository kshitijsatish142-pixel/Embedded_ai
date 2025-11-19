# Blind Navigation - Obstacle Detection System

A computer vision system for obstacle detection to assist blind and visually impaired individuals in navigation. Optimized for Raspberry Pi 4B with 5MP camera.

## 🎯 Overview

This system detects obstacles and navigation aids in real-time using a camera, providing audio/visual feedback to help blind users navigate safely.

## 📋 Ideal Objects to Detect

See [OBSTACLE_PRIORITIES.md](OBSTACLE_PRIORITIES.md) for complete list.

### Critical (Priority 1)
- **Vehicles**: Cars, buses, trucks, motorcycles
- **People**: Pedestrians, cyclists
- **Stairs**: Upward/downward steps
- **Doors**: Open/closed doors
- **Walls & Barriers**: Obstacles to avoid

### Important (Priority 2)
- Furniture, poles, curbs, ground obstacles, overhead hazards

### Navigation Aids (Priority 3)
- Crosswalks, traffic signs, elevators, handrails

## 🚀 Quick Start

### 1. Collect Training Data

**Option A: COCO Dataset (Recommended for person & vehicle)**
```bash
# Show instructions
python3 blind_navigation/quick_start_data.py --instructions

# Download COCO from: https://cocodataset.org/#download
# Then extract:
python3 blind_navigation/download_coco_data.py extract \
    --coco-dir ./coco_dataset \
    --output-dir ./blind_navigation/dataset/train \
    --classes person vehicle \
    --max-images 500
```

**Option B: Quick Setup (All methods)**
```bash
# Show all data collection options
python3 blind_navigation/quick_start_data.py --instructions

# Create dataset structure
python3 blind_navigation/quick_start_data.py --setup --output-dir ./dataset
```

**Option C: Manual Collection**
- Use Google Images, Unsplash, or your own photos
- See `quick_start_data.py --instructions` for details

### 2. Annotate Images

```bash
# Install annotation tool
pip install labelImg

# Annotate images
labelImg dataset/train/person/

# Or create template
python3 blind_navigation/prepare_data.py template --output annotations_template.json
```

### 3. Organize Dataset

```json
{
  "images": [
    {
      "file": "image1.jpg",
      "objects": [
        {
          "class": "person",
          "bbox": [100, 150, 80, 200]
        },
        {
          "class": "vehicle",
          "bbox": [300, 200, 150, 100]
        }
      ]
    }
  ]
}
```

**Bbox format**: `[x, y, width, height]` in pixels (top-left corner origin)

### 4. Train Model

```bash
python3 blind_navigation/train_obstacle_detector.py \
    --data-dir ./dataset \
    --output-dir ./models/obstacle_detector \
    --epochs 50 \
    --batch-size 16 \
    --device cuda  # or cpu
```

### 5. Deploy to Raspberry Pi

```bash
# Copy model folder to Pi
scp -r ./models/obstacle_detector pi@raspberrypi.local:/home/pi/models/

# On Pi, install dependencies
pip install torch torchvision opencv-python pillow numpy

# Run detection
python3 blind_navigation/pi_inference.py \
    --model-dir /home/pi/models/obstacle_detector \
    --camera 0
```

## 📁 Project Structure

```
blind_navigation/
├── OBSTACLE_PRIORITIES.md      # Object detection priorities
├── train_obstacle_detector.py  # Training script
├── pi_inference.py             # Pi inference script
├── prepare_data.py             # Data preparation utilities
├── README.md                   # This file
└── models/                     # Trained models
    └── obstacle_detector/
        ├── best_model.pt
        ├── model_metadata.json
        └── ...
```

## 🎓 Training Workflow

### Step 1: Collect Data

**Minimum requirements:**
- Critical objects: 500+ images per class
- Important objects: 300+ images per class
- Navigation aids: 200+ images per class

**Data diversity:**
- Different lighting (day, night, dusk)
- Different weather (clear, rain, fog)
- Different environments (indoor, outdoor, urban)
- Different angles and distances

### Step 2: Annotate Images

Use annotation tools:
- **LabelImg**: https://github.com/tzutalin/labelImg
- **CVAT**: https://cvat.org/
- **Roboflow**: https://roboflow.com/

Export as JSON with format:
```json
{
  "file": "image.jpg",
  "objects": [
    {"class": "person", "bbox": [x, y, w, h]}
  ]
}
```

### Step 3: Organize Dataset

```bash
python3 blind_navigation/prepare_data.py organize \
    --source ./all_images \
    --output ./dataset \
    --train-ratio 0.8
```

This creates:
```
dataset/
├── train/
│   ├── annotations.json
│   ├── image1.jpg
│   └── ...
└── val/
    ├── annotations.json
    ├── image1.jpg
    └── ...
```

### Step 4: Train

```bash
python3 blind_navigation/train_obstacle_detector.py \
    --data-dir ./dataset \
    --output-dir ./models/obstacle_detector \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001
```

**Training tips:**
- Start with 5-10 critical classes
- Use data augmentation (enabled by default)
- Monitor validation loss
- Save best model automatically

## 📱 Using on Raspberry Pi

### Setup

```bash
# Install dependencies
pip install torch torchvision opencv-python pillow numpy

# For audio announcements (optional)
pip install pyttsx3
```

### Run Detection

```bash
# Basic usage
python3 blind_navigation/pi_inference.py \
    --model-dir /home/pi/models/obstacle_detector

# With USB camera
python3 blind_navigation/pi_inference.py \
    --model-dir /home/pi/models/obstacle_detector \
    --camera 1

# Adjust confidence threshold
python3 blind_navigation/pi_inference.py \
    --model-dir /home/pi/models/obstacle_detector \
    --confidence 0.6
```

### Integration Example

```python
from blind_navigation.pi_inference import ObstacleDetector
import cv2

# Initialize
detector = ObstacleDetector("/home/pi/models/obstacle_detector")

# Capture frame
camera = cv2.VideoCapture(0)
ret, frame = camera.read()

# Detect
detections = detector.detect(frame)

# Announce critical obstacles
detector.announce(detections, audio_enabled=True)
```

## 🎯 Model Architecture

- **Backbone**: MobileNetV3-Small (lightweight, fast)
- **Detection Head**: Custom lightweight head
- **Input Size**: 640x640 pixels
- **Output**: 5 bounding boxes per image
- **Classes**: Configurable (start with 5-10 critical classes)

**Performance on Pi 4B:**
- Inference: ~50-100ms per frame
- FPS: 10-20 FPS (depending on resolution)
- Memory: ~200-300 MB

## 📊 Performance Metrics

Track during training:
- **mAP (mean Average Precision)**: Overall detection accuracy
- **Precision**: Correct detections / Total detections
- **Recall**: Detected objects / Total objects
- **FPS**: Frames per second on target device

**Target metrics:**
- mAP@0.5: >0.75 for critical objects
- Precision: >0.85 for critical objects
- FPS: >10 on Pi 4B

## 🔧 Advanced Usage

### Custom Classes

Edit `annotations.json` to add new classes:

```json
{
  "images": [
    {
      "file": "image.jpg",
      "objects": [
        {"class": "your_new_class", "bbox": [x, y, w, h]}
      ]
    }
  ]
}
```

The model will automatically detect all classes in your annotations.

### Distance Estimation

The system estimates distance based on bounding box size. Calibrate by:

1. Measure real distances
2. Record bbox sizes at those distances
3. Update `_estimate_distance()` in `pi_inference.py`

### Audio Announcements

Enable text-to-speech:

```python
detector.announce(detections, audio_enabled=True)
```

Requires: `pip install pyttsx3`

## 🐛 Troubleshooting

### Training Issues

**"Annotations file not found"**
- Create `annotations.json` in train/val folders
- Use `prepare_data.py template` to create template

**"No images found"**
- Check image file extensions (.jpg, .png, etc.)
- Verify images are in correct folders

**"CUDA out of memory"**
- Reduce batch size: `--batch-size 8`
- Use CPU: `--device cpu`

### Pi Deployment Issues

**"Model weights not found"**
- Copy entire model folder to Pi
- Check file permissions

**"Camera not opening"**
- For Pi Camera: Use `picamera2` library
- For USB: Check device index (`--camera 1`, `--camera 2`, etc.)

**"Too slow on Pi"**
- Reduce input size in training (e.g., 416x416)
- Use smaller batch size
- Close other applications

### Detection Issues

**"No detections"**
- Lower confidence threshold: `--confidence 0.3`
- Check if objects are in training data
- Verify model was trained properly

**"False positives"**
- Increase confidence threshold: `--confidence 0.7`
- Add more negative examples to training
- Train for more epochs

## 📚 Resources

### Datasets
- **COCO**: https://cocodataset.org/ (general objects)
- **Open Images**: https://storage.googleapis.com/openimages/web/index.html
- **Custom**: Collect your own for specific use cases

### Annotation Tools
- **LabelImg**: https://github.com/tzutalin/labelImg
- **CVAT**: https://cvat.org/
- **Roboflow**: https://roboflow.com/

### Model Optimization
- **TensorRT**: For NVIDIA devices
- **ONNX**: Cross-platform optimization
- **Quantization**: Reduce model size

## 🎯 Next Steps

1. **Start Small**: Train with 5 critical classes first
2. **Collect Data**: Gather diverse images of obstacles
3. **Iterate**: Improve model with more data
4. **Deploy**: Test on Pi with real camera
5. **Enhance**: Add more classes, improve accuracy

## 📝 Notes

- Model is optimized for Pi 4B (CPU inference)
- Real-time performance: 10-20 FPS
- Memory efficient: ~200-300 MB
- Supports both Pi Camera and USB cameras
- Audio announcements optional (requires pyttsx3)

## 🤝 Contributing

Feel free to:
- Add new object classes
- Improve distance estimation
- Optimize for better performance
- Add new features

---

**Ready to start?** Begin with [OBSTACLE_PRIORITIES.md](OBSTACLE_PRIORITIES.md) to understand what to detect!

