# Quick Start - Obstacle Detection for Blind Navigation

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install torch torchvision opencv-python pillow numpy
```

### 2. Prepare Your Data

```bash
# Create dataset structure
mkdir -p dataset/train dataset/val

# Copy your images
cp your_images/*.jpg dataset/train/

# Create annotations.json (see format below)
```

### 3. Annotate Images

Create `dataset/train/annotations.json`:

```json
{
  "images": [
    {
      "file": "image1.jpg",
      "objects": [
        {"class": "person", "bbox": [100, 150, 80, 200]},
        {"class": "vehicle", "bbox": [300, 200, 150, 100]}
      ]
    }
  ]
}
```

**Bbox format**: `[x, y, width, height]` in pixels

### 4. Train Model

```bash
python3 blind_navigation/train_obstacle_detector.py \
    --data-dir ./dataset \
    --output-dir ./models/obstacle_detector \
    --epochs 30
```

### 5. Test on Pi

```bash
# Copy to Pi
scp -r ./models/obstacle_detector pi@raspberrypi.local:/home/pi/models/

# Run on Pi
python3 blind_navigation/pi_inference.py \
    --model-dir /home/pi/models/obstacle_detector
```

## Recommended Starting Classes

Start with these 5 critical classes:

1. **person** - Pedestrians
2. **vehicle** - Cars, buses, trucks
3. **stairs** - Steps up/down
4. **door** - Open/closed doors
5. **wall** - Walls and barriers

## Data Collection Tips

- **Minimum**: 100 images per class
- **Better**: 500+ images per class
- **Diversity**: Different lighting, angles, distances
- **Quality**: Clear images, good annotations

## Annotation Tools

- **LabelImg**: `pip install labelImg && labelImg`
- **Online**: Use Roboflow or CVAT
- **Manual**: Edit JSON directly

## That's It!

Once trained, the model will:
- ✅ Detect obstacles in real-time
- ✅ Estimate distances
- ✅ Prioritize critical objects
- ✅ Work on Raspberry Pi 4B

For detailed instructions, see [README.md](README.md).

