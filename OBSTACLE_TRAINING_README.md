# Obstacle Detection Training - Following YOLO Notebook Workflow

This follows the exact workflow from `Train_YOLO_Models.ipynb` for obstacle detection.

## Data Structure Required

Your obstacle dataset must be organized in this structure:

```
obstacle_data/
├── images/          # All your obstacle images (.jpg, .png, etc.)
├── labels/          # YOLO format annotation files (.txt files, one per image)
└── classes.txt     # List of obstacle classes (one per line)
```

### classes.txt format:
```
obstacle
person
chair
table
door
wall
```

### YOLO Label Format (.txt files):
Each image needs a corresponding .txt file with the same name.
Format: `class_id center_x center_y width height` (normalized 0-1)

Example: `0 0.5 0.5 0.3 0.4` means:
- Class 0 (obstacle)
- Center at (50%, 50%)
- Width 30%, Height 40%

## Training Steps

### Option 1: Use the automated script (recommended)
```bash
cd Train-and-Deploy-YOLO-Models
python train_obstacle_yolo.py
```

This script will:
1. Split your data into train/validation (90/10)
2. Create data.yaml config file
3. Train the model

### Option 1b: Direct Python training (if CLI doesn't work)
```bash
# First prepare data and config
python utils/train_val_split.py --datapath="obstacle_data" --train_pct=0.9
python create_data_yaml.py

# Then train
python train_obstacle_yolo_direct.py
```

### Option 2: Manual steps (following notebook exactly)

1. **Split data into train/validation:**
```bash
python utils/train_val_split.py --datapath="obstacle_data" --train_pct=0.9
```

2. **Create data.yaml config:**
```bash
python create_data_yaml.py
```

3. **Train model:**
```bash
yolo detect train data=data.yaml model=yolo11s.pt epochs=60 imgsz=640
```

## Training Parameters

- **Model size**: `yolo11n.pt` (fastest), `yolo11s.pt` (recommended), `yolo11m.pt`, `yolo11l.pt`, `yolo11xl.pt` (most accurate)
- **Epochs**: 60 if <200 images, 40 if >200 images
- **Resolution**: 640 (standard), or 480 for faster inference

## Output

- Best model: `runs/detect/train/weights/best.pt`
- Training results: `runs/detect/train/results.png`

## Testing the Model

```bash
python yolo_detect.py --model runs/detect/train/weights/best.pt --source path/to/test/image.jpg
```

## Raspberry Pi Deployment

The trained model can be used with `yolo_detect.py` on Raspberry Pi:
```bash
python yolo_detect.py --model best.pt --source picamera0 --resolution 640x480
```

