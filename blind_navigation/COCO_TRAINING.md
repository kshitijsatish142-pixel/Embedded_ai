# COCO Training - Complete Guide

## Step 1: Download COCO Dataset

```bash
# Download these files from: https://cocodataset.org/#download
# - 2017 Train images (~18 GB): train2017.zip
# - 2017 Train annotations (~241 MB): annotations_trainval2017.zip
```

## Step 2: Extract

```bash
# Extract images
unzip train2017.zip -d coco_dataset/

# Extract annotations  
unzip annotations_trainval2017.zip -d coco_dataset/
```

## Step 3: Run Training (One Command)

```bash
python3 blind_navigation/use_coco_and_train.py \
    --coco-images ./coco_dataset/train2017 \
    --coco-annotations ./coco_dataset/annotations/instances_train2017.json \
    --epochs 50 \
    --batch-size 8
```

That's it! This will:
1. Extract person and vehicle images from COCO
2. Create annotations automatically
3. Start training immediately

## Quick Start (If COCO Already Downloaded)

```bash
# Just point to your COCO directory
python3 blind_navigation/use_coco_and_train.py \
    --coco-images /path/to/coco/train2017 \
    --coco-annotations /path/to/coco/annotations/instances_train2017.json
```

## Options

```bash
--max-images 500      # Images per class (default: 500)
--epochs 50          # Training epochs (default: 50)
--batch-size 8       # Batch size (default: 8, use 4 for CPU)
--device cpu         # cpu or cuda
--skip-training      # Only extract, don't train
```

## What You Get

- Extracted images: `blind_navigation/dataset/train/person/` and `vehicle/`
- Annotations: `blind_navigation/dataset/train/annotations.json`
- Trained model: `blind_navigation/models/obstacle_detector/`

