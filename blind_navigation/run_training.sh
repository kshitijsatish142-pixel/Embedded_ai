#!/bin/bash
# Extract and train once COCO is downloaded
# All data stays in current folder (snn)

cd "$(dirname "$0")/.." || exit

echo "📦 Extracting train images to current folder..."
unzip -q train2017.zip -d .

echo "📦 Extracting annotations to current folder..."
unzip -q annotations_trainval2017.zip -d .

echo "🚀 Starting training..."
python3 blind_navigation/use_coco_and_train.py \
    --coco-images ./train2017 \
    --coco-annotations ./annotations/instances_train2017.json \
    --epochs 50 \
    --batch-size 4

