"""
One-command setup: Download COCO data and start training immediately.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def check_coco_available(coco_dir: Path) -> bool:
    """Check if COCO dataset is available."""
    coco_dir = Path(coco_dir)
    images_dir = coco_dir / "train2017"
    annotations_file = coco_dir / "annotations" / "instances_train2017.json"
    return images_dir.exists() and annotations_file.exists()


def main():
    parser = argparse.ArgumentParser(
        description="Setup COCO data and start training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--coco-dir",
        type=Path,
        help="Path to COCO dataset (if already downloaded)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("blind_navigation/dataset"),
        help="Where to extract training data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("blind_navigation/models/obstacle_detector"),
        help="Where to save trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device",
    )
    
    args = parser.parse_args()
    
    print("🚀 Setting up real training data...\n")
    
    # Check if COCO is available
    if args.coco_dir and check_coco_available(args.coco_dir):
        print(f"✅ COCO dataset found at {args.coco_dir}")
        print("📥 Extracting person and vehicle classes...\n")
        
        # Extract from COCO
        extract_cmd = [
            sys.executable,
            "blind_navigation/download_coco_data.py",
            "extract",
            "--coco-dir", str(args.coco_dir),
            "--output-dir", str(args.data_dir / "train"),
            "--classes", "person", "vehicle",
            "--max-images", "500",
        ]
        
        subprocess.run(extract_cmd, check=True)
        
        print("\n✅ Data extracted! Starting training...\n")
        
    else:
        print("""
⚠️  COCO dataset not found!

To get real training data, you have 3 options:

OPTION 1: Download COCO (Recommended - Real, annotated data)
──────────────────────────────────────────────────────────────
1. Download from: https://cocodataset.org/#download
   - 2017 Train images (~18 GB)
   - 2017 Train annotations (~241 MB)

2. Extract:
   unzip train2017.zip -d coco_dataset/
   unzip annotations_trainval2017.zip -d coco_dataset/

3. Run again with:
   python3 blind_navigation/setup_and_train.py --coco-dir ./coco_dataset

OPTION 2: Use Your Own Images
──────────────────────────────────────────────────────────────
1. Collect images and organize:
   dataset/train/person/
   dataset/train/vehicle/
   dataset/train/stairs/
   dataset/train/door/
   dataset/train/wall/

2. Annotate with LabelImg:
   pip install labelImg
   labelImg dataset/train/

3. Start training:
   python3 blind_navigation/train_obstacle_detector.py \\
       --data-dir ./dataset \\
       --output-dir ./models

OPTION 3: Download from APIs (Requires API keys)
──────────────────────────────────────────────────────────────
python3 blind_navigation/download_real_data.py \\
    --pexels-key YOUR_KEY \\
    --unsplash-key YOUR_KEY
""")
        return
    
    # Start training
    train_cmd = [
        sys.executable,
        "blind_navigation/train_obstacle_detector.py",
        "--data-dir", str(args.data_dir),
        "--output-dir", str(args.output_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
    ]
    
    print("🚀 Starting training...\n")
    subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()

