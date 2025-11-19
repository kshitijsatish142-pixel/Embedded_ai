"""
One-command script to start training - handles data setup automatically.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import requests
from PIL import Image
import io
import time


def download_sample_images(output_dir: Path, class_name: str, count: int = 20) -> int:
    """Download sample images from Unsplash for quick start."""
    print(f"📥 Downloading {count} sample images for '{class_name}'...")
    
    queries = {
        "person": "person walking",
        "vehicle": "car street",
        "stairs": "stairs steps",
        "door": "door entrance",
        "wall": "wall barrier",
    }
    
    query = queries.get(class_name, class_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    collected = 0
    base_url = "https://source.unsplash.com/640x480/?"
    
    for i in range(count):
        try:
            url = f"{base_url}{query}"
            response = requests.get(url, timeout=10, stream=True)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                img = img.convert("RGB")
                img_path = output_dir / f"{class_name}_{i:04d}.jpg"
                img.save(img_path, "JPEG", quality=85)
                collected += 1
                print(f"  ✓ Downloaded {i+1}/{count}", end="\r")
                time.sleep(0.3)  # Rate limiting
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue
    
    print(f"\n  ✅ Collected {collected} images for {class_name}")
    return collected


def create_dummy_annotations(image_dir: Path, class_name: str) -> None:
    """Create placeholder annotations (user needs to annotate properly later)."""
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not images:
        return
    
    annotations = {
        "images": [
            {
                "file": img.name,
                "objects": [
                    {
                        "class": class_name,
                        "bbox": [50, 50, 200, 300],  # Placeholder - needs real annotation
                    }
                ]
            }
            for img in images[:50]  # Limit for quick start
        ]
    }
    
    annotations_file = image_dir.parent / "annotations.json"
    
    # Merge with existing if exists
    if annotations_file.exists():
        with open(annotations_file, "r") as f:
            existing = json.load(f)
        existing["images"].extend(annotations["images"])
        annotations = existing
    
    with open(annotations_file, "w") as f:
        json.dump(annotations, f, indent=2)


def setup_quick_dataset(output_dir: Path, classes: List[str], images_per_class: int = 20) -> Path:
    """Set up a quick dataset for immediate training."""
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Setting up quick dataset...")
    print(f"   Classes: {', '.join(classes)}")
    print(f"   Images per class: {images_per_class}\n")
    
    total_collected = 0
    
    for class_name in classes:
        class_dir = train_dir / class_name
        collected = download_sample_images(class_dir, class_name, images_per_class)
        total_collected += collected
        
        if collected > 0:
            create_dummy_annotations(class_dir, class_name)
    
    print(f"\n✅ Dataset ready: {total_collected} total images")
    print(f"   Location: {train_dir}")
    print(f"\n⚠️  NOTE: Annotations are placeholders!")
    print(f"   For real training, annotate properly with LabelImg:")
    print(f"   pip install labelImg && labelImg {train_dir}")
    
    return train_dir


def check_data_exists(data_dir: Path) -> bool:
    """Check if dataset exists and has images."""
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    
    if not train_dir.exists():
        return False
    
    # Check for images
    image_extensions = {".jpg", ".jpeg", ".png"}
    total_images = 0
    
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions]
            total_images += len(images)
    
    return total_images > 0


def start_training(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 8,
    device: str = "cpu",
    quick_setup: bool = False,
) -> None:
    """Start training process."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Check if data exists
    if not check_data_exists(data_dir):
        if quick_setup:
            print("📦 No data found, setting up quick dataset...")
            setup_quick_dataset(
                data_dir,
                classes=["person", "vehicle", "stairs", "door", "wall"],
                images_per_class=20,
            )
        else:
            print(f"❌ No data found in {data_dir}")
            print(f"\n   Quick setup options:")
            print(f"   1. Run with --quick-setup to download sample images")
            print(f"   2. Or prepare data manually (see README.md)")
            print(f"   3. Or use COCO dataset (see download_coco_data.py)")
            return
    
    # Check for annotations
    train_dir = data_dir / "train"
    annotations_file = train_dir / "annotations.json"
    
    if not annotations_file.exists():
        print(f"⚠️  No annotations.json found in {train_dir}")
        print(f"   Creating placeholder annotations...")
        
        # Create basic annotations for all images
        all_images = []
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                class_name = class_dir.name
                for img in images:
                    all_images.append({
                        "file": f"{class_name}/{img.name}",
                        "objects": [{"class": class_name, "bbox": [50, 50, 200, 300]}],
                    })
        
        if all_images:
            annotations = {"images": all_images[:100]}  # Limit for quick start
            with open(annotations_file, "w") as f:
                json.dump(annotations, f, indent=2)
            print(f"   ✅ Created placeholder annotations")
            print(f"   ⚠️  For real training, annotate properly with LabelImg!")
    
    # Start training
    print(f"\n🚀 Starting training...")
    print(f"   Data: {data_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}\n")
    
    # Import and run training
    try:
        from blind_navigation.train_obstacle_detector import train_model
        
        train_model(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
    except ImportError:
        # Run as subprocess
        script_path = Path(__file__).parent / "train_obstacle_detector.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--device", device,
        ]
        
        print(f"Running: {' '.join(cmd)}\n")
        subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Start training obstacle detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("blind_navigation/dataset"),
        help="Dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("blind_navigation/models/obstacle_detector"),
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (use 8 for CPU, 16+ for GPU)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--quick-setup",
        action="store_true",
        help="Automatically download sample images if no data exists",
    )
    
    args = parser.parse_args()
    
    start_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        quick_setup=args.quick_setup,
    )


if __name__ == "__main__":
    main()

