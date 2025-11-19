"""
Start training immediately with whatever data you have.
Works with any images you provide - just organize them and run!
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def find_images_in_dir(directory: Path) -> list[Path]:
    """Find all images in directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for ext in image_extensions:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(images)


def auto_create_annotations(data_dir: Path) -> bool:
    """Auto-create annotations.json from existing images."""
    train_dir = data_dir / "train"
    
    if not train_dir.exists():
        return False
    
    all_annotations = []
    has_images = False
    
    # Find all class directories
    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        images = find_images_in_dir(class_dir)
        
        if images:
            has_images = True
            print(f"  📦 Found {len(images)} images in '{class_name}'")
            
            for img in images:
                all_annotations.append({
                    "file": f"{class_name}/{img.name}",
                    "objects": [
                        {
                            "class": class_name,
                            "bbox": [100, 100, 200, 300],  # Placeholder - will need real annotation
                        }
                    ]
                })
    
    if has_images:
        annotations_file = train_dir / "annotations.json"
        with open(annotations_file, "w") as f:
            json.dump({"images": all_annotations}, f, indent=2)
        
        print(f"\n  ✅ Created annotations.json with {len(all_annotations)} images")
        print(f"  ⚠️  NOTE: Bounding boxes are placeholders!")
        print(f"     For best results, annotate properly with LabelImg:")
        print(f"     pip install labelImg && labelImg {train_dir}")
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Start training with your images - works immediately!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("blind_navigation/dataset"),
        help="Dataset directory (should have train/<class_name>/ folders with images)",
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
        default=50,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (use 4-8 for CPU, 16+ for GPU)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device",
    )
    parser.add_argument(
        "--skip-annotations",
        action="store_true",
        help="Skip auto-creating annotations (use existing)",
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting training setup...\n")
    
    # Check for existing data
    train_dir = args.data_dir / "train"
    
    if not train_dir.exists():
        print(f"❌ Dataset directory not found: {train_dir}")
        print("\n📝 Quick setup:")
        print("   1. Create folder structure:")
        print(f"      mkdir -p {train_dir}/person")
        print(f"      mkdir -p {train_dir}/vehicle")
        print(f"      mkdir -p {train_dir}/stairs")
        print(f"      mkdir -p {train_dir}/door")
        print(f"      mkdir -p {train_dir}/wall")
        print("\n   2. Add your images to each folder")
        print("\n   3. Run this script again!")
        print("\n   Or use COCO dataset:")
        print("   python3 blind_navigation/setup_and_train.py --coco-dir <path>")
        return
    
    # Auto-create annotations if needed
    annotations_file = train_dir / "annotations.json"
    if not annotations_file.exists() and not args.skip_annotations:
        print("📝 Creating annotations from existing images...")
        if not auto_create_annotations(args.data_dir):
            print("  ⚠️  No images found in class folders")
            print(f"     Add images to: {train_dir}/<class_name>/")
            return
    
    # Check if we have data
    if annotations_file.exists():
        with open(annotations_file, "r") as f:
            data = json.load(f)
        num_images = len(data.get("images", []))
        if num_images == 0:
            print("❌ No images in annotations.json")
            return
        print(f"✅ Found {num_images} images ready for training\n")
    else:
        print("❌ No annotations.json found")
        return
    
    # Start training
    print("🚀 Starting training...")
    print(f"   Data: {args.data_dir}")
    print(f"   Output: {args.output_dir}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Device: {args.device}\n")
    
    train_cmd = [
        sys.executable,
        "blind_navigation/train_obstacle_detector.py",
        "--data-dir", str(args.data_dir),
        "--output-dir", str(args.output_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
    ]
    
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        print("\n💡 Tips:")
        print("   - Check that images are in correct folders")
        print("   - Verify annotations.json format")
        print("   - Try smaller batch size: --batch-size 4")
        sys.exit(1)


if __name__ == "__main__":
    main()

