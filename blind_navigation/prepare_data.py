"""
Utilities to prepare and organize training data for obstacle detection.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def create_annotations_template(output_file: Path):
    """Create a template annotations.json file."""
    template = {
        "images": [
            {
                "file": "example1.jpg",
                "objects": [
                    {
                        "class": "person",
                        "bbox": [100, 150, 80, 200]  # [x, y, width, height]
                    },
                    {
                        "class": "vehicle",
                        "bbox": [300, 200, 150, 100]
                    }
                ]
            }
        ]
    }
    
    with open(output_file, "w") as f:
        json.dump(template, f, indent=2)
    
    print(f"✅ Created template: {output_file}")
    print("\nFormat:")
    print("  - 'file': Image filename")
    print("  - 'objects': List of detected objects")
    print("  - 'class': Object class name")
    print("  - 'bbox': [x, y, width, height] in pixels")


def organize_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
):
    """
    Organize images into train/val folders.
    
    Args:
        source_dir: Directory with all images
        output_dir: Output directory for organized dataset
        train_ratio: Ratio of images for training (rest for validation)
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Create directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in source_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"❌ No images found in {source_dir}")
        return
    
    # Split train/val
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Copy images
    print(f"Organizing {len(images)} images...")
    
    for img in train_images:
        shutil.copy2(img, train_dir / img.name)
    
    for img in val_images:
        shutil.copy2(img, val_dir / img.name)
    
    print(f"✅ Organized dataset:")
    print(f"   Training: {len(train_images)} images -> {train_dir}")
    print(f"   Validation: {len(val_images)} images -> {val_dir}")
    print(f"\n⚠️  Don't forget to create annotations.json in each folder!")


def validate_annotations(data_dir: Path):
    """Validate annotations.json file."""
    data_dir = Path(data_dir)
    annotations_file = data_dir / "annotations.json"
    
    if not annotations_file.exists():
        print(f"❌ annotations.json not found in {data_dir}")
        return False
    
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
    
    # Check structure
    if "images" not in annotations:
        print("❌ Missing 'images' key in annotations")
        return False
    
    # Validate each image
    image_dir = data_dir
    valid_count = 0
    invalid_count = 0
    
    for item in annotations["images"]:
        img_file = image_dir / item["file"]
        if not img_file.exists():
            print(f"⚠️  Image not found: {item['file']}")
            invalid_count += 1
            continue
        
        if "objects" not in item:
            print(f"⚠️  No 'objects' key for {item['file']}")
            invalid_count += 1
            continue
        
        for obj in item["objects"]:
            if "class" not in obj or "bbox" not in obj:
                print(f"⚠️  Invalid object in {item['file']}")
                invalid_count += 1
                break
            
            if len(obj["bbox"]) != 4:
                print(f"⚠️  Invalid bbox format in {item['file']}")
                invalid_count += 1
                break
        
        valid_count += 1
    
    print(f"✅ Validation complete:")
    print(f"   Valid: {valid_count}")
    print(f"   Invalid: {invalid_count}")
    
    return invalid_count == 0


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create template
    template_parser = subparsers.add_parser("template", help="Create annotations template")
    template_parser.add_argument("--output", type=Path, default=Path("annotations_template.json"))
    
    # Organize dataset
    organize_parser = subparsers.add_parser("organize", help="Organize images into train/val")
    organize_parser.add_argument("--source", type=Path, required=True, help="Source directory with images")
    organize_parser.add_argument("--output", type=Path, required=True, help="Output directory")
    organize_parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio")
    
    # Validate
    validate_parser = subparsers.add_parser("validate", help="Validate annotations")
    validate_parser.add_argument("--data-dir", type=Path, required=True, help="Data directory with annotations.json")
    
    args = parser.parse_args()
    
    if args.command == "template":
        create_annotations_template(args.output)
    elif args.command == "organize":
        organize_dataset(args.source, args.output, args.train_ratio)
    elif args.command == "validate":
        validate_annotations(args.data_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

