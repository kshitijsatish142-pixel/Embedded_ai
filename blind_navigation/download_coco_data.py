"""
Download and extract images from COCO dataset for critical classes.
COCO has person and vehicles - perfect for starting!
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import List, Set

import requests
from tqdm import tqdm


# COCO class mappings
COCO_CLASSES = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
}

# Map to our classes
CLASS_MAPPING = {
    "person": [1],  # person
    "vehicle": [3, 6, 8],  # car, bus, truck
}


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download large file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    
    with open(output_path, "wb") as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_coco_images(
    coco_dir: Path,
    output_dir: Path,
    classes: List[str],
    max_images: int = 500,
    split: str = "train2017",
) -> None:
    """
    Extract images from COCO dataset for specified classes.
    
    Args:
        coco_dir: Directory containing COCO dataset
        output_dir: Where to save extracted images
        classes: List of class names to extract
        max_images: Maximum images per class
        split: Dataset split (train2017 or val2017)
    """
    # Load annotations
    annotations_file = coco_dir / f"annotations/instances_{split}.json"
    if not annotations_file.exists():
        print(f"❌ Annotations not found: {annotations_file}")
        print(f"   Download from: https://cocodataset.org/#download")
        return
    
    print(f"📖 Loading COCO annotations...")
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)
    
    # Build image ID to class mapping
    image_to_classes: dict[int, Set[str]] = {}
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    
    print(f"📊 Processing {len(coco_data['annotations'])} annotations...")
    for ann in tqdm(coco_data["annotations"]):
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        category_name = category_id_to_name[category_id]
        
        # Map to our classes
        our_class = None
        if category_id == 1:  # person
            our_class = "person"
        elif category_id in [3, 6, 8]:  # car, bus, truck
            our_class = "vehicle"
        
        if our_class and our_class in classes:
            if image_id not in image_to_classes:
                image_to_classes[image_id] = set()
            image_to_classes[image_id].add(our_class)
    
    # Build image ID to filename mapping
    image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}
    
    # Extract images
    images_dir = coco_dir / split
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        print(f"   Download from: https://cocodataset.org/#download")
        return
    
    print(f"\n📥 Extracting images...")
    
    class_counts = {cls: 0 for cls in classes}
    
    for image_id, detected_classes in tqdm(image_to_classes.items()):
        if all(class_counts[cls] >= max_images for cls in detected_classes):
            continue
        
        image_file = image_id_to_file[image_id]
        source_path = images_dir / image_file
        
        if not source_path.exists():
            continue
        
        # Copy to each detected class folder
        for cls in detected_classes:
            if class_counts[cls] >= max_images:
                continue
            
            dest_dir = output_dir / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            dest_path = dest_dir / f"{cls}_{class_counts[cls]:05d}.jpg"
            shutil.copy2(source_path, dest_path)
            class_counts[cls] += 1
    
    print(f"\n✅ Extraction complete!")
    for cls, count in class_counts.items():
        print(f"   {cls}: {count} images")


def setup_coco_download_instructions(output_dir: Path) -> None:
    """Create instructions for downloading COCO dataset."""
    instructions = """
# COCO Dataset Download Instructions

## Step 1: Download COCO Dataset

1. Go to: https://cocodataset.org/#download
2. Download these files:
   - **2017 Train images** (~18 GB) - `train2017.zip`
   - **2017 Val images** (~1 GB) - `val2017.zip`
   - **2017 Train/Val annotations** (~241 MB) - `annotations_trainval2017.zip`

## Step 2: Extract Files

```bash
# Extract images
unzip train2017.zip -d coco_dataset/
unzip val2017.zip -d coco_dataset/

# Extract annotations
unzip annotations_trainval2017.zip -d coco_dataset/
```

## Step 3: Run Extraction Script

```bash
python3 blind_navigation/download_coco_data.py extract \\
    --coco-dir ./coco_dataset \\
    --output-dir ./blind_navigation/dataset/train \\
    --classes person vehicle \\
    --max-images 500
```

## What You'll Get

- **person**: Images with people
- **vehicle**: Images with cars, buses, trucks

## Note

COCO does NOT have:
- stairs
- doors  
- walls

You'll need to collect these from other sources (see COLLECTION_INSTRUCTIONS.md)

## Alternative: Use Pre-extracted Data

If you have COCO already downloaded, just point to the directory:

```bash
python3 blind_navigation/download_coco_data.py extract \\
    --coco-dir /path/to/coco \\
    --output-dir ./dataset/train
```
"""
    
    readme_path = output_dir / "COCO_DOWNLOAD_INSTRUCTIONS.md"
    readme_path.write_text(instructions)
    print(f"✅ Instructions saved: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and extract COCO data")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract images from COCO")
    extract_parser.add_argument(
        "--coco-dir",
        type=Path,
        required=True,
        help="Directory containing COCO dataset",
    )
    extract_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for extracted images",
    )
    extract_parser.add_argument(
        "--classes",
        nargs="+",
        default=["person", "vehicle"],
        help="Classes to extract",
    )
    extract_parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Maximum images per class",
    )
    extract_parser.add_argument(
        "--split",
        type=str,
        default="train2017",
        choices=["train2017", "val2017"],
        help="Dataset split",
    )
    
    # Instructions command
    instructions_parser = subparsers.add_parser(
        "instructions",
        help="Show download instructions",
    )
    instructions_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("blind_navigation/dataset"),
        help="Where to save instructions",
    )
    
    args = parser.parse_args()
    
    if args.command == "extract":
        extract_coco_images(
            coco_dir=args.coco_dir,
            output_dir=args.output_dir,
            classes=args.classes,
            max_images=args.max_images,
            split=args.split,
        )
    elif args.command == "instructions":
        setup_coco_download_instructions(args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

