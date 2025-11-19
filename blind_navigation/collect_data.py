"""
Automated data collection for obstacle detection training.
Downloads images from public datasets and organizes them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import requests
from PIL import Image
import io


# Critical classes to collect
CRITICAL_CLASSES = {
    "person": ["person", "pedestrian", "people", "human"],
    "vehicle": ["car", "vehicle", "automobile", "bus", "truck", "motorcycle"],
    "stairs": ["stairs", "steps", "staircase", "escalator"],
    "door": ["door", "doorway", "entrance", "exit"],
    "wall": ["wall", "barrier", "fence", "railing"],
}

# Alternative class mappings for different datasets
CLASS_MAPPINGS = {
    "coco": {
        "person": 1,  # COCO class IDs
        "vehicle": [3, 6, 8],  # car, bus, truck
        "stairs": None,  # Not in COCO
        "door": None,  # Not in COCO
        "wall": None,  # Not in COCO
    }
}


def download_image(url: str, timeout: int = 10) -> Image.Image | None:
    """Download image from URL."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"  ⚠️  Failed to download {url}: {e}")
        return None


def collect_from_unsplash(
    query: str,
    count: int = 50,
    output_dir: Path,
    class_name: str,
) -> int:
    """
    Collect images from Unsplash (free, high-quality images).
    Note: Requires Unsplash API key for large-scale collection.
    """
    print(f"📥 Collecting '{class_name}' from Unsplash (query: {query})...")
    
    # For demo, we'll use Unsplash Source API (no key needed, limited)
    # In production, use Unsplash API with key for better results
    collected = 0
    
    # Unsplash Source API (no authentication, but limited)
    base_url = "https://source.unsplash.com/640x480/?"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(min(count, 20)):  # Limit for demo
        try:
            url = f"{base_url}{query}"
            img = download_image(url)
            if img:
                img_path = output_dir / f"{class_name}_{i:04d}.jpg"
                img.save(img_path, "JPEG", quality=85)
                collected += 1
                time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
    
    print(f"  ✅ Collected {collected} images")
    return collected


def collect_from_pexels(
    query: str,
    count: int = 50,
    output_dir: Path,
    class_name: str,
) -> int:
    """
    Collect images from Pexels (free stock photos).
    Note: Requires API key for programmatic access.
    """
    print(f"📥 Collecting '{class_name}' from Pexels (query: {query})...")
    
    # Pexels requires API key - provide instructions
    print(f"  ℹ️  Pexels requires API key. Get one at: https://www.pexels.com/api/")
    print(f"  ℹ️  Or use manual download from: https://www.pexels.com/search/{query}/")
    
    return 0


def create_annotations_from_images(
    image_dir: Path,
    class_name: str,
    output_file: Path,
) -> None:
    """Create basic annotations.json file for images (requires manual bbox annotation)."""
    image_dir = Path(image_dir)
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    annotations = {
        "images": [
            {
                "file": img.name,
                "objects": [
                    {
                        "class": class_name,
                        "bbox": [0, 0, 0, 0],  # Placeholder - needs manual annotation
                    }
                ]
            }
            for img in images
        ]
    }
    
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=2)
    
    print(f"  ✅ Created annotations template: {output_file}")
    print(f"  ⚠️  You need to annotate bounding boxes manually using LabelImg")


def download_coco_subset(
    classes: List[str],
    output_dir: Path,
    max_images_per_class: int = 100,
) -> None:
    """
    Download subset of COCO dataset for specified classes.
    COCO has person and vehicles, but not stairs/doors/walls.
    """
    print("📥 Downloading COCO dataset subset...")
    print("  ℹ️  COCO has: person, vehicles")
    print("  ⚠️  COCO does NOT have: stairs, doors, walls")
    print("  ℹ️  You'll need to collect these from other sources")
    
    # Instructions for COCO download
    print("\n  To download COCO:")
    print("  1. Visit: https://cocodataset.org/#download")
    print("  2. Download: 2017 Train/Val images")
    print("  3. Download: 2017 Train/Val annotations")
    print("  4. Use COCO API to extract images for your classes")
    print("\n  Or use this script with COCO API installed:")
    print("    pip install pycocotools")
    
    # Placeholder - would need COCO API
    output_dir.mkdir(parents=True, exist_ok=True)


def collect_from_google_images(
    query: str,
    count: int = 50,
    output_dir: Path,
    class_name: str,
) -> int:
    """
    Collect images using Google Images (via simple method).
    Note: For production, use official APIs or web scraping tools.
    """
    print(f"📥 Collecting '{class_name}' from Google Images (query: {query})...")
    print("  ℹ️  Using simple method - for better results, use:")
    print("     - Google Custom Search API (requires API key)")
    print("     - Or use: pip install google-images-download")
    
    # Simple approach: provide instructions
    print(f"\n  Manual method:")
    print(f"  1. Go to: https://www.google.com/search?tbm=isch&q={query}")
    print(f"  2. Use browser extension to download images")
    print(f"  3. Save to: {output_dir}")
    
    return 0


def setup_dataset_structure(output_dir: Path) -> None:
    """Create dataset directory structure."""
    output_dir = Path(output_dir)
    
    for split in ["train", "val"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created dataset structure: {output_dir}")


def collect_critical_classes(
    output_dir: Path,
    images_per_class: int = 100,
    source: str = "unsplash",
) -> None:
    """
    Collect images for all critical classes.
    
    Args:
        output_dir: Where to save collected images
        images_per_class: Target number of images per class
        source: Data source ('unsplash', 'pexels', 'google', 'coco')
    """
    output_dir = Path(output_dir)
    setup_dataset_structure(output_dir)
    
    print(f"🚀 Collecting data for critical classes...")
    print(f"   Target: {images_per_class} images per class")
    print(f"   Source: {source}\n")
    
    total_collected = 0
    
    for class_name, queries in CRITICAL_CLASSES.items():
        print(f"\n📦 Class: {class_name}")
        class_dir = output_dir / "train" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        collected = 0
        
        if source == "unsplash":
            # Try first query
            collected = collect_from_unsplash(
                query=queries[0],
                count=images_per_class,
                output_dir=class_dir,
                class_name=class_name,
            )
        elif source == "google":
            collected = collect_from_google_images(
                query=queries[0],
                count=images_per_class,
                output_dir=class_dir,
                class_name=class_name,
            )
        elif source == "coco":
            if class_name in ["person", "vehicle"]:
                download_coco_subset([class_name], class_dir, images_per_class)
            else:
                print(f"  ⚠️  {class_name} not in COCO, skipping")
        
        total_collected += collected
        
        # Create annotations template
        if collected > 0:
            annotations_file = output_dir / "train" / f"{class_name}_annotations.json"
            create_annotations_from_images(class_dir, class_name, annotations_file)
    
    print(f"\n✅ Collection complete!")
    print(f"   Total images collected: {total_collected}")
    print(f"\n📝 Next steps:")
    print(f"   1. Review images in: {output_dir}/train/")
    print(f"   2. Annotate bounding boxes using LabelImg")
    print(f"   3. Merge annotations into single annotations.json")
    print(f"   4. Split into train/val sets")


def create_quick_start_dataset(output_dir: Path) -> None:
    """
    Create a minimal dataset structure with instructions for manual collection.
    """
    output_dir = Path(output_dir)
    setup_dataset_structure(output_dir)
    
    instructions = f"""
# Data Collection Instructions

## Critical Classes to Collect

1. **person** - Pedestrians, people walking
2. **vehicle** - Cars, buses, trucks, motorcycles
3. **stairs** - Stairs going up/down
4. **door** - Open and closed doors
5. **wall** - Walls, barriers, fences

## Recommended Sources

### Option 1: Public Datasets (Easiest)
- **COCO Dataset**: https://cocodataset.org/
  - Has: person, vehicles
  - Download: 2017 Train/Val images + annotations
  - Use COCO API to extract classes

- **Open Images**: https://storage.googleapis.com/openimages/web/index.html
  - Large dataset with many classes
  - Download specific classes

### Option 2: Google Images (Manual)
1. Go to: https://www.google.com/images
2. Search for each class (e.g., "person walking", "car street")
3. Use browser extension to download (e.g., "Download All Images")
4. Save to: {output_dir}/train/<class_name>/

### Option 3: Unsplash/Pexels (Free Stock)
- **Unsplash**: https://unsplash.com/
- **Pexels**: https://www.pexels.com/
- Search and download images
- Free for commercial use

### Option 4: Your Own Photos
- Take photos with your phone/camera
- Focus on diverse: lighting, angles, distances
- Minimum: 50-100 images per class to start

## Quick Start (50 images per class)

For a quick start, collect:
- 50 images of people
- 50 images of vehicles  
- 50 images of stairs
- 50 images of doors
- 50 images of walls

Total: 250 images minimum

## After Collection

1. Organize: Put images in {output_dir}/train/<class_name>/
2. Annotate: Use LabelImg to draw bounding boxes
3. Validate: Check annotations.json format
4. Train: Run training script

See README.md for detailed instructions.
"""
    
    readme_path = output_dir / "COLLECTION_INSTRUCTIONS.md"
    readme_path.write_text(instructions)
    
    print(f"✅ Created dataset structure: {output_dir}")
    print(f"✅ Instructions saved: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect training data for obstacle detection")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect images from source")
    collect_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("blind_navigation/dataset"),
        help="Output directory for dataset",
    )
    collect_parser.add_argument(
        "--images-per-class",
        type=int,
        default=100,
        help="Target number of images per class",
    )
    collect_parser.add_argument(
        "--source",
        type=str,
        choices=["unsplash", "google", "coco", "pexels"],
        default="unsplash",
        help="Data source",
    )
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Create dataset structure with instructions")
    setup_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("blind_navigation/dataset"),
        help="Output directory for dataset",
    )
    
    args = parser.parse_args()
    
    if args.command == "collect":
        collect_critical_classes(
            output_dir=args.output_dir,
            images_per_class=args.images_per_class,
            source=args.source,
        )
    elif args.command == "setup":
        create_quick_start_dataset(args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

