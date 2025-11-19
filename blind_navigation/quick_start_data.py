"""
Quick start script to set up data collection for critical classes.
This will guide you through the easiest ways to get training data.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def print_setup_instructions():
    """Print step-by-step setup instructions."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║   Quick Start: Data Collection for Critical Classes        ║
╚══════════════════════════════════════════════════════════════╝

We need data for 5 critical classes:
  1. person
  2. vehicle (cars, buses, trucks)
  3. stairs
  4. door
  5. wall

═══════════════════════════════════════════════════════════════

OPTION 1: COCO Dataset (Easiest - Has person & vehicle)
───────────────────────────────────────────────────────────────

✅ Pros: High quality, already annotated, free
⚠️  Cons: Only has person & vehicle (need stairs/door/wall elsewhere)

Steps:
  1. Download COCO dataset:
     https://cocodataset.org/#download
     
     Download:
     - 2017 Train images (~18 GB)
     - 2017 Train annotations (~241 MB)
     
  2. Extract:
     unzip train2017.zip -d coco_dataset/
     unzip annotations_trainval2017.zip -d coco_dataset/
     
  3. Extract our classes:
     python3 blind_navigation/download_coco_data.py extract \\
         --coco-dir ./coco_dataset \\
         --output-dir ./blind_navigation/dataset/train \\
         --classes person vehicle \\
         --max-images 500

═══════════════════════════════════════════════════════════════

OPTION 2: Google Images (Manual - All Classes)
───────────────────────────────────────────────────────────────

✅ Pros: Can get all classes, easy to use
⚠️  Cons: Manual work, need to annotate yourself

Steps:
  1. Go to: https://www.google.com/images
  
  2. Search and download for each class:
     - "person walking street" → save to dataset/train/person/
     - "car vehicle street" → save to dataset/train/vehicle/
     - "stairs steps" → save to dataset/train/stairs/
     - "door entrance" → save to dataset/train/door/
     - "wall barrier" → save to dataset/train/wall/
  
  3. Use browser extension:
     - "Download All Images" extension
     - Or manually save 50-100 images per class
  
  4. Annotate with LabelImg:
     pip install labelImg
     labelImg dataset/train/person/

═══════════════════════════════════════════════════════════════

OPTION 3: Your Own Photos (Best Quality)
───────────────────────────────────────────────────────────────

✅ Pros: Real-world scenarios, perfect for your use case
⚠️  Cons: Takes time to collect

Steps:
  1. Take photos with phone/camera:
     - Walk around and photograph obstacles
     - Focus on: person, vehicles, stairs, doors, walls
     - Get different: lighting, angles, distances
  
  2. Organize:
     - Put in: dataset/train/<class_name>/
     - Minimum: 50 images per class to start
  
  3. Annotate:
     - Use LabelImg to draw bounding boxes
     - Save as YOLO or Pascal VOC format

═══════════════════════════════════════════════════════════════

OPTION 4: Hybrid Approach (Recommended)
───────────────────────────────────────────────────────────────

1. Use COCO for person & vehicle (high quality, annotated)
2. Use Google Images for stairs, door, wall
3. Add your own photos for real-world scenarios

═══════════════════════════════════════════════════════════════

After Collecting Data:
───────────────────────────────────────────────────────────────

1. Organize structure:
   dataset/
   ├── train/
   │   ├── person/
   │   ├── vehicle/
   │   ├── stairs/
   │   ├── door/
   │   └── wall/
   └── val/
       └── (same structure)

2. Annotate bounding boxes:
   pip install labelImg
   labelImg dataset/train/

3. Create annotations.json:
   python3 blind_navigation/prepare_data.py template

4. Validate:
   python3 blind_navigation/prepare_data.py validate --data-dir dataset/train

5. Train:
   python3 blind_navigation/train_obstacle_detector.py --data-dir dataset

═══════════════════════════════════════════════════════════════

Quick Commands:
───────────────────────────────────────────────────────────────

# Setup dataset structure
python3 blind_navigation/collect_data.py setup --output-dir ./dataset

# Get COCO instructions
python3 blind_navigation/download_coco_data.py instructions

# Organize your images
python3 blind_navigation/prepare_data.py organize \\
    --source ./my_images \\
    --output ./dataset

═══════════════════════════════════════════════════════════════
""")


def create_minimal_dataset_structure(output_dir: Path):
    """Create minimal dataset structure with example."""
    output_dir = Path(output_dir)
    
    # Create structure
    for split in ["train", "val"]:
        for class_name in ["person", "vehicle", "stairs", "door", "wall"]:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme = """# Dataset Structure

Put your images here:
- train/person/ - Images with people
- train/vehicle/ - Images with vehicles
- train/stairs/ - Images with stairs
- train/door/ - Images with doors
- train/wall/ - Images with walls

After adding images, annotate with LabelImg:
  pip install labelImg
  labelImg train/person/

Then create annotations.json in each folder.
"""
    
    (output_dir / "README.md").write_text(readme)
    print(f"✅ Created dataset structure: {output_dir}")
    print(f"   Add your images to: {output_dir}/train/<class_name>/")


def main():
    parser = argparse.ArgumentParser(
        description="Quick start data collection setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create dataset structure",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("blind_navigation/dataset"),
        help="Output directory",
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Show setup instructions",
    )
    
    args = parser.parse_args()
    
    if args.instructions or (not args.setup):
        print_setup_instructions()
    
    if args.setup:
        create_minimal_dataset_structure(args.output_dir)


if __name__ == "__main__":
    main()

