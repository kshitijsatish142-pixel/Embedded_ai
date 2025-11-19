"""
Complete COCO setup and training - one command does everything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

try:
    from pycocotools.coco import COCO
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False


def install_pycocotools():
    """Install pycocotools if not available."""
    print("📦 Installing pycocotools...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pycocotools"], check=True)
    print("✅ Installed")


def extract_from_coco(
    coco_images_dir: Path,
    coco_annotations_file: Path,
    output_dir: Path,
    classes: List[str],
    max_images: int = 500,
) -> None:
    """Extract images from COCO dataset."""
    if not COCO_AVAILABLE:
        install_pycocotools()
        from pycocotools.coco import COCO
    
    print(f"📖 Loading COCO annotations...")
    coco = COCO(str(coco_annotations_file))
    
    # Class mappings
    class_mapping = {
        "person": [1],  # person
        "vehicle": [3, 6, 8],  # car, bus, truck
    }
    
    category_names = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}
    
    # Build image to class mapping
    image_to_classes: Dict[int, Set[str]] = {}
    
    print(f"📊 Processing annotations...")
    for class_name, cat_ids in class_mapping.items():
        if class_name not in classes:
            continue
        
        for cat_id in cat_ids:
            img_ids = coco.getImgIds(catIds=[cat_id])
            
            for img_id in img_ids[:max_images]:
                if img_id not in image_to_classes:
                    image_to_classes[img_id] = set()
                image_to_classes[img_id].add(class_name)
    
    # Get image info
    img_ids = list(image_to_classes.keys())
    images = coco.loadImgs(img_ids)
    
    # Extract images
    print(f"\n📥 Extracting {len(images)} images...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    from tqdm import tqdm
    
    class_counts = {cls: 0 for cls in classes}
    all_annotations = []
    
    for img_info in tqdm(images):
        img_id = img_info["id"]
        img_file = img_info["file_name"]
        source_path = coco_images_dir / img_file
        
        if not source_path.exists():
            continue
        
        detected_classes = image_to_classes[img_id]
        
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        
        for class_name in detected_classes:
            if class_counts[class_name] >= max_images:
                continue
            
            # Create class directory
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy image
            dest_path = class_dir / f"{class_name}_{class_counts[class_name]:05d}.jpg"
            shutil.copy2(source_path, dest_path)
            
            # Find bbox for this class
            bbox = None
            for ann in anns:
                cat_id = ann["category_id"]
                if class_name == "person" and cat_id == 1:
                    bbox = ann["bbox"]  # [x, y, w, h]
                    break
                elif class_name == "vehicle" and cat_id in [3, 6, 8]:
                    if bbox is None:
                        bbox = ann["bbox"]
                    break
            
            if bbox is None:
                bbox = [100, 100, 200, 200]  # Default
            
            # Add to annotations
            all_annotations.append({
                "file": f"{class_name}/{dest_path.name}",
                "objects": [
                    {
                        "class": class_name,
                        "bbox": bbox,  # COCO format: [x, y, w, h]
                    }
                ]
            })
            
            class_counts[class_name] += 1
    
    # Save annotations
    annotations_file = output_dir / "annotations.json"
    with open(annotations_file, "w") as f:
        json.dump({"images": all_annotations}, f, indent=2)
    
    print(f"\n✅ Extraction complete!")
    for cls, count in class_counts.items():
        print(f"   {cls}: {count} images")
    print(f"   Annotations: {annotations_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract from COCO and train immediately",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--coco-images",
        type=Path,
        required=True,
        help="COCO images directory (e.g., coco_dataset/train2017)",
    )
    parser.add_argument(
        "--coco-annotations",
        type=Path,
        required=True,
        help="COCO annotations file (e.g., coco_dataset/annotations/instances_train2017.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("blind_navigation/dataset/train"),
        help="Output directory for extracted data",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("blind_navigation/models/obstacle_detector"),
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Max images per class",
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
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only extract, don't train",
    )
    
    args = parser.parse_args()
    
    # Validate COCO paths
    if not args.coco_images.exists():
        print(f"❌ COCO images directory not found: {args.coco_images}")
        print("\nDownload COCO from: https://cocodataset.org/#download")
        print("Then extract and provide paths:")
        print("  --coco-images <path>/train2017")
        print("  --coco-annotations <path>/annotations/instances_train2017.json")
        return
    
    if not args.coco_annotations.exists():
        print(f"❌ COCO annotations not found: {args.coco_annotations}")
        return
    
    # Extract
    print("🚀 Extracting COCO data...\n")
    extract_from_coco(
        coco_images_dir=args.coco_images,
        coco_annotations_file=args.coco_annotations,
        output_dir=args.output_dir,
        classes=["person", "vehicle"],
        max_images=args.max_images,
    )
    
    if args.skip_training:
        print("\n✅ Extraction complete. Training skipped.")
        return
    
    # Train
    print("\n🚀 Starting training...\n")
    train_cmd = [
        sys.executable,
        "blind_navigation/train_obstacle_detector.py",
        "--data-dir", str(args.output_dir.parent),
        "--output-dir", str(args.model_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
    ]
    
    subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()

