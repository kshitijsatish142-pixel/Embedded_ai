"""
Download real images for training from reliable sources.
Uses multiple methods to get actual photos.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import requests
from PIL import Image
import io


def download_from_pexels_api(query: str, count: int, api_key: str, output_dir: Path) -> int:
    """Download images from Pexels API (requires free API key)."""
    if not api_key:
        return 0
    
    headers = {"Authorization": api_key}
    url = f"https://api.pexels.com/v1/search"
    
    collected = 0
    page = 1
    
    while collected < count:
        params = {"query": query, "per_page": min(80, count - collected), "page": page}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for photo in data.get("photos", []):
                if collected >= count:
                    break
                
                # Get medium size image
                img_url = photo.get("src", {}).get("medium", "")
                if img_url:
                    img_response = requests.get(img_url, timeout=10)
                    if img_response.status_code == 200:
                        img = Image.open(io.BytesIO(img_response.content))
                        img = img.convert("RGB")
                        img_path = output_dir / f"{query.replace(' ', '_')}_{collected:04d}.jpg"
                        img.save(img_path, "JPEG", quality=90)
                        collected += 1
                        print(f"  ✓ Downloaded {collected}/{count}", end="\r")
                        time.sleep(0.2)
            
            if not data.get("photos"):
                break
            page += 1
            
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            break
    
    return collected


def download_from_unsplash_api(query: str, count: int, api_key: str, output_dir: Path) -> int:
    """Download images from Unsplash API (requires free API key)."""
    if not api_key:
        return 0
    
    headers = {"Authorization": f"Client-ID {api_key}"}
    url = "https://api.unsplash.com/search/photos"
    
    collected = 0
    page = 1
    
    while collected < count:
        params = {"query": query, "per_page": min(30, count - collected), "page": page}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for photo in data.get("results", []):
                if collected >= count:
                    break
                
                img_url = photo.get("urls", {}).get("regular", "")
                if img_url:
                    img_response = requests.get(img_url, timeout=10)
                    if img_response.status_code == 200:
                        img = Image.open(io.BytesIO(img_response.content))
                        img = img.convert("RGB")
                        img_path = output_dir / f"{query.replace(' ', '_')}_{collected:04d}.jpg"
                        img.save(img_path, "JPEG", quality=90)
                        collected += 1
                        print(f"  ✓ Downloaded {collected}/{count}", end="\r")
                        time.sleep(0.2)
            
            if not data.get("results"):
                break
            page += 1
            
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            break
    
    return collected


def create_annotations_for_images(image_dir: Path, class_name: str) -> None:
    """Create annotations.json for downloaded images (needs manual bbox annotation)."""
    images = sorted(list(image_dir.glob("*.jpg")))
    
    if not images:
        return
    
    annotations = {
        "images": [
            {
                "file": f"{class_name}/{img.name}",
                "objects": [
                    {
                        "class": class_name,
                        "bbox": [0, 0, 0, 0],  # Placeholder - needs annotation
                    }
                ]
            }
            for img in images
        ]
    }
    
    annotations_file = image_dir.parent / "annotations.json"
    
    # Merge if exists
    if annotations_file.exists():
        with open(annotations_file, "r") as f:
            existing = json.load(f)
        existing["images"].extend(annotations["images"])
        annotations = existing
    
    with open(annotations_file, "w") as f:
        json.dump(annotations, f, indent=2)


def download_real_images(
    output_dir: Path,
    classes: List[str],
    images_per_class: int = 100,
    pexels_key: str = "",
    unsplash_key: str = "",
) -> None:
    """Download real images for training."""
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    queries = {
        "person": "person walking street",
        "vehicle": "car vehicle street",
        "stairs": "stairs steps",
        "door": "door entrance",
        "wall": "wall barrier fence",
    }
    
    print(f"📥 Downloading real images for training...")
    print(f"   Classes: {', '.join(classes)}")
    print(f"   Target: {images_per_class} images per class\n")
    
    if not pexels_key and not unsplash_key:
        print("⚠️  No API keys provided!")
        print("\n   To get real images, you need API keys (both free):")
        print("   1. Pexels: https://www.pexels.com/api/")
        print("   2. Unsplash: https://unsplash.com/developers")
        print("\n   Or use COCO dataset (no key needed):")
        print("   python3 blind_navigation/download_coco_data.py extract --coco-dir <path>")
        print("\n   Or download manually from Google Images and organize in dataset/train/")
        return
    
    total_collected = 0
    
    for class_name in classes:
        query = queries.get(class_name, class_name)
        class_dir = train_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📦 {class_name}...")
        
        collected = 0
        
        # Try Pexels first
        if pexels_key:
            collected += download_from_pexels_api(query, images_per_class, pexels_key, class_dir)
        
        # Then Unsplash
        if collected < images_per_class and unsplash_key:
            remaining = images_per_class - collected
            collected += download_from_unsplash_api(query, remaining, unsplash_key, class_dir)
        
        if collected > 0:
            create_annotations_for_images(class_dir, class_name)
            total_collected += collected
            print(f"\n  ✅ Collected {collected} images")
        else:
            print(f"  ⚠️  No images collected (check API keys)")
    
    print(f"\n✅ Total: {total_collected} images downloaded")
    print(f"   Location: {train_dir}")
    print(f"\n⚠️  Next step: Annotate bounding boxes with LabelImg:")
    print(f"   pip install labelImg")
    print(f"   labelImg {train_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download real images for training")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("blind_navigation/dataset"),
        help="Output directory",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["person", "vehicle", "stairs", "door", "wall"],
        help="Classes to download",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=100,
        help="Images per class",
    )
    parser.add_argument(
        "--pexels-key",
        type=str,
        default="",
        help="Pexels API key (get from https://www.pexels.com/api/)",
    )
    parser.add_argument(
        "--unsplash-key",
        type=str,
        default="",
        help="Unsplash API key (get from https://unsplash.com/developers)",
    )
    
    args = parser.parse_args()
    
    download_real_images(
        output_dir=args.output_dir,
        classes=args.classes,
        images_per_class=args.images_per_class,
        pexels_key=args.pexels_key,
        unsplash_key=args.unsplash_key,
    )


if __name__ == "__main__":
    main()

