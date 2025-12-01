#!/usr/bin/env python3
"""
Import obstacle dataset for blind people
This script helps organize your obstacle images and labels into the YOLO format
"""
import os
import shutil
import sys
from pathlib import Path

def import_from_folder(source_folder, dest_folder="obstacle_data"):
    """
    Import images and labels from a source folder
    Assumes source folder has images and corresponding label files
    """
    source = Path(source_folder)
    dest_images = Path(dest_folder) / "images"
    dest_labels = Path(dest_folder) / "labels"
    
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    
    if not source.exists():
        print(f"ERROR: Source folder not found: {source_folder}")
        return False
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = []
    for ext in image_extensions:
        images.extend(source.rglob(f'*{ext}'))
    
    if not images:
        print(f"ERROR: No images found in {source_folder}")
        print("Looking for: .jpg, .jpeg, .png files")
        return False
    
    print(f"Found {len(images)} images")
    
    # Copy images and find corresponding labels
    copied_images = 0
    copied_labels = 0
    
    for img_path in images:
        # Copy image
        dest_img = dest_images / img_path.name
        shutil.copy2(img_path, dest_img)
        copied_images += 1
        
        # Look for corresponding label file (.txt)
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            dest_label = dest_labels / label_path.name
            shutil.copy2(label_path, dest_label)
            copied_labels += 1
        else:
            print(f"Warning: No label file found for {img_path.name}")
    
    print(f"\n✅ Imported:")
    print(f"  Images: {copied_images}")
    print(f"  Labels: {copied_labels}")
    
    if copied_labels < copied_images:
        print(f"\n⚠️  Warning: {copied_images - copied_labels} images are missing label files!")
        print("You'll need to create YOLO format .txt label files for these images.")
    
    return True

def import_from_zip(zip_path, dest_folder="obstacle_data"):
    """Import from a zip file (like from Label Studio)"""
    import zipfile
    
    dest = Path(dest_folder)
    dest.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest / "temp")
    
    # Look for images and labels folders in extracted content
    temp_folder = dest / "temp"
    images_folder = None
    labels_folder = None
    
    # Check common structures
    if (temp_folder / "images").exists():
        images_folder = temp_folder / "images"
    if (temp_folder / "labels").exists():
        labels_folder = temp_folder / "labels"
    
    # Also check if images/labels are directly in temp
    if not images_folder:
        # Look for any folder with images
        for item in temp_folder.iterdir():
            if item.is_dir():
                # Check if it contains images
                for ext in ['.jpg', '.png', '.jpeg']:
                    if list(item.rglob(f'*{ext}')):
                        images_folder = item
                        break
    
    if images_folder:
        print(f"Found images in: {images_folder}")
        # Copy to final location
        dest_images = dest / "images"
        dest_labels = dest / "labels"
        
        for img in images_folder.rglob('*.jpg'):
            shutil.copy2(img, dest_images / img.name)
        for img in images_folder.rglob('*.png'):
            shutil.copy2(img, dest_images / img.name)
        for img in images_folder.rglob('*.jpeg'):
            shutil.copy2(img, dest_images / img.name)
        
        if labels_folder:
            for label in labels_folder.rglob('*.txt'):
                shutil.copy2(label, dest_labels / label.name)
        
        # Clean up temp
        shutil.rmtree(temp_folder)
        print("✅ Import completed!")
        return True
    else:
        print("ERROR: Could not find images folder in zip file")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python import_obstacle_data.py <source_folder>")
        print("  python import_obstacle_data.py <source_zip_file>")
        print("\nExample:")
        print("  python import_obstacle_data.py /path/to/obstacle/images")
        print("  python import_obstacle_data.py obstacle_dataset.zip")
        sys.exit(1)
    
    source = sys.argv[1]
    
    if not os.path.exists(source):
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)
    
    if source.endswith('.zip'):
        success = import_from_zip(source)
    else:
        success = import_from_folder(source)
    
    if success:
        print("\n✅ Dataset imported successfully!")
        print("Next steps:")
        print("1. Check obstacle_data/classes.txt has your obstacle classes")
        print("2. Run: python train_obstacle_yolo.py")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

