#!/usr/bin/env python3
"""
Train YOLO model for obstacle detection
Following exact workflow from Train_YOLO_Models.ipynb
"""
import os
import subprocess
import sys

# Step 1: Prepare data structure (assumes images and labels are in obstacle_data/)
# Step 2: Split train/val (using train_val_split.py)
# Step 3: Create data.yaml
# Step 4: Train model

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    # Step 1: Check if data needs to be split
    custom_data_path = "obstacle_data"
    data_path = "data"
    
    if not os.path.exists(os.path.join(data_path, "train", "images")):
        print("Step 1: Splitting data into train/validation...")
        if not os.path.exists(custom_data_path):
            print(f"ERROR: {custom_data_path} directory not found!")
            print("Please organize your obstacle images in:")
            print(f"  {custom_data_path}/images/  (image files)")
            print(f"  {custom_data_path}/labels/  (YOLO format .txt annotation files)")
            print(f"  {custom_data_path}/classes.txt  (class names, one per line)")
            sys.exit(1)
        
        # Run train_val_split.py
        cmd = [sys.executable, "utils/train_val_split.py", 
               f"--datapath={custom_data_path}", "--train_pct=0.9"]
        result = subprocess.run(cmd, check=True)
        print("Data split completed!")
    
    # Step 2: Create data.yaml
    print("\nStep 2: Creating data.yaml config file...")
    cmd = [sys.executable, "create_data_yaml.py"]
    result = subprocess.run(cmd, check=True)
    
    # Step 3: Train model
    print("\nStep 3: Starting YOLO training...")
    print("Following notebook command: yolo detect train data=data.yaml model=yolo11s.pt epochs=60 imgsz=640")
    
    # Training parameters (adjust as needed)
    model = "yolo11s.pt"  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11xl.pt
    epochs = 60  # Use 60 if <200 images, 40 if >200 images
    imgsz = 640  # Standard YOLO resolution
    
    # Use ultralytics CLI (yolo command or python -m ultralytics)
    import shutil
    yolo_cmd = shutil.which("yolo")
    if yolo_cmd:
        cmd = [
            "yolo", "detect", "train",
            f"data=data.yaml",
            f"model={model}",
            f"epochs={epochs}",
            f"imgsz={imgsz}"
        ]
    else:
        # Fallback to python -m ultralytics
        cmd = [
            sys.executable, "-m", "ultralytics", "yolo", "detect", "train",
            f"data=data.yaml",
            f"model={model}",
            f"epochs={epochs}",
            f"imgsz={imgsz}"
        ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    
    print("\n✅ Training completed!")
    print(f"Best model saved at: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()

