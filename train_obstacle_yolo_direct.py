#!/usr/bin/env python3
"""
Train YOLO model for obstacle detection - Direct Python approach
Following exact workflow from Train_YOLO_Models.ipynb
This uses the YOLO class directly instead of CLI commands
"""
import os
import sys
from ultralytics import YOLO

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    # Check if data.yaml exists
    if not os.path.exists("data.yaml"):
        print("ERROR: data.yaml not found!")
        print("Please run: python create_data_yaml.py first")
        print("Or run: python train_obstacle_yolo.py (which does all steps)")
        sys.exit(1)
    
    # Training parameters (following notebook)
    model = "yolo11s.pt"  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11xl.pt
    epochs = 60  # Use 60 if <200 images, 40 if >200 images
    imgsz = 640  # Standard YOLO resolution
    
    print("Loading model...")
    yolo_model = YOLO(model)
    
    print(f"\nStarting training with:")
    print(f"  Model: {model}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Data config: data.yaml\n")
    
    # Train (following notebook: yolo detect train data=data.yaml model=yolo11s.pt epochs=60 imgsz=640)
    results = yolo_model.train(
        data="data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        project="runs/detect",
        name="train",
        exist_ok=True
    )
    
    print("\n✅ Training completed!")
    print(f"Best model saved at: runs/detect/train/weights/best.pt")
    print(f"Results saved at: runs/detect/train/")

if __name__ == "__main__":
    main()

