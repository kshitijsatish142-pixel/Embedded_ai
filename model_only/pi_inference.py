"""
Standalone inference for obstacle detection on Raspberry Pi.
Works with Pi Camera Module v2 (5MP) or USB camera.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class LightweightObstacleDetector(nn.Module):
    """Same model architecture as training."""
    
    def __init__(self, num_classes: int, input_size: tuple[int, int] = (640, 640)):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        from torchvision.models import mobilenet_v3_small
        backbone = mobilenet_v3_small(pretrained=False)
        self.backbone = nn.Sequential(*list(backbone.features.children()))
        
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, (5 + num_classes) * 5),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.detection_head(features)
        batch_size = x.size(0)
        output = output.view(batch_size, 5, 5 + self.num_classes)
        return output


class ObstacleDetector:
    """
    Obstacle detection system for blind navigation.
    
    Usage:
        detector = ObstacleDetector(model_dir="/path/to/model")
        results = detector.detect(frame)
        detector.announce(results)
    """
    
    # Priority levels for different object types
    PRIORITY_LEVELS = {
        "person": 1,
        "vehicle": 1,
        "car": 1,
        "bus": 1,
        "truck": 1,
        "motorcycle": 1,
        "stairs": 1,
        "step": 1,
        "door": 1,
        "wall": 1,
        "barrier": 1,
        "furniture": 2,
        "chair": 2,
        "table": 2,
        "pole": 2,
        "post": 2,
        "curb": 2,
        "pothole": 2,
        "crosswalk": 3,
        "sign": 3,
    }
    
    def __init__(self, model_dir: str | Path, confidence_threshold: float = 0.5):
        self.model_dir = Path(model_dir)
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cpu")
        
        # Load metadata
        metadata_path = self.model_dir / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        self.classes = self.metadata["classes"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.input_size = tuple(self.metadata["input_size"])
        
        # Load model (check multiple possible filenames)
        model_path = self.model_dir / "model_weights.pt"
        if not model_path.exists():
            model_path = self.model_dir / "best_model.pt"
        if not model_path.exists():
            model_path = self.model_dir / "final_model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found in {self.model_dir}. Expected: model_weights.pt, best_model.pt, or final_model.pt")
        
        self.model = LightweightObstacleDetector(
            num_classes=len(self.classes),
            input_size=self.input_size,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"✅ Obstacle detector loaded")
        print(f"   Classes: {', '.join(self.classes)}")
        print(f"   Input size: {self.input_size}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect obstacles in a frame.
        
        Args:
            frame: BGR image from camera (numpy array)
        
        Returns:
            List of detections with keys: class, confidence, bbox, priority, distance
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize and preprocess
        image_resized = pil_image.resize(self.input_size, Image.BILINEAR)
        image_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Parse outputs (simplified - you'd implement proper NMS here)
        detections = []
        h, w = frame.shape[:2]
        
        # Process each detection box
        for box_idx in range(5):  # 5 boxes per image
            box_output = outputs[0, box_idx, :]
            
            # Extract bbox, confidence, class probs
            bbox = box_output[:4].cpu().numpy()  # [cx, cy, w, h] normalized
            conf = torch.sigmoid(box_output[4]).item()
            class_probs = torch.softmax(box_output[5:], dim=0).cpu().numpy()
            
            if conf < self.confidence_threshold:
                continue
            
            # Get best class
            class_idx = np.argmax(class_probs)
            class_name = self.idx_to_class[class_idx]
            class_conf = class_probs[class_idx]
            
            # Convert normalized bbox to pixel coordinates
            cx, cy, bw, bh = bbox
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            # Estimate distance (simplified - use bbox size)
            # Larger objects = closer, smaller = farther
            bbox_area = bw * bh * (w * h)
            distance = self._estimate_distance(bbox_area, class_name)
            
            # Get priority
            priority = self.PRIORITY_LEVELS.get(class_name.lower(), 3)
            
            detections.append({
                "class": class_name,
                "confidence": float(conf * class_conf),
                "bbox": [x1, y1, x2, y2],
                "priority": priority,
                "distance": distance,
            })
        
        # Sort by priority and distance
        detections.sort(key=lambda x: (x["priority"], x["distance"]))
        
        return detections
    
    def _estimate_distance(self, bbox_area: float, class_name: str) -> float:
        """Estimate distance in meters based on bbox size."""
        # Simplified estimation - you'd calibrate this with real measurements
        # Typical sizes at 1 meter distance (in pixels^2)
        typical_sizes = {
            "person": 50000,
            "vehicle": 200000,
            "car": 200000,
            "door": 30000,
            "stairs": 40000,
        }
        
        typical_size = typical_sizes.get(class_name.lower(), 50000)
        # Inverse relationship: larger area = closer
        distance = max(0.5, min(20.0, typical_size / (bbox_area + 1) * 2))
        return distance
    
    def announce(self, detections: List[Dict], audio_enabled: bool = False):
        """
        Announce detected obstacles (text output or audio).
        
        Args:
            detections: List of detections from detect()
            audio_enabled: If True, use text-to-speech (requires pyttsx3)
        """
        if not detections:
            return
        
        # Get critical detections (priority 1, within 5 meters)
        critical = [d for d in detections if d["priority"] == 1 and d["distance"] < 5.0]
        
        if critical:
            for det in critical[:3]:  # Top 3 critical
                message = f"{det['class']} detected {det['distance']:.1f} meters ahead"
                print(f"⚠️  {message}")
                
                if audio_enabled:
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.say(message)
                        engine.runAndWait()
                    except ImportError:
                        pass
        
        # Summary
        if detections:
            summary = f"Detected {len(detections)} objects"
            print(f"📊 {summary}")


def run_camera_detection(model_dir: str | Path, camera_index: int = 0, confidence: float = 0.5, use_pi_camera: bool = True):
    """
    Run real-time obstacle detection from camera.
    
    Args:
        model_dir: Path to model directory
        camera_index: Camera device index (for USB cameras, ignored for Pi Camera)
        confidence: Confidence threshold (0.0-1.0)
        use_pi_camera: If True, use Pi Camera Module (picamera2), else use USB camera (OpenCV)
    """
    detector = ObstacleDetector(model_dir, confidence_threshold=confidence)
    
    # Initialize camera
    camera = None
    pi_camera = None
    
    if use_pi_camera:
        try:
            from picamera2 import Picamera2
            pi_camera = Picamera2()
            # Configure camera for preview
            preview_config = pi_camera.create_preview_configuration(main={"size": (640, 480)})
            pi_camera.configure(preview_config)
            pi_camera.start()
            print("📹 Pi Camera started. Press 'q' to quit.")
        except ImportError:
            print("⚠️  picamera2 not found. Install with: pip install picamera2")
            print("📹 Falling back to USB camera...")
            use_pi_camera = False
        except Exception as e:
            print(f"⚠️  Pi Camera error: {e}")
            print("📹 Falling back to USB camera...")
            use_pi_camera = False
    
    if not use_pi_camera:
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            print(f"❌ Could not open USB camera {camera_index}")
            return
        print("📹 USB Camera started. Press 'q' to quit.")
    
    fps_counter = 0
    fps_start_time = time.time()
    
    try:
        while True:
            if use_pi_camera and pi_camera:
                # Capture from Pi Camera
                frame = pi_camera.capture_array()
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # Capture from USB camera
                ret, frame = camera.read()
                if not ret:
                    break
            
            # Detect obstacles
            detections = detector.detect(frame)
            
            # Draw detections on frame
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                color = (0, 0, 255) if det["priority"] == 1 else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class']} {det['distance']:.1f}m"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Announce critical detections
            detector.announce(detections, audio_enabled=False)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                print(f"FPS: {fps:.1f}")
            
            # Display frame (optional - comment out for headless)
            cv2.imshow("Obstacle Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        if use_pi_camera and pi_camera:
            pi_camera.stop()
        elif camera:
            camera.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run obstacle detection on Raspberry Pi")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("."),
        help="Directory containing model files (default: current directory)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (for USB cameras only)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--usb-camera",
        action="store_true",
        help="Use USB camera instead of Pi Camera Module",
    )
    
    args = parser.parse_args()
    
    run_camera_detection(args.model_dir, args.camera, args.confidence, use_pi_camera=not args.usb_camera)


if __name__ == "__main__":
    main()

