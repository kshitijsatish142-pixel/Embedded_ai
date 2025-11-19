"""
Camera module for Raspberry Pi using rpicam/libcamera.
Uses OpenCV with v4l2 backend to access libcamera devices.
Includes obstacle detection functionality.
"""

from __future__ import annotations

import cv2
import subprocess
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List

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
    """Obstacle detection system."""
    
    PRIORITY_LEVELS = {
        "person": 1, "vehicle": 1, "car": 1, "bus": 1, "truck": 1,
        "motorcycle": 1, "stairs": 1, "step": 1, "door": 1, "wall": 1,
        "barrier": 1, "furniture": 2, "chair": 2, "table": 2, "pole": 2,
        "post": 2, "curb": 2, "pothole": 2, "crosswalk": 3, "sign": 3,
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
        
        # Load model
        model_path = self.model_dir / "model_weights.pt"
        if not model_path.exists():
            model_path = self.model_dir / "best_model.pt"
        if not model_path.exists():
            model_path = self.model_dir / "final_model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found in {self.model_dir}")
        
        self.model = LightweightObstacleDetector(
            num_classes=len(self.classes),
            input_size=self.input_size,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"✅ Obstacle detector loaded - Classes: {', '.join(self.classes)}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect obstacles in a frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        image_resized = pil_image.resize(self.input_size, Image.BILINEAR)
        image_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        detections = []
        h, w = frame.shape[:2]
        
        for box_idx in range(5):
            box_output = outputs[0, box_idx, :]
            bbox = box_output[:4].cpu().numpy()
            conf = torch.sigmoid(box_output[4]).item()
            class_probs = torch.softmax(box_output[5:], dim=0).cpu().numpy()
            
            if conf < self.confidence_threshold:
                continue
            
            class_idx = np.argmax(class_probs)
            class_name = self.idx_to_class[class_idx]
            class_conf = class_probs[class_idx]
            
            cx, cy, bw, bh = bbox
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            bbox_area = bw * bh * (w * h)
            distance = self._estimate_distance(bbox_area, class_name)
            priority = self.PRIORITY_LEVELS.get(class_name.lower(), 3)
            
            detections.append({
                "class": class_name,
                "confidence": float(conf * class_conf),
                "bbox": [x1, y1, x2, y2],
                "priority": priority,
                "distance": distance,
            })
        
        detections.sort(key=lambda x: (x["priority"], x["distance"]))
        return detections
    
    def _estimate_distance(self, bbox_area: float, class_name: str) -> float:
        """Estimate distance in meters."""
        typical_sizes = {
            "person": 50000, "vehicle": 200000, "car": 200000,
            "door": 30000, "stairs": 40000,
        }
        typical_size = typical_sizes.get(class_name.lower(), 50000)
        distance = max(0.5, min(20.0, typical_size / (bbox_area + 1) * 2))
        return distance
    
    def announce(self, detections: List[Dict], audio_enabled: bool = False):
        """Announce detected obstacles."""
        if not detections:
            return
        
        critical = [d for d in detections if d["priority"] == 1 and d["distance"] < 5.0]
        if critical:
            for det in critical[:3]:
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
        
        if detections:
            print(f"📊 Detected {len(detections)} objects")


class Camera:
    """
    Camera interface using rpicam/libcamera via v4l2.
    Includes optional obstacle detection.
    
    Usage:
        # Camera only
        camera = Camera(use_pi_camera=True)
        ret, frame = camera.read()
        
        # Camera with detection
        camera = Camera(use_pi_camera=True, model_dir=".")
        ret, frame = camera.read()
        detections = camera.detect(frame)
    """
    
    def __init__(
        self,
        use_pi_camera: bool = True,
        camera_index: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        model_dir: Optional[str | Path] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize camera.
        
        Args:
            use_pi_camera: If True, use Pi Camera Module via libcamera/v4l2, else USB camera
            camera_index: Camera device index (0 for Pi Camera via /dev/video0, or USB camera index)
            resolution: Camera resolution (width, height)
            model_dir: Optional path to model directory for obstacle detection
            confidence_threshold: Detection confidence threshold (0.0-1.0)
        """
        self.use_pi_camera = use_pi_camera
        self.camera_index = camera_index
        self.resolution = resolution
        self.camera = None
        self.detector = None
        self._initialized = False
        
        # Initialize detector if model_dir provided
        if model_dir:
            try:
                self.detector = ObstacleDetector(model_dir, confidence_threshold=confidence_threshold)
            except Exception as e:
                print(f"⚠️  Could not load detector: {e}")
                print("   Continuing without detection...")
        
        self._initialize()
    
    def _find_libcamera_device(self) -> Optional[int]:
        """Find libcamera v4l2 device."""
        # Check common libcamera device paths
        for i in range(10):
            device_path = Path(f"/dev/video{i}")
            if device_path.exists():
                # Check if it's a libcamera device by checking if rpicam can see it
                try:
                    result = subprocess.run(
                        ["rpicam-hello", "--list-cameras"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0 and "Available cameras" in result.stdout:
                        return i
                except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                    pass
        return None
    
    def _initialize(self):
        """Initialize the camera."""
        if self.use_pi_camera:
            # Try to find libcamera device
            libcamera_idx = self._find_libcamera_device()
            
            if libcamera_idx is not None:
                # Use libcamera via v4l2
                # OpenCV with v4l2 backend for libcamera
                self.camera = cv2.VideoCapture(libcamera_idx, cv2.CAP_V4L2)
                if not self.camera.isOpened():
                    # Try without explicit backend
                    self.camera = cv2.VideoCapture(libcamera_idx)
            else:
                # Fallback to /dev/video0 (common libcamera device)
                self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
                if not self.camera.isOpened():
                    self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                raise RuntimeError("Could not open Pi Camera. Make sure rpicam/libcamera is installed.")
            
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Set buffer size to reduce latency
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print("📹 Pi Camera (rpicam/libcamera) started.")
        else:
            # USB camera
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                raise RuntimeError(f"Could not open USB camera {self.camera_index}")
            
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            print(f"📹 USB Camera {self.camera_index} started.")
        
        self._initialized = True
    
    def read(self) -> Optional[Tuple[bool, any]]:
        """
        Read a frame from the camera.
        
        Returns:
            (ret, frame) tuple, or None if camera not initialized
        """
        if not self._initialized or not self.camera:
            return None
        
        return self.camera.read()
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect obstacles in a frame.
        
        Args:
            frame: BGR image frame from camera
        
        Returns:
            List of detections with keys: class, confidence, bbox, priority, distance
            Returns empty list if detector not initialized
        """
        if not self.detector:
            return []
        
        return self.detector.detect(frame)
    
    def read_and_detect(self) -> Optional[Tuple[bool, np.ndarray, List[Dict]]]:
        """
        Read a frame and detect obstacles in one call.
        
        Returns:
            (ret, frame, detections) tuple, or None if camera not initialized
            detections will be empty list if detector not initialized
        """
        result = self.read()
        if result is None:
            return None
        
        ret, frame = result
        if not ret:
            return (False, None, [])
        
        detections = self.detect(frame) if self.detector else []
        return (ret, frame, detections)
    
    def announce(self, detections: List[Dict], audio_enabled: bool = False):
        """
        Announce detected obstacles.
        
        Args:
            detections: List of detections from detect()
            audio_enabled: If True, use text-to-speech
        """
        if self.detector:
            self.detector.announce(detections, audio_enabled=audio_enabled)
    
    def is_opened(self) -> bool:
        """Check if camera is opened and ready."""
        if self.camera:
            return self.camera.isOpened()
        return False
    
    def release(self):
        """Release camera resources."""
        if self.camera:
            self.camera.release()
            self.camera = None
        self._initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.release()


def test_camera():
    """Test camera functionality with optional detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test camera")
    parser.add_argument(
        "--usb-camera",
        action="store_true",
        help="Use USB camera instead of Pi Camera",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (for USB cameras)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model directory for obstacle detection (optional)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold",
    )
    
    args = parser.parse_args()
    
    try:
        with Camera(
            use_pi_camera=not args.usb_camera,
            camera_index=args.camera,
            model_dir=args.model_dir,
            confidence_threshold=args.confidence
        ) as camera:
            print("Camera initialized. Press 'q' to quit.")
            if camera.detector:
                print("✅ Detection enabled")
            
            while True:
                if camera.detector:
                    # Use read_and_detect for convenience
                    result = camera.read_and_detect()
                    if result is None:
                        break
                    
                    ret, frame, detections = result
                    if not ret:
                        break
                    
                    # Draw detections
                    for det in detections:
                        x1, y1, x2, y2 = det["bbox"]
                        color = (0, 0, 255) if det["priority"] == 1 else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{det['class']} {det['distance']:.1f}m"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Announce critical detections
                    camera.announce(detections)
                else:
                    # Camera only
                    result = camera.read()
                    if result is None:
                        break
                    
                    ret, frame = result
                    if not ret:
                        break
                
                # Display frame
                cv2.imshow("Camera Test", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()

