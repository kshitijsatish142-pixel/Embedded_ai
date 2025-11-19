"""
Training script for obstacle detection model for blind navigation.
Optimized for Raspberry Pi 4B deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class ObstacleDataset(Dataset):
    """Dataset for obstacle detection with bounding boxes."""
    
    def __init__(
        self,
        data_dir: Path,
        image_size: tuple[int, int] = (640, 640),
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Load annotations
        annotations_file = self.data_dir / "annotations.json"
        if not annotations_file.exists():
            raise FileNotFoundError(
                f"Annotations file not found: {annotations_file}\n"
                f"Create annotations.json with format:\n"
                f'{{"images": [{{"file": "img1.jpg", "objects": [{{"class": "person", "bbox": [x, y, w, h]}}]}}]}}'
            )
        
        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)
        
        # Class mapping
        self.classes = self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Transformations
        if augment:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def _get_classes(self) -> List[str]:
        """Extract unique classes from annotations."""
        classes = set()
        for item in self.annotations.get("images", []):
            for obj in item.get("objects", []):
                classes.add(obj["class"])
        return sorted(list(classes))
    
    def __len__(self) -> int:
        return len(self.annotations.get("images", []))
    
    def __getitem__(self, idx: int):
        item = self.annotations["images"][idx]
        img_path = self.data_dir / item["file"]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        original_size = image.size
        
        # Resize image
        image = image.resize(self.image_size, Image.BILINEAR)
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Prepare targets (bounding boxes + classes)
        targets = []
        for obj in item.get("objects", []):
            cls = obj["class"]
            bbox = obj["bbox"]  # [x, y, w, h] in original image coordinates
            
            # Normalize bbox to [0, 1] and adjust for resizing
            x, y, w, h = bbox
            x_norm = (x / original_size[0]) * (self.image_size[0] / original_size[0])
            y_norm = (y / original_size[1]) * (self.image_size[1] / original_size[1])
            w_norm = (w / original_size[0]) * (self.image_size[0] / original_size[0])
            h_norm = (h / original_size[1]) * (self.image_size[1] / original_size[1])
            
            # Convert to center format [cx, cy, w, h]
            cx = x_norm + w_norm / 2
            cy = y_norm + h_norm / 2
            
            targets.append({
                "class": self.class_to_idx[cls],
                "bbox": [cx, cy, w_norm, h_norm],
            })
        
        return image_tensor, targets


class LightweightObstacleDetector(nn.Module):
    """
    Lightweight object detection model optimized for Raspberry Pi.
    Based on MobileNet backbone with detection head.
    """
    
    def __init__(self, num_classes: int, input_size: tuple[int, int] = (640, 640)):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # MobileNet-like backbone
        from torchvision.models import mobilenet_v3_small
        backbone = mobilenet_v3_small(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.features.children()))
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, (5 + num_classes) * 5),  # 5 boxes * (4 bbox + 1 conf + num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.detection_head(features)
        batch_size = x.size(0)
        output = output.view(batch_size, 5, 5 + self.num_classes)
        return output


def train_model(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    device: str = "cpu",
):
    """Train obstacle detection model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device)
    
    # Create datasets
    train_dataset = ObstacleDataset(data_dir / "train", augment=True)
    
    # Create val dataset if exists, otherwise use train for validation
    val_dir = data_dir / "val"
    if (val_dir / "annotations.json").exists():
        val_dataset = ObstacleDataset(val_dir, augment=False)
    else:
        print("⚠️  No validation set found, using training set for validation")
        val_dataset = ObstacleDataset(data_dir / "train", augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model
    num_classes = len(train_dataset.classes)
    model = LightweightObstacleDetector(num_classes=num_classes)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    print(f"Training model with {num_classes} classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            
            # Forward pass (simplified - you'd implement proper loss here)
            outputs = model(images)
            
            # Loss computation (placeholder - implement proper detection loss)
            loss = nn.functional.mse_loss(outputs, torch.zeros_like(outputs))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                outputs = model(images)
                loss = nn.functional.mse_loss(outputs, torch.zeros_like(outputs))
                val_loss += loss.item()
        
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"✅ Saved best model (val loss: {avg_val_loss:.4f})")
    
    # Save final model and metadata
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    
    metadata = {
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "num_classes": num_classes,
        "input_size": train_dataset.image_size,
        "epochs": epochs,
    }
    
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {output_dir / 'best_model.pt'}")
    print(f"   Metadata saved to: {output_dir / 'model_metadata.json'}")


def main():
    parser = argparse.ArgumentParser(description="Train obstacle detection model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing train/val folders with images and annotations.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("blind_navigation/models"),
        help="Directory to save trained model",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()

