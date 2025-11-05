# CNN training script for book classification
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ===== 1. Hyperparameters =====
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
image_size = 128
num_classes = 2  # Book / Non-book if binary, else auto-detected

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Deterministic seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ===== 2. Data Preparation =====
# Resolve paths relative to this file, not current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "books"))

if not (os.path.isdir(os.path.join(data_dir, "train")) and os.path.isdir(os.path.join(data_dir, "valid"))):
    raise FileNotFoundError(f"Expected 'train' and 'valid' directories under {data_dir}")

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=transform)

# Ensure class mapping is consistent between splits
if train_dataset.classes != val_dataset.classes:
    raise ValueError(f"Class mismatch between train {train_dataset.classes} and valid {val_dataset.classes}")

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Empty dataset detected. Ensure images exist under train/ and valid/ subfolders.")

num_workers = 0  # Safe default across platforms; increase if desired
pin_memory = device.type == "cuda"

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

# ===== 3. Model Definition =====
class BookCNN(nn.Module):
    def __init__(self, num_classes):
        super(BookCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (image_size // 8) * (image_size // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

num_classes = len(train_dataset.classes)
model = BookCNN(num_classes).to(device)
print("Classes:", train_dataset.classes)
print(f"Samples -> train: {len(train_dataset)} | valid: {len(val_dataset)}")

# ===== 4. Loss & Optimizer =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===== 5. Training Loop =====
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

# ===== 6. Save Model =====
models_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))
os.makedirs(models_dir, exist_ok=True)
save_path = os.path.join(models_dir, "book_cnn.pth")
torch.save(model.state_dict(), save_path)
print(f"✅ Model saved to {save_path}")
