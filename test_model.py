import torch
import sys
import os
from torchvision import transforms
from PIL import Image
import random

# Add src to path to import BookCNN
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import BookCNN

# --- Paths ---
model_path = "models/book_cnn.pth"
test_data_path = "data/books/valid"  # using validation data for testing

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get class names
classes = sorted([d for d in os.listdir(test_data_path)
                  if os.path.isdir(os.path.join(test_data_path, d))])
num_classes = len(classes)
print(f"Classes: {classes}")

model = BookCNN(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")

# --- Transform (same as training) ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Pick random test images from each class ---
print("\nTesting model on sample images:\n")
for class_name in classes:
    class_path = os.path.join(test_data_path, class_name)
    images = [f for f in os.listdir(class_path)
              if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        continue

    # Test 3 random images from this class
    for _ in range(min(3, len(images))):
        img_name = random.choice(images)
        img_path = os.path.join(class_path, img_name)

        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        # --- Predict ---
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = classes[predicted.item()]
        print(f"Image: {class_name}/{img_name}")
        print(f"  True class: {class_name}")
        print(f"  Predicted: {predicted_class} (confidence: {confidence.item():.2%})")
        print(f"  {'✓ CORRECT' if predicted_class == class_name else '✗ WRONG'}\n")
