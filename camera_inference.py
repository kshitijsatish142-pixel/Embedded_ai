#!/usr/bin/env python3
"""
Real-time book detection using Raspberry Pi camera
"""
import torch
import sys
import os
from torchvision import transforms
from PIL import Image
import time
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import BookCNN

class BookDetector:
    def __init__(self, model_path="models/book_cnn.pth", image_size=128):
        """Initialize the book detector"""
        self.image_size = image_size
        self.device = torch.device("cpu")  # Pi uses CPU

        # Load model
        print("Loading model...")
        self.classes = ['book', 'other']
        self.model = BookCNN(num_classes=len(self.classes), image_size=image_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded! Classes: {self.classes}")

        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        """Predict if image contains a book"""
        # Convert to PIL Image if it's a numpy array
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Transform and predict
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = self.classes[predicted.item()]
        confidence_score = confidence.item()

        return predicted_class, confidence_score

    def run_camera(self, camera_index=0, show_fps=True):
        """Run real-time detection from camera"""
        print(f"\nStarting camera {camera_index}...")
        print("Press 'q' to quit\n")

        # Initialize camera
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Make prediction
                start_time = time.time()
                predicted_class, confidence = self.predict(frame)
                inference_time = (time.time() - start_time) * 1000  # ms

                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_start_time > 1:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()

                # Determine color based on prediction
                if predicted_class == "book":
                    color = (0, 255, 0)  # Green for book
                    text = f"BOOK DETECTED! ({confidence:.1%})"
                else:
                    color = (0, 0, 255)  # Red for no book
                    text = f"No book ({confidence:.1%})"

                # Draw on frame
                cv2.rectangle(frame, (10, 10), (630, 100), color, -1)
                cv2.putText(frame, text, (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                # Show inference time and FPS
                if show_fps:
                    info_text = f"Inference: {inference_time:.1f}ms | FPS: {current_fps:.1f}"
                    cv2.putText(frame, info_text, (10, 470),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Display frame
                cv2.imshow('Book Detection', frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released")

def main():
    """Main function"""
    print("=" * 50)
    print("Real-time Book Detection")
    print("=" * 50)

    # Check if model exists
    model_path = "models/book_cnn.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python src/train_cnn.py")
        return

    # Initialize detector
    detector = BookDetector(model_path)

    # Run camera detection
    detector.run_camera(camera_index=0)

if __name__ == "__main__":
    main()
