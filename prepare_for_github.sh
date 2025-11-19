#!/bin/bash
# Prepare and push to GitHub repo

cd "$(dirname "$0")" || exit

echo "📦 Preparing for GitHub..."

# Create .gitignore if doesn't exist
if [ ! -f .gitignore ]; then
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/

# Model files (large)
*.pt
*.pth
*.h5
*.ckpt

# Data
train2017/
val2017/
annotations/
*.zip
dataset/
coco_dataset/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
runs/
wandb/

# Keep pi_package but ignore large model files there
blind_navigation/models/obstacle_detector/*.pt
blind_navigation/models/pi_package/model_weights.pt
EOF
    echo "✅ Created .gitignore"
fi

# Copy essential files to root for repo structure
echo "📁 Organizing files..."

# Create src structure if needed
mkdir -p src/blind_navigation

# Copy key files
cp -r blind_navigation/models/pi_package src/ 2>/dev/null || true
cp blind_navigation/pi_inference.py src/ 2>/dev/null || true
cp blind_navigation/train_obstacle_detector.py src/blind_navigation/ 2>/dev/null || true
cp blind_navigation/package_for_pi.py src/blind_navigation/ 2>/dev/null || true

# Create requirements files
if [ ! -f requirements.txt ]; then
    cat > requirements.txt << EOF
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
numpy>=1.20.0
Pillow>=8.0.0
pycocotools>=2.0.0
tqdm>=4.60.0
EOF
    echo "✅ Created requirements.txt"
fi

if [ ! -f requirements_pi.txt ]; then
    cp blind_navigation/models/pi_package/requirements_pi.txt . 2>/dev/null || true
fi

# Create camera inference if doesn't exist
if [ ! -f camera_inference.py ]; then
    cat > camera_inference.py << 'EOF'
"""
Camera inference for obstacle detection on Raspberry Pi.
"""

from pi_inference import ObstacleDetector
import cv2

def main():
    detector = ObstacleDetector("pi_package")
    
    # Try Pi Camera first, then USB
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("❌ Could not open camera")
        return
    
    print("📹 Camera started. Press 'q' to quit.")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        detector.announce(detections, audio_enabled=False)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 0, 255) if det["priority"] == 1 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['distance']:.1f}m"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow("Obstacle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
EOF
    echo "✅ Created camera_inference.py"
fi

echo ""
echo "✅ Preparation complete!"
echo ""
echo "Next steps:"
echo "1. Initialize git (if not already):"
echo "   git init"
echo ""
echo "2. Add remote:"
echo "   git remote add origin https://github.com/kshitijsatish142-pixel/Embedded_ai.git"
echo ""
echo "3. Add and commit:"
echo "   git add ."
echo "   git commit -m 'Add obstacle detection model and Pi package'"
echo ""
echo "4. Push:"
echo "   git push -u origin main"

