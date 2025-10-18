# Raspberry Pi Setup Guide

This guide will help you deploy the book detection model on your Raspberry Pi with a camera for real-time detection.

## Prerequisites

- Raspberry Pi 3/4 (4 GB RAM recommended)
- Raspberry Pi Camera Module or USB webcam
- Raspberry Pi OS (64-bit recommended)
- Internet connection

## Step 1: Clone the Repository on Raspberry Pi

```bash
# SSH into your Raspberry Pi
ssh pi@raspberrypi.local

# Clone the repository
git clone https://github.com/kshitijsatish142-pixel/Embedded_ai.git
cd Embedded_ai
```

## Step 2: Set Up Python Environment

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y python3-pip python3-venv
sudo apt-get install -y libopencv-dev python3-opencv
sudo apt-get install -y libatlas-base-dev libopenblas-dev

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 3: Install Python Dependencies

```bash
# Install lightweight requirements for Pi
pip install -r requirements_pi.txt

# This will install:
# - PyTorch (CPU version)
# - TorchVision
# - OpenCV
# - Pillow
# - NumPy
```

**Note**: Installing PyTorch on Raspberry Pi can take 10-30 minutes. Be patient!

## Step 4: Enable Camera

### For Raspberry Pi Camera Module:

```bash
# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Reboot
sudo reboot
```

### For USB Webcam:
No additional setup needed!

## Step 5: Download the Trained Model

You have two options:

### Option A: Pull from GitHub (if model is committed)
```bash
git pull origin main
```

### Option B: Transfer model manually via SCP
```bash
# Run this on your Mac/PC (not on Pi)
scp models/book_cnn.pth pi@raspberrypi.local:~/Embedded_ai/models/
```

## Step 6: Run Real-time Detection

```bash
# Make sure you're in the project directory
cd ~/Embedded_ai

# Activate virtual environment
source venv/bin/activate

# Run camera inference
python camera_inference.py
```

**Controls:**
- Press `q` to quit the application

## Expected Performance

- **Inference Speed**: 50-200ms per frame on Pi 4
- **FPS**: 5-20 FPS depending on Pi model
- **Accuracy**: ~80% (same as trained model)

## Troubleshooting

### Camera not detected
```bash
# For Pi Camera Module
vcgencmd get_camera

# For USB webcam
ls /dev/video*
```

### "No module named cv2"
```bash
sudo apt-get install -y python3-opencv
pip install opencv-python
```

### PyTorch installation fails
```bash
# Install dependencies first
sudo apt-get install -y libopenblas-dev libblas-dev m4 cmake
pip install torch torchvision --no-cache-dir
```

### Low FPS / Slow inference
- Use a Pi 4 with 4GB+ RAM
- Close other applications
- Consider using a smaller image size (64x64) in camera_inference.py
- Reduce camera resolution

## Optimizing for Better Performance

### 1. Reduce Image Size
Edit `camera_inference.py`:
```python
detector = BookDetector(model_path, image_size=64)  # Instead of 128
```

### 2. Lower Camera Resolution
Edit `camera_inference.py`:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Instead of 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Instead of 480
```

### 3. Use Headless Mode (No Display)
For deployment without monitor, modify the script to save detections or send alerts instead of displaying.

## Next Steps

- Mount your Pi with the camera in a fixed position
- Point the camera at books to test detection
- Experiment with different lighting conditions
- Consider adding LED indicators for book detection status
- Log detection events to a file for analysis

## Remote Access (Optional)

To view the camera feed remotely:

```bash
# Install VNC
sudo apt-get install realvnc-vnc-server

# Enable VNC
sudo raspi-config
# Interface Options > VNC > Enable
```

Then connect using VNC Viewer from your computer.

## Support

For issues, check:
- GitHub repository: https://github.com/kshitijsatish142-pixel/Embedded_ai
- Ensure all files are transferred correctly
- Check camera connections
- Verify model file exists in models/
