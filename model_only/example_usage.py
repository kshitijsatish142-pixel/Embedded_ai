"""
Example: Use obstacle detector on Raspberry Pi
"""

from pi_inference import ObstacleDetector
import cv2

# Initialize detector
detector = ObstacleDetector(".")

# Open camera (0 for Pi Camera, 1+ for USB)
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    # Detect obstacles
    detections = detector.detect(frame)
    
    # Announce critical obstacles
    detector.announce(detections, audio_enabled=False)
    
    # Draw on frame (optional)
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
