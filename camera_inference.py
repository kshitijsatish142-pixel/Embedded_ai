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
