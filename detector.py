from ultralytics import YOLO
import cv2
import numpy as np
from main import VIDEO_PATH
class PersonDetector:
    def __init__(self, model_name='yolov8x.pt'):
        """
        download model YOLO
        yolov8n.pt = fast but less accurate model
        yolov8m.pt = mid model
        yolov8x.pt = large but more accurate model
        """
        self.model = YOLO(model_name)
        self.person_class_id = 0  # person class ID in COCO dataset
    
    def detect_persons(self, frame):
        """
        detect persons in a given frame
        returns list of bounding boxes [x1, y1, x2, y2, confidence]
        """
        results = self.model(frame, classes=[self.person_class_id], verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                # filter by confidence threshold
                if confidence > 0.5:
                    detections.append([x1, y1, x2, y2, confidence])
        
        return np.array(detections)

# create detector instance
detector = PersonDetector()

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

if ret:
    detections = detector.detect_persons(frame)
    print(f"detected {len(detections)} persons in the first frame")
    
    # رسم المربعات
    for det in detections:
        x1, y1, x2, y2, conf = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{conf:.2f}', (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite('outputs/detection_test.jpg', frame)
    print("output saved to outputs/detection_test.jpg")

cap.release()