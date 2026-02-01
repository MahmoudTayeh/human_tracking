"""
Person Detector using YOLO
"""

import cv2
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """
    Detects people in images and video streams
    """
    
    def __init__(self, model_path='yolov8x.pt', confidence=0.5, device='cuda'):
        """
        Initialize the detector
        
        Args:
            model_path: Path to the YOLO model
            confidence: Detection confidence threshold
            device: Device to run the model on (cuda/cpu)
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.person_class_id = 0  # Person class in COCO dataset
        
        print(f"✅ YOLO model loaded: {model_path}")
        print(f"🖥️  Device in use: {device}")
    
    def detect(self, frame):
        """
        Detect people in a single frame
        
        Args:
            frame: Input frame to process
            
        Returns:
            np.array: [[x1, y1, x2, y2, confidence], ...]
        """
        # Run the model
        results = self.model(
            frame,
            classes=[self.person_class_id],
            conf=self.confidence,
            device=self.device,
            verbose=False
        )
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf])
        
        return np.array(detections) if detections else np.array([])
    
    def convert_to_deepsort_format(self, detections):
        """
        Convert YOLO detections to DeepSort format
        
        Args:
            detections: YOLO detections [[x1, y1, x2, y2, conf], ...]
            
        Returns:
            list: [[[x, y, w, h], conf, 'person'], ...]
        """
        deepsort_format = []
        
        for det in detections:
            if len(det) == 0:
                continue
                
            x1, y1, x2, y2, conf = det
            w = x2 - x1
            h = y2 - y1
            
            deepsort_format.append([[x1, y1, w, h], conf, 'person'])
        
        return deepsort_format
