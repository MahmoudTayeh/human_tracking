from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
from detector import PersonDetector
from main import VIDEO_PATH
class HumanTracker:
    def __init__(self):
        """
        configure person detector and DeepSort tracker
        """
        self.detector = PersonDetector()
        
        # DeepSort configuration
        self.tracker = DeepSort(
            max_age=30,              # Number of frames before a track is deleted
            n_init=3,                # Number of detections required to confirm a track
            max_iou_distance=0.7,    # IoU distance threshold
            max_cosine_distance=0.3, # Appearance feature distance threshold
            embedder="mobilenet",    # Feature extraction model
            half=True,               # Use FP16 for faster inference
            embedder_gpu=True        # Use GPU for feature extraction
        )

        
        self.colors = {}  # color for each trackID
    
    def get_color(self, track_id):
        """
        one stable color per track ID
        """
        if track_id not in self.colors:
            np.random.seed(track_id)
            self.colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.colors[track_id]
    
    def process_frame(self, frame):
        """
        process a single frame: detect persons, track them, and annotate the frame
        """
        # 1. detect persons
        detections = self.detector.detect_persons(frame)
        
        # 2. change format for DeepSort
        # deepsort expects [ [x, y, w, h], confidence, class_name ]
        deepsort_input = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            w = x2 - x1
            h = y2 - y1
            deepsort_input.append([[x1, y1, w, h], conf, 'person'])
        
        # 3. update tracker
        tracks = self.tracker.update_tracks(deepsort_input, frame=frame)
        
        # 4. draw tracks
        annotated_frame = frame.copy()
        active_tracks = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()  # [left, top, right, bottom]
            
            x1, y1, x2, y2 = map(int, bbox)
            color = self.get_color(track_id)
            
            # draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # write track ID
            label = f'ID: {track_id}'
            cv2.putText(annotated_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            active_tracks.append({
                'id': track_id,
                'bbox': [x1, y1, x2, y2],
                'center': [(x1+x2)//2, (y1+y2)//2]
            })
        
        return annotated_frame, active_tracks

# test tracker
tracker = HumanTracker()

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

if ret:
    result_frame, tracks = tracker.process_frame(frame)
    print(f" tracked {len(tracks)} people")
    cv2.imwrite('outputs/tracking_test.jpg', result_frame)

cap.release()