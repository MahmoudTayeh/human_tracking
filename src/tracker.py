"""
Tracking system using DeepSort with Face Recognition
"""

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.detector import PersonDetector

# Import face recognizer if available
try:
    from src.face_recognizer import FaceRecognizerInsightFace
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️  Face recognition not available. Install insightface to enable.")


class HumanTracker:
    """
    Integrated system for person detection, tracking, and face recognition
    """
    
    def __init__(self, config):
        """
        Initialize the tracking system
        
        Args:
            config: Configuration dictionary
        """
        # Initialize detector
        self.detector = PersonDetector(
            model_path=config['model']['yolo_model'],
            confidence=config['model']['confidence_threshold'],
            device=config['model']['device']
        )
        
        # Initialize tracker
        self.tracker = DeepSort(
            max_age=config['tracker']['max_age'],
            n_init=config['tracker']['n_init'],
            max_iou_distance=config['tracker']['max_iou_distance'],
            max_cosine_distance=config['tracker']['max_cosine_distance'],
            embedder=config['tracker']['embedder'],
            half=True,
            embedder_gpu=config['tracker']['use_gpu']
        )
        
        # Initialize face recognizer if enabled
        self.face_recognizer = None
        if FACE_RECOGNITION_AVAILABLE and config.get('face_recognition', {}).get('enabled', False):
            try:
                self.face_recognizer = FaceRecognizerInsightFace(
                    database_path=config['face_recognition'].get('database_path', 'data/face_database.pkl'),
                    similarity_threshold=config['face_recognition'].get('similarity_threshold', 0.4),
                    recognition_interval=config['face_recognition'].get('recognition_interval', 10)
                )
                print("✅ Face recognition enabled")
            except Exception as e:
                print(f"⚠️  Could not initialize face recognition: {e}")
                self.face_recognizer = None
        
        # Store colors for each track ID
        self.colors = {}
        
        # Store track histories
        self.track_history = {}
        
        print("✅ Tracking system initialized")
    
    def get_color(self, track_id):
        """
        Get a consistent color for each track ID
        """
        if track_id not in self.colors:
            # Convert track_id to integer if it's a string
            if isinstance(track_id, str):
                seed = abs(hash(track_id)) % (2**32)
            else:
                seed = int(track_id)

            np.random.seed(seed)
            self.colors[track_id] = tuple(
                np.random.randint(0, 255, 3).tolist()
            )
        return self.colors[track_id]
    
    def process_frame(self, frame):
        """
        Process a single frame with detection, tracking, and face recognition
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (annotated_frame, active_tracks)
        """
        # 1. Detect people
        detections = self.detector.detect(frame)
        
        # 2. Convert detections to DeepSort format
        deepsort_input = self.detector.convert_to_deepsort_format(detections)
        
        # 3. Update tracker
        tracks = self.tracker.update_tracks(deepsort_input, frame=frame)
        
        # 4. Process tracking results
        annotated_frame = frame.copy()
        active_tracks = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            
            color = self.get_color(track_id)
            center = [(x1 + x2) // 2, (y1 + y2) // 2]
            
            # Save position to history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(center)
            
            active_tracks.append({
                'id': track_id,
                'bbox': [x1, y1, x2, y2],
                'center': center,
                'color': color
            })
        
        # 5. Run face recognition if enabled
        track_names = {}
        if self.face_recognizer is not None and len(active_tracks) > 0:
            track_names = self.face_recognizer.recognize_faces_in_tracks(frame, active_tracks)
        
        # 6. Draw results
        for track_data in active_tracks:
            track_id = track_data['id']
            x1, y1, x2, y2 = track_data['bbox']
            color = track_data['color']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and name
            name = track_names.get(track_id, "Unknown")
            if name != "Unknown":
                label = f'ID: {track_id} - {name}'
                # Use green for recognized faces
                label_color = (0, 255, 0)
            else:
                label = f'ID: {track_id}'
                label_color = color
            
            # Draw label with background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - 25),
                (x1 + label_size[0] + 10, y1),
                label_color,
                -1
            )
            cv2.putText(
                annotated_frame, label, (x1 + 5, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        return annotated_frame, active_tracks

    
    def get_statistics(self):
        """
        Get tracking statistics
        
        Returns:
            dict: Statistics for each track
        """
        stats = {}
        
        for track_id, positions in self.track_history.items():
            # Calculate total traveled distance
            total_distance = 0
            for i in range(1, len(positions)):
                x1, y1 = positions[i-1]
                x2, y2 = positions[i]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                total_distance += distance
            
            # Get name if face recognition is enabled
            name = "Unknown"
            if self.face_recognizer is not None:
                name = self.face_recognizer.get_name_for_track(track_id)
            
            stats[track_id] = {
                'name': name,
                'frames': len(positions),
                'total_distance': total_distance,
                'start_position': positions[0],
                'end_position': positions[-1]
            }
        
        return stats