"""
Face Recognition Module using InsightFace
Integrates with YOLO + DeepSort for person identification
"""

import cv2
import numpy as np
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import insightface
from insightface.app import FaceAnalysis


class FaceRecognizerInsightFace:
    """
    Face recognition system using InsightFace
    More accurate than face_recognition library, better for production
    """
    
    def __init__(self, 
                 database_path: str = "data/face_database.pkl",
                 similarity_threshold: float = 0.4,
                 recognition_interval: int = 10,
                 det_size: tuple = (640, 640)):
        """
        Initialize InsightFace recognizer
        
        Args:
            database_path: Path to pickle file storing face embeddings
            similarity_threshold: Lower = stricter (0.25-0.45 recommended)
                                 Cosine similarity: 1.0 = identical, 0.0 = completely different
            recognition_interval: Run face recognition every N frames
            det_size: Detection size for face detection (larger = slower but more accurate)
        """
        self.database_path = database_path
        self.similarity_threshold = similarity_threshold
        self.recognition_interval = recognition_interval
        
        # Create database directory if needed
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        
        # Initialize InsightFace
        print("[FaceRecognizer] Initializing InsightFace model...")
        self.app = FaceAnalysis(
            name='buffalo_l',  # High quality model
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=det_size)
        print("[FaceRecognizer] ✓ InsightFace initialized")
        
        # Load known faces
        self.known_face_embeddings = []  # List of numpy arrays
        self.known_face_names = []       # List of names
        self.known_face_metadata = {}    # Dict of metadata
        self.load_database()
        
        # Track which person ID has which name
        self.track_id_to_name = {}
        self.track_id_confidence = {}
        
        # Frame counter for interval-based recognition
        self.frame_count = 0
        
        print(f"[FaceRecognizer] Initialized with {len(self.known_face_names)} known faces")
    
    
    def load_database(self):
        """Load face embeddings from pickle database"""
        if not os.path.exists(self.database_path):
            print(f"[FaceRecognizer] No database found at {self.database_path}")
            return
        
        try:
            with open(self.database_path, 'rb') as f:
                data = pickle.load(f)
            
            self.known_face_embeddings = []
            self.known_face_names = []
            self.known_face_metadata = {}
            
            for person_name, person_data in data.items():
                embeddings = person_data.get('embeddings', [])
                for embedding in embeddings:
                    self.known_face_embeddings.append(np.array(embedding))
                    self.known_face_names.append(person_name)
                
                # Store metadata
                self.known_face_metadata[person_name] = {
                    'num_encodings': len(embeddings),
                    'registered_date': person_data.get('registered_date', 'Unknown'),
                    'last_seen': person_data.get('last_seen', 'Never')
                }
            
            print(f"[FaceRecognizer] Loaded {len(self.known_face_names)} face embeddings")
            
        except Exception as e:
            print(f"[FaceRecognizer] Error loading database: {e}")
    
    
    def save_database(self):
        """Save face embeddings to pickle database"""
        # Organize data by person
        database = {}
        
        for name, embedding in zip(self.known_face_names, self.known_face_embeddings):
            if name not in database:
                database[name] = {
                    'embeddings': [],
                    'registered_date': self.known_face_metadata.get(name, {}).get('registered_date', datetime.now().isoformat()),
                    'last_seen': self.known_face_metadata.get(name, {}).get('last_seen', 'Never')
                }
            database[name]['embeddings'].append(embedding.tolist())
        
        try:
            with open(self.database_path, 'wb') as f:
                pickle.dump(database, f)
            print(f"[FaceRecognizer] Database saved to {self.database_path}")
        except Exception as e:
            print(f"[FaceRecognizer] Error saving database: {e}")
    
    
    def register_person(self, name: str, image_paths: List[str]) -> bool:
        """
        Register a new person with multiple images
        
        Args:
            name: Person's name
            image_paths: List of paths to person's images
            
        Returns:
            True if registration successful
        """
        embeddings_added = 0
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"[FaceRecognizer] Image not found: {img_path}")
                continue
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"[FaceRecognizer] Could not read image: {img_path}")
                continue
            
            # Detect faces and get embeddings
            faces = self.app.get(image)
            
            if len(faces) == 0:
                print(f"[FaceRecognizer] No face found in {img_path}")
                continue
            
            if len(faces) > 1:
                print(f"[FaceRecognizer] Multiple faces in {img_path}, using largest one")
            
            # Use the largest face (most prominent)
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Add embedding
            self.known_face_embeddings.append(face.embedding)
            self.known_face_names.append(name)
            embeddings_added += 1
            
            print(f"[FaceRecognizer] Added embedding for {name} from {img_path}")
        
        if embeddings_added > 0:
            # Update metadata
            self.known_face_metadata[name] = {
                'num_encodings': embeddings_added,
                'registered_date': datetime.now().isoformat(),
                'last_seen': 'Never'
            }
            
            # Save to database
            self.save_database()
            print(f"[FaceRecognizer] Successfully registered {name} with {embeddings_added} embeddings")
            return True
        
        return False
    
    
    def recognize_faces_in_tracks(self, frame: np.ndarray, active_tracks: List[Dict]) -> Dict[int, str]:
        """
        Recognize faces for tracked persons in the current frame
        Designed to work with your HumanTracker output format
        
        Args:
            frame: Current video frame (BGR)
            active_tracks: List of dicts from HumanTracker.process_frame()
                          Each dict: {'id': track_id, 'bbox': [x1,y1,x2,y2], 'center': [x,y], 'color': (r,g,b)}
            
        Returns:
            Dictionary mapping track_id to person name
        """
        self.frame_count += 1
        
        # Only run recognition every N frames (performance optimization)
        if self.frame_count % self.recognition_interval != 0:
            return self.track_id_to_name
        
        # Process each tracked person
        for track in active_tracks:
            track_id = track['id']
            x1, y1, x2, y2 = track['bbox']
            
            # Add padding to bbox to ensure we get the full face
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            # Extract person region
            person_roi = frame[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                continue
            
            # Detect faces in this person's region
            faces = self.app.get(person_roi)
            
            # Match face if found
            if len(faces) > 0:
                # Use the largest face in the region
                face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                
                name, similarity = self._match_face(face.embedding)
                
                if name != "Unknown":
                    # Update track-to-name mapping with confidence
                    if track_id not in self.track_id_confidence:
                        self.track_id_confidence[track_id] = {}
                    
                    if name not in self.track_id_confidence[track_id]:
                        self.track_id_confidence[track_id][name] = 0
                    
                    self.track_id_confidence[track_id][name] += 1
                    
                    # Assign name if confidence is high enough (seen 3+ times)
                    if self.track_id_confidence[track_id][name] >= 3:
                        self.track_id_to_name[track_id] = name
                        
                        # Update last seen
                        if name in self.known_face_metadata:
                            self.known_face_metadata[name]['last_seen'] = datetime.now().isoformat()
        
        return self.track_id_to_name
    
    
    def _match_face(self, face_embedding: np.ndarray) -> Tuple[str, float]:
        """
        Match a face embedding against known faces using cosine similarity
        
        Args:
            face_embedding: 512-dimensional face embedding from InsightFace
            
        Returns:
            Tuple of (name, similarity_score)
        """
        if len(self.known_face_embeddings) == 0:
            return "Unknown", 0.0
        
        # Calculate cosine similarity with all known faces
        similarities = []
        for known_embedding in self.known_face_embeddings:
            # Cosine similarity
            similarity = np.dot(face_embedding, known_embedding) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding)
            )
            similarities.append(similarity)
        
        # Find best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Check if similarity is above threshold
        if best_similarity >= self.similarity_threshold:
            name = self.known_face_names[best_match_idx]
            print(f"[FaceRecognizer] Matched: {name} (similarity: {best_similarity:.3f})")
            return name, best_similarity
        
        return "Unknown", best_similarity
    
    
    def get_name_for_track(self, track_id: int) -> str:
        """
        Get the name associated with a track ID
        
        Args:
            track_id: DeepSort track ID
            
        Returns:
            Person name or "Unknown"
        """
        return self.track_id_to_name.get(track_id, "Unknown")
    
    
    def clear_track_memory(self, track_id: int):
        """
        Clear the name association for a track ID
        (call when track is lost)
        
        Args:
            track_id: DeepSort track ID
        """
        if track_id in self.track_id_to_name:
            del self.track_id_to_name[track_id]
        if track_id in self.track_id_confidence:
            del self.track_id_confidence[track_id]
    
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return {
            'total_known_faces': len(set(self.known_face_names)),
            'total_embeddings': len(self.known_face_embeddings),
            'active_tracks': len(self.track_id_to_name),
            'known_people': list(set(self.known_face_names)),
            'metadata': self.known_face_metadata
        }
    
    
    def remove_person(self, name: str) -> bool:
        """
        Remove a person from the database
        
        Args:
            name: Person's name
            
        Returns:
            True if removal successful
        """
        # Remove all embeddings for this person
        indices_to_remove = [i for i, n in enumerate(self.known_face_names) if n == name]
        
        if not indices_to_remove:
            print(f"[FaceRecognizer] Person '{name}' not found in database")
            return False
        
        # Remove in reverse order to avoid index issues
        for i in sorted(indices_to_remove, reverse=True):
            del self.known_face_embeddings[i]
            del self.known_face_names[i]
        
        # Remove metadata
        if name in self.known_face_metadata:
            del self.known_face_metadata[name]
        
        # Clear from active tracks
        tracks_to_clear = [tid for tid, n in self.track_id_to_name.items() if n == name]
        for tid in tracks_to_clear:
            self.clear_track_memory(tid)
        
        # Save updated database
        self.save_database()
        
        print(f"[FaceRecognizer] Removed {len(indices_to_remove)} embeddings for '{name}'")
        return True

