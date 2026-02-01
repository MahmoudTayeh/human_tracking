"""
Face Registration Script (InsightFace version)
Register new people for face recognition system
"""

import os
import sys
import argparse
import cv2
from pathlib import Path

# Import face recognizer
try:
    from src.face_recognizer import FaceRecognizerInsightFace
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.face_recognizer import FaceRecognizerInsightFace


def register_from_images(name: str, image_dir: str, recognizer: FaceRecognizerInsightFace):
    """
    Register a person from a directory of images
    
    Args:
        name: Person's name
        image_dir: Directory containing person's images
        recognizer: FaceRecognizerInsightFace instance
    """
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' not found")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(f'*{ext}'))
        image_paths.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    if not image_paths:
        print(f"Error: No images found in '{image_dir}'")
        return
    
    print(f"Found {len(image_paths)} images for '{name}'")
    
    # Register person
    success = recognizer.register_person(name, [str(p) for p in image_paths])
    
    if success:
        print(f"\n✓ Successfully registered '{name}'!")
    else:
        print(f"\n✗ Failed to register '{name}'")


def register_from_webcam(name: str, recognizer: FaceRecognizerInsightFace, num_photos: int = 5):
    """
    Register a person by taking photos from webcam
    
    Args:
        name: Person's name
        recognizer: FaceRecognizerInsightFace instance
        num_photos: Number of photos to capture
    """
    # Create temp directory for captured images
    temp_dir = f"data/temp_capture_{name}"
    os.makedirs(temp_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print(f"\nCapturing {num_photos} photos for '{name}'")
    print("Instructions:")
    print("  - Look at the camera")
    print("  - Press SPACE to capture a photo")
    print("  - Turn your head slightly between photos (front, left, right)")
    print("  - Press ESC to cancel")
    
    captured = 0
    image_paths = []
    
    while captured < num_photos:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display instructions
        display = frame.copy()
        cv2.putText(display, f"Photo {captured + 1}/{num_photos}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "Press SPACE to capture", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw guide box
        h, w = frame.shape[:2]
        box_size = min(h, w) // 2
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, "Position face here", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Face Registration', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space bar
            # Save image
            img_path = os.path.join(temp_dir, f"photo_{captured + 1}.jpg")
            cv2.imwrite(img_path, frame)
            image_paths.append(img_path)
            captured += 1
            print(f"  Captured photo {captured}/{num_photos}")
            
            # Brief pause to avoid accidental double-capture
            cv2.waitKey(300)
            
        elif key == 27:  # ESC
            print("Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Register person with captured images
    print(f"\nRegistering '{name}' with {len(image_paths)} photos...")
    success = recognizer.register_person(name, image_paths)
    
    if success:
        print(f"\n✓ Successfully registered '{name}'!")
        # Clean up temp images
        for img_path in image_paths:
            try:
                os.remove(img_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
    else:
        print(f"\n✗ Failed to register '{name}'")
        print(f"   Images saved in: {temp_dir}")


def list_registered_people(recognizer: FaceRecognizerInsightFace):
    """List all registered people"""
    stats = recognizer.get_statistics()
    
    print("\n" + "="*50)
    print("REGISTERED PEOPLE")
    print("="*50)
    
    if not stats['known_people']:
        print("No people registered yet")
        return
    
    for person in sorted(set(stats['known_people'])):
        metadata = stats['metadata'].get(person, {})
        num_encodings = metadata.get('num_encodings', 0)
        registered = metadata.get('registered_date', 'Unknown')
        last_seen = metadata.get('last_seen', 'Never')
        
        print(f"\n{person}:")
        print(f"  Embeddings: {num_encodings}")
        print(f"  Registered: {registered[:10] if registered != 'Unknown' else 'Unknown'}")
        print(f"  Last seen: {last_seen[:10] if last_seen != 'Never' else 'Never'}")
    
    print(f"\nTotal: {stats['total_known_faces']} people")
    print("="*50)


def remove_person(name: str, recognizer: FaceRecognizerInsightFace):
    """Remove a person from the database"""
    print(f"\nRemoving '{name}' from database...")
    
    # Confirm
    confirm = input(f"Are you sure you want to remove '{name}'? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Cancelled")
        return
    
    success = recognizer.remove_person(name)
    
    if success:
        print(f"✓ Successfully removed '{name}'")
    else:
        print(f"✗ Failed to remove '{name}'")


def main():
    parser = argparse.ArgumentParser(description='Register faces for recognition system (InsightFace)')
    
    parser.add_argument('--mode', type=str, choices=['images', 'webcam', 'list', 'remove'],
                       default='list',
                       help='Registration mode')
    
    parser.add_argument('--name', type=str,
                       help='Person name to register/remove')
    
    parser.add_argument('--images', type=str,
                       help='Directory containing person images')
    
    parser.add_argument('--num-photos', type=int, default=5,
                       help='Number of photos to capture in webcam mode')
    
    parser.add_argument('--database', type=str, default='data/face_database.pkl',
                       help='Path to face database (pickle file)')
    
    parser.add_argument('--similarity-threshold', type=float, default=0.4,
                       help='Face matching threshold (0.25-0.45 recommended)')
    
    args = parser.parse_args()
    
    # Initialize face recognizer
    print("Initializing InsightFace...")
    recognizer = FaceRecognizerInsightFace(
        database_path=args.database,
        similarity_threshold=args.similarity_threshold
    )
    
    if args.mode == 'list':
        list_registered_people(recognizer)
    
    elif args.mode == 'images':
        if not args.name or not args.images:
            print("Error: --name and --images are required for 'images' mode")
            return
        register_from_images(args.name, args.images, recognizer)
    
    elif args.mode == 'webcam':
        if not args.name:
            print("Error: --name is required for 'webcam' mode")
            return
        register_from_webcam(args.name, recognizer, args.num_photos)
    
    elif args.mode == 'remove':
        if not args.name:
            print("Error: --name is required for 'remove' mode")
            return
        remove_person(args.name, recognizer)


if __name__ == "__main__":
    main()