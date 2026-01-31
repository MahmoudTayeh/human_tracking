import cv2
import os
# create necessary directories
os.makedirs('videos', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)
# test video path
VIDEO_PATH = 'videos/input_video.mp4'  
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error in opening video file.")
else:
    # info about video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"FPS: {fps}")
    print(f"Dimensions: {width}x{height}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f} seconds")

cap.release()