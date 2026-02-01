"""
Helper functions
"""

import yaml
import os
import cv2


def load_config(config_path='configs/config.yaml'):
    """
    Load configuration file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration settings
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config):
    """
    Create required directories
    
    Args:
        config: Configuration settings
    """
    directories = [
        'outputs/videos',
        'outputs/statistics',
        'outputs/visualizations',
        'data/videos',
        'data/models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories created successfully")


def get_video_info(video_path):
    """
    Retrieve video information
    
    Args:
        video_path: Path to the video
        
    Returns:
        dict: Video information
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    
    info = {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS))
    }
    
    cap.release()
    return info


def print_video_info(info):
    """
    Print video information
    """
    print("\n📹 Video Information:")
    print(f"   FPS: {info['fps']}")
    print(f"   Resolution: {info['width']}x{info['height']}")
    print(f"   Total Frames: {info['total_frames']}")
    print(f"   Duration: {info['duration']:.2f} seconds\n")
