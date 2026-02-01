"""
Main program for human tracking in video
"""

import cv2
import time
from tqdm import tqdm
import argparse

from src.tracker import HumanTracker
from src.visualizer import Visualizer
from src.utils import load_config, create_directories, get_video_info, print_video_info


def main(config_path='configs/config.yaml'):
    """
    Main program
    """
    # 1. Load configuration
    print("⚙️  Loading configuration...")
    config = load_config(config_path)
    
    # 2. Create required directories
    create_directories(config)
    
    # 3. Video information
    video_path = config['video']['input_path']
    video_info = get_video_info(video_path)
    print_video_info(video_info)
    
    # 4. Initialize tracking system
    print("🚀 Initializing tracking system...")
    tracker = HumanTracker(config)
    # 5. Open video
    cap = cv2.VideoCapture(video_path)
    
    # 6. Video writer setup
    if config['video']['save_video']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            config['video']['output_path'],
            fourcc,
            video_info['fps'],
            (video_info['width'], video_info['height'])
        )
    
    # 7. Video processing
    print("\n🎬 Starting processing...")
    frame_count = 0
    processing_times = []
    
    pbar = tqdm(total=video_info['total_frames'], desc="Processing")
    
    # Store first frame as background
    first_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if first_frame is None:
            first_frame = frame.copy()
        
        frame_count += 1
        start_time = time.time()
        
        # Process frame
        result_frame, tracks = tracker.process_frame(frame)

        # Overlay information
        if config['visualization']['show_statistics']:
            info_text = f"Frame: {frame_count}/{video_info['total_frames']} | Persons: {len(tracks)}"
            cv2.putText(
                result_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
        
        # Save frame
        if config['video']['save_video']:
            out.write(result_frame)
        
        # Show preview
        if config['video']['show_preview']:
            cv2.imshow('Human Tracking', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  Processing stopped by user")
                break
        
        # Timing
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        pbar.update(1)
    
    # 9. Release resources
    pbar.close()
    cap.release()
    if config['video']['save_video']:
        out.release()
    cv2.destroyAllWindows()
    
    # 10. Print performance statistics
    avg_time = sum(processing_times) / len(processing_times)
    avg_fps = 1 / avg_time
    
    print("\n" + "=" * 60)
    print("✅ Processing completed!")
    print("=" * 60)
    print(f"⏱️  Average time per frame: {avg_time * 1000:.2f} ms")
    print(f"📊 Average FPS: {avg_fps:.2f}")
    print(f"👥 Number of tracked persons: {len(tracker.track_history)}")
    
    if config['video']['save_video']:
        print(f"💾 Video saved at: {config['video']['output_path']}")
    
    # 11. Generate statistics and visualizations
    print("\n📊 Generating statistics...")
    
    # Plot statistics
    stats_df = Visualizer.plot_statistics(
        tracker.track_history,
        video_info['fps'],
        'outputs/statistics/tracking_stats.png'
    )
    print("✅ Statistics saved in: outputs/statistics/") 
    print("\n🎉 All tasks completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human Tracking System')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args.config)
