"""
Visualization of results and statistics
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Visualizer:
    """
    Visual result rendering
    """
    
    @staticmethod
    def plot_training_curves(history, save_path):
        """
        Plot training curves (for other projects)
        """
        pass
    
    @staticmethod
    def plot_statistics(track_history, fps, save_path):
        """
        Plot tracking statistics
        
        Args:
            track_history: Track history
            fps: Frames per second
            save_path: Output save path
        """
        stats = []
        
        for track_id, positions in track_history.items():
            # Calculate statistics
            total_distance = 0
            for i in range(1, len(positions)):
                x1, y1 = positions[i - 1]
                x2, y2 = positions[i]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                total_distance += distance
            
            duration = len(positions) / fps
            avg_speed = total_distance / duration if duration > 0 else 0
            
            stats.append({
                'Track ID': track_id,
                'Duration (sec)': duration,
                'Total Distance (px)': total_distance,
                'Avg Speed (px/s)': avg_speed,
                'Frames': len(positions)
            })
        
        df = pd.DataFrame(stats)
        df = df.sort_values('Duration (sec)', ascending=False)
        
        # Plot charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Appearance duration
        axes[0, 0].barh(df['Track ID'].astype(str), df['Duration (sec)'])
        axes[0, 0].set_xlabel('Duration (seconds)')
        axes[0, 0].set_ylabel('Track ID')
        axes[0, 0].set_title('Appearance Duration per Person')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Total distance traveled
        axes[0, 1].barh(df['Track ID'].astype(str), df['Total Distance (px)'])
        axes[0, 1].set_xlabel('Distance (pixels)')
        axes[0, 1].set_ylabel('Track ID')
        axes[0, 1].set_title('Total Distance Traveled')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Average speed
        axes[1, 0].barh(df['Track ID'].astype(str), df['Avg Speed (px/s)'])
        axes[1, 0].set_xlabel('Speed (pixels/second)')
        axes[1, 0].set_ylabel('Track ID')
        axes[1, 0].set_title('Average Speed')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Number of frames
        axes[1, 1].barh(df['Track ID'].astype(str), df['Frames'])
        axes[1, 1].set_xlabel('Number of Frames')
        axes[1, 1].set_ylabel('Track ID')
        axes[1, 1].set_title('Tracking Frames')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return df
    
    @staticmethod
    def plot_heatmap(track_history, frame_shape, save_path):
        """
        Plot movement heatmap
        
        Args:
            track_history: Track history
            frame_shape: Frame shape (height, width)
            save_path: Output save path
        """
        height, width = frame_shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Aggregate all positions
        for positions in track_history.values():
            for x, y in positions:
                if 0 <= x < width and 0 <= y < height:
                    heatmap[y, x] += 1
        
        # Apply Gaussian blur
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap, cmap='hot', interpolation='bilinear')
        plt.colorbar(label='Movement Density')
        plt.title('Movement Heatmap')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def draw_trajectories_map(track_history, background_frame, save_path):
        """
        Draw trajectory map over a static background
        
        Args:
            track_history: Track history
            background_frame: Background frame
            save_path: Output save path
        """
        overlay = background_frame.copy()
        
        for track_id, positions in track_history.items():
            if len(positions) < 2:
                continue
            
            # Random color per track
            np.random.seed(track_id)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Draw trajectory
            for i in range(1, len(positions)):
                cv2.line(overlay, positions[i - 1], positions[i], color, 2)
            
            # Start point
            start = positions[0]
            cv2.circle(overlay, start, 8, color, -1)
            cv2.putText(
                overlay, f'Start {track_id}',
                (start[0] + 10, start[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # End point
            end = positions[-1]
            cv2.circle(overlay, end, 8, (0, 0, 255), -1)
        
        # Blend with background
        result = cv2.addWeighted(background_frame, 0.5, overlay, 0.5, 0)
        
        cv2.imwrite(save_path, result)
        return result
