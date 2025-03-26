import os
import cv2
import numpy as np
import math

def create_grid_comparison_video(video_paths, output_file, fps=None, title="Video Grid Comparison"):
    """
    Create a video comparing multiple input videos in a grid layout.
    
    Args:
        video_paths (list): List of paths to the input videos
        output_file (str): Path to the output video file
        fps (int, optional): Frames per second for the output video. If None, uses the fps of the first video.
        title (str): Title to display at the top of the video
    """
    if not video_paths:
        print("Error: No video paths provided.")
        return
    
    # Open all video files
    caps = []
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video {path}")
            return
        caps.append(cap)
    
    # Get video properties
    video_properties = []
    for cap in caps:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_properties.append((width, height, frame_count))
    
    # Determine common dimensions for all videos
    max_width = max(prop[0] for prop in video_properties)
    max_height = max(prop[1] for prop in video_properties)
    min_frames = min(prop[2] for prop in video_properties)
    
    # Determine the grid dimensions
    num_videos = len(video_paths)
    grid_size = math.ceil(math.sqrt(num_videos))  # Square grid
    grid_width = min(grid_size, num_videos)
    grid_height = math.ceil(num_videos / grid_width)
    
    # Determine the output video's fps
    if fps is None:
        fps = caps[0].get(cv2.CAP_PROP_FPS)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' for AVI format
    title_height = 50  # Height for main title
    label_height = 30  # Height for video labels
    
    # Calculate dimensions for the full output video
    video_width = max_width * grid_width
    video_height = title_height + (max_height + label_height) * grid_height
    
    out = cv2.VideoWriter(output_file, fourcc, fps, (video_width, video_height))
    
    # Define font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (255, 255, 255)  # White
    
    # Get video filenames for labels
    video_names = [os.path.basename(path) for path in video_paths]
    
    # Process frames
    frame_index = 0
    
    while frame_index < min_frames:
        # Read frames from all videos
        frames = []
        all_read = True
        
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                all_read = False
                break
            
            # Resize frame if necessary
            if frame.shape[0] != max_height or frame.shape[1] != max_width:
                frame = cv2.resize(frame, (max_width, max_height))
            
            frames.append(frame)
        
        if not all_read:
            break
        
        # Create a blank space for the main title
        title_space = np.zeros((title_height, video_width, 3), dtype=np.uint8)
        
        # Add title text
        text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        text_x = (video_width - text_size[0]) // 2
        text_y = 35  # Position from top
        cv2.putText(title_space, title, (text_x, text_y), font, font_scale, font_color, font_thickness)
        
        # Add frame number
        frame_text = f"Frame: {frame_index:04d}"
        cv2.putText(title_space, frame_text, (10, 35), font, 0.5, font_color, 1)
        
        # Create grid layout
        grid_rows = []
        
        # For each row in the grid
        for row in range(grid_height):
            row_frames = []
            
            # For each column in the grid
            for col in range(grid_width):
                idx = row * grid_width + col
                
                # If we still have videos to display
                if idx < len(frames):
                    # Create label space for this video
                    label_space = np.zeros((label_height, max_width, 3), dtype=np.uint8)
                    
                    # Add video name label with increased font size
                    video_label = video_names[idx]
                    font_scale_label = 0.7  # Increased from 0.5
                    font_thickness_label = 2  # Increased from 1
                    text_size = cv2.getTextSize(video_label, font, font_scale_label, font_thickness_label)[0]
                    text_x = (max_width - text_size[0]) // 2
                    cv2.putText(label_space, video_label, (text_x, 20), font, font_scale_label, font_color, font_thickness_label)
                    
                    # Combine frame and its label
                    frame_with_label = np.vstack((frames[idx], label_space))
                    row_frames.append(frame_with_label)
                else:
                    # Fill empty space if needed
                    empty_frame = np.zeros((max_height + label_height, max_width, 3), dtype=np.uint8)
                    row_frames.append(empty_frame)
            
            # Combine all frames in this row
            row_combined = np.hstack(row_frames)
            grid_rows.append(row_combined)
        
        # Combine all rows
        grid_combined = np.vstack(grid_rows)
        
        # Combine title and grid
        full_frame = np.vstack((title_space, grid_combined))
        
        # Write the frame
        out.write(full_frame)
        
        # Display progress
        if frame_index % 10 == 0:
            print(f"Processing frame {frame_index}/{min_frames}...")
        
        frame_index += 1
    
    # Release resources
    for cap in caps:
        cap.release()
    out.release()
    
    # Report any frame count differences
    if any(prop[2] != min_frames for prop in video_properties):
        print("Warning: Videos have different frame counts:")
        for i, prop in enumerate(video_properties):
            print(f"  {video_names[i]}: {prop[2]} frames")
        print(f"Processed {frame_index} frames (the shortest video length).")
    
    print(f"Video comparison grid saved as {output_file}")

def main():
    # Get all MP4 files in the current directory
    video_dir = "."  # Current directory, change if needed
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                  if f.lower().endswith('.mp4') and f != "comparison_grid.mp4"]  # Exclude output file
    
    if not video_paths:
        print("No MP4 files found in the directory.")
        return
    
    print(f"Found {len(video_paths)} video files:")
    for path in video_paths:
        print(f"  - {os.path.basename(path)}")
    
    output_file = "comparison_grid.mp4"
    title = f"Grid Comparison of {len(video_paths)} Videos"
    
    create_grid_comparison_video(video_paths, output_file, fps=None, title=title)
    
if __name__ == "__main__":
    main()