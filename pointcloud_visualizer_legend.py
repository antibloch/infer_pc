import open3d as o3d
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import json
import matplotlib.patches as mpatches

def visualize_pointcloud_360_with_zoom(ply_path, json_path, num_frames, output_dir="./frames", video_path="pointcloud_360.mp4", fps=30, max_zoom=2.0):
    """
    Visualize a point cloud in a 360-degree rotation with constant zoom using matplotlib.
    Saves frames and converts them to a video using OpenCV.
    
    Args:
        ply_path (str): Path to the .ply file
        json_path (str): Path to the JSON file containing the label dictionary
        num_frames (int): Number of frames for the 360-degree rotation
        output_dir (str): Directory to save the frames
        video_path (str): Path for the output video file
        fps (int): Frames per second for the output video
        max_zoom (float): Constant zoom factor (2.0 = 2x zoom in, 1.0 = no zoom, 0.5 = zoom out)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the point cloud
    print(f"Loading point cloud from {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # Load the label dictionary
    print(f"Loading label dictionary from {json_path}")
    with open(json_path, 'r') as f:
        label_dict = json.load(f)
    
    # Check if the point cloud has points
    if not pcd.has_points():
        print("Error: The point cloud has no points!")
        return
    
    # Print basic information about the point cloud
    original_point_count = len(pcd.points)
    print(f"Point cloud loaded with {original_point_count} points.")
    print(f"Point cloud has colors: {pcd.has_colors()}")
    print(f"Point cloud has normals: {pcd.has_normals()}")
    
    # Get points from the point cloud
    points = np.asarray(pcd.points)
    
    # If the number of points is greater than 10 million, downsample by a factor of 10
    if points.shape[0] > 10_000_000:
        print("Large point cloud detected (> 10 million points), downsampling by a factor of 10.")
        points = points[::10]
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)[::10]
        else:
            norm = plt.Normalize(points[:, 2].min(), points[:, 2].max())
            colors = cm.viridis(norm(points[:, 2]))[:, :3]  # Exclude alpha channel
    else:
        # If the point cloud has colors, get them, otherwise use a default color
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            norm = plt.Normalize(points[:, 2].min(), points[:, 2].max())
            colors = cm.viridis(norm(points[:, 2]))[:, :3]  # Exclude alpha channel
    
    # Calculate center for rotation
    center = np.mean(points, axis=0)
    
    # Find the bounding box to determine appropriate axis limits
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    extent = np.max(max_bound - min_bound)
    
    # Fixed axis limits for consistent view
    view_radius = extent / 1.5  # Adjust for good default view
    
    print(f"Starting matplotlib 360-degree visualization with constant {max_zoom}x zoom, {num_frames} frames...")
    
    # Prepare legend patches
    legend_patches = []
    for class_name, color_values in label_dict.items():
        # Convert the color values to RGB format (0-1 range for matplotlib)
        rgb_color = [c/255.0 for c in color_values]
        patch = mpatches.Patch(color=rgb_color, label=class_name)
        legend_patches.append(patch)
    
    # Create frames for the video
    for i in range(num_frames):
        # Calculate the rotation angle for this frame
        angle = i * (360.0 / num_frames)
        
        # Create a rotation matrix around the y-axis
        rotation_matrix = np.array([
            [np.cos(np.radians(angle)), 0, np.sin(np.radians(angle))],
            [0, 1, 0],
            [-np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]
        ])
        
        # Apply a constant zoom factor to all frames
        current_zoom = max_zoom
        
        # Apply rotation to the points relative to the center
        centered_points = points - center
        
        # Apply zoom by scaling the points (zoom in = points appear larger)
        zoomed_points = centered_points * current_zoom
        
        # Apply rotation and shift back
        rotated_points = np.dot(zoomed_points, rotation_matrix.T) + center
        
        # Set up the figure - higher DPI for better quality
        plt.figure(figsize=(10, 8), dpi=120)
        ax = plt.subplot(111, projection='3d')
        
        # Plot the rotated points with their colors
        ax.scatter(
            rotated_points[:, 0], 
            rotated_points[:, 1], 
            rotated_points[:, 2],
            c=colors, 
            s=0.5,  # Small point size for better performance
            alpha=0.7
        )
        
        # Set equal aspect ratio for consistent visualization
        ax.set_box_aspect([1, 1, 1])
        
        # Set consistent view limits centered on the center point
        ax.set_xlim(center[0] - view_radius, center[0] + view_radius)
        ax.set_ylim(center[1] - view_radius, center[1] + view_radius)
        ax.set_zlim(center[2] - view_radius, center[2] + view_radius)
        
        # Add legend
        ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1, 1),
                  fancybox=True, shadow=True, ncol=1, fontsize='small',
                  framealpha=0.8, facecolor='lightgray', edgecolor='black')
        
        # Hide axis labels and ticks for cleaner visualization
        ax.set_axis_off()
        
        # Set a dark background color
        ax.set_facecolor((0.1, 0.1, 0.1))
        plt.gcf().set_facecolor((0.1, 0.1, 0.1))
        
        # Set elevation with slight variation to add dynamic movement
        elev = 20 + 10 * np.sin(np.radians(angle))
        ax.view_init(elev=elev, azim=angle)
        
        # Save the image without displaying it
        output_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor=(0.1, 0.1, 0.1))
        plt.close()  # Close the figure to free memory
        
        # Print progress
        print(f"Saved frame {i+1}/{num_frames} - {angle:.1f} degrees", end="\r")
    
    print("\nAll frames generated. Converting to video...")
    
    # Create video from the frames using OpenCV
    create_video_from_frames(output_dir, video_path, fps)


def create_video_from_frames(frame_dir, output_path, fps=30):
    """
    Create a video from the saved frames using OpenCV.
    
    Args:
        frame_dir (str): Directory containing the frames
        output_path (str): Path to save the video
        fps (int): Frames per second for the output video
    """
    # Get all PNG files in the directory
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    
    if not frame_files:
        print(f"No frames found in {frame_dir}")
        return
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width, layers = first_frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4 format
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each frame to the video
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        # Check if frame was loaded properly
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
            
        video.write(frame)
    
    # Release the video writer
    video.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a .ply point cloud with a 360-degree rotation and zoom using matplotlib")
    parser.add_argument("ply_path", help="Path to the .ply file")
    parser.add_argument("--json_path", required=True, help="Path to the JSON file containing the label dictionary")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames for the 360-degree rotation (default: 60)")
    parser.add_argument("--output", default="./frames", help="Directory to save the frames (default: ./frames)")
    parser.add_argument("--video", default="pointcloud_360.mp4", help="Path for the output video file (default: pointcloud_360.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video (default: 30)")
    parser.add_argument("--zoom", type=float, default=6.0, help="Constant zoom factor: 2.0 = 2x zoom in, 1.0 = no zoom, 0.5 = zoom out (default: 6.0)")
    
    args = parser.parse_args()
    
    visualize_pointcloud_360_with_zoom(args.ply_path, args.json_path, args.frames, args.output, args.video, args.fps, args.zoom)
