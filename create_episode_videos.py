"""
Script to concatenate front and wrist view images horizontally and create videos for each episode.
"""
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def concatenate_images_horizontally(front_img_path, wrist_img_path):
    """
    Concatenate two images horizontally.
    
    Args:
        front_img_path: Path to front view image
        wrist_img_path: Path to wrist view image
    
    Returns:
        Concatenated image as numpy array
    """
    front_img = cv2.imread(str(front_img_path))
    wrist_img = cv2.imread(str(wrist_img_path))
    
    if front_img is None or wrist_img is None:
        return None
    
    # Ensure both images have the same height
    if front_img.shape[0] != wrist_img.shape[0]:
        # Resize to match the smaller height
        target_height = min(front_img.shape[0], wrist_img.shape[0])
        front_img = cv2.resize(front_img, (int(front_img.shape[1] * target_height / front_img.shape[0]), target_height))
        wrist_img = cv2.resize(wrist_img, (int(wrist_img.shape[1] * target_height / wrist_img.shape[0]), target_height))
    
    # Concatenate horizontally
    concatenated = np.hstack([front_img, wrist_img])
    return concatenated


def create_video_for_episode(episode_num, base_path, output_dir, fps=30):
    """
    Create a video for a specific episode by concatenating front and wrist images.
    
    Args:
        episode_num: Episode number (e.g., 0, 1, 2, ...)
        base_path: Base path to the images directory
        output_dir: Directory to save the output videos
        fps: Frames per second for the output video
    """
    episode_name = f"episode_{episode_num:06d}"
    
    front_dir = base_path / "observation.images.front_view" / episode_name
    wrist_dir = base_path / "observation.images.wrist_view" / episode_name
    
    # Check if episode directories exist
    if not front_dir.exists() or not wrist_dir.exists():
        print(f"Skipping {episode_name}: directories not found")
        return False
    
    # Get list of frame files
    front_frames = sorted(front_dir.glob("frame_*.png"))
    wrist_frames = sorted(wrist_dir.glob("frame_*.png"))
    
    if len(front_frames) == 0 or len(wrist_frames) == 0:
        print(f"Skipping {episode_name}: no frames found")
        return False
    
    # Ensure we have the same number of frames
    num_frames = min(len(front_frames), len(wrist_frames))
    
    if num_frames == 0:
        print(f"Skipping {episode_name}: no matching frames")
        return False
    
    # Get dimensions from first concatenated image
    first_concat = concatenate_images_horizontally(front_frames[0], wrist_frames[0])
    if first_concat is None:
        print(f"Skipping {episode_name}: failed to load first frame")
        return False
    
    height, width, _ = first_concat.shape
    
    # Create output video path
    output_path = output_dir / f"{episode_name}.mp4"
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Process each frame
    for i in range(num_frames):
        concat_img = concatenate_images_horizontally(front_frames[i], wrist_frames[i])
        if concat_img is not None:
            video_writer.write(concat_img)
    
    video_writer.release()
    print(f"Created video: {output_path} ({num_frames} frames)")
    return True


def main():
    # Define paths
    base_path = Path("/home/recherche-a/OneDrive_recherche_a/Linux_onedrive/Projects_linux/Isaac-GR00T/test_results_20251020_191220/images")
    output_dir = Path("/home/recherche-a/OneDrive_recherche_a/Linux_onedrive/Projects_linux/Isaac-GR00T/test_results_20251020_191220/episode_videos")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get list of episodes
    front_view_dir = base_path / "observation.images.front_view"
    episodes = sorted([d.name for d in front_view_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    
    print(f"Found {len(episodes)} episodes")
    print(f"Output directory: {output_dir}")
    print("Creating videos...")
    
    success_count = 0
    
    # Process each episode with progress bar
    for episode_name in tqdm(episodes, desc="Processing episodes"):
        episode_num = int(episode_name.split("_")[-1])
        if create_video_for_episode(episode_num, base_path, output_dir):
            success_count += 1
    
    print(f"\nCompleted! Successfully created {success_count}/{len(episodes)} videos")
    print(f"Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
