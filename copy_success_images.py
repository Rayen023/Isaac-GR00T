import os
import shutil
import pandas as pd
import random
from pathlib import Path


def copy_last_frames(test_results_dir, episode_type):
    if episode_type not in ['success', 'failure']:
        raise ValueError("episode_type must be either 'success' or 'failure'")
    
    csv_path = os.path.join(test_results_dir, "results.csv")
    images_base_path = os.path.join(test_results_dir, "images/observation.images.front_view")
    
    folder_num = "1" if episode_type == "success" else "0"
    destination_path = os.path.expanduser(
        f"~/OneDrive_recherche_a/Linux_onedrive/Projects_linux/result_classifier/images/{folder_num}"
    )
    
    os.makedirs(destination_path, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    success_value = 1 if episode_type == "success" else 0
    filtered_episodes = df[df['success'] == success_value]
    
    print(f"Found {len(filtered_episodes)} {episode_type} episodes out of {len(df)} total episodes")
    
    copied_count = 0
    
    for idx, row in filtered_episodes.iterrows():
        position_num = row['position_num']
        episode_folder = f"episode_{position_num:06d}"
        episode_path = os.path.join(images_base_path, episode_folder)
        
        if not os.path.exists(episode_path):
            print(f"Warning: Episode folder {episode_folder} not found")
            continue
        
        frame_files = sorted([f for f in os.listdir(episode_path) if f.startswith('frame_') and f.endswith('.png')])
        
        if not frame_files:
            print(f"Warning: No frames found in {episode_folder}")
            continue
        
        last_frame = frame_files[-1]
        source_path = os.path.join(episode_path, last_frame)
        
        dest_filename = f"episode_{position_num:06d}_last_frame.png"
        dest_path = os.path.join(destination_path, dest_filename)
        
        shutil.copy2(source_path, dest_path)
        copied_count += 1
            
    print(f"\nCompleted! Copied {copied_count} {episode_type} images to {destination_path}")


def copy_random_first_frames(test_results_dir, num_images):
    images_base_path = os.path.join(test_results_dir, "images/observation.images.front_view")
    
    destination_path = os.path.expanduser(
        f"~/OneDrive_recherche_a/Linux_onedrive/Projects_linux/result_classifier/images/0"
    )
    
    os.makedirs(destination_path, exist_ok=True)
    
    if not os.path.exists(images_base_path):
        print(f"Error: Images directory not found at {images_base_path}")
        return
    
    # Get all episode folders
    episode_folders = sorted([f for f in os.listdir(images_base_path) if f.startswith('episode_')])
    
    if not episode_folders:
        print(f"Error: No episode folders found in {images_base_path}")
        return
    
    print(f"Found {len(episode_folders)} total episodes")
    
    # Randomly select episodes
    num_to_select = min(num_images, len(episode_folders))
    selected_episodes = random.sample(episode_folders, num_to_select)
    
    print(f"Randomly selected {num_to_select} episodes")
    
    copied_count = 0
    
    for episode_folder in selected_episodes:
        episode_path = os.path.join(images_base_path, episode_folder)
        
        if not os.path.exists(episode_path):
            print(f"Warning: Episode folder {episode_folder} not found")
            continue
        
        frame_files = sorted([f for f in os.listdir(episode_path) if f.startswith('frame_') and f.endswith('.png')])
        
        if not frame_files:
            print(f"Warning: No frames found in {episode_folder}")
            continue
        
        # Get the first frame
        first_frame = frame_files[0]
        source_path = os.path.join(episode_path, first_frame)
        
        dest_filename = f"{episode_folder}_first_frame.png"
        dest_path = os.path.join(destination_path, dest_filename)
        
        shutil.copy2(source_path, dest_path)
        copied_count += 1
        print(f"Copied first frame from {episode_folder}")
            
    print(f"\nCompleted! Copied {copied_count} first frames to {destination_path}")


if __name__ == "__main__":
    # copy_last_frames("test_results_20251020_191220", "failure")
    copy_random_first_frames("test_results_20251020_191220", 5)
