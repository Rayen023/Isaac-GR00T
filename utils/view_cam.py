import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use('Agg')  # Use non-interactive backend



def view_img(img_dict):
    """Save camera images like in eval_lerobot.py"""
    if isinstance(img_dict, dict):
        # Stack images horizontally
        img = np.concatenate([img_dict[k] for k in img_dict], axis=1)
    else:
        img = img_dict
    
    # Create output directory if it doesn't exist
    output_dir = "camera_feed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save image with timestamp
    timestamp = int(time.time() * 1000)
    filename = f"{output_dir}/camera_feed_{timestamp}.png"
    
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.title("Camera View - Real-time Feed")
    plt.axis("off")
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()
    
    print(f"Camera feed saved to: {filename}")