#!/usr/bin/env python3
"""
Minimalist OpenCV script to display live feeds from all connected cameras.
Press 'q' or Ctrl+C to exit.
Press 's' to save image from camera 2.
"""

import cv2
import sys
import os
from datetime import datetime


def main():
    # Create directory for saved images if it doesn't exist
    save_dir = "camera_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Find all available cameras (usually 0-9 should cover most cases)
    cameras = []
    camera_windows = []
    camera_indices = []  # Keep track of original camera indices
    
    print("Detecting cameras...")
    
    # Try to open cameras from index 0 to 9
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Test if we can read a frame
            ret, frame = cap.read()
            if ret:
                cameras.append(cap)
                camera_windows.append(f"Camera {i}")
                camera_indices.append(i)
                print(f"Found camera {i}")
            else:
                cap.release()
        else:
            cap.release()
    
    if not cameras:
        print("No cameras found!")
        return
    
    print(f"Found {len(cameras)} camera(s). Press 'q' to quit, 's' to save image from camera 2.")
    
    try:
        # Store the original frames for saving (before resizing)
        original_frames = []
        
        while True:
            original_frames.clear()  # Clear previous frames
            
            for i, cap in enumerate(cameras):
                ret, frame = cap.read()
                if ret:
                    # Store original frame for potential saving
                    original_frames.append(frame.copy())
                    
                    # Resize frame for better display (optional)
                    height, width = frame.shape[:2]
                    if width > 640:
                        scale = 640 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    cv2.imshow(camera_windows[i], frame)
                else:
                    print(f"Failed to read from camera {camera_indices[i]}")
                    original_frames.append(None)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save image from camera 2
                # Find the index in our cameras list that corresponds to camera 2
                camera_2_list_index = None
                for i, camera_idx in enumerate(camera_indices):
                    if camera_idx == 2:
                        camera_2_list_index = i
                        break
                
                if camera_2_list_index is not None and camera_2_list_index < len(original_frames) and original_frames[camera_2_list_index] is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{save_dir}/camera_2_{timestamp}.jpg"
                    
                    # Save the original resolution image
                    success = cv2.imwrite(filename, original_frames[camera_2_list_index])
                    if success:
                        print(f"Image saved: {filename}")
                    else:
                        print(f"Failed to save image: {filename}")
                else:
                    print("Camera 2 not available or no frame captured")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up
        print("Cleaning up...")
        for cap in cameras:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()