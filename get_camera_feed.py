#!/usr/bin/env python3
"""
Minimalist OpenCV script to display live feeds from all connected cameras.
Press 'q' or Ctrl+C to exit.
"""

import cv2
import sys


def main():
    # Find all available cameras (usually 0-9 should cover most cases)
    cameras = []
    camera_windows = []
    
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
                print(f"Found camera {i}")
            else:
                cap.release()
        else:
            cap.release()
    
    if not cameras:
        print("No cameras found!")
        return
    
    print(f"Found {len(cameras)} camera(s). Press 'q' to quit.")
    
    try:
        while True:
            for i, cap in enumerate(cameras):
                ret, frame = cap.read()
                if ret:
                    # Resize frame for better display (optional)
                    height, width = frame.shape[:2]
                    if width > 640:
                        scale = 640 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    cv2.imshow(camera_windows[i], frame)
                else:
                    print(f"Failed to read from camera {i}")
            
            # Check for 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
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