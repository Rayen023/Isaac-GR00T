#!/usr/bin/env python3
"""
Script to view saved object positions from CSV on live camera feed.
Use left/right arrow keys to navigate between saved positions.
Press 'q' to quit.
"""

import cv2
import numpy as np
import csv
import sys


def overlay_transparent(background, overlay, x, y, transparency=0.7):
    """
    Overlay a transparent PNG image on a background image at position (x, y).
    
    Args:
        background: Background image (BGR)
        overlay: Overlay image with alpha channel (BGRA)
        x: X position for overlay
        y: Y position for overlay
        transparency: Additional transparency to apply (0.0 = invisible, 1.0 = original opacity)
    
    Returns:
        Background image with overlay applied
    """
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]
    
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + ol_w)
    y2 = min(bg_h, y + ol_h)
    
    ol_x1 = max(0, -x)
    ol_y1 = max(0, -y)
    ol_x2 = ol_x1 + (x2 - x1)
    ol_y2 = ol_y1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return background
    
    roi = background[y1:y2, x1:x2]
    overlay_region = overlay[ol_y1:ol_y2, ol_x1:ol_x2]
    
    if overlay_region.shape[2] == 4:
        overlay_bgr = overlay_region[:, :, :3]
        alpha = overlay_region[:, :, 3] / 255.0
        alpha = alpha * (1.0 - transparency)
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
        blended = (alpha_3ch * overlay_bgr + (1 - alpha_3ch) * roi).astype(np.uint8)
        background[y1:y2, x1:x2] = blended
    else:
        background[y1:y2, x1:x2] = overlay_region
    
    return background


def load_positions_from_csv(csv_filename):
    """Load all saved positions from CSV file."""
    positions = []
    try:
        with open(csv_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                positions.append({
                    'timestamp': row['timestamp'],
                    'cup_x': int(row['cup_x']),
                    'cup_y': int(row['cup_y']),
                    'block_x': int(row['block_x']),
                    'block_y': int(row['block_y'])
                })
    except FileNotFoundError:
        print(f"Error: {csv_filename} not found")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    return positions


def main():
    camera_index = 2
    csv_filename = "object_positions.csv"
    
    # Load positions from CSV
    print(f"Loading positions from {csv_filename}...")
    positions = load_positions_from_csv(csv_filename)
    
    if positions is None or len(positions) == 0:
        print("No positions found in CSV file")
        sys.exit(1)
    
    print(f"Loaded {len(positions)} position sets")
    
    # Load overlay images
    print("Loading overlay images...")
    cup_img = cv2.imread("cup.png", cv2.IMREAD_UNCHANGED)
    block_img = cv2.imread("legoblock.png", cv2.IMREAD_UNCHANGED)
    
    if cup_img is None:
        print("Error: Could not load cup.png")
        sys.exit(1)
    
    if block_img is None:
        print("Error: Could not load legoblock.png")
        sys.exit(1)
    
    # Open camera
    print(f"Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        sys.exit(1)
    
    print(f"Camera {camera_index} opened successfully!")
    print("\nControls:")
    print("  Left Arrow  - Previous position")
    print("  Right Arrow - Next position")
    print("  'q'         - Quit")
    
    # Start at first position
    current_index = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read from camera")
                break
            
            # Get current position
            pos = positions[current_index]
            
            # Create display frame
            display_frame = frame.copy()
            
            # Overlay cup and block at saved positions
            display_frame = overlay_transparent(display_frame, cup_img, 
                                               pos['cup_x'], pos['cup_y'])
            display_frame = overlay_transparent(display_frame, block_img, 
                                               pos['block_x'], pos['block_y'])
            
            # Add text overlay showing current index and timestamp
            info_text = f"Position {current_index + 1}/{len(positions)} - {pos['timestamp']}"
            cv2.putText(display_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow(f"Camera {camera_index} - Saved Positions", display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == 81 or key == 2:  # Left arrow (81 on some systems, 2 on others)
                current_index = (current_index - 1) % len(positions)
                print(f"Viewing position {current_index + 1}/{len(positions)}: {positions[current_index]['timestamp']}")
            elif key == 83 or key == 3:  # Right arrow (83 on some systems, 3 on others)
                current_index = (current_index + 1) % len(positions)
                print(f"Viewing position {current_index + 1}/{len(positions)}: {positions[current_index]['timestamp']}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
