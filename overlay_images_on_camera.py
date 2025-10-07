#!/usr/bin/env python3
"""
Script to display live feed from camera 2 with cup and lego block images
overlaid at random positions. The images have transparent backgrounds.
Press 'q' to quit, 'r' to randomize positions, 'y' to save positions to CSV.
"""

import cv2
import numpy as np
import sys
import random
import csv
import os
from datetime import datetime


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
    # Get dimensions
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]
    
    # Calculate the region to overlay
    # Make sure we don't go out of bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + ol_w)
    y2 = min(bg_h, y + ol_h)
    
    # Adjust overlay coordinates if position is negative
    ol_x1 = max(0, -x)
    ol_y1 = max(0, -y)
    ol_x2 = ol_x1 + (x2 - x1)
    ol_y2 = ol_y1 + (y2 - y1)
    
    # If overlay is completely outside the frame, return original background
    if x2 <= x1 or y2 <= y1:
        return background
    
    # Extract the region of interest from background
    roi = background[y1:y2, x1:x2]
    
    # Extract the overlay region and alpha channel
    overlay_region = overlay[ol_y1:ol_y2, ol_x1:ol_x2]
    
    # Check if overlay has alpha channel
    if overlay_region.shape[2] == 4:
        # Split the overlay into color and alpha channels
        overlay_bgr = overlay_region[:, :, :3]
        alpha = overlay_region[:, :, 3] / 255.0
        
        # Apply additional transparency (reduce alpha by transparency factor)
        alpha = alpha * (1.0 - transparency)
        
        # Expand alpha to 3 channels
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
        
        # Blend the images
        blended = (alpha_3ch * overlay_bgr + (1 - alpha_3ch) * roi).astype(np.uint8)
        
        # Place the blended region back into the background
        background[y1:y2, x1:x2] = blended
    else:
        # If no alpha channel, just place it directly
        background[y1:y2, x1:x2] = overlay_region
    
    return background


def is_in_exclusion_zone(x, y, img_width, img_height):
    """
    Check if an image at position (x, y) overlaps with the exclusion zone.
    Exclusion zone: x from 253 to 388, y from 95 to 147
    
    Args:
        x: X position of image top-left corner
        y: Y position of image top-left corner
        img_width: Width of the image
        img_height: Height of the image
    
    Returns:
        True if image overlaps with exclusion zone, False otherwise
    """
    # Exclusion zone boundaries
    excl_x1, excl_y1 = 253, 95
    excl_x2, excl_y2 = 388, 147
    
    # Image boundaries
    img_x1, img_y1 = x, y
    img_x2, img_y2 = x + img_width, y + img_height
    
    # Check for overlap (rectangles overlap if they don't NOT overlap)
    overlap = not (img_x2 < excl_x1 or img_x1 > excl_x2 or 
                   img_y2 < excl_y1 or img_y1 > excl_y2)
    
    return overlap


def get_random_position(frame_width, frame_height, img_width, img_height, 
                       other_pos=None, other_size=None, margin=50):
    """
    Get a random position within defined boundaries, avoiding exclusion zone
    and maintaining minimum distance from other object.
    
    Boundaries:
    - Outer box: x from 125 to 515, y from 95 to 400
    - Exclusion zone: x from 253 to 388, y from 95 to 147
    - Minimum distance from other object: 200 pixels in both axes
    
    Args:
        frame_width: Width of the video frame (not used with fixed boundaries)
        frame_height: Height of the video frame (not used with fixed boundaries)
        img_width: Width of the overlay image
        img_height: Height of the overlay image
        other_pos: Position of other object as (x, y) tuple, or None
        other_size: Size of other object as (width, height) tuple, or None
        margin: Not used (kept for compatibility)
    
    Returns:
        Tuple of (x, y) coordinates
    """
    # Define boundaries
    min_x, max_x = 40, 600
    min_y, max_y = 80, 440
    
    # Adjust max values to account for image size
    max_x = max_x - img_width
    max_y = max_y - img_height
    
    # Minimum distance requirement
    min_distance = 100
    
    # Try to find a valid position (max 1000 attempts)
    for attempt in range(1000):
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)
        
        # Check if position is in exclusion zone
        if is_in_exclusion_zone(x, y, img_width, img_height):
            continue
        
        # Check distance from other object if provided
        if other_pos is not None and other_size is not None:
            other_x, other_y = other_pos
            other_w, other_h = other_size
            
            # Calculate center-to-center distance in both axes
            center_x = x + img_width // 2
            center_y = y + img_height // 2
            other_center_x = other_x + other_w // 2
            other_center_y = other_y + other_h // 2
            
            dx = abs(center_x - other_center_x)
            dy = abs(center_y - other_center_y)
            
            # Both axes must have at least min_distance separation
            if dx < min_distance or dy < min_distance:
                continue
        
        # Valid position found
        return x, y
    
    # If no valid position found after many attempts, return a default position
    print(f"Warning: Could not find valid position after 1000 attempts, using fallback")
    return min_x, min_y


def main():
    # Camera index
    camera_index = 2
    
    # CSV file setup
    csv_filename = "object_positions.csv"
    csv_exists = os.path.exists(csv_filename)
    
    # Load the PNG images with alpha channel
    print("Loading images...")
    cup_img = cv2.imread("cup.png", cv2.IMREAD_UNCHANGED)
    block_img = cv2.imread("legoblock.png", cv2.IMREAD_UNCHANGED)
    
    if cup_img is None:
        print("Error: Could not load cup.png")
        sys.exit(1)
    
    if block_img is None:
        print("Error: Could not load legoblock.png")
        sys.exit(1)
    
    print(f"Cup image shape: {cup_img.shape}")
    print(f"Lego block image shape: {block_img.shape}")
    
    # Open camera
    print(f"Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        sys.exit(1)
    
    print(f"Camera {camera_index} opened successfully!")
    print("Press 'q' to quit, 'r' to randomize positions, 'y' to save positions")
    
    # Initialize positions
    cup_pos = None
    block_pos = None
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read from camera")
                break
            
            # Get frame dimensions
            frame_h, frame_w = frame.shape[:2]
            
            # Initialize random positions on first frame or when requested
            if cup_pos is None:
                # Get cup position first
                cup_pos = get_random_position(frame_w, frame_h, 
                                              cup_img.shape[1], cup_img.shape[0])
                # Get block position, ensuring it's at least 200 pixels away from cup
                block_pos = get_random_position(frame_w, frame_h, 
                                               block_img.shape[1], block_img.shape[0],
                                               other_pos=cup_pos,
                                               other_size=(cup_img.shape[1], cup_img.shape[0]))
                print(f"Cup position: {cup_pos}")
                print(f"Block position: {block_pos}")
            
            # Create a copy of the frame to overlay images on
            display_frame = frame.copy()
            
            # Overlay cup image
            display_frame = overlay_transparent(display_frame, cup_img, 
                                               cup_pos[0], cup_pos[1])
            
            # Overlay lego block image
            display_frame = overlay_transparent(display_frame, block_img, 
                                               block_pos[0], block_pos[1])
            
            # Display the frame
            cv2.imshow(f"Camera {camera_index} with Overlays", display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Randomize positions
                # Get cup position first
                cup_pos = get_random_position(frame_w, frame_h, 
                                              cup_img.shape[1], cup_img.shape[0])
                # Get block position, ensuring it's at least 200 pixels away from cup
                block_pos = get_random_position(frame_w, frame_h, 
                                               block_img.shape[1], block_img.shape[0],
                                               other_pos=cup_pos,
                                               other_size=(cup_img.shape[1], cup_img.shape[0]))
                print(f"New cup position: {cup_pos}")
                print(f"New block position: {block_pos}")
            elif key == ord('y'):
                # Save positions to CSV
                with open(csv_filename, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write header if file is new
                    if not csv_exists:
                        writer.writerow(['timestamp', 'cup_x', 'cup_y', 'block_x', 'block_y'])
                        csv_exists = True
                    
                    # Write positions
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([timestamp, cup_pos[0], cup_pos[1], block_pos[0], block_pos[1]])
                    print(f"✓ Saved positions to {csv_filename}")
                    print(f"  Cup: {cup_pos}, Block: {block_pos}")
                
                # Automatically randomize after saving
                cup_pos = get_random_position(frame_w, frame_h, 
                                              cup_img.shape[1], cup_img.shape[0])
                block_pos = get_random_position(frame_w, frame_h, 
                                               block_img.shape[1], block_img.shape[0],
                                               other_pos=cup_pos,
                                               other_size=(cup_img.shape[1], cup_img.shape[0]))
                print(f"New cup position: {cup_pos}")
                print(f"New block position: {block_pos}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
