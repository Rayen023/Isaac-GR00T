#!/usr/bin/env python3
"""
Script to view saved object positions from CSV on live camera feed using matplotlib.
Use left/right arrow keys to navigate between saved positions.
Press 'q' or close window to quit.
"""

import matplotlib
# Set interactive backend before importing pyplot
matplotlib.use('TkAgg')  # or 'Qt5Agg' if TkAgg doesn't work

import cv2
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from PIL import Image


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


def overlay_image_on_frame(frame, overlay_img, x, y, transparency=0.7):
    """
    Overlay a PNG image on a frame at position (x, y).
    
    Args:
        frame: Background image (RGB)
        overlay_img: Overlay image with alpha channel (RGBA)
        x: X position for overlay
        y: Y position for overlay
        transparency: Additional transparency to apply (0.0 = invisible, 1.0 = original opacity)
    
    Returns:
        Frame with overlay applied
    """
    frame = frame.copy()
    frame_h, frame_w = frame.shape[:2]
    ol_h, ol_w = overlay_img.shape[:2]
    
    # Calculate bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + ol_w)
    y2 = min(frame_h, y + ol_h)
    
    ol_x1 = max(0, -x)
    ol_y1 = max(0, -y)
    ol_x2 = ol_x1 + (x2 - x1)
    ol_y2 = ol_y1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return frame
    
    roi = frame[y1:y2, x1:x2]
    overlay_region = overlay_img[ol_y1:ol_y2, ol_x1:ol_x2]
    
    # Extract RGB and alpha channels
    if overlay_region.shape[2] == 4:
        overlay_rgb = overlay_region[:, :, :3]
        alpha = overlay_region[:, :, 3] / 255.0
        alpha = alpha * transparency
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
        
        # Blend the overlay with the background
        blended = (alpha_3ch * overlay_rgb + (1 - alpha_3ch) * roi).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended
    else:
        frame[y1:y2, x1:x2] = overlay_region
    
    return frame


class PositionViewer:
    """Interactive position viewer using matplotlib."""
    
    def __init__(self, camera_index=2, csv_filename="object_positions.csv"):
        self.camera_index = camera_index
        self.csv_filename = csv_filename
        self.current_index = 0
        self.positions = None
        self.cap = None
        self.cup_img = None
        self.block_img = None
        self.fig = None
        self.ax = None
        self.img_display = None
        self.running = True
        
    def load_data(self):
        """Load positions from CSV and overlay images."""
        print(f"Loading positions from {self.csv_filename}...")
        self.positions = load_positions_from_csv(self.csv_filename)
        
        if self.positions is None or len(self.positions) == 0:
            print("No positions found in CSV file")
            return False
        
        print(f"Loaded {len(self.positions)} position sets")
        
        # Load overlay images using PIL for RGBA support
        print("Loading overlay images...")
        try:
            self.cup_img = np.array(Image.open("cup.png").convert('RGBA'))
            self.block_img = np.array(Image.open("legoblock.png").convert('RGBA'))
        except Exception as e:
            print(f"Error loading overlay images: {e}")
            return False
        
        return True
    
    def open_camera(self):
        """Open the camera."""
        print(f"Opening camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        print(f"Camera {self.camera_index} opened successfully!")
        return True
    
    def get_frame_with_overlays(self):
        """Get current camera frame with overlays."""
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        # Convert BGR to RGB for matplotlib
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get current position
        pos = self.positions[self.current_index]
        
        # Overlay cup and block at saved positions
        frame = overlay_image_on_frame(frame, self.cup_img, pos['cup_x'], pos['cup_y'])
        frame = overlay_image_on_frame(frame, self.block_img, pos['block_x'], pos['block_y'])
        
        return frame, pos
    
    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'q' or event.key == 'escape':
            self.running = False
            plt.close(self.fig)
        elif event.key == 'left':
            self.current_index = (self.current_index - 1) % len(self.positions)
            print(f"Viewing position {self.current_index + 1}/{len(self.positions)}: "
                  f"{self.positions[self.current_index]['timestamp']}")
        elif event.key == 'right':
            self.current_index = (self.current_index + 1) % len(self.positions)
            print(f"Viewing position {self.current_index + 1}/{len(self.positions)}: "
                  f"{self.positions[self.current_index]['timestamp']}")
    
    def on_close(self, event):
        """Handle window close event."""
        self.running = False
    
    def update_frame(self, frame_num):
        """Update the displayed frame (called by animation)."""
        if not self.running:
            return []
        
        result = self.get_frame_with_overlays()
        if result is None:
            return []
        
        frame, pos = result
        
        # Update the image
        self.img_display.set_array(frame)
        
        # Update the title
        info_text = f"Position {self.current_index + 1}/{len(self.positions)} - {pos['timestamp']}"
        self.ax.set_title(info_text, fontsize=12, pad=10)
        
        return [self.img_display]
    
    def run(self):
        """Run the interactive viewer."""
        if not self.load_data():
            return False
        
        if not self.open_camera():
            return False
        
        print("\nControls:")
        print("  Left Arrow  - Previous position")
        print("  Right Arrow - Next position")
        print("  'q' or ESC  - Quit")
        print()
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.fig.canvas.manager.set_window_title(f'Camera {self.camera_index} - Saved Positions')
        
        # Get initial frame
        result = self.get_frame_with_overlays()
        if result is None:
            print("Failed to read from camera")
            return False
        
        frame, pos = result
        
        # Display initial frame
        self.img_display = self.ax.imshow(frame)
        self.ax.axis('off')
        info_text = f"Position {self.current_index + 1}/{len(self.positions)} - {pos['timestamp']}"
        self.ax.set_title(info_text, fontsize=12, pad=10)
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # Start animation (updates at ~30 fps)
        anim = FuncAnimation(self.fig, self.update_frame, interval=33, blit=True, cache_frame_data=False)
        
        plt.tight_layout()
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        if self.cap is not None:
            self.cap.release()
        plt.close('all')


def main():
    """Run the position viewer. Returns True on success, False on failure."""
    viewer = PositionViewer(camera_index=2, csv_filename="object_positions.csv")
    success = viewer.run()
    return success


if __name__ == "__main__":
    # Only exit if running as standalone script
    success = main()
    sys.exit(0 if success else 1)
