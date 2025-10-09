import cv2
import numpy as np
import csv
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from PIL import Image

def overlay(frame, img, x, y):
    frame = frame.copy()
    h, w = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    ox1, oy1 = max(0, -x), max(0, -y)
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return frame
    
    roi = frame[y1:y2, x1:x2]
    o = img[oy1:oy2, ox1:ox2]
    
    if o.shape[2] == 4:
        rgb = o[:, :, :3]
        alpha = (o[:, :, 3] / 255.0 * 0.7)[:, :, None]
        frame[y1:y2, x1:x2] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    
    return frame

def view_position(pos_num):
    with open("object_positions.csv", 'r') as f:
        positions = list(csv.DictReader(f))
    
    pos = positions[pos_num - 1]
    cup = np.array(Image.open("cup.png").convert('RGBA'))
    block = np.array(Image.open("legoblock.png").convert('RGBA'))
    cap = cv2.VideoCapture(2)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')
    
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = overlay(frame, cup, int(pos['cup_x']), int(pos['cup_y']))
    frame = overlay(frame, block, int(pos['block_x']), int(pos['block_y']))
    #frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    img_display = ax.imshow(frame)
    
    def update(frame_num):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = overlay(frame, cup, int(pos['cup_x']), int(pos['cup_y']))
            frame = overlay(frame, block, int(pos['block_x']), int(pos['block_y']))
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            img_display.set_array(frame)
        return [img_display]
    
    running = [True]
    def on_key(event):
        if event.key in ['q', 'escape']:
            running[0] = False
            plt.close()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, interval=16, blit=True, cache_frame_data=False)
    
    plt.show()
    cap.release()

if __name__ == "__main__":
    view_position(1)
