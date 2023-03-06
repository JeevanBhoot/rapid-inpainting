import numpy as np
import cv2

def generate_random_mask(height, width, factor=0.5):
    # Create a blank image
    mask = np.ones((height, width), dtype=np.uint8)

    # Draw 3 random rectangles
    for _ in range(3):
        x1, y1 = np.random.randint(0, width, 2)
        x2, y2 = np.random.randint(0, factor*width, 2)
        x2, y2 = min(x1+x2, width), min(y1+y2, height)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    # Draw 10 random brush strokes
    for _ in range(10):
        x1, y1 = np.random.randint(0, width, 2)
        x2, y2 = np.random.randint(0, width, 2)
        thickness = np.random.randint(1, 6)
        cv2.line(mask, (x1, y1), (x2, y2), 0, thickness)

    return mask