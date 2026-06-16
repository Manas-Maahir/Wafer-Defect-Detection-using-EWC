import cv2
import numpy as np

def generate_synthetic_wafer(filename, defect_type="edge_ring"):
    # Create a blank black image (background)
    wafer = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Draw dark gray circle for the wafer (intensity 100)
    center = (150, 150)
    radius = 120
    cv2.circle(wafer, center, radius, (100, 100, 100), -1)
    
    # Add grid lines to simulate dies
    for i in range(30, 270, 10):
        cv2.line(wafer, (i, 30), (i, 270), (80, 80, 80), 1)
        cv2.line(wafer, (30, i), (270, i), (80, 80, 80), 1)
    
    # Re-mask with circle to avoid grid outside wafer
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)
    wafer[mask == 0] = (0, 0, 0)
    
    # Add defect
    if defect_type == "edge_ring":
        # Draw some bright dies near the edge to form a ring
        cv2.circle(wafer, center, radius-5, (255, 255, 255), 2)
    elif defect_type == "scratch":
        # Draw a scratch
        cv2.line(wafer, (100, 100), (200, 180), (255, 255, 255), 3)
    
    cv2.imwrite(filename, wafer)
    print(f"Saved {filename}")

if __name__ == "__main__":
    generate_synthetic_wafer("test_edge_ring.png", "edge_ring")
    generate_synthetic_wafer("test_scratch.png", "scratch")
    generate_synthetic_wafer("test_normal.png", "normal")
    print("Test images generated successfully.")
