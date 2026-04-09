import cv2
import numpy as np

def center_of_mass(mask):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        # Fallback to geometric centre when wafer map is empty
        h, w = mask.shape
        return np.array([h / 2.0, w / 2.0])
    return coords.mean(axis=0)
def detect_wafer(wafer_map):
    """
    Detects the wafer center and radius from a 2D wafer map.
    Assumes 0 for background, 1 for normal dies, 2 for defects.
    """
    # Create mask of dies (normal or defect)
    mask = (wafer_map > 0).astype(np.uint8)
    
    # Calculate moments for center
    y, x = center_of_mass(mask)
    center = (int(x), int(y))
    
    # Estimate radius (max distance from center to any die)
    rows, cols = np.where(mask > 0)
    distances = np.sqrt((rows - y)**2 + (cols - x)**2)
    radius = int(np.max(distances))
    
    return center, radius

def extract_edge_ring(wafer_map, center, radius, edge_depth=5):
    """
    Extracts the annular edge ring region.
    """
    mask = np.zeros_like(wafer_map, dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)
    
    inner_radius = max(0, radius - edge_depth)
    cv2.circle(mask, center, inner_radius, 0, -1)
    
    # Isolate the ring
    ring_region = wafer_map * mask
    return ring_region

def cartesian_to_polar(wafer_map, center, radius, edge_depth=5, output_shape=(360, 64)):
    """
    Converts the edge ring from Cartesian to Polar coordinates.
    Horizontal axis = theta (0-360)
    Vertical axis = radial depth (distance from periphery inward)
    """
    theta_res, radial_res = output_shape
    
    # Create an empty strip
    polar_strip = np.zeros((radial_res, theta_res), dtype=np.float32)
    
    theta = np.linspace(0, 2 * np.pi, theta_res)
    # radial depth goes from radius down to radius - edge_depth
    r = np.linspace(radius, radius - edge_depth, radial_res)
    
    # Compute coordinates
    T, R = np.meshgrid(theta, r)
    X = center[0] + R * np.cos(T)
    Y = center[1] + R * np.sin(T)
    
    # Remap using OpenCV for efficiency
    # Note: cv2.remap expects map_x and map_y as float32
    polar_strip = cv2.remap(wafer_map.astype(np.float32), 
                            X.astype(np.float32), 
                            Y.astype(np.float32), 
                            cv2.INTER_LINEAR)
    
    return polar_strip

def preprocess_wafer(wafer_map, edge_depth=5):
    """
    Full pipeline: Detection -> Ring Extraction -> Polar Transform
    """
    try:
        center, radius = detect_wafer(wafer_map)
        polar_strip = cartesian_to_polar(wafer_map, center, radius, edge_depth=edge_depth)
        return polar_strip
    except Exception as e:
        # Fallback or error handling for invalid/empty maps
        return np.zeros((64, 360), dtype=np.float32)

if __name__ == "__main__":
    # Test with a dummy circular map
    dummy = np.zeros((100, 100))
    cv2.circle(dummy, (50, 50), 40, 1, -1)
    cv2.circle(dummy, (50, 50), 38, 0, 1) # some "defects" near edge
    
    strip = preprocess_wafer(dummy, edge_depth=10)
    print(f"Polar strip shape: {strip.shape}")
    # strip shape will be (64, 360) by default
