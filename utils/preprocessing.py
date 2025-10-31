@@ -0,0 +1,124 @@
"""
Preprocessing utilities for Matka Cleaning Monitoring System
"""

import os
import numpy as np
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import torch
        import torchvision
        import ultralytics
        import numpy
        import cv2
        import matplotlib
        import PIL
        import tensorflow
        logger.info("All dependencies are available.")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def create_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['data/train/clean', 'data/train/dirty', 'data/val/clean', 'data/val/dirty', 'models']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info("Directories created successfully.")

def generate_synthetic_ir_image(size=(224, 224), is_clean=True):
    """
    Generate synthetic IR image simulating matka heat patterns
    Args:
        size: Tuple of (height, width)
        is_clean: If True, generate clean pattern, else dirty
    Returns:
        numpy array: Synthetic IR image
    """
    # Create base image
    image = np.zeros(size, dtype=np.uint8)

    # Create circular matka shape
    center = (size[0] // 2, size[1] // 2)
    radius = min(size) // 3

    # Generate heat pattern
    y, x = np.ogrid[:size[0], :size[1]]
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Base temperature gradient
    base_temp = 100 + 50 * np.exp(-dist_from_center / (radius * 0.8))

    if is_clean:
        # Clean: uniform, smooth heat distribution
        noise = np.random.normal(0, 5, size)
        temperature = base_temp + noise
    else:
        # Dirty: uneven, patchy heat distribution
        # Add random hot/cold spots
        num_spots = np.random.randint(3, 8)
        temperature = base_temp.copy()

        for _ in range(num_spots):
            spot_center = (
                np.random.randint(radius//2, size[0]-radius//2),
                np.random.randint(radius//2, size[1]-radius//2)
            )
            spot_radius = np.random.randint(10, 30)
            spot_intensity = np.random.choice([-30, 30])  # Hot or cold spot

            spot_dist = np.sqrt((x - spot_center[1])**2 + (y - spot_center[0])**2)
            spot_mask = spot_dist < spot_radius
            temperature[spot_mask] += spot_intensity * np.exp(-spot_dist[spot_mask] / spot_radius)

        # Add noise
        noise = np.random.normal(0, 10, size)
        temperature += noise

    # Clip to valid range and convert to uint8
    temperature = np.clip(temperature, 0, 255).astype(np.uint8)

    # Apply circular mask
    mask = dist_from_center <= radius
    image[mask] = temperature[mask]

    return image

def augment_image(image):
    """
    Apply data augmentation to image
    Args:
        image: Input image
    Returns:
        Augmented image
    """
    # Random flip
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Random brightness adjustment
    brightness = np.random.uniform(0.8, 1.2)
    image = np.clip(image * brightness, 0, 255).astype(np.uint8)

    # Slight blur
    if np.random.random() > 0.5:
        kernel_size = np.random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return image

def save_image(image, path):
    """Save image to specified path"""
    cv2.imwrite(str(path), image)
