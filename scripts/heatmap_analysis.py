
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,163 @@
#!/usr/bin/env python3
"""
Heatmap Analysis Script for Matka Cleaning Monitoring System
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def analyze_thermal_image(image_path, threshold_temp=127, uniformity_threshold=80):
    """
    Analyze thermal image to determine cleaning status
    Args:
        image_path: Path to thermal image
        threshold_temp: Temperature threshold for hot/cold classification
        uniformity_threshold: Threshold for uniformity percentage
    Returns:
        Tuple of (is_clean, heat_percentage, uniformity_score)
    """
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
    else:
        # Assume it's already an image array
        image = image_path
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Threshold to find hot areas
    _, hot_mask = cv2.threshold(blurred, threshold_temp, 255, cv2.THRESH_BINARY)

    # Calculate percentage of hot surface area
    total_pixels = np.sum(image > 0)  # Non-zero pixels (matka area)
    hot_pixels = np.sum(hot_mask > 0)
    heat_percentage = (hot_pixels / total_pixels * 100) if total_pixels > 0 else 0

    # Calculate uniformity score
    # Uniformity is based on temperature variance
    matka_pixels = image[image > 0]
    if len(matka_pixels) > 0:
        mean_temp = np.mean(matka_pixels)
        std_temp = np.std(matka_pixels)
        uniformity_score = max(0, 100 - (std_temp / mean_temp * 100))
    else:
        uniformity_score = 0

    # Classify as clean if uniformity is high enough
    is_clean = uniformity_score >= uniformity_threshold

    logger.info(f"Heat percentage: {heat_percentage:.1f}%, Uniformity: {uniformity_score:.1f}%")
    logger.info(f"Classification: {'CLEAN' if is_clean else 'NOT CLEAN'}")

    return is_clean, heat_percentage, uniformity_score

def create_heat_overlay(image, hot_mask):
    """
    Create heat overlay on image
    Args:
        image: Original image
        hot_mask: Binary mask of hot areas
    Returns:
        Image with heat overlay
    """
    # Convert to BGR if needed
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    # Create colored overlay
    hot_overlay = np.zeros_like(overlay)
    hot_overlay[hot_mask > 0] = [0, 0, 255]  # Red for hot areas

    # Blend with original
    result = cv2.addWeighted(overlay, 0.7, hot_overlay, 0.3, 0)

    return result

def process_image(image_path, display=True, save_path=None):
    """
    Process a single thermal image
    Args:
        image_path: Path to image or image array
        display: Whether to display results
        save_path: Path to save result image
    Returns:
        Tuple of (is_clean, heat_percentage, uniformity_score)
    """
    try:
        is_clean, heat_percentage, uniformity_score = analyze_thermal_image(image_path)

        if display or save_path:
            # Load image for visualization
            if isinstance(image_path, str):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = image_path

            # Create overlay
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            _, hot_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

            result_image = create_heat_overlay(image, hot_mask)

            # Add text overlay
            status = "CLEAN" if is_clean else "NOT CLEAN"
            color = (0, 255, 0) if is_clean else (0, 0, 255)

            cv2.putText(result_image, f"Status: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(result_image, f"Heat Area: {heat_percentage:.1f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_image, f"Uniformity: {uniformity_score:.1f}%", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if display:
                cv2.imshow("Thermal Analysis", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(save_path, result_image)
                logger.info(f"Result saved to {save_path}")

        return is_clean, heat_percentage, uniformity_score

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return False, 0, 0

def main():
    """Main entry point for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze thermal image for matka cleaning")
    parser.add_argument("image_path", help="Path to thermal image")
    parser.add_argument("--save", help="Save result image to path")
    parser.add_argument("--no-display", action="store_true", help="Don't display result")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    display = not args.no_display
    process_image(args.image_path, display=display, save_path=args.save)

if __name__ == "__main__":
    main()
