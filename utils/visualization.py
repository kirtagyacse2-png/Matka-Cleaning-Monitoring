"""
Visualization utilities for Matka Cleaning Monitoring System
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def create_heatmap_overlay(image, heat_percentage, threshold=80):
    """
    Create a heatmap overlay on the image

    Args:
        image: Input image
        heat_percentage: Percentage of hot surface area
        threshold: Threshold for clean/dirty classification

    Returns:
        Image with heatmap overlay
    """
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        display_image = image.copy()

    # Create heatmap based on heat percentage
    if heat_percentage >= threshold:
        # Green for clean
        color = (0, 255, 0)
        status = "CLEAN"
    else:
        # Red for dirty
        color = (0, 0, 255)
        status = "NOT CLEAN"

    # Add colored overlay
    overlay = display_image.copy()
    cv2.rectangle(overlay, (10, 10), (200, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, display_image, 0.7, 0, display_image)

    # Add text
    cv2.putText(display_image, f"Status: {status}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display_image, f"Heat Area: {heat_percentage:.1f}%", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return display_image

def plot_training_history(history):
    """
    Plot training history

    Args:
        history: Training history dictionary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    ax1.plot(history.get('accuracy', []), label='Train Accuracy')
    ax1.plot(history.get('val_accuracy', []), label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss
    ax2.plot(history.get('loss', []), label='Train Loss')
    ax2.plot(history.get('val_loss', []), label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Training history plot saved to models/training_history.png")

def display_image_with_info(image, title="Thermal Image", heat_percentage=None):
    """
    Display image with information

    Args:
        image: Image to display
        title: Window title
        heat_percentage: Heat percentage to display
    """
    if heat_percentage is not None:
        image = create_heatmap_overlay(image, heat_percentage)

    cv2.imshow(title, image)
    cv2.waitKey(1)  # Allow GUI to update

def save_result_image(image, heat_percentage, output_path="outputs/result.jpg"):
    """
    Save result image with overlay

    Args:
        image: Input image
        heat_percentage: Heat percentage
        output_path: Path to save the image
    """
    result_image = create_heatmap_overlay(image, heat_percentage)

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(output_path, result_image)
    logger.info(f"Result image saved to {output_path}")

def create_progress_bar(percentage, width=50):
    """
    Create a text-based progress bar

    Args:
        percentage: Percentage (0-100)
        width: Width of the progress bar

    Returns:
        String representation of progress bar
    """
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percentage:.1f}%"