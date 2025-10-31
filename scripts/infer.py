@@ -0,0 +1,200 @@
#!/usr/bin/env python3
"""
Inference script for Matka Cleaning Monitoring System
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
import time
from datetime import datetime

# Import utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import generate_synthetic_ir_image
from utils.visualization import create_heatmap_overlay, display_image_with_info, save_result_image
from scripts.heatmap_analysis import analyze_thermal_image

logger = logging.getLogger(__name__)

class MatkaClassifier:
    """Matka cleaning classifier using trained model"""

    def __init__(self, model_path='models/matka_ir_cleaning.keras'):
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            if self.model_path.exists():
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}. Using fallback analysis.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to model input size
        image = cv2.resize(image, (224, 224))

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=[0, -1])

        return image

    def predict(self, image):
        """
        Predict cleaning status
        Args:
            image: Input image array
        Returns:
            Tuple of (prediction, confidence)
        """
        if self.model is None:
            # Fallback to heatmap analysis
            is_clean, heat_percentage, uniformity = analyze_thermal_image(image)
            confidence = uniformity / 100.0
            prediction = "clean" if is_clean else "dirty"
        else:
            # Use trained model
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            prediction = "clean" if class_idx == 0 else "dirty"

        return prediction, confidence

def process_webcam(classifier, output_dir="outputs"):
    """Process webcam feed for real-time inference"""
    logger.info("Starting webcam inference...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return

    Path(output_dir).mkdir(exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for thermal simulation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Predict
            prediction, confidence = classifier.predict(gray)

            # Create overlay
            overlay_frame = create_heatmap_overlay(gray, confidence * 100)

            # Add prediction text
            status = "CLEAN" if prediction == "clean" else "NOT CLEAN"
            color = (0, 255, 0) if prediction == "clean" else (0, 0, 255)

            cv2.putText(overlay_frame, f"Status: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(overlay_frame, f"Confidence: {confidence:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display
            cv2.imshow("Matka Cleaning Monitor", overlay_frame)

            # Save periodic results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if int(time.time()) % 30 == 0:  # Save every 30 seconds
                save_path = f"{output_dir}/webcam_{timestamp}.jpg"
                cv2.imwrite(save_path, overlay_frame)
                logger.info(f"Webcam result saved to {save_path}")

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

def process_image_file(image_path, classifier, output_dir="outputs"):
    """Process a single image file"""
    logger.info(f"Processing image: {image_path}")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return

    # Predict
    prediction, confidence = classifier.predict(image)

    # Create result
    heat_percentage = confidence * 100  # Simplified
    result_image = create_heatmap_overlay(image, heat_percentage)

    # Display results
    print(f"Image: {image_path}")
    print(f"Status: {'CLEAN' if prediction == 'clean' else 'NOT CLEAN'}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Heat Distribution: {heat_percentage:.1f}%")

    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{output_dir}/result_{timestamp}.jpg"
    save_result_image(image, heat_percentage, save_path)

    # Display
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(source, model_path='models/matka_ir_cleaning.keras'):
    """Main inference function"""
    logging.basicConfig(level=logging.INFO)

    # Initialize classifier
    classifier = MatkaClassifier(model_path)

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    if source.lower() == 'webcam':
        process_webcam(classifier, output_dir)
    else:
        image_path = Path(source)
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return
        process_image_file(image_path, classifier, output_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference for Matka Cleaning Monitoring")
    parser.add_argument("--source", required=True, help="Image path or 'webcam'")
    parser.add_argument("--model", default="models/matka_ir_cleaning.keras", help="Model path")

    args = parser.parse_args()
    main(args.source, args.model)
