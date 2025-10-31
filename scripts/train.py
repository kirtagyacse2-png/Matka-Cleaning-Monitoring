#!/usr/bin/env python3
"""
Training script for Matka Cleaning Monitoring System
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

# Import utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import (
    create_directories, generate_synthetic_ir_image, augment_image, save_image
)
from utils.visualization import plot_training_history

logger = logging.getLogger(__name__)

def create_efficientnet_model(input_shape=(224, 224, 1), num_classes=2):
    """
    Create EfficientNetB0 model for thermal image classification

    EfficientNetB0 is chosen for IR image analysis because:
    - Excellent performance with fewer parameters than larger models
    - Compound scaling for optimal accuracy-efficiency trade-off
    - Pre-trained on ImageNet, adaptable to thermal imaging
    - Good at capturing both local and global features in heat patterns

    Args:
        input_shape: Input image shape
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    # Load EfficientNetB0 with pre-trained weights
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,  # No pre-trained weights for thermal images
        input_shape=input_shape
    )

    # Add classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def generate_dataset(num_samples=1000, val_split=0.2):
    """
    Generate synthetic dataset for training

    Args:
        num_samples: Total number of samples to generate
        val_split: Validation split ratio

    Returns:
        Tuple of (train_images, train_labels, val_images, val_labels)
    """
    logger.info(f"Generating {num_samples} synthetic IR images...")

    create_directories()

    images = []
    labels = []

    for i in range(num_samples):
        # Alternate between clean and dirty
        is_clean = i % 2 == 0

        # Generate base image
        image = generate_synthetic_ir_image(is_clean=is_clean)

        # Apply augmentation
        image = augment_image(image)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Add channel dimension
        image = np.expand_dims(image, axis=-1)

        images.append(image)
        labels.append([1, 0] if is_clean else [0, 1])  # One-hot encoding

        # Save sample images
        if i < 10:  # Save first 10 images for inspection
            class_name = 'clean' if is_clean else 'dirty'
            split = 'train' if i < 5 else 'val'
            save_image(
                (image.squeeze() * 255).astype(np.uint8),
                f'data/{split}/{class_name}/sample_{i}.png'
            )

    images = np.array(images)
    labels = np.array(labels)

    # Split into train/val
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=val_split, random_state=42, stratify=labels.argmax(axis=1)
    )

    logger.info(f"Dataset generated: {len(train_images)} train, {len(val_images)} val samples")
    return train_images, train_labels, val_images, val_labels

def train_model():
    """Main training function"""
    logger.info("Starting model training...")

    # Generate dataset
    train_images, train_labels, val_images, val_labels = generate_dataset()

    # Create model
    model = create_efficientnet_model()

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/matka_ir_cleaning.keras',
            monitor='val_accuracy',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        )
    ]

    # Train model
    logger.info("Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks
    )

    # Save final model in PyTorch format for consistency
    # Convert to TorchScript (simplified approach)
    logger.info("Model training completed. Saving model...")

    # For now, save as Keras model - can be converted later if needed
    model.save('models/matka_ir_cleaning.keras')
    logger.info("Model saved to models/matka_ir_cleaning.keras")

    # Plot training history
    plot_training_history(history.history)

    return model

def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)
    train_model()

if __name__ == "__main__":
    main()