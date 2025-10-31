# Matka Cleaning Monitoring System

A computer vision system that uses thermal imaging and machine learning to automatically monitor and classify the cleaning status of traditional Indian cooking vessels (matkas) in real-time.

## Overview

This system analyzes thermal images captured from matkas during the cleaning process to determine whether they are properly cleaned. It uses a combination of computer vision techniques and deep learning models to:

- Detect heat patterns in thermal images
- Classify matkas as "clean" or "not clean" based on temperature uniformity
- Provide real-time monitoring through webcam integration
- Generate synthetic training data for model training
- Visualize results with heatmaps and overlays

## Features

- **Real-time Monitoring**: Webcam integration for live cleaning status assessment
- **Image Analysis**: Process individual thermal images for cleaning verification
- **Synthetic Data Generation**: Create training datasets with realistic thermal patterns
- **Deep Learning Classification**: EfficientNet-based model for accurate classification
- **Heatmap Analysis**: Fallback analysis using thermal uniformity metrics
- **Visualization Tools**: Training history plots, result overlays, and progress indicators
- **Modular Architecture**: Separate scripts for training, inference, and analysis

## How It Works

### Thermal Analysis Principle
- **Clean Matka**: Exhibits uniform heat distribution across the surface
- **Dirty Matka**: Shows uneven temperature patterns with hot/cold spots

The system uses two complementary approaches:

1. **Deep Learning Model**: Trained EfficientNet model that learns to classify thermal patterns
2. **Rule-based Analysis**: Heatmap analysis that measures temperature uniformity and hot surface area percentage

## Installation

### Prerequisites
- Python 3.8+
- Webcam (for real-time monitoring)

### Setup
1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies
- `torch` & `torchvision` - PyTorch for deep learning
- `ultralytics` - YOLO models (future expansion)
- `tensorflow` - Keras for EfficientNet model
- `opencv-python` - Computer vision operations
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `pillow` - Image processing
- `scikit-learn` - Machine learning utilities

## Usage

### Training the Model
Generate synthetic dataset and train the classification model:
```bash
python main.py --mode train
```

This will:
- Generate 1000 synthetic thermal images (500 clean, 500 dirty)
- Train an EfficientNetB0 model for 50 epochs
- Save the trained model to `models/matka_ir_cleaning.keras`
- Generate training history plots

### Real-time Inference (Webcam)
Monitor cleaning in real-time using webcam:
```bash
python main.py --mode infer --source webcam
```

Features:
- Live thermal analysis overlay
- Real-time clean/dirty classification
- Confidence scores display
- Automatic result saving every 30 seconds

### Single Image Analysis
Analyze a specific thermal image:
```bash
python main.py --mode infer --source path/to/thermal_image.jpg
```

### Advanced Usage

#### Custom Model Path
```bash
python main.py --mode infer --source webcam --model_path custom_model.keras
```

#### Heatmap Analysis Only
Analyze thermal images using rule-based approach:
```bash
python scripts/heatmap_analysis.py path/to/image.jpg --save result.jpg
```

## Project Structure

```
matka_cleaning_monitoring/
├── main.py                 # Main entry point with CLI interface
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── scripts/
│   ├── train.py           # Model training script
│   ├── infer.py           # Inference and webcam processing
│   └── heatmap_analysis.py # Rule-based thermal analysis
├── utils/
│   ├── preprocessing.py   # Data generation and preprocessing
│   └── visualization.py   # Plotting and image overlays
├── data/                  # Training data directory (auto-generated)
│   ├── train/
│   │   ├── clean/         # Clean matka thermal images
│   │   └── dirty/         # Dirty matka thermal images
│   └── val/               # Validation data
├── models/                # Trained models directory
└── outputs/               # Inference results and logs
```

## Technical Details

### Model Architecture
- **Base Model**: EfficientNetB0 (pre-trained on ImageNet, fine-tuned for thermal images)
- **Input Size**: 224x224 grayscale thermal images
- **Output**: Binary classification (clean/dirty)
- **Training**: Adam optimizer, categorical cross-entropy loss
- **Augmentation**: Random flips, rotations, brightness adjustments

### Synthetic Data Generation
- Circular matka shape simulation
- Realistic heat gradients from center to edges
- Clean patterns: uniform temperature distribution
- Dirty patterns: random hot/cold spots with noise
- Data augmentation for robustness

### Thermal Analysis Metrics
- **Heat Percentage**: Percentage of surface area above temperature threshold
- **Uniformity Score**: Measure of temperature distribution consistency
- **Classification Threshold**: 80% uniformity for clean classification

## Configuration

### Training Parameters
- Dataset size: 1000 samples (configurable in `train.py`)
- Validation split: 20%
- Batch size: 32
- Epochs: 50 (with early stopping)
- Learning rate: 0.001 (with reduction on plateau)

### Analysis Parameters
- Temperature threshold: 127 (configurable)
- Uniformity threshold: 80%
- Image size: 224x224 for model input

## Output Formats

### Training Outputs
- `models/matka_ir_cleaning.keras` - Trained Keras model
- `models/training_history.png` - Training curves plot

### Inference Outputs
- Console output with classification results
- Timestamped result images in `outputs/` directory
- Real-time webcam overlay with status indicators

## Troubleshooting

### Common Issues
1. **Webcam not detected**: Ensure camera permissions and no other applications using it
2. **Model loading fails**: Check model path and ensure training completed successfully
3. **Low accuracy**: Increase training dataset size or adjust synthetic data parameters
4. **Memory errors**: Reduce batch size or image resolution

### Performance Tips
- Use GPU for faster training (CUDA-enabled PyTorch/TensorFlow)
- Reduce synthetic dataset size for quicker experimentation
- Enable early stopping to prevent overfitting

## Future Enhancements

- Integration with actual thermal cameras (FLIR, etc.)
- Multi-class classification (partially clean, heavily soiled)
- Batch processing for multiple matkas
- Web interface for remote monitoring
- Mobile app integration
- Integration with cleaning equipment automation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open-source. Please check individual component licenses for compliance.

## Contact

For questions or support, please create an issue in the repository.
