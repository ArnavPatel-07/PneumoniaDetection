# Model Training Documentation

This document describes the training pipeline for the Pneumonia Detection ResNet50 model.

## Training Overview

The model uses **ResNet50 transfer learning** with a two-phase training approach:
1. **Phase 1**: Train classifier head with frozen base model
2. **Phase 2**: Fine-tune deeper layers with lower learning rate

## Key Features

- **Medical-grade preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced X-ray contrast
- **Class imbalance handling**: Weighted loss function based on dataset distribution
- **Comprehensive evaluation**: AUC-ROC, Precision-Recall curves, Confusion Matrix, Grad-CAM visualizations
- **Clinical threshold analysis**: Sensitivity/Specificity trade-offs for different thresholds

## Dataset Structure

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Training Configuration

- **Image Size**: 384×384 pixels
- **Batch Size**: 32
- **Base Model**: ResNet50 (ImageNet weights)
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy with class weights
- **Metrics**: Accuracy, AUC-ROC, Precision, Recall

## Preprocessing Pipeline

1. Load image and convert to grayscale
2. Apply CLAHE enhancement (clipLimit=2.0, tileGridSize=8×8)
3. Convert back to RGB (3 channels for ResNet50)
4. Normalize to [0, 1] range
5. Resize to 384×384

## Data Augmentation (Training Only)

- Rotation: ±12 degrees
- Translation: ±15% width/height
- Zoom: ±15%
- Horizontal flip
- Brightness: ±15%

## Model Architecture

```
ResNet50 Base (frozen early layers)
    ↓
GlobalAveragePooling2D
    ↓
Attention Mechanism (Dense + Multiply)
    ↓
Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(256) → Dropout(0.3)
    ↓
Dense(128) → Dropout(0.2)
    ↓
Dense(1, sigmoid) → Binary Output
```

## Training Phases

### Phase 1: Head Training
- **Epochs**: 15 (with early stopping)
- **Learning Rate**: 1e-3
- **Base Model**: Frozen
- **Monitor**: Validation AUC-ROC
- **Patience**: 5 epochs

### Phase 2: Fine-Tuning
- **Epochs**: 20 (with early stopping)
- **Learning Rate**: 1e-5
- **Trainable Layers**: Layers 100+ unfrozen
- **Monitor**: Validation AUC-ROC
- **Patience**: 8 epochs

## Evaluation Metrics

The model is evaluated using:
- **AUC-ROC**: Area under ROC curve
- **Accuracy**: Overall classification accuracy
- **Sensitivity (Recall)**: True Positive Rate
- **Specificity**: True Negative Rate
- **Precision**: Positive Predictive Value
- **F1-Score**: Harmonic mean of precision and recall

## Model Files

- `resnet50_phase1_best.h5`: Best model from Phase 1
- `resnet50_final_best.h5`: Best model from Phase 2 (final)
- `pneumonia_resnet50_final.h5`: Final saved model

## Usage Notes

⚠️ **Important**: The training script uses CLAHE preprocessing, but the current FastAPI backend uses standard ResNet50 preprocessing. For production deployment, ensure preprocessing matches training pipeline.

## Training Script Location

The complete training script is provided in the user's training notebook. Key components:
- Data exploration and class imbalance analysis
- Medical preprocessing with CLAHE
- Two-phase training with callbacks
- Comprehensive evaluation and visualization
- Grad-CAM for interpretability

## Performance Expectations

Based on the training pipeline, expected performance:
- **AUC-ROC**: > 0.95
- **Accuracy**: > 90%
- **Sensitivity**: > 90% (critical for medical screening)
- **Specificity**: > 85%

## Next Steps for Production

1. Align preprocessing between training and inference
2. Implement CLAHE preprocessing in FastAPI backend
3. Add model versioning and A/B testing
4. Implement logging and monitoring
5. Add authentication and rate limiting
6. Deploy with proper error handling and validation
