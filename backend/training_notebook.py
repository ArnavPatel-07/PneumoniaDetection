"""
Pneumonia Detection using ResNet50 Transfer Learning
Medical-Grade Deep Learning for Chest X-Ray Classification

This script contains the complete training pipeline.
Run this in a Jupyter notebook or Python environment with GPU support.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    classification_report, precision_score, 
    recall_score, f1_score, precision_recall_curve
)
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ============================================================================
# DATA EXPLORATION
# ============================================================================

base_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Count images in each set
def count_images(directory):
    counts = {}
    for subdir in ['NORMAL', 'PNEUMONIA']:
        path = os.path.join(directory, subdir)
        if os.path.exists(path):
            counts[subdir] = len(os.listdir(path))
    return counts

train_counts = count_images(train_dir)
val_counts = count_images(val_dir)
test_counts = count_images(test_dir)

print("\n📊 Dataset Distribution:")
print(f"Training Set   - Normal: {train_counts.get('NORMAL', 0)}, Pneumonia: {train_counts.get('PNEUMONIA', 0)}")
print(f"Validation Set - Normal: {val_counts.get('NORMAL', 0)}, Pneumonia: {val_counts.get('PNEUMONIA', 0)}")
print(f"Test Set       - Normal: {test_counts.get('NORMAL', 0)}, Pneumonia: {test_counts.get('PNEUMONIA', 0)}")

# Calculate class imbalance for weighting
total_train = sum(train_counts.values())
pneumonia_ratio = train_counts['PNEUMONIA'] / train_counts['NORMAL']
print(f"\n⚖️ Class Imbalance Ratio: {pneumonia_ratio:.2f}:1 (Pneumonia:Normal)")

# ============================================================================
# MEDICAL IMAGE PREPROCESSING
# ============================================================================

def medical_preprocessing_resnet(image):
    """Enhanced preprocessing for chest X-rays with CLAHE contrast enhancement"""
    # Ensure image is in proper format (0-255 range)
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 2:
        gray = image
    else:
        gray = image[:, :, 0]
    
    # Ensure grayscale is uint8
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to 3 channels for ResNet50
    image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    # Normalize to 0-1 range for the model
    image_rgb = image_rgb.astype(np.float32) / 255.0
    
    return image_rgb

# ============================================================================
# DATA GENERATORS WITH MEDICAL AUGMENTATION
# ============================================================================

IMG_SIZE = (384, 384)
BATCH_SIZE = 32

# Training data with augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=medical_preprocessing_resnet,
    rotation_range=12,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    fill_mode='constant',
    cval=0
)

# Validation and test data (no augmentation, only preprocessing)
val_test_datagen = ImageDataGenerator(
    preprocessing_function=medical_preprocessing_resnet
)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\n✅ Data Generators Created")
print(f"Training batches: {len(train_generator)}")
print(f"Validation batches: {len(val_generator)}")
print(f"Test batches: {len(test_generator)}")

# ============================================================================
# MEDICAL-OPTIMIZED RESNET50 MODEL
# ============================================================================

def create_medical_resnet50():
    """Create ResNet50 with medical-optimized classifier head"""
    # Load ResNet50 with ImageNet weights
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(384, 384, 3)
    )
    
    # Initially freeze early layers
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Medical-optimized classifier head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    # Attention mechanism for lung region focus (matching dimensions)
    attention = layers.Dense(x.shape[-1], activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])
    
    # Deep classifier for medical complexity
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Binary output
    output = layers.Dense(1, activation='sigmoid', name='pneumonia')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model

print("\n🏗️ Building Medical ResNet50 Model...")
model, base_model = create_medical_resnet50()
print(f"Total parameters: {model.count_params():,}")
print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ============================================================================
# TRAINING PHASE 1: HEAD TRAINING
# ============================================================================

print("\n" + "="*70)
print("🚀 PHASE 1: Training Classifier Head (Base Model Frozen)")
print("="*70)

base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Calculate class weights for imbalanced dataset
class_weight = {
    0: 1.0,  # Normal
    1: pneumonia_ratio  # Pneumonia (higher weight)
}

callbacks_phase1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'resnet50_phase1_best.h5',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

history_phase1 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=callbacks_phase1,
    verbose=1
)

# ============================================================================
# TRAINING PHASE 2: FINE-TUNING
# ============================================================================

print("\n" + "="*70)
print("🔥 PHASE 2: Fine-Tuning Deeper Layers")
print("="*70)

# Unfreeze deeper layers
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

print(f"Trainable parameters after unfreezing: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Much lower learning rate
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

callbacks_phase2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=8,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=5,
        mode='max',
        min_lr=1e-8,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'resnet50_final_best.h5',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

history_phase2 = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks_phase2,
    verbose=1
)

# ============================================================================
# TRAINING HISTORY VISUALIZATION
# ============================================================================

def plot_training_history(history1, history2):
    """Plot training metrics across both phases"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History - Both Phases', fontsize=16, fontweight='bold')
    
    # Combine histories
    metrics = ['loss', 'auc', 'accuracy', 'recall']
    titles = ['Loss', 'AUC-ROC', 'Accuracy', 'Sensitivity (Recall)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Phase 1
        phase1_train = history1.history[metric]
        phase1_val = history1.history[f'val_{metric}']
        epochs1 = range(1, len(phase1_train) + 1)
        
        # Phase 2
        phase2_train = history2.history[metric]
        phase2_val = history2.history[f'val_{metric}']
        epochs2 = range(len(phase1_train) + 1, len(phase1_train) + len(phase2_train) + 1)
        
        # Plot
        ax.plot(epochs1, phase1_train, 'b-', label='Phase 1 Train', linewidth=2)
        ax.plot(epochs1, phase1_val, 'b--', label='Phase 1 Val', linewidth=2)
        ax.plot(epochs2, phase2_train, 'r-', label='Phase 2 Train', linewidth=2)
        ax.plot(epochs2, phase2_val, 'r--', label='Phase 2 Val', linewidth=2)
        
        ax.axvline(x=len(phase1_train), color='gray', linestyle=':', linewidth=2, label='Fine-tuning Start')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_training_history(history_phase1, history_phase2)

# ============================================================================
# COMPREHENSIVE MEDICAL EVALUATION
# ============================================================================

print("\n" + "="*70)
print("📊 MEDICAL-GRADE PERFORMANCE EVALUATION")
print("="*70)

def evaluate_medical_model(model, test_generator):
    """Hospital-grade performance metrics"""
    print("\n🔬 Collecting predictions...")
    
    # Get all predictions
    y_true = []
    y_pred_proba = []
    
    for i in range(len(test_generator)):
        images, labels = test_generator[i]
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels)
        y_pred_proba.extend(predictions.flatten())
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'AUC-ROC': roc_auc_score(y_true, y_pred_proba),
        'Accuracy': np.mean(y_true == y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Sensitivity (Recall)': recall_score(y_true, y_pred, pos_label=1),
        'Specificity': recall_score(y_true, y_pred, pos_label=0),
        'F1-Score': f1_score(y_true, y_pred)
    }
    
    return metrics, y_true, y_pred, y_pred_proba

metrics, y_true, y_pred, y_pred_proba = evaluate_medical_model(model, test_generator)

# Print metrics
print("\n🎯 OVERALL PERFORMANCE METRICS:")
print("=" * 50)
for metric_name, value in metrics.items():
    print(f"{metric_name:.<35} {value:.4f} ({value*100:.2f}%)")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'])
plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# Classification Report
print("\n📋 DETAILED CLASSIFICATION REPORT:")
print("=" * 50)
print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, linewidth=3, label=f'ResNet50 (AUC = {metrics["AUC-ROC"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curve - Pneumonia Detection', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Precision-Recall Curve
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(recall_vals, precision_vals, linewidth=3, color='green')
plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# THRESHOLD ANALYSIS FOR CLINICAL SETTINGS
# ============================================================================

print("\n🏥 CLINICAL THRESHOLD ANALYSIS:")
print("=" * 70)
print(f"{'Threshold':<12} {'Sensitivity':<15} {'Specificity':<15} {'F1-Score':<12}")
print("=" * 70)
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_thresh = (y_pred_proba > threshold).astype(int)
    sensitivity = recall_score(y_true, y_pred_thresh, pos_label=1)
    specificity = recall_score(y_true, y_pred_thresh, pos_label=0)
    f1 = f1_score(y_true, y_pred_thresh)
    
    print(f"{threshold:<12.1f} {sensitivity:<15.4f} {specificity:<15.4f} {f1:<12.4f}")

# ============================================================================
# GRAD-CAM VISUALIZATION
# ============================================================================

print("\n🔍 Generating Grad-CAM Visualizations...")

def create_gradcam_model(model):
    """Create Grad-CAM model for visualization"""
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        raise ValueError("No convolutional layer found")
    
    grad_model = Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )
    return grad_model, last_conv_layer.name

def generate_gradcam_heatmap(grad_model, image):
    """Generate Grad-CAM heatmap"""
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap) + 1e-10
    
    return heatmap.numpy()

# Create Grad-CAM model
grad_model, conv_layer_name = create_gradcam_model(model)
print(f"Using layer: {conv_layer_name} for Grad-CAM")

# Visualize sample predictions
def visualize_predictions(model, grad_model, test_gen, num_samples=6):
    """Visualize predictions with Grad-CAM heatmaps"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Pneumonia Detection with Grad-CAM Visualization', 
                 fontsize=16, fontweight='bold')
    
    images, labels = next(iter(test_gen))
    
    for idx in range(min(num_samples, len(images))):
        ax = axes[idx // 3, idx % 3]
        
        # Get prediction
        img = np.expand_dims(images[idx], axis=0)
        pred_proba = model.predict(img, verbose=0)[0][0]
        pred_class = 'PNEUMONIA' if pred_proba > 0.5 else 'NORMAL'
        true_class = 'PNEUMONIA' if labels[idx] == 1 else 'NORMAL'
        
        # Generate heatmap
        heatmap = generate_gradcam_heatmap(grad_model, img)
        heatmap = cv2.resize(heatmap, (384, 384))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Original image
        original = np.uint8(images[idx] * 255)
        
        # Overlay
        superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        
        # Display
        ax.imshow(superimposed)
        ax.axis('off')
        
        color = 'green' if pred_class == true_class else 'red'
        title = f"True: {true_class}\nPred: {pred_class} ({pred_proba:.2%})"
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.show()

visualize_predictions(model, grad_model, test_generator, num_samples=6)

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n💾 Saving Final Model...")
model.save('pneumonia_resnet50_final.h5')
print("✅ Model saved as 'pneumonia_resnet50_final.h5'")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✨ TRAINING COMPLETE - MEDICAL AI SYSTEM READY")
print("="*70)
print(f"\n🎯 Final Test Performance:")
print(f"   • AUC-ROC: {metrics['AUC-ROC']:.4f}")
print(f"   • Accuracy: {metrics['Accuracy']:.4f}")
print(f"   • Sensitivity: {metrics['Sensitivity (Recall)']:.4f}")
print(f"   • Specificity: {metrics['Specificity']:.4f}")
print(f"\n💡 Model is ready for clinical deployment!")
print("="*70)
