<div style="font-size:2em; font-weight:bold; text-align:center; margin-top:20px;">Utils Directory</div>

## Table of Contents
<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1-overview"><i><b>1. Overview</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#2-directory-structure"><i><b>2. Directory Structure</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#3-data-utilities"><i><b>3. Data Utilities</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#4-visualization-utilities"><i><b>4. Visualization Utilities</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#5-model-utilities"><i><b>5. Model Utilities</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#6-performance-utilities"><i><b>6. Performance Utilities</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#7-deployment-utilities"><i><b>7. Deployment Utilities</b></i></a></div>
&nbsp;

## 1. Overview

The `utils` directory contains reusable utility functions and helper classes that support various aspects of the CNN-MNIST project. These utilities handle data processing, visualization, model evaluation, performance metrics, and deployment assistance, providing a clean separation of concerns from the main application code.

## 2. Directory Structure

```
Folder PATH listing
.
+---data                       <-- Data handling utilities
|   +---__init__.py            <-- Package initialization
|   +---augmentation.py        <-- Data augmentation functions
|   +---loader.py              <-- Data loading functions
|   +---preprocessing.py       <-- Data preprocessing functions
|
+---deployment                 <-- Deployment utilities
|   +---__init__.py            <-- Package initialization
|   +---model_export.py        <-- Model export functions
|   +---optimizations.py       <-- Model optimization utilities
|   +---serving.py             <-- Model serving utilities
|
+---model                      <-- Model utilities
|   +---__init__.py            <-- Package initialization
|   +---callbacks.py           <-- Custom callback implementations
|   +---checkpoints.py         <-- Checkpoint management functions
|   +---metrics.py             <-- Custom metrics implementations
|
+---performance                <-- Performance utilities
|   +---__init__.py            <-- Package initialization
|   +---metrics_eval.py        <-- Performance metrics calculation
|   +---profiling.py           <-- Model profiling utilities
|
+---visualization              <-- Visualization utilities
|   +---__init__.py            <-- Package initialization
|   +---confusion_matrix.py    <-- Confusion matrix visualization
|   +---gradcam.py             <-- Grad-CAM visualization utilities
|   +---plotting.py            <-- General plotting functions
|
+---__init__.py                <-- Package initialization
+---README.md                  <-- This documentation file
+---toc_generator.py           <-- Table of contents generator for markdown files
```

## 3. Data Utilities

The `utils/data` module provides functions for working with the MNIST dataset and general data processing tasks.

### 3.1 Data Loading (`loader.py`)

Functions for loading the MNIST dataset and managing data pipeline:

```python
def load_mnist_data(data_dir="data/processed"):
    """
    Load preprocessed MNIST data from numpy files.
    
    Args:
        data_dir (str): Directory containing processed data files
        
    Returns:
        tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_tf_dataset(images, labels, batch_size=32, shuffle=True, augment=False):
    """
    Create a TensorFlow dataset from numpy arrays.
    
    Args:
        images (np.ndarray): Image data
        labels (np.ndarray): Label data
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        augment (bool): Whether to apply data augmentation
        
    Returns:
        tf.data.Dataset: TensorFlow dataset
    """
    # Implementation...
```

### 3.2 Preprocessing (`preprocessing.py`)

Functions for preprocessing raw MNIST data:

```python
def normalize_images(images, method='divide'):
    """
    Normalize image pixel values.
    
    Args:
        images (np.ndarray): Input images
        method (str): Normalization method ('divide', 'standardize')
        
    Returns:
        np.ndarray: Normalized images
    """
    if method == 'divide':
        # Simple division by 255
        return images.astype('float32') / 255.0
    elif method == 'standardize':
        # Standardize to mean=0, std=1
        mean = np.mean(images)
        std = np.std(images)
        return (images.astype('float32') - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def reshape_for_cnn(images):
    """
    Reshape images for CNN input (adds channel dimension).
    
    Args:
        images (np.ndarray): Input images with shape (n, 28, 28)
        
    Returns:
        np.ndarray: Reshaped images with shape (n, 28, 28, 1)
    """
    # Implementation...


def one_hot_encode(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoded vectors.
    
    Args:
        labels (np.ndarray): Integer labels
        num_classes (int): Number of classes
        
    Returns:
        np.ndarray: One-hot encoded labels
    """
    # Implementation...
```

### 3.3 Augmentation (`augmentation.py`)

Functions for data augmentation:

```python
def create_augmentation_generator(rotation_range=10, width_shift_range=0.1,
                                 height_shift_range=0.1, zoom_range=0.1):
    """
    Create an ImageDataGenerator for data augmentation.
    
    Args:
        rotation_range (int): Rotation range in degrees
        width_shift_range (float): Width shift range as fraction of width
        height_shift_range (float): Height shift range as fraction of height
        zoom_range (float): Zoom range as fraction
        
    Returns:
        ImageDataGenerator: Configured data generator
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        fill_mode='nearest'
    )


def apply_augmentation(images, labels, generator, samples_multiplier=2):
    """
    Apply augmentation to create additional samples.
    
    Args:
        images (np.ndarray): Original images
        labels (np.ndarray): Original labels
        generator (ImageDataGenerator): Configured generator
        samples_multiplier (int): Number of times to multiply the dataset
        
    Returns:
        tuple: (augmented_images, augmented_labels)
    """
    # Implementation...
```

## 4. Visualization Utilities

The `utils/visualization` module provides functions for visualizing data, model performance, and model interpretability.

### 4.1 General Plotting (`plotting.py`)

Functions for general plotting tasks:

```python
def plot_sample_images(images, labels, num_samples=10, predictions=None):
    """
    Plot sample images with their labels.
    
    Args:
        images (np.ndarray): Images to plot
        labels (np.ndarray): True labels (one-hot encoded or integers)
        num_samples (int): Number of samples to plot
        predictions (np.ndarray, optional): Model predictions
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert one-hot encoded labels to integers if needed
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    
    # Select random samples
    indices = np.random.choice(range(len(images)), num_samples, replace=False)
    
    # Create the plot
    fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
    
    for i, idx in enumerate(indices):
        # Reshape image if needed
        img = images[idx]
        if len(img.shape) > 2 and img.shape[2] == 1:
            img = img.reshape(img.shape[0], img.shape[1])
        
        # Plot image
        axes[i].imshow(img, cmap='gray')
        title = f"Label: {labels[idx]}"
        
        if predictions is not None:
            pred = np.argmax(predictions[idx]) if len(predictions[idx].shape) > 0 else predictions[idx]
            title += f"\nPred: {pred}"
            
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """
    Plot training history metrics.
    
    Args:
        history: Keras training history object
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Implementation...
```

### 4.2 Confusion Matrix (`confusion_matrix.py`)

Functions for confusion matrix visualization:

```python
def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        y_true (np.ndarray): True labels (integers)
        y_pred (np.ndarray): Predicted labels (integers)
        class_names (list, optional): List of class names
        normalize (bool): Whether to normalize the confusion matrix
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert one-hot encoded labels to integers if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    
    # Add class labels
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    
    return fig
```

### 4.3 Grad-CAM Visualization (`gradcam.py`)

Functions for model interpretability using Grad-CAM:

```python
def apply_gradcam(model, image, layer_name=None, pred_index=None):
    """
    Apply Grad-CAM to visualize model attention.
    
    Args:
        model: Keras model
        image (np.ndarray): Input image (should be preprocessed)
        layer_name (str, optional): Target layer name (uses last conv if None)
        pred_index (int, optional): Index of the class to explain
        
    Returns:
        tuple: (heatmap, overlaid_image)
    """
    import tensorflow as tf
    import numpy as np
    import cv2
    
    # Implementation...
```

## 5. Model Utilities

The `utils/model` module provides functions for model management, custom metrics, and callbacks.

### 5.1 Checkpoint Management (`checkpoints.py`)

Functions for managing model checkpoints:

```python
def create_checkpoint_callback(checkpoint_dir, monitor='val_accuracy', 
                              save_best_only=True, save_weights_only=False):
    """
    Create a ModelCheckpoint callback.
    
    Args:
        checkpoint_dir (str): Directory to save checkpoints
        monitor (str): Metric to monitor
        save_best_only (bool): Whether to save only the best model
        save_weights_only (bool): Whether to save only weights
        
    Returns:
        tf.keras.callbacks.ModelCheckpoint: Checkpoint callback
    """
    import os
    import tensorflow as tf
    
    # Ensure directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create filepath pattern
    filepath = os.path.join(
        checkpoint_dir, 
        'model_{epoch:02d}_{' + monitor + ':.4f}.h5'
    )
    
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        verbose=1
    )


def load_latest_checkpoint(checkpoint_dir, custom_objects=None):
    """
    Load the latest checkpoint from the given directory.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        custom_objects (dict, optional): Dictionary mapping names to custom classes/functions
        
    Returns:
        tf.keras.Model: Loaded model or None if no checkpoint found
    """
    # Implementation...
```

### 5.2 Custom Metrics (`metrics.py`)

Custom metric implementations:

```python
def f1_score(y_true, y_pred):
    """
    F1 score metric implementation for Keras.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        tf.Tensor: F1 score
    """
    import tensorflow as tf
    import keras.backend as K
    
    # Implementation...


def precision_at_k(k=5):
    """
    Precision at K metric for Keras.
    
    Args:
        k (int): Number of top predictions to consider
        
    Returns:
        function: Metric function
    """
    # Implementation...
```

### 5.3 Custom Callbacks (`callbacks.py`)

Custom callback implementations:

```python
class LearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Custom learning rate scheduler with warmup.
    
    Args:
        initial_lr (float): Initial learning rate
        warmup_epochs (int): Number of warmup epochs
        decay_factor (float): Factor to decay learning rate
        decay_epochs (int): Epoch interval for decay
    """
    # Implementation...


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    """
    Callback to generate confusion matrix after each epoch.
    
    Args:
        validation_data (tuple): Validation data (X, y)
        log_dir (str): Directory to save confusion matrix plots
    """
    # Implementation...
```

## 6. Performance Utilities

The `utils/performance` module provides functions for evaluating and profiling model performance.

### 6.1 Metrics Evaluation (`metrics_eval.py`)

Functions for calculating performance metrics:

```python
def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities or labels
        
    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.metrics import f1_score, roc_auc_score
    import numpy as np
    
    # Convert one-hot encoded to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true
        
    # Handle different prediction formats
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        # We have probability predictions
        y_pred_proba = y_pred
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        # We have class predictions
        y_pred_labels = y_pred
        y_pred_proba = None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true_labels, y_pred_labels),
        'precision_micro': precision_score(y_true_labels, y_pred_labels, average='micro'),
        'precision_macro': precision_score(y_true_labels, y_pred_labels, average='macro'),
        'recall_micro': recall_score(y_true_labels, y_pred_labels, average='micro'),
        'recall_macro': recall_score(y_true_labels, y_pred_labels, average='macro'),
        'f1_micro': f1_score(y_true_labels, y_pred_labels, average='micro'),
        'f1_macro': f1_score(y_true_labels, y_pred_labels, average='macro'),
    }
    
    # Add ROC AUC if we have probability predictions
    if y_pred_proba is not None:
        metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    
    return metrics


def per_class_metrics(y_true, y_pred, class_names=None):
    """
    Calculate per-class performance metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (list, optional): List of class names
        
    Returns:
        pd.DataFrame: DataFrame with per-class metrics
    """
    # Implementation...
```

### 6.2 Profiling (`profiling.py`)

Functions for profiling model performance:

```python
def profile_model_inference(model, input_data, batch_size=32, num_runs=100):
    """
    Profile model inference performance.
    
    Args:
        model: Keras model
        input_data (np.ndarray): Input data
        batch_size (int): Batch size for inference
        num_runs (int): Number of runs to average
        
    Returns:
        dict: Dictionary with profiling results
    """
    import time
    import numpy as np
    
    # Warm-up run
    _ = model.predict(input_data[:batch_size])
    
    # Batch prediction timing
    batch_times = []
    for _ in range(num_runs):
        batch = input_data[:batch_size]
        start_time = time.time()
        _ = model.predict(batch)
        end_time = time.time()
        batch_times.append(end_time - start_time)
    
    # Single sample prediction timing
    single_times = []
    for _ in range(num_runs):
        sample = np.expand_dims(input_data[0], axis=0)
        start_time = time.time()
        _ = model.predict(sample)
        end_time = time.time()
        single_times.append(end_time - start_time)
    
    results = {
        'batch_size': batch_size,
        'avg_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'avg_batch_fps': batch_size / np.mean(batch_times),
        'avg_single_time': np.mean(single_times),
        'std_single_time': np.std(single_times),
        'avg_single_fps': 1 / np.mean(single_times)
    }
    
    return results


def calculate_model_size(model):
    """
    Calculate model size in terms of parameters and memory.
    
    Args:
        model: Keras model
        
    Returns:
        dict: Dictionary with model size information
    """
    # Implementation...
```

## 7. Deployment Utilities

The `utils/deployment` module provides utilities for model export, optimization, and serving.

### 7.1 Model Export (`model_export.py`)

Functions for exporting models to different formats:

```python
def export_saved_model(model, export_dir):
    """
    Export model to SavedModel format.
    
    Args:
        model: Keras model
        export_dir (str): Directory to save the model
        
    Returns:
        str: Path to the exported model
    """
    import os
    import tensorflow as tf
    
    # Create directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    
    # Define export path
    saved_model_path = os.path.join(export_dir, 'saved_model')
    
    # Export model
    tf.saved_model.save(model, saved_model_path)
    
    return saved_model_path


def export_tflite_model(model, export_path, quantize=False):
    """
    Export model to TensorFlow Lite format.
    
    Args:
        model: Keras model
        export_path (str): Path to save the TFLite model
        quantize (bool): Whether to apply quantization
        
    Returns:
        str: Path to the exported model
    """
    import os
    import tensorflow as tf
    
    # Create directory if needed
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    # Convert model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Save model
    with open(export_path, 'wb') as f:
        f.write(tflite_model)
    
    return export_path


def export_onnx_model(model, export_path, input_shape=None):
    """
    Export model to ONNX format.
    
    Args:
        model: Keras model
        export_path (str): Path to save the ONNX model
        input_shape (tuple, optional): Input shape for the model
        
    Returns:
        str: Path to the exported model
    """
    # Implementation...
```

### 7.2 Model Optimization (`optimizations.py`)

Functions for optimizing models for deployment:

```python
def quantize_model(model, representative_dataset=None):
    """
    Apply post-training quantization to a model.
    
    Args:
        model: Keras model
        representative_dataset: Function that returns representative dataset
        
    Returns:
        bytes: Quantized TFLite model
    """
    # Implementation...


def prune_model(model, pruning_params):
    """
    Apply weight pruning to a model.
    
    Args:
        model: Keras model
        pruning_params (dict): Pruning parameters
        
    Returns:
        tf.keras.Model: Pruned model
    """
    # Implementation...
```

### 7.3 Model Serving (`serving.py`)

Utilities for model serving:

```python
def create_serving_function(model):
    """
    Create a serving function for the model.
    
    Args:
        model: Keras model
        
    Returns:
        function: Serving function that takes raw input and returns prediction
    """
    # Implementation...


def create_prediction_api(model, app=None):
    """
    Create a Flask API for model serving.
    
    Args:
        model: Keras model
        app (Flask, optional): Flask app to add routes to
        
    Returns:
        Flask: Flask application
    """
    # Implementation...
``` 