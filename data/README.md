<div style="font-size:2em; font-weight:bold; text-align:center; margin-top:20px;">Data Directory</div>

## Table of Contents
<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1-overview"><i><b>1. Overview</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#2-directory-structure"><i><b>2. Directory Structure</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#3-data-processing-pipeline"><i><b>3. Data Processing Pipeline</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#4-dataset-details"><i><b>4. Dataset Details</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#5-data-usage-guidelines"><i><b>5. Data Usage Guidelines</b></i></a></div>
&nbsp;

## 1. Overview

This directory contains the MNIST dataset and its processed versions for training, evaluating, and testing CNN models for handwritten digit classification. The data is organized in a structured pipeline from raw data to processed formats optimized for model training and inference.

## 2. Directory Structure

```
Folder PATH listing
.
+---mnist                      <-- Processed MNIST dataset
|   +---x_test.npy             <-- Test images (10,000 samples)
|   +---x_train.npy            <-- Training images (54,000 samples)
|   +---x_val.npy              <-- Validation images (6,000 samples)
|   +---y_test.npy             <-- Test labels (one-hot encoded)
|   +---y_train.npy            <-- Training labels (one-hot encoded)
|   +---y_val.npy              <-- Validation labels (one-hot encoded)
|
+---mnist_samples              <-- Sample digit images extracted from dataset
|   +---digit_0_sample_0.png   <-- Sample of digit 0
|   +---digit_0_sample_1.png   <-- Sample of digit 0
|   +---digit_0_sample_2.png   <-- Sample of digit 0
|   +---digit_0_sample_3.png   <-- Sample of digit 0
|   +---digit_0_sample_4.png   <-- Sample of digit 0
|   +---digit_1_sample_0.png   <-- Sample of digit 1
|   +---digit_1_sample_1.png   <-- Sample of digit 1
|   +---...                    <-- More sample images
|
+---README.md                  <-- This documentation file
```

## 3. Data Processing Pipeline

### 3.1 Data Acquisition

The MNIST dataset is downloaded automatically from the official source using the `download_mnist.py` script in the `scripts` directory:

```bash
python scripts/download_mnist.py --output_dir data/raw
```

### 3.2 Preprocessing Steps

The raw MNIST data undergoes the following preprocessing steps:

1. **Reading binary files**: Convert binary MNIST format to numpy arrays
2. **Normalization**: Scale pixel values from [0, 255] to [0, 1]
3. **Reshaping**: Reshape images to (28, 28, 1) format for CNN input
4. **Train/Val/Test Split**: Split data into training (80%), validation (10%), and testing (10%) sets
5. **One-hot encoding**: Convert class labels to one-hot encoded vectors

The preprocessing can be executed using:

```bash
python scripts/preprocess_data.py --input_dir data/raw --output_dir data/processed
```

### 3.3 Data Augmentation

Data augmentation increases the diversity of training data by applying various transformations:

1. **Rotation**: Random rotation within ±10 degrees
2. **Shift**: Random horizontal and vertical shifts up to 10% of image dimensions
3. **Zoom**: Random zoom between 90-110% of original size
4. **Shear**: Random shearing transformations

Augmentation can be applied using:

```bash
python scripts/augment_data.py --input_dir data/processed --output_dir data/augmented
```

### 3.4 Metadata Generation

Statistical information about the dataset is extracted and stored as JSON files:

```bash
python scripts/generate_metadata.py --input_dir data/processed --output_dir data/metadata
```

## 4. Dataset Details

### 4.1 MNIST Dataset

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9):

- 60,000 training examples
- 10,000 test examples
- Image dimensions: 28x28 pixels
- Grayscale (1 channel)
- 10 classes (digits 0-9)

### 4.2 Data Format

After preprocessing, the data is stored in numpy (.npy) format:

- **Images**: Float32 arrays with shape (n_samples, 28, 28, 1) and values normalized to [0, 1]
- **Labels**: One-hot encoded vectors with shape (n_samples, 10)

Example of accessing the data:

```python
import numpy as np

# Load preprocessed data
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')

# Check shapes
print(f"X_train shape: {X_train.shape}")  # (48000, 28, 28, 1)
print(f"y_train shape: {y_train.shape}")  # (48000, 10)

# Access a single example
sample_image = X_train[0]
sample_label = y_train[0]
sample_digit = np.argmax(sample_label)
```

### 4.3 Class Distribution

The MNIST dataset has a relatively balanced distribution of classes:

| Digit | Training Count | Test Count |
|-------|---------------|------------|
| 0     | 5,923         | 980        |
| 1     | 6,742         | 1,135      |
| 2     | 5,958         | 1,032      |
| 3     | 6,131         | 1,010      |
| 4     | 5,842         | 982        |
| 5     | 5,421         | 892        |
| 6     | 5,918         | 958        |
| 7     | 6,265         | 1,028      |
| 8     | 5,851         | 974        |
| 9     | 5,949         | 1,009      |

### 4.4 Dataset Statistics

Statistical properties of the processed dataset:

- **Mean pixel value**: 0.1307
- **Standard deviation**: 0.3081
- **Min value**: 0.0
- **Max value**: 1.0

## 5. Data Usage Guidelines

### 5.1 Training/Validation/Test Split

Always maintain the separation between training, validation, and test sets:

- Use **training data** (`X_train.npy`, `y_train.npy`) for model training
- Use **validation data** (`X_val.npy`, `y_val.npy`) for hyperparameter tuning
- Only use **test data** (`X_test.npy`, `y_test.npy`) for final model evaluation

### 5.2 Data Loading in Training Scripts

The recommended way to load data in training scripts:

```python
def load_data(data_dir="data/processed"):
    """Load preprocessed MNIST data."""
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
```

### 5.3 Data Augmentation Usage

When using data augmentation, consider:

1. Only apply augmentation to the training set, never to validation or test sets
2. Use the TensorFlow/Keras data generator for on-the-fly augmentation:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data generator with augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)

# Fit the generator on the training data
datagen.fit(X_train)

# Use the generator in training
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10
)
```

### 5.4 Custom Dataset Creation

To create custom variants of the dataset:

1. Start with the raw data in `data/raw/`
2. Modify the preprocessing script or create a new one
3. Save the custom dataset in a new subdirectory (e.g., `data/processed_custom/`)
4. Document the preprocessing changes in the metadata

### 5.5 Data Version Control

When making changes to the dataset:

1. Document all changes in `data/metadata/preprocessing_params.json`
2. Include timestamp and version information
3. For significant changes, create a new subdirectory with a version suffix
4. Update model training scripts to point to the correct data version