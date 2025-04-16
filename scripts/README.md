<div style="font-size:2em; font-weight:bold; text-align:center; margin-top:20px;">Scripts Directory</div>

## Table of Contents
<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#1-overview"><i><b>1. Overview</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#2-scripts-inventory"><i><b>2. Scripts Inventory</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#3-common-usage-patterns"><i><b>3. Common Usage Patterns</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#4-development-guidelines"><i><b>4. Development Guidelines</b></i></a>
</div>
&nbsp;

## 1. Overview

This directory contains Python scripts for data processing, model training, evaluation, and deployment in our CNN-based MNIST classification project. These scripts automate common tasks, ensure reproducibility, and provide utilities for working with the MNIST dataset and TensorFlow/Keras models.

## 2. Scripts Inventory

```
Folder PATH listing
.
+---data_prep.py               <-- Download and preprocess MNIST data
+---extract_sample_images.py   <-- Extract sample images for visualization
+---README.md                  <-- This documentation file
+---train_cnn.py               <-- Train the CNN model
+---visualize_features.py      <-- Generate feature map visualizations
```

### 2.1 Data Processing Scripts

#### `download_data.py`
Downloads the MNIST dataset from the official source and saves it to the `data/raw` directory.

```bash
python scripts/download_data.py --output_dir data/raw
```

#### `preprocess_data.py`
Preprocesses raw MNIST data by normalizing, reshaping, and splitting into training, validation, and test sets.

```bash
python scripts/preprocess_data.py --raw_dir data/raw --processed_dir data/processed --val_split 0.1
```

#### `augment_data.py`
Creates augmented versions of the training data to improve model generalization.

```bash
python scripts/augment_data.py --input_dir data/processed --output_dir data/augmented --augmentation_factor 2
```

#### `utils/data_helpers.py`
Contains utility functions for data loading, transformation, and visualization.

```python
from scripts.utils.data_helpers import load_mnist_data, visualize_samples
```

### 2.2 Model Training Scripts

#### `train_model.py`
Trains a CNN model on the MNIST dataset with specified hyperparameters.

```bash
python scripts/train_model.py --model_type basic_cnn --data_dir data/processed --model_dir models/basic_cnn --epochs 10 --batch_size 64
```

#### `hyperparameter_search.py`
Performs hyperparameter optimization using grid or random search strategies.

```bash
python scripts/hyperparameter_search.py --model_type advanced_cnn --search_method random --trials 20 --output_dir models/hyperparameter_results
```

#### `train_distributed.py`
Trains models using distributed training across multiple GPUs or machines.

```bash
python scripts/train_distributed.py --num_gpus 2 --model_type mobilenet --data_dir data/processed --model_dir models/mobilenet_distributed
```

### 2.3 Evaluation Scripts

#### `evaluate_model.py`
Evaluates a trained model on the test set and generates performance metrics.

```bash
python scripts/evaluate_model.py --model_path models/basic_cnn/final_model.h5 --data_dir data/processed --output_dir results/basic_cnn
```

#### `visualize_results.py`
Generates visualizations for model predictions, confusion matrices, and performance metrics.

```bash
python scripts/visualize_results.py --results_dir results/basic_cnn --output_dir visualizations/basic_cnn
```

#### `benchmark_models.py`
Benchmarks multiple models and compares their performance metrics.

```bash
python scripts/benchmark_models.py --model_dirs models/basic_cnn,models/advanced_cnn,models/mobilenet --output_dir benchmarks
```

### 2.4 Model Export and Deployment Scripts

#### `export_model.py`
Exports trained models to various formats (TensorFlow SavedModel, TFLite, ONNX).

```bash
python scripts/export_model.py --model_path models/basic_cnn/final_model.h5 --export_dir exports --formats savedmodel,tflite,onnx
```

#### `optimize_model.py`
Applies optimization techniques to models for deployment (quantization, pruning).

```bash
python scripts/optimize_model.py --model_path models/basic_cnn/final_model.h5 --output_dir models/optimized --technique quantization
```

#### `deploy_model.py`
Prepares a model for deployment to target platforms (TF Serving, TF.js, edge devices).

```bash
python scripts/deploy_model.py --model_path exports/basic_cnn.tflite --target edge --output_dir deployment
```

### 2.5 Inference Scripts

#### `predict.py`
Makes predictions on new data using a trained model.

```bash
python scripts/predict.py --model_path models/basic_cnn/final_model.h5 --image_path data/samples/sample_digits/7/sample_7_01.png
```

#### `predict_batch.py`
Processes batches of images for inference.

```bash
python scripts/predict_batch.py --model_path models/basic_cnn/final_model.h5 --image_dir data/samples/difficult_samples --output_file predictions.csv
```

### 2.6 Monitoring and Logging Scripts

#### `monitor_training.py`
Monitors and logs model training metrics in real-time.

```bash
python scripts/monitor_training.py --log_dir logs/training --port 8080
```

#### `utils/logging_config.py`
Provides configuration for consistent logging across scripts.

```python
from scripts.utils.logging_config import setup_logger
logger = setup_logger(__name__)
```

## 3. Common Usage Patterns

### 3.1 Complete Training Pipeline

The following shows a typical sequence for a complete training pipeline:

```bash
# 1. Download the data
python scripts/download_data.py --output_dir data/raw

# 2. Preprocess the data
python scripts/preprocess_data.py --raw_dir data/raw --processed_dir data/processed

# 3. Create augmented training data
python scripts/augment_data.py --input_dir data/processed --output_dir data/augmented

# 4. Train the model
python scripts/train_model.py --model_type advanced_cnn --data_dir data/processed --augmented_dir data/augmented --model_dir models/advanced_cnn

# 5. Evaluate the model
python scripts/evaluate_model.py --model_path models/advanced_cnn/final_model.h5 --data_dir data/processed --output_dir results/advanced_cnn

# 6. Export the model for deployment
python scripts/export_model.py --model_path models/advanced_cnn/final_model.h5 --export_dir exports --formats tflite
```

### 3.2 Hyperparameter Optimization

```bash
# 1. Perform hyperparameter search
python scripts/hyperparameter_search.py --model_type basic_cnn --search_method random --trials 20 --output_dir models/hyperparameter_results

# 2. Train with best parameters
python scripts/train_model.py --model_type basic_cnn --config models/hyperparameter_results/best_params.json --data_dir data/processed --model_dir models/basic_cnn_optimized
```

### 3.3 Model Comparison and Benchmarking

```bash
# 1. Train multiple models
python scripts/train_model.py --model_type basic_cnn --data_dir data/processed --model_dir models/basic_cnn
python scripts/train_model.py --model_type advanced_cnn --data_dir data/processed --model_dir models/advanced_cnn
python scripts/train_model.py --model_type mobilenet --data_dir data/processed --model_dir models/mobilenet

# 2. Benchmark the models
python scripts/benchmark_models.py --model_dirs models/basic_cnn,models/advanced_cnn,models/mobilenet --output_dir benchmarks

# 3. Visualize comparative results
python scripts/visualize_results.py --results_dir benchmarks --output_dir visualizations/model_comparison --comparison
```

## 4. Development Guidelines

### 4.1 Script Structure

All scripts should follow this basic structure:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script description and purpose.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import project modules
from scripts.utils.logging_config import setup_logger

# Setup logging
logger = setup_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Script description')
    # Add arguments
    parser.add_argument('--required_arg', type=str, required=True, help='Required argument')
    parser.add_argument('--optional_arg', type=int, default=10, help='Optional argument with default')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    logger.info(f"Starting script with args: {args}")
    
    # Script logic here
    
    logger.info("Script completed successfully")

if __name__ == "__main__":
    main()
```

### 4.2 Coding Standards

1. **Documentation**: All scripts must have docstrings and clear argument help text.
2. **Error Handling**: Use try-except blocks for robust error handling.
3. **Logging**: Use the project's logging utilities rather than print statements.
4. **Configuration**: Use command-line arguments or config files, not hardcoded values.
5. **Testing**: Include unit tests for critical functions in the `tests` directory.

### 4.3 Adding New Scripts

When adding new scripts:

1. Follow the established naming conventions
2. Include comprehensive documentation
3. Update this README.md file
4. Ensure compatibility with existing scripts
5. Add example usage commands

### 4.4 Dependencies

Scripts should declare their dependencies properly. If a script requires packages beyond the core project dependencies, add them to a requirements section in the script's docstring.

### 4.5 Performance Considerations

- Use vectorized operations when possible (NumPy/TensorFlow)
- For large datasets, implement batch processing
- Include progress indicators for long-running operations
- Add timestamps to logs for performance analysis
