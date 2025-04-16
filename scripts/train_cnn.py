#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train CNN models for MNIST digit classification.

This script provides functionality to train different CNN architectures
on the MNIST dataset, with options for model selection, hyperparameters,
and training configuration.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Add parent directory to path to import from models
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_registry import ModelRegistry
from models.model_factory import save_model
from models.training.trainer import Trainer
from models.training.callbacks import ConfusionMatrixCallback, FeatureMapVisualizer
from models.evaluation.evaluator import Evaluator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CNN models for MNIST digit classification')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/mnist',
                        help='Directory containing MNIST data')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='simple_cnn',
                        choices=ModelRegistry.list_models(),
                        help=f'Type of model to train. Available options: {ModelRegistry.list_models()}')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save trained models')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train for (default: use model config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (default: use model config)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of training data to use for validation')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    
    # Visualization arguments
    parser.add_argument('--fig_dir', type=str, default='figures',
                        help='Directory to save figures')
    
    return parser.parse_args()

def load_data(data_dir):
    """
    Load preprocessed MNIST data.
    
    Args:
        data_dir (str): Directory containing the data
        
    Returns:
        tuple: ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    """
    print(f"Loading data from {data_dir}...")
    
    if os.path.exists(os.path.join(data_dir, 'x_train.npy')):
        # Load preprocessed data
        x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        x_val = np.load(os.path.join(data_dir, 'x_val.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        print(f"Loaded preprocessed data:")
        print(f"  Training: {x_train.shape[0]} samples")
        print(f"  Validation: {x_val.shape[0]} samples")
        print(f"  Test: {x_test.shape[0]} samples")
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    else:
        # If preprocessed data doesn't exist, try to load from TensorFlow
        try:
            import tensorflow_datasets as tfds
            print("Preprocessed data not found. Loading MNIST from TensorFlow Datasets...")
            
            # Load MNIST dataset
            (ds_train, ds_test), ds_info = tfds.load(
                'mnist',
                split=['train', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True,
            )
            
            # Define preprocessing function
            def normalize_img(image, label):
                """Normalize image to [0, 1]."""
                return tf.cast(image, tf.float32) / 255.0, label
            
            # Apply preprocessing
            ds_train = ds_train.map(normalize_img)
            ds_test = ds_test.map(normalize_img)
            
            # Convert to NumPy arrays
            print("Converting TensorFlow datasets to NumPy arrays...")
            x_train_all = np.zeros((60000, 28, 28, 1), dtype=np.float32)
            y_train_all = np.zeros(60000, dtype=np.int32)
            
            for i, (image, label) in enumerate(tfds.as_numpy(ds_train)):
                x_train_all[i] = image
                y_train_all[i] = label
            
            x_test = np.zeros((10000, 28, 28, 1), dtype=np.float32)
            y_test = np.zeros(10000, dtype=np.int32)
            
            for i, (image, label) in enumerate(tfds.as_numpy(ds_test)):
                x_test[i] = image
                y_test[i] = label
            
            # Convert labels to one-hot encoding
            y_train_all = tf.keras.utils.to_categorical(y_train_all, 10)
            y_test = tf.keras.utils.to_categorical(y_test, 10)
            
            # Split training data into train and validation
            indices = np.random.permutation(len(x_train_all))
            val_size = int(len(x_train_all) * args.val_split)
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            x_train = x_train_all[train_indices]
            y_train = y_train_all[train_indices]
            x_val = x_train_all[val_indices]
            y_val = y_train_all[val_indices]
            
            # Create the data directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Save the preprocessed data
            print(f"Saving preprocessed data to {data_dir}...")
            np.save(os.path.join(data_dir, 'x_train.npy'), x_train)
            np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(data_dir, 'x_val.npy'), x_val)
            np.save(os.path.join(data_dir, 'y_val.npy'), y_val)
            np.save(os.path.join(data_dir, 'x_test.npy'), x_test)
            np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
            
            print(f"Processed and saved data:")
            print(f"  Training: {x_train.shape[0]} samples")
            print(f"  Validation: {x_val.shape[0]} samples")
            print(f"  Test: {x_test.shape[0]} samples")
            
            return (x_train, y_train), (x_val, y_val), (x_test, y_test)
            
        except ImportError:
            print("Error: tensorflow_datasets not available. Please run data_prep.py first.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

def main(args):
    """Main function."""
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    
    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(args.data_dir)
    
    # Create model and configuration
    print(f"Creating {args.model_type} model...")
    model = ModelRegistry.create_model(args.model_type)
    config = ModelRegistry.create_config(args.model_type)
    
    # Override configuration with command line arguments
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.no_augmentation:
        config.use_data_augmentation = False
    
    # Print model summary
    model.summary()
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Create custom callbacks
    callbacks = []
    
    # Confusion matrix callback
    confusion_matrix_dir = os.path.join(args.fig_dir, 'confusion_matrices')
    os.makedirs(confusion_matrix_dir, exist_ok=True)
    confusion_callback = ConfusionMatrixCallback(
        validation_data=(x_val, y_val),
        output_dir=confusion_matrix_dir,
        freq=5
    )
    callbacks.append(confusion_callback)
    
    # Feature map visualizer callback
    if x_test.shape[0] > 0:
        feature_maps_dir = os.path.join(args.fig_dir, 'feature_maps')
        os.makedirs(feature_maps_dir, exist_ok=True)
        feature_map_callback = FeatureMapVisualizer(
            test_image=x_test[0:1],
            output_dir=feature_maps_dir,
            freq=10
        )
        callbacks.append(feature_map_callback)
    
    # Train the model
    print(f"Training {args.model_type} model...")
    history = trainer.train(
        (x_train, y_train),
        (x_val, y_val),
        epochs=args.epochs
    )
    
    # Plot training history
    print("Plotting training history...")
    history_plot = trainer.plot_training_history()
    history_plot.savefig(os.path.join(args.fig_dir, f"{args.model_type}_training_history.png"))
    
    # Save the final model
    final_model_path = os.path.join(args.model_dir, f"{args.model_type}_final.h5")
    trainer.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Evaluate the model on test data
    print("Evaluating model on test data...")
    evaluator = Evaluator(model, (x_test, y_test))
    metrics = evaluator.evaluate()
    
    # Print evaluation metrics
    print("\nTest Metrics:")
    for name, value in metrics.items():
        if isinstance(value, np.ndarray):
            if value.size <= 10:  # Only print if it's a reasonably sized array
                print(f"  {name}: {value}")
        else:
            print(f"  {name}: {value}")
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    cm_plot = evaluator.plot_confusion_matrix(normalize=True)
    cm_plot.savefig(os.path.join(args.fig_dir, f"{args.model_type}_confusion_matrix.png"))
    
    # Plot misclassified examples
    print("Plotting misclassified examples...")
    misclassified_plot = evaluator.plot_misclassified_examples(num_examples=10)
    misclassified_plot.savefig(os.path.join(args.fig_dir, f"{args.model_type}_misclassified.png"))
    
    print(f"Training and evaluation of {args.model_type} model completed successfully.")

if __name__ == '__main__':
    args = parse_args()
    main(args)
