#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and preprocess the MNIST dataset.
This script handles:
- Downloading MNIST data if not available
- Normalizing pixel values to [0, 1]
- Reshaping data to include channel dimension
- Splitting data into training, validation, and test sets
- Saving processed data for later use
"""

import os
import argparse
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download and preprocess MNIST data')
    parser.add_argument('--data_dir', type=str, default='data/mnist',
                        help='Directory to save the MNIST data')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of training data to use for validation')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def preprocess_data(x_train, y_train, x_test, y_test, val_split=0.1, random_seed=42):
    """Preprocess MNIST data."""
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to include channel dimension (28x28x1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Split training data into train and validation sets
    np.random.seed(random_seed)
    indices = np.random.permutation(len(x_train))
    val_size = int(len(x_train) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def main():
    """Main function."""
    args = parse_args()
    
    # Create data directory
    create_directory(args.data_dir)
    
    print("Downloading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print("Preprocessing data...")
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test, 
        val_split=args.val_split, 
        random_seed=args.random_seed
    )
    
    # Save preprocessed data
    print(f"Saving preprocessed data to {args.data_dir}...")
    np.save(os.path.join(args.data_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(args.data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(args.data_dir, 'x_val.npy'), x_val)
    np.save(os.path.join(args.data_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(args.data_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(args.data_dir, 'y_test.npy'), y_test)
    
    print("Data preparation completed successfully.")
    print(f"Training set: {x_train.shape[0]} samples")
    print(f"Validation set: {x_val.shape[0]} samples")
    print(f"Test set: {x_test.shape[0]} samples")

if __name__ == '__main__':
    main()
