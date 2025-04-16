#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract sample images from the MNIST dataset for visualization.
This script:
- Loads the MNIST dataset
- Selects representative samples from each digit class
- Saves the sample images to the data/mnist_samples directory
- Generates a grid visualization of the samples
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract sample images from MNIST dataset')
    parser.add_argument('--data_dir', type=str, default='data/mnist',
                        help='Directory with MNIST data')
    parser.add_argument('--samples_dir', type=str, default='data/mnist_samples',
                        help='Directory to save sample images')
    parser.add_argument('--output_file', type=str, default='figures/mnist_samples.png',
                        help='Path to save grid visualization')
    parser.add_argument('--samples_per_class', type=int, default=5,
                        help='Number of samples to extract per class')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def extract_samples(x_data, y_data, samples_per_class=5, random_seed=42):
    """Extract sample images from each class."""
    np.random.seed(random_seed)
    samples = []
    
    # For each digit class (0-9)
    for digit in range(10):
        # Find indices of images with this digit
        indices = np.where(y_data == digit)[0]
        
        # Randomly select samples_per_class images
        selected_indices = np.random.choice(indices, samples_per_class, replace=False)
        
        # Add selected images to samples list
        for i, idx in enumerate(selected_indices):
            samples.append((x_data[idx], digit, i))
    
    return samples

def save_samples(samples, samples_dir):
    """Save individual sample images."""
    for img, digit, idx in samples:
        filename = os.path.join(samples_dir, f"digit_{digit}_sample_{idx}.png")
        plt.imsave(filename, img, cmap='gray')
    
    print(f"Saved {len(samples)} sample images to {samples_dir}")

def create_grid_visualization(samples, output_file, samples_per_class=5):
    """Create a grid visualization of sample images."""
    plt.figure(figsize=(12, 10))
    
    for i, (img, digit, idx) in enumerate(samples):
        plt.subplot(10, samples_per_class, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Digit: {digit}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created grid visualization: {output_file}")

def main():
    """Main function."""
    args = parse_args()
    
    # Create directories
    create_directory(args.samples_dir)
    create_directory(os.path.dirname(args.output_file))
    
    # Check if preprocessed data exists
    if os.path.exists(os.path.join(args.data_dir, 'x_test.npy')):
        print("Loading preprocessed MNIST data...")
        x_test = np.load(os.path.join(args.data_dir, 'x_test.npy'))
        y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
        
        # Convert one-hot encoded labels back to class indices
        if len(y_test.shape) > 1:
            y_test = np.argmax(y_test, axis=1)
        
        # Reshape images to 2D for visualization
        if len(x_test.shape) > 3:  # If shape is (n, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28)
    else:
        print("Preprocessed data not found. Loading original MNIST dataset...")
        (_, _), (x_test, y_test) = mnist.load_data()
    
    print("Extracting sample images...")
    samples = extract_samples(
        x_test, y_test, 
        samples_per_class=args.samples_per_class, 
        random_seed=args.random_seed
    )
    
    print("Saving sample images...")
    save_samples(samples, args.samples_dir)
    
    print("Creating grid visualization...")
    create_grid_visualization(
        samples, args.output_file, 
        samples_per_class=args.samples_per_class
    )
    
    print("Sample extraction completed successfully.")

if __name__ == '__main__':
    main()
