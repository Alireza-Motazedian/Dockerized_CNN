#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize features learned by CNN models for MNIST digit classification.

This script generates visualizations of CNN feature maps, activations,
and filters to help understand what the model has learned.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

# Add parent directory to path to import from models
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_factory import load_pretrained_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize CNN features for MNIST')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='simple_cnn',
                        help='Type of model (if loading weights only)')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/mnist',
                        help='Directory containing MNIST data')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='figures/feature_maps',
                        help='Directory to save visualizations')
    
    # Visualization types
    parser.add_argument('--visualize_filters', action='store_true',
                        help='Visualize convolutional filters')
    parser.add_argument('--visualize_activations', action='store_true',
                        help='Visualize layer activations')
    parser.add_argument('--visualize_gradcam', action='store_true',
                        help='Visualize Grad-CAM heatmaps')
    
    return parser.parse_args()

def load_data(data_dir, num_samples=5):
    """
    Load a few test samples from MNIST dataset.
    
    Args:
        data_dir (str): Directory containing the data
        num_samples (int): Number of samples to load
        
    Returns:
        tuple: (x_samples, y_samples)
    """
    print(f"Loading test data from {data_dir}...")
    
    if os.path.exists(os.path.join(data_dir, 'x_test.npy')):
        # Load preprocessed data
        x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        # Select a few samples (one from each class if possible)
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            # One-hot encoded
            y_test_indices = np.argmax(y_test, axis=1)
        else:
            y_test_indices = y_test
        
        # Try to select one example for each digit
        unique_classes = np.unique(y_test_indices)
        
        samples_per_class = max(1, num_samples // len(unique_classes))
        remaining_samples = num_samples - samples_per_class * len(unique_classes)
        
        selected_indices = []
        for cls in unique_classes:
            class_indices = np.where(y_test_indices == cls)[0]
            if len(class_indices) > 0:
                cls_samples = min(samples_per_class, len(class_indices))
                selected_indices.extend(np.random.choice(class_indices, cls_samples, replace=False))
        
        # Add any remaining samples
        if remaining_samples > 0 and len(y_test_indices) > len(selected_indices):
            remaining_indices = np.setdiff1d(range(len(y_test_indices)), selected_indices)
            additional_indices = np.random.choice(
                remaining_indices, 
                min(remaining_samples, len(remaining_indices)), 
                replace=False
            )
            selected_indices.extend(additional_indices)
        
        x_samples = x_test[selected_indices]
        y_samples = y_test[selected_indices]
        
        print(f"Selected {len(selected_indices)} samples for visualization")
        
        return x_samples, y_samples
    else:
        print(f"Error: Test data not found in {data_dir}")
        print("Please run data_prep.py and train_cnn.py first.")
        sys.exit(1)

def get_conv_layers(model):
    """
    Get all convolutional layers from a model.
    
    Args:
        model: Keras model
        
    Returns:
        list: List of convolutional layers
    """
    conv_layers = []
    for layer in model.layers:
        # Check if it's a convolutional layer
        if 'conv' in layer.name.lower() or isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer)
    return conv_layers

def visualize_filters(model, output_dir):
    """
    Visualize convolutional filters.
    
    Args:
        model: Keras model
        output_dir (str): Directory to save visualizations
    """
    print("Visualizing convolutional filters...")
    
    # Get convolutional layers
    conv_layers = get_conv_layers(model)
    
    if not conv_layers:
        print("No convolutional layers found in the model.")
        return
    
    # Create output directory for filters
    filter_dir = os.path.join(output_dir, 'filters')
    os.makedirs(filter_dir, exist_ok=True)
    
    # Visualize filters for each convolutional layer
    for layer in conv_layers:
        # Get the weights
        weights, biases = layer.get_weights()
        
        # Normalize weights to [0, 1] for visualization
        min_val = np.min(weights)
        max_val = np.max(weights)
        weights = (weights - min_val) / (max_val - min_val + 1e-10)
        
        # Determine grid size
        num_filters = weights.shape[3]
        grid_size = int(np.ceil(np.sqrt(num_filters)))
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        fig.suptitle(f"Filters for layer: {layer.name}", fontsize=16)
        
        # Plot each filter
        for i in range(grid_size * grid_size):
            ax = axes[i // grid_size, i % grid_size]
            if i < num_filters:
                # Get the filter
                kernel = weights[:, :, :, i]
                
                # For RGB filters (3 channels), create a color image
                if kernel.shape[2] == 3:
                    ax.imshow(kernel)
                # For single channel filters, average across input channels
                else:
                    ax.imshow(np.mean(kernel, axis=2), cmap='viridis')
                
                ax.set_title(f"Filter {i}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save the figure
        filter_path = os.path.join(filter_dir, f"filters_{layer.name}.png")
        plt.savefig(filter_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved filter visualization for layer {layer.name}")

def visualize_activations(model, samples, output_dir):
    """
    Visualize feature maps (activations) for a set of samples.
    
    Args:
        model: Keras model
        samples: Input samples
        output_dir (str): Directory to save visualizations
    """
    print("Visualizing layer activations...")
    
    # Create output directory for activations
    activation_dir = os.path.join(output_dir, 'activations')
    os.makedirs(activation_dir, exist_ok=True)
    
    # Get convolutional layers
    layers_to_visualize = []
    for layer in model.layers:
        # Include convolutional and pooling layers
        if ('conv' in layer.name.lower() or 
            'pool' in layer.name.lower() or
            isinstance(layer, tf.keras.layers.Conv2D) or
            isinstance(layer, tf.keras.layers.MaxPooling2D) or
            isinstance(layer, tf.keras.layers.AveragePooling2D)):
            layers_to_visualize.append(layer)
    
    if not layers_to_visualize:
        print("No layers suitable for activation visualization found.")
        return
    
    # Create activation models for each layer
    activation_models = []
    for layer in layers_to_visualize:
        activation_model = Model(inputs=model.input, outputs=layer.output)
        activation_models.append((layer.name, activation_model))
    
    # Visualize activations for each sample
    for i, sample in enumerate(samples):
        print(f"  Visualizing activations for sample {i}...")
        
        # Add batch dimension if not present
        if len(sample.shape) == 3:
            sample = np.expand_dims(sample, axis=0)
        
        for layer_name, activation_model in activation_models:
            # Get activations
            activations = activation_model.predict(sample)
            
            # Determine grid size
            num_filters = activations.shape[3]
            grid_size = min(8, int(np.ceil(np.sqrt(num_filters))))
            
            # Create figure
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
            fig.suptitle(f"Activations for layer: {layer_name} (Sample {i})", fontsize=16)
            
            # Display original image
            sample_img = sample[0]
            if sample_img.shape[2] == 1:
                sample_img = sample_img.reshape(sample_img.shape[0], sample_img.shape[1])
            
            # Plot each activation channel
            for j in range(grid_size * grid_size):
                ax = axes[j // grid_size, j % grid_size]
                if j < num_filters:
                    ax.imshow(activations[0, :, :, j], cmap='viridis')
                    ax.set_title(f"Channel {j}")
                else:
                    ax.axis('off')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Save the figure
            activation_path = os.path.join(activation_dir, f"activation_sample{i}_{layer_name}.png")
            plt.savefig(activation_path, dpi=150, bbox_inches='tight')
            plt.close()

def gradcam(model, image, layer_name=None, class_idx=None):
    """
    Compute Grad-CAM heatmap for a model.
    
    Args:
        model: Keras model
        image: Input image (should be preprocessed)
        layer_name (str, optional): Target layer name (uses last conv if None)
        class_idx (int, optional): Index of the class to explain (uses predicted class if None)
        
    Returns:
        tuple: (heatmap, overlaid_image)
    """
    # Find the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
        if layer_name is None:
            print("No convolutional layer found in the model.")
            return None, None
    
    # Create a model that maps the input to the output of the target layer
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Cast to float32
        image = tf.cast(image, tf.float32)
        # Forward pass through the model
        conv_output, predictions = grad_model(image)
        
        # Get the predicted class if not specified
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        
        # Get the output for the specified class
        class_output = predictions[:, class_idx]
    
    # Compute the gradient of the class output with respect to the feature map
    grads = tape.gradient(class_output, conv_output)
    
    # Global average pooling for the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the feature map with the gradient
    last_conv_layer_output = conv_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()
    
    # Resize heatmap to original image size
    import cv2
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert image to RGB if it's grayscale
    if image.shape[3] == 1:
        img = np.squeeze(image[0]) * 255
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = np.uint8(image[0] * 255)
    
    # Superimpose the heatmap on the image
    heatmap = heatmap.astype(np.float32) / 255
    img = img.astype(np.float32) / 255
    overlaid_img = heatmap * 0.4 + img
    overlaid_img = np.clip(overlaid_img, 0, 1)
    
    return heatmap, overlaid_img

def visualize_gradcam(model, samples, labels, output_dir):
    """
    Visualize Grad-CAM heatmaps for samples.
    
    Args:
        model: Keras model
        samples: Input samples
        labels: Sample labels (one-hot encoded or class indices)
        output_dir (str): Directory to save visualizations
    """
    print("Visualizing Grad-CAM heatmaps...")
    
    # Create output directory
    gradcam_dir = os.path.join(output_dir, 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # Convert labels to class indices if one-hot encoded
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        true_classes = np.argmax(labels, axis=1)
    else:
        true_classes = labels
    
    # Get convolutional layers for Grad-CAM
    conv_layers = get_conv_layers(model)
    
    if not conv_layers:
        print("No convolutional layers found for Grad-CAM.")
        return
    
    # Use the last convolutional layer
    target_layer = conv_layers[-1].name
    
    # Process each sample
    for i, (sample, true_class) in enumerate(zip(samples, true_classes)):
        print(f"  Generating Grad-CAM for sample {i} (digit {true_class})...")
        
        # Get prediction
        pred_probs = model.predict(np.expand_dims(sample, axis=0))[0]
        pred_class = np.argmax(pred_probs)
        
        # Generate Grad-CAM for both true and predicted class
        classes_to_visualize = [true_class]
        if pred_class != true_class:
            classes_to_visualize.append(pred_class)
        
        # Create figure
        fig, axes = plt.subplots(1, len(classes_to_visualize) + 1, figsize=(15, 5))
        fig.suptitle(f"Grad-CAM for Sample {i} (Digit {true_class})", fontsize=16)
        
        # Display original image
        sample_img = sample.copy()
        if sample_img.shape[2] == 1:
            sample_img = sample_img.reshape(sample_img.shape[0], sample_img.shape[1])
        
        axes[0].imshow(sample_img, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Generate and display Grad-CAM for each class
        for j, class_idx in enumerate(classes_to_visualize):
            _, cam_img = gradcam(model, sample, target_layer, class_idx)
            
            axes[j + 1].imshow(cam_img)
            if class_idx == true_class:
                title = f"Class {class_idx} (True)"
            else:
                title = f"Class {class_idx} (Predicted)"
            axes[j + 1].set_title(title)
            axes[j + 1].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save the figure
        gradcam_path = os.path.join(gradcam_dir, f"gradcam_sample{i}.png")
        plt.savefig(gradcam_path, dpi=150, bbox_inches='tight')
        plt.close()

def main(args):
    """Main function."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    try:
        if args.model_path.endswith(".h5") and "_weights" in args.model_path:
            # Load model with weights
            model = load_pretrained_model(args.model_type, args.model_path)
        else:
            # Load entire model
            model = tf.keras.models.load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Display model summary
    model.summary()
    
    # Load test samples
    samples, labels = load_data(args.data_dir, args.num_samples)
    
    # Create a figure showing the test samples
    print("Visualizing test samples...")
    fig, axes = plt.subplots(1, len(samples), figsize=(3*len(samples), 3))
    
    for i, (sample, label) in enumerate(zip(samples, labels)):
        # Get true label
        if len(label.shape) > 0 and label.shape[0] > 1:
            true_label = np.argmax(label)
        else:
            true_label = label
        
        # Display sample
        img = sample.copy()
        if img.shape[2] == 1:
            img = img.reshape(img.shape[0], img.shape[1])
        
        if len(samples) > 1:
            ax = axes[i]
        else:
            ax = axes
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Digit: {true_label}")
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    samples_path = os.path.join(args.output_dir, "test_samples.png")
    plt.savefig(samples_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Perform visualizations
    if args.visualize_filters:
        visualize_filters(model, args.output_dir)
    
    if args.visualize_activations:
        visualize_activations(model, samples, args.output_dir)
    
    if args.visualize_gradcam:
        visualize_gradcam(model, samples, labels, args.output_dir)
    
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
