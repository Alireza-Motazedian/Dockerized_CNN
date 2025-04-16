#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom callbacks for model training.

This module implements custom callbacks to be used during model training.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix

class LearningRateSchedulerCallback(Callback):
    """
    Learning rate scheduler with various schedules.
    
    Args:
        initial_lr (float): Initial learning rate
        schedule (str): Type of schedule ('constant', 'step', 'exponential', 'cosine')
        decay_steps (int): Number of steps before decay (for step and exponential)
        decay_rate (float): Decay rate for step and exponential schedules
        total_epochs (int): Total number of epochs (for cosine decay)
    """
    
    def __init__(self, initial_lr=0.001, schedule='constant', 
                 decay_steps=5, decay_rate=0.5, total_epochs=20):
        super().__init__()
        self.initial_lr = initial_lr
        self.schedule = schedule
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.total_epochs = total_epochs
        self.history = []
    
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            return
        
        # Get the current learning rate
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        
        # Calculate new learning rate based on schedule
        if self.schedule == 'constant':
            new_lr = self.initial_lr
        elif self.schedule == 'step':
            # Step decay: reduce by decay_rate every decay_steps epochs
            new_lr = self.initial_lr * self.decay_rate ** (epoch // self.decay_steps)
        elif self.schedule == 'exponential':
            # Exponential decay: continuous exponential decay
            new_lr = self.initial_lr * np.exp(-self.decay_rate * epoch / self.decay_steps)
        elif self.schedule == 'cosine':
            # Cosine annealing: cosine decay to zero
            new_lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / self.total_epochs))
        else:
            raise ValueError(f"Unknown learning rate schedule: {self.schedule}")
        
        # Set the learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        self.history.append(new_lr)
        
        print(f"\nEpoch {epoch+1}/{self.total_epochs}: Learning rate set to {new_lr:.6f}")
    
    def plot_history(self):
        """Plot the learning rate history."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.history) + 1), self.history, marker='o')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.tight_layout()
        return plt.gcf()

class ConfusionMatrixCallback(Callback):
    """
    Callback to generate confusion matrix after specified epochs.
    
    Args:
        validation_data (tuple): Validation data (x_val, y_val)
        output_dir (str): Directory to save confusion matrix plots
        class_names (list): List of class names
        freq (int): Frequency of epochs to generate confusion matrix
    """
    
    def __init__(self, validation_data, output_dir='figures', 
                 class_names=None, freq=5):
        super().__init__()
        self.x_val, self.y_val = validation_data
        self.output_dir = output_dir
        self.freq = freq
        
        # If class_names is not provided, use digit labels
        if class_names is None:
            self.class_names = [str(i) for i in range(10)]
        else:
            self.class_names = class_names
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        # Only generate confusion matrix at specified frequency
        if (epoch + 1) % self.freq != 0:
            return
        
        # Get predictions
        y_pred = self.model.predict(self.x_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # If y_val is one-hot encoded, convert to class indices
        if len(self.y_val.shape) > 1 and self.y_val.shape[1] > 1:
            y_true = np.argmax(self.y_val, axis=1)
        else:
            y_true = self.y_val
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.colorbar()
        
        # Add labels
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
        plt.close()

class FeatureMapVisualizer(Callback):
    """
    Callback to visualize feature maps of convolutional layers.
    
    Args:
        test_image: A single test image to visualize feature maps for
        output_dir (str): Directory to save feature map visualizations
        layer_names (list): List of layer names to visualize
        freq (int): Frequency of epochs to generate visualizations
    """
    
    def __init__(self, test_image, output_dir='figures/feature_maps', 
                 layer_names=None, freq=10):
        super().__init__()
        # Add batch dimension if not present
        if len(test_image.shape) == 3:
            self.test_image = np.expand_dims(test_image, axis=0)
        else:
            self.test_image = test_image
        
        self.output_dir = output_dir
        self.layer_names = layer_names
        self.freq = freq
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        # Only generate visualizations at specified frequency
        if (epoch + 1) % self.freq != 0:
            return
        
        # If layer_names not specified, find all Conv2D layers
        if self.layer_names is None:
            self.layer_names = [layer.name for layer in self.model.layers 
                               if 'conv' in layer.name.lower()]
        
        # Create feature map models for each layer
        for layer_name in self.layer_names:
            try:
                # Create a model that outputs the feature maps
                layer = self.model.get_layer(layer_name)
                feature_map_model = tf.keras.Model(inputs=self.model.inputs,
                                                  outputs=layer.output)
                
                # Get feature maps
                feature_maps = feature_map_model.predict(self.test_image)
                
                # Plot feature maps
                self._plot_feature_maps(feature_maps, layer_name, epoch)
            except:
                print(f"Could not visualize feature maps for layer: {layer_name}")
    
    def _plot_feature_maps(self, feature_maps, layer_name, epoch):
        """Plot feature maps for a specific layer."""
        # Determine number of filters to display (at most 16)
        num_filters = min(16, feature_maps.shape[-1])
        
        # Create figure
        plt.figure(figsize=(20, 10))
        plt.suptitle(f'Feature Maps - Layer: {layer_name} - Epoch: {epoch+1}')
        
        # Plot feature maps
        for i in range(num_filters):
            plt.subplot(4, 4, i + 1)
            plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
            plt.title(f'Filter {i}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, 
                                f'feature_maps_{layer_name}_epoch_{epoch+1}.png'))
        plt.close() 