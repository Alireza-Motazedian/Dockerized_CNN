#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluator class for model evaluation.

This module implements an Evaluator class for evaluating trained models
on test data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

from models.evaluation.metrics import precision_recall_f1, top_k_accuracy, per_class_accuracy

class Evaluator:
    """
    Class for evaluating trained models on test data.
    
    Args:
        model: Trained Keras model
        test_data: Tuple of (X_test, y_test) or tf.data.Dataset
    """
    
    def __init__(self, model, test_data):
        self.model = model
        
        # Handle different types of test data
        if isinstance(test_data, tuple):
            self.X_test, self.y_test = test_data
        else:
            # Extract data from tf.data.Dataset
            # Note: This will consume the dataset, so it can't be used again
            import tensorflow as tf
            self.X_test = []
            self.y_test = []
            for x, y in test_data:
                self.X_test.append(x.numpy())
                self.y_test.append(y.numpy())
            self.X_test = np.concatenate(self.X_test, axis=0)
            self.y_test = np.concatenate(self.y_test, axis=0)
        
        # Store predictions for later use
        self.y_pred = None
    
    def evaluate(self):
        """
        Evaluate the model on test data.
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Get model predictions if not already cached
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)
        
        # Get built-in evaluation metrics
        metrics = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        results = dict(zip(self.model.metrics_names, metrics))
        
        # Add custom metrics
        # Precision, recall, F1 score
        prf_metrics = precision_recall_f1(self.y_test, self.y_pred)
        results.update(prf_metrics)
        
        # Top-k accuracy for k=3 and k=5
        results['top_3_accuracy'] = top_k_accuracy(self.y_test, self.y_pred, k=3)
        results['top_5_accuracy'] = top_k_accuracy(self.y_test, self.y_pred, k=5)
        
        # Per-class accuracy
        results['per_class_accuracy'] = per_class_accuracy(self.y_test, self.y_pred)
        
        return results
    
    def evaluate_per_class(self):
        """
        Evaluate the model performance per class.
        
        Returns:
            pd.DataFrame: DataFrame with per-class metrics
        """
        # Get predictions if not already cached
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)
        
        # Convert predictions to class indices
        y_pred_classes = np.argmax(self.y_pred, axis=1)
        
        # Convert true labels to class indices if one-hot encoded
        if len(self.y_test.shape) > 1 and self.y_test.shape[1] > 1:
            y_true_classes = np.argmax(self.y_test, axis=1)
        else:
            y_true_classes = self.y_test
        
        # Get metrics per class
        per_class_acc = per_class_accuracy(y_true_classes, y_pred_classes)
        
        # Get precision, recall, F1 for each class
        metrics = precision_recall_f1(y_true_classes, y_pred_classes)
        
        # Create DataFrame with per-class metrics
        df = pd.DataFrame({
            'Class': range(10),
            'Accuracy': per_class_acc,
            'Precision': metrics['precision_per_class'],
            'Recall': metrics['recall_per_class'],
            'F1 Score': metrics['f1_per_class']
        })
        
        # Count samples per class
        class_counts = np.bincount(y_true_classes, minlength=10)
        df['Sample Count'] = class_counts
        
        return df
    
    def confusion_matrix(self, normalize=False):
        """
        Generate confusion matrix for the model.
        
        Args:
            normalize (bool): Whether to normalize the confusion matrix
            
        Returns:
            np.ndarray: Confusion matrix
        """
        # Get predictions if not already cached
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)
        
        # Convert predictions to class indices
        y_pred_classes = np.argmax(self.y_pred, axis=1)
        
        # Convert true labels to class indices if one-hot encoded
        if len(self.y_test.shape) > 1 and self.y_test.shape[1] > 1:
            y_true_classes = np.argmax(self.y_test, axis=1)
        else:
            y_true_classes = self.y_test
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return cm
    
    def plot_confusion_matrix(self, normalize=False, class_names=None, figsize=(10, 8)):
        """
        Plot confusion matrix for the model.
        
        Args:
            normalize (bool): Whether to normalize the confusion matrix
            class_names (list): List of class names
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Confusion matrix plot
        """
        # Get confusion matrix
        cm = self.confusion_matrix(normalize=normalize)
        
        # Set up class names
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im)
        
        # Set up axis labels
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        
        return fig
    
    def misclassified_examples(self, num_examples=10):
        """
        Get examples of misclassified images.
        
        Args:
            num_examples (int): Number of examples to return
            
        Returns:
            tuple: (misclassified_images, true_labels, predicted_labels)
        """
        # Get predictions if not already cached
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)
        
        # Convert predictions to class indices
        y_pred_classes = np.argmax(self.y_pred, axis=1)
        
        # Convert true labels to class indices if one-hot encoded
        if len(self.y_test.shape) > 1 and self.y_test.shape[1] > 1:
            y_true_classes = np.argmax(self.y_test, axis=1)
        else:
            y_true_classes = self.y_test
        
        # Find indices of misclassified examples
        misclassified_indices = np.where(y_pred_classes != y_true_classes)[0]
        
        # Select a random subset of misclassified examples
        if len(misclassified_indices) > num_examples:
            selected_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
        else:
            selected_indices = misclassified_indices
        
        # Get the misclassified images and labels
        misclassified_images = self.X_test[selected_indices]
        true_labels = y_true_classes[selected_indices]
        predicted_labels = y_pred_classes[selected_indices]
        
        return misclassified_images, true_labels, predicted_labels
    
    def plot_misclassified_examples(self, num_examples=10, figsize=(15, 10)):
        """
        Plot examples of misclassified images.
        
        Args:
            num_examples (int): Number of examples to plot
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure with misclassified examples
        """
        # Get misclassified examples
        images, true_labels, pred_labels = self.misclassified_examples(num_examples)
        
        # Determine grid dimensions
        if num_examples <= 5:
            grid_dim = (1, num_examples)
        else:
            cols = min(5, num_examples)
            rows = (num_examples + cols - 1) // cols  # Ceiling division
            grid_dim = (rows, cols)
        
        # Create figure
        fig, axes = plt.subplots(*grid_dim, figsize=figsize)
        axes = axes.flatten() if num_examples > 1 else [axes]
        
        # Plot each misclassified example
        for i, (img, true_label, pred_label) in enumerate(zip(images, true_labels, pred_labels)):
            if i >= num_examples:
                break
                
            # Reshape image for display if needed
            if img.shape[-1] == 1:
                img = img.reshape(img.shape[0], img.shape[1])
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {true_label}, Pred: {pred_label}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Misclassified Examples', fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        return fig 