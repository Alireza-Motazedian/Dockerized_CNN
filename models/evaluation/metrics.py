#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom metrics for model evaluation.

This module implements custom metrics for evaluating model performance.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score as sklearn_f1_score

def precision_recall_f1(y_true, y_pred):
    """
    Calculate precision, recall, and F1 score for each class.
    
    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        
    Returns:
        tuple: (precision, recall, f1)
    """
    # Convert one-hot encoded labels to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Calculate metrics for each class
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = sklearn_f1_score(y_true, y_pred, average=None)
    
    # Also calculate micro and macro averages
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_micro = sklearn_f1_score(y_true, y_pred, average='micro')
    f1_macro = sklearn_f1_score(y_true, y_pred, average='macro')
    
    # Create a dictionary with all metrics
    metrics = {
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro
    }
    
    return metrics

def top_k_accuracy(y_true, y_pred, k=3):
    """
    Calculate top-k accuracy.
    
    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted probabilities
        k (int): Number of top predictions to consider
        
    Returns:
        float: Top-k accuracy
    """
    # Convert one-hot encoded labels to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Ensure y_pred is probabilities
    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
        raise ValueError("y_pred must be predicted probabilities, not class indices")
    
    # For each sample, check if true label is in top k predictions
    top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]  # Get top k indices
    correct = 0
    
    for i, true_label in enumerate(y_true):
        if true_label in top_k_indices[i]:
            correct += 1
    
    return correct / len(y_true)

def f1_score_keras(y_true, y_pred):
    """
    F1 score as a Keras metric.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        
    Returns:
        tf.Tensor: F1 score
    """
    # Convert probabilities to class indices
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true_classes = tf.argmax(y_true, axis=1)
    
    # Calculate true positives, false positives, false negatives
    true_positives = tf.reduce_sum(
        tf.cast(tf.equal(y_pred_classes, y_true_classes), tf.float32)
    )
    false_positives = tf.reduce_sum(
        tf.cast(tf.not_equal(y_pred_classes, y_true_classes), tf.float32)
    )
    false_negatives = false_positives  # For multi-class, single-label case
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
    recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    
    return f1

def per_class_accuracy(y_true, y_pred, num_classes=10):
    """
    Calculate per-class accuracy.
    
    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or probabilities)
        num_classes (int): Number of classes
        
    Returns:
        np.ndarray: Per-class accuracy
    """
    # Convert one-hot encoded labels to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Convert predictions to class indices if needed
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Initialize array for per-class accuracy
    per_class_acc = np.zeros(num_classes)
    
    # Calculate accuracy for each class
    for c in range(num_classes):
        # Get indices of samples with this true class
        class_indices = np.where(y_true == c)[0]
        
        if len(class_indices) > 0:
            # Calculate accuracy for this class
            correct = np.sum(y_pred[class_indices] == c)
            per_class_acc[c] = correct / len(class_indices)
    
    return per_class_acc 