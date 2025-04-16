#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model factory for creating model instances.

This module provides functions to create and load CNN models
for MNIST digit classification.
"""

import os
import importlib
import tensorflow as tf

def create_model(model_name, input_shape=(28, 28, 1), num_classes=10, **kwargs):
    """
    Create a model based on model name.
    
    Args:
        model_name (str): Name of the model architecture
        input_shape (tuple): Input shape of the model
        num_classes (int): Number of output classes
        **kwargs: Additional keyword arguments for the model
        
    Returns:
        tf.keras.Model: Created model
    """
    # Map model names to creation functions in architectures module
    model_name = model_name.lower()
    
    try:
        # Import the appropriate module
        module_name = f"models.architectures.{model_name}"
        module = importlib.import_module(module_name)
        
        # Get the model creation function (expected naming convention: create_<model_name>)
        create_func_name = f"create_{model_name}"
        if not hasattr(module, create_func_name):
            raise ValueError(f"Module {module_name} does not have a {create_func_name} function")
        
        create_func = getattr(module, create_func_name)
        
        # Create the model
        model = create_func(input_shape=input_shape, num_classes=num_classes, **kwargs)
        
        return model
    
    except ImportError:
        raise ValueError(f"Model architecture '{model_name}' not found")
    except Exception as e:
        raise ValueError(f"Error creating model '{model_name}': {e}")

def load_pretrained_model(model_name, model_path, custom_objects=None):
    """
    Load a pretrained model from disk.
    
    Args:
        model_name (str): Name of the model architecture
        model_path (str): Path to the model file or weights file
        custom_objects (dict): Dictionary mapping names to custom classes or functions
        
    Returns:
        tf.keras.Model: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # If it's a weights file (h5 with specific naming convention)
    if model_path.endswith(".h5") and "_weights" in model_path:
        # Create the model
        model = create_model(model_name)
        
        # Load weights
        model.load_weights(model_path)
        return model
    
    # Otherwise, load the entire model
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        raise ValueError(f"Error loading model from {model_path}: {e}")

def save_model(model, save_path, save_weights_only=False):
    """
    Save a model to disk.
    
    Args:
        model: Keras model to save
        save_path (str): Path to save the model
        save_weights_only (bool): Whether to save only weights
        
    Returns:
        str: Path where the model was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if save_weights_only:
        # Ensure path ends with '_weights.h5' for consistency
        if not save_path.endswith("_weights.h5"):
            base_path = save_path.rsplit(".", 1)[0] if "." in save_path else save_path
            save_path = f"{base_path}_weights.h5"
        
        model.save_weights(save_path)
    else:
        model.save(save_path)
    
    return save_path 