#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model registry of available models.

This module provides a registry of available models and utility functions
to create and manage them.
"""

from models.model_factory import create_model

class ModelRegistry:
    """Registry of available models."""
    
    # Dictionary mapping model names to info
    _models = {
        'simple_cnn': {
            'name': 'Simple CNN',
            'description': 'A basic CNN model with two convolutional layers',
            'module': 'models.architectures.simple_cnn',
            'params': {
                'input_shape': (28, 28, 1),
                'num_classes': 10
            },
            'config_module': 'models.configs.simple_cnn_config',
            'config_class': 'SimpleCNNConfig'
        },
        'lenet5': {
            'name': 'LeNet-5',
            'description': 'Classic LeNet-5 architecture by Yann LeCun',
            'module': 'models.architectures.lenet5',
            'params': {
                'input_shape': (28, 28, 1),
                'num_classes': 10
            },
            'config_module': 'models.configs.lenet5_config',
            'config_class': 'LeNet5Config'
        },
        'resnet': {
            'name': 'ResNet',
            'description': 'A smaller version of ResNet adapted for MNIST',
            'module': 'models.architectures.resnet',
            'params': {
                'input_shape': (28, 28, 1),
                'num_classes': 10,
                'blocks_per_group': 2
            },
            'config_module': 'models.configs.resnet_config',
            'config_class': 'ResNetConfig'
        },
        'custom_cnn': {
            'name': 'Custom CNN',
            'description': 'A custom CNN architecture optimized for MNIST',
            'module': 'models.architectures.custom_cnn',
            'params': {
                'input_shape': (28, 28, 1),
                'num_classes': 10,
                'dropout_rate': 0.4
            },
            'config_module': 'models.configs.custom_cnn_config',
            'config_class': 'CustomCNNConfig'
        }
    }
    
    @classmethod
    def list_models(cls):
        """
        List all available models.
        
        Returns:
            list: List of available model names
        """
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_name):
        """
        Get information about a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Model information
        """
        model_name = model_name.lower()
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        return cls._models[model_name]
    
    @classmethod
    def create_model(cls, model_name, **kwargs):
        """
        Create a model by name with optional parameters.
        
        Args:
            model_name (str): Name of the model
            **kwargs: Additional parameters to override defaults
            
        Returns:
            tf.keras.Model: Created model
        """
        model_name = model_name.lower()
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        # Get default parameters
        params = cls._models[model_name]['params'].copy()
        
        # Override with provided parameters
        params.update(kwargs)
        
        # Create and return the model
        return create_model(model_name, **params)
    
    @classmethod
    def create_config(cls, model_name):
        """
        Create a configuration object for a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            object: Configuration object
        """
        model_name = model_name.lower()
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        # Get config information
        config_info = cls._models[model_name]
        config_module_name = config_info['config_module']
        config_class_name = config_info['config_class']
        
        # Import the module
        import importlib
        config_module = importlib.import_module(config_module_name)
        
        # Get the config class
        config_class = getattr(config_module, config_class_name)
        
        # Create and return the config
        return config_class() 