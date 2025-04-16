#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base configuration for model training.

This module defines the base configuration class with
default parameters for model training.
"""

class BaseConfig:
    """Base configuration class for model training."""
    
    def __init__(self):
        # Data parameters
        self.batch_size = 64
        self.use_data_augmentation = True
        
        # Training parameters
        self.epochs = 20
        self.initial_learning_rate = 0.001
        self.learning_rate_schedule = 'step'  # ['constant', 'step', 'exponential', 'cosine']
        self.lr_decay_steps = 5
        self.lr_decay_rate = 0.5
        
        # Regularization
        self.l2_weight_decay = 0.0001
        self.dropout_rate = 0.5
        self.use_batch_norm = True
        
        # Optimizer parameters
        self.optimizer = 'adam'  # ['adam', 'sgd', 'rmsprop']
        self.momentum = 0.9  # For SGD
        
        # Checkpoint parameters
        self.save_best_only = True
        self.checkpoint_path = 'models/checkpoints/'
        
        # Early stopping parameters
        self.early_stopping = True
        self.patience = 5
        self.min_delta = 0.001 