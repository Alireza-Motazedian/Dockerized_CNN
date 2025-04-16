#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom CNN model configuration.

This module defines the configuration for the Custom CNN model.
"""

from models.configs.base_config import BaseConfig

class CustomCNNConfig(BaseConfig):
    """Configuration for the Custom CNN model."""
    
    def __init__(self):
        super().__init__()
        self.model_name = 'custom_cnn'
        self.batch_size = 128
        self.epochs = 25
        self.optimizer = 'adam'
        self.initial_learning_rate = 0.001
        self.learning_rate_schedule = 'cosine'
        self.dropout_rate = 0.4
        self.use_batch_norm = True
        self.early_stopping = True
        self.patience = 10
        self.min_delta = 0.0005 