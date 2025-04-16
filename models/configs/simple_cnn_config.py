#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple CNN model configuration.

This module defines the configuration for the Simple CNN model.
"""

from models.configs.base_config import BaseConfig

class SimpleCNNConfig(BaseConfig):
    """Configuration for the Simple CNN model."""
    
    def __init__(self):
        super().__init__()
        self.model_name = 'simple_cnn'
        self.batch_size = 128
        self.epochs = 15
        self.dropout_rate = 0.5
        self.use_batch_norm = False 