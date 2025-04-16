#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeNet-5 model configuration.

This module defines the configuration for the LeNet-5 model.
"""

from models.configs.base_config import BaseConfig

class LeNet5Config(BaseConfig):
    """Configuration for the LeNet-5 model."""
    
    def __init__(self):
        super().__init__()
        self.model_name = 'lenet5'
        self.batch_size = 64
        self.epochs = 20
        self.optimizer = 'sgd'
        self.initial_learning_rate = 0.01
        self.use_batch_norm = False 