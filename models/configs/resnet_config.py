#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet model configuration.

This module defines the configuration for the ResNet model.
"""

from models.configs.base_config import BaseConfig

class ResNetConfig(BaseConfig):
    """Configuration for the ResNet model."""
    
    def __init__(self):
        super().__init__()
        self.model_name = 'resnet'
        self.batch_size = 64
        self.epochs = 25
        self.optimizer = 'adam'
        self.initial_learning_rate = 0.0005
        self.learning_rate_schedule = 'exponential'
        self.lr_decay_steps = 4
        self.lr_decay_rate = 0.7
        self.dropout_rate = 0.3
        self.use_batch_norm = True
        self.blocks_per_group = 2 