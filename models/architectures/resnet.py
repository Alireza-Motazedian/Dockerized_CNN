#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet-based CNN architecture for MNIST digit classification.

This module implements a smaller version of ResNet architecture
adapted for the MNIST dataset.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model

def residual_block(x, filters, kernel_size=3, stride=1, use_bias=True, name=None):
    """
    Create a residual block for a ResNet architecture.
    
    Args:
        x: Input tensor
        filters (int): Number of filters
        kernel_size (int): Size of the kernel
        stride (int): Stride for the convolution
        use_bias (bool): Whether to use bias
        name (str): Name prefix for the layers
        
    Returns:
        tf.Tensor: Output tensor
    """
    name_prefix = '' if name is None else name + '_'
    
    # Shortcut connection
    shortcut = x
    
    # First convolutional block
    x = Conv2D(filters, kernel_size, strides=stride, padding='same',
               use_bias=use_bias, name=name_prefix + 'conv1')(x)
    x = BatchNormalization(name=name_prefix + 'bn1')(x)
    x = Activation('relu', name=name_prefix + 'relu1')(x)
    
    # Second convolutional block
    x = Conv2D(filters, kernel_size, padding='same',
               use_bias=use_bias, name=name_prefix + 'conv2')(x)
    x = BatchNormalization(name=name_prefix + 'bn2')(x)
    
    # If dimensions change, apply a 1x1 convolution to shortcut
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same',
                          use_bias=use_bias, name=name_prefix + 'shortcut_conv')(shortcut)
        shortcut = BatchNormalization(name=name_prefix + 'shortcut_bn')(shortcut)
    
    # Add shortcut to output
    x = Add(name=name_prefix + 'add')([x, shortcut])
    x = Activation('relu', name=name_prefix + 'relu2')(x)
    
    return x

def create_resnet(input_shape=(28, 28, 1), num_classes=10, blocks_per_group=2):
    """
    Create a ResNet model for MNIST digit classification.
    
    Args:
        input_shape (tuple): Input shape of the images
        num_classes (int): Number of output classes
        blocks_per_group (int): Number of residual blocks per group
        
    Returns:
        tf.keras.Model: A compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial convolutional layer
    x = Conv2D(32, 3, strides=1, padding='same', use_bias=True, name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='relu1')(x)
    
    # First group of residual blocks (32 filters)
    for i in range(blocks_per_group):
        x = residual_block(x, 32, name=f'group1_block{i+1}')
    
    # Second group of residual blocks (64 filters)
    x = residual_block(x, 64, stride=2, name='group2_block1')
    for i in range(1, blocks_per_group):
        x = residual_block(x, 64, name=f'group2_block{i+1}')
    
    # Third group of residual blocks (128 filters)
    x = residual_block(x, 128, stride=2, name='group3_block1')
    for i in range(1, blocks_per_group):
        x = residual_block(x, 128, name=f'group3_block{i+1}')
    
    # Global average pooling and output layer
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    outputs = Dense(num_classes, activation='softmax', name='fc')(x)
    
    # Create model
    model = Model(inputs, outputs, name='resnet_mnist')
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model 