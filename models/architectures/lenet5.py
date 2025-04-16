#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeNet-5 CNN architecture for MNIST digit classification.

This module implements the classic LeNet-5 architecture by Yann LeCun,
adapted for the MNIST dataset.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

def create_lenet5(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a LeNet-5 model for MNIST digit classification.
    
    Architecture:
    - Conv2D(6, 5x5) -> tanh -> AveragePooling(2x2)
    - Conv2D(16, 5x5) -> tanh -> AveragePooling(2x2)
    - Flatten -> Dense(120) -> tanh -> Dense(84) -> tanh -> Dense(10) -> Softmax
    
    Args:
        input_shape (tuple): Input shape of the images
        num_classes (int): Number of output classes
        
    Returns:
        tf.keras.Model: A compiled Keras model
    """
    model = Sequential([
        # First convolutional block
        Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=input_shape),
        AveragePooling2D(pool_size=(2, 2)),
        
        # Second convolutional block
        Conv2D(16, kernel_size=(5, 5), activation='tanh'),
        AveragePooling2D(pool_size=(2, 2)),
        
        # Fully connected layers
        Flatten(),
        Dense(120, activation='tanh'),
        Dense(84, activation='tanh'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model 