#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple CNN architecture for MNIST digit classification.

This module implements a basic CNN with two convolutional layers,
followed by max pooling and fully connected layers.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_simple_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a simple CNN model for MNIST digit classification.
    
    Architecture:
    - Conv2D(32, 3x3) -> ReLU -> MaxPooling(2x2)
    - Conv2D(64, 3x3) -> ReLU -> MaxPooling(2x2)
    - Flatten -> Dense(128) -> ReLU -> Dropout(0.5) -> Dense(10) -> Softmax
    
    Args:
        input_shape (tuple): Input shape of the images
        num_classes (int): Number of output classes
        
    Returns:
        tf.keras.Model: A compiled Keras model
    """
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model 