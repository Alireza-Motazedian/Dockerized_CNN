#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom CNN architecture for MNIST digit classification.

This module implements a custom CNN architecture optimized for MNIST
through experimentation.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout, LeakyReLU

def create_custom_cnn(input_shape=(28, 28, 1), num_classes=10, dropout_rate=0.4):
    """
    Create a custom CNN model optimized for MNIST through experimentation.
    
    Architecture:
    - Conv2D(32, 3x3) -> LeakyReLU -> BatchNorm
    - Conv2D(32, 3x3) -> LeakyReLU -> MaxPooling(2x2) -> BatchNorm -> Dropout
    - Conv2D(64, 3x3) -> LeakyReLU -> BatchNorm
    - Conv2D(64, 3x3) -> LeakyReLU -> MaxPooling(2x2) -> BatchNorm -> Dropout
    - Conv2D(128, 3x3) -> LeakyReLU -> MaxPooling(2x2) -> BatchNorm -> Dropout
    - Flatten -> Dense(128) -> LeakyReLU -> BatchNorm -> Dropout -> Dense(10) -> Softmax
    
    Args:
        input_shape (tuple): Input shape of the images
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: A compiled Keras model
    """
    model = Sequential()
    
    # First convolutional block with dual Conv2D layers
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Second convolutional block with dual Conv2D layers
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Third convolutional block
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model 