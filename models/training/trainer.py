#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer class for model training.

This module implements a Trainer class that handles the training
process for MNIST CNN models.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

from models.training.callbacks import LearningRateSchedulerCallback

class Trainer:
    """
    Trainer class for model training.
    
    Args:
        model: Keras model to train
        config: Training configuration
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.history = None
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.config.checkpoint_path, exist_ok=True)
    
    def train(self, train_data, val_data, epochs=None):
        """
        Train the model.
        
        Args:
            train_data: Training data (can be tuple of (x_train, y_train) or tf.data.Dataset)
            val_data: Validation data (can be tuple of (x_val, y_val) or tf.data.Dataset)
            epochs (int, optional): Number of epochs to train for (defaults to config.epochs)
            
        Returns:
            History object with training history
        """
        # Determine the number of epochs
        if epochs is None:
            epochs = self.config.epochs
        
        # Prepare callbacks
        callbacks = self._create_callbacks(epochs)
        
        # Train the model using the appropriate method based on data type
        if isinstance(train_data, tf.data.Dataset):
            self.history = self._train_with_dataset(train_data, val_data, epochs, callbacks)
        else:
            self.history = self._train_with_numpy(train_data, val_data, epochs, callbacks)
        
        return self.history
    
    def _train_with_dataset(self, train_dataset, val_dataset, epochs, callbacks):
        """Train the model using tf.data.Dataset."""
        return self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    
    def _train_with_numpy(self, train_data, val_data, epochs, callbacks):
        """Train the model using numpy arrays."""
        x_train, y_train = train_data
        x_val, y_val = val_data
        
        # Use data augmentation if specified in config
        if self.config.use_data_augmentation:
            return self._train_with_augmentation(
                x_train, y_train, x_val, y_val, epochs, callbacks
            )
        else:
            return self.model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                batch_size=self.config.batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
    
    def _train_with_augmentation(self, x_train, y_train, x_val, y_val, epochs, callbacks):
        """Train the model with data augmentation."""
        # Create data generator for augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False
        )
        
        # Fit data generator on training data
        datagen.fit(x_train)
        
        # Train the model with data generator
        return self.model.fit(
            datagen.flow(x_train, y_train, batch_size=self.config.batch_size),
            validation_data=(x_val, y_val),
            steps_per_epoch=len(x_train) // self.config.batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    
    def _create_callbacks(self, epochs):
        """Create callbacks for training."""
        callbacks = []
        
        # Learning rate scheduler
        if self.config.learning_rate_schedule != 'constant':
            lr_scheduler = LearningRateSchedulerCallback(
                initial_lr=self.config.initial_learning_rate,
                schedule=self.config.learning_rate_schedule,
                decay_steps=self.config.lr_decay_steps,
                decay_rate=self.config.lr_decay_rate,
                total_epochs=epochs
            )
            callbacks.append(lr_scheduler)
        
        # Early stopping
        if self.config.early_stopping:
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                mode='max',
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Model checkpointing
        checkpoint_path = os.path.join(
            self.config.checkpoint_path,
            f"{self.config.model_name}_{{epoch:02d}}_{{val_accuracy:.4f}}.h5"
        )
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=self.config.save_best_only,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def plot_training_history(self, figsize=(12, 8), save_path=None):
        """
        Plot training history.
        
        Args:
            figsize (tuple): Figure size
            save_path (str, optional): Path to save figure
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet.")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True)
        ax1.legend(loc='lower right')
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        ax2.legend(loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if specified
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def save_model(self, save_path, save_weights_only=False):
        """
        Save the model.
        
        Args:
            save_path (str): Path to save the model
            save_weights_only (bool): Whether to save only weights
            
        Returns:
            str: The save path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if save_weights_only:
            self.model.save_weights(save_path)
        else:
            self.model.save(save_path)
        
        print(f"Model saved to {save_path}")
        return save_path 