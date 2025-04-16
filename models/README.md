<div style="font-size:2em; font-weight:bold; text-align:center; margin-top:20px;">Models Directory</div>

## Table of Contents
<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1-overview"><i><b>1. Overview</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#2-directory-structure"><i><b>2. Directory Structure</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#3-model-architectures"><i><b>3. Model Architectures</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#4-training-configurations"><i><b>4. Training Configurations</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#5-model-evaluation"><i><b>5. Model Evaluation</b></i></a></div>
&nbsp;

<div>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#6-usage-examples"><i><b>6. Usage Examples</b></i></a></div>
&nbsp;

## 1. Overview

The `models` directory contains implementations of Convolutional Neural Network (CNN) architectures designed for the MNIST handwritten digit classification task. Each model is implemented as a separate module, with supporting files for configuration, training, and evaluation. This documentation provides details on the available architectures, training configurations, evaluation metrics, and usage examples.

## 2. Directory Structure

```
Folder PATH listing
.
+---architectures              <-- Model architecture implementations
|   +---__init__.py            <-- Package initialization
|   +---custom_cnn.py          <-- Custom CNN implementation
|   +---lenet5.py              <-- LeNet-5 CNN architecture
|   +---resnet.py              <-- ResNet-based architecture
|   +---simple_cnn.py          <-- Simple CNN implementation
|
+---checkpoints                <-- Model checkpoints during training
|
+---configs                    <-- Model configuration files
|   +---__init__.py            <-- Package initialization
|   +---base_config.py         <-- Base configuration class
|   +---custom_cnn_config.py   <-- Custom CNN configuration
|   +---lenet5_config.py       <-- LeNet-5 configuration
|   +---resnet_config.py       <-- ResNet configuration
|   +---simple_cnn_config.py   <-- Simple CNN configuration
|
+---evaluation                 <-- Model evaluation utilities
|   +---__init__.py            <-- Package initialization
|   +---evaluator.py           <-- Evaluator class for model evaluation
|   +---metrics.py             <-- Custom metrics for evaluation
|
+---pretrained                 <-- Pretrained model weights
|   +---lenet5                 <-- LeNet-5 pretrained weights
|   +---resnet                 <-- ResNet pretrained weights
|   +---simple_cnn             <-- Simple CNN pretrained weights
|
+---training                   <-- Model training utilities
|   +---__init__.py            <-- Package initialization
|   +---callbacks.py           <-- Custom callbacks for training
|   +---trainer.py             <-- Trainer class for model training
|
+---__init__.py                <-- Package initialization
+---mnist_cnn_best.h5          <-- Best model based on validation accuracy
+---mnist_cnn_final.h5         <-- Final trained model
+---model_factory.py           <-- Factory for creating model instances
+---model_registry.py          <-- Registry of available models
+---README.md                  <-- This documentation file
+---simple_cnn_final.h5        <-- Trained simple CNN model
```

## 3. Model Architectures

The project includes several CNN architectures designed for MNIST digit classification:

### 3.1 Simple CNN (`simple_cnn.py`)

A basic CNN model with two convolutional layers for beginners and baseline performance.

```python
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
```

**Key Features:**
- Simple architecture with 2 convolutional layers
- Dropout regularization to prevent overfitting
- Suited for quick prototyping and baseline establishment
- ~400K parameters

### 3.2 LeNet-5 (`lenet5.py`)

An implementation of the classic LeNet-5 architecture by Yann LeCun, adapted for MNIST.

```python
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
    # Implementation...
```

**Key Features:**
- Classic architecture that pioneered CNNs for digit recognition
- Uses tanh activation functions and average pooling
- Historically significant architecture
- ~60K parameters

### 3.3 ResNet-based (`resnet.py`)

A smaller version of ResNet adapted for MNIST digit classification.

```python
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
    # Implementation...


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
    # Implementation...
```

**Key Features:**
- Residual connections to enable deeper networks
- Batch normalization for faster convergence
- Higher capacity for more complex patterns
- ~1.2M parameters

### 3.4 Custom CNN (`custom_cnn.py`)

A custom CNN architecture optimized for MNIST through experimentation.

```python
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
    # Implementation...
```

**Key Features:**
- Deeper architecture with 5 convolutional layers
- LeakyReLU activations for better gradient flow
- Batch normalization for faster convergence
- Dropout at multiple levels for strong regularization
- ~800K parameters

## 4. Training Configurations

Model training configurations are defined in the `configs` directory:

### 4.1 Base Configuration (`base_config.py`)

```python
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
        self.checkpoint_path = 'checkpoints/'
        
        # Early stopping parameters
        self.early_stopping = True
        self.patience = 5
        self.min_delta = 0.001
```

### 4.2 Model-Specific Configurations

Each model has its own configuration that inherits from the base configuration and overrides specific parameters:

```python
class SimpleCNNConfig(BaseConfig):
    """Configuration for the Simple CNN model."""
    
    def __init__(self):
        super().__init__()
        self.model_name = 'simple_cnn'
        self.batch_size = 128
        self.epochs = 15
        self.dropout_rate = 0.5
        self.use_batch_norm = False
        

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
```

## 5. Model Evaluation

The `evaluation` directory contains utilities for evaluating trained models:

### 5.1 Evaluator (`evaluator.py`)

The `Evaluator` class provides methods for evaluating models on test data:

```python
class Evaluator:
    """
    Class for evaluating trained models on test data.
    
    Args:
        model: Trained Keras model
        test_data: Tuple of (X_test, y_test) or tf.data.Dataset
    """
    
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
    
    def evaluate(self):
        """
        Evaluate the model on test data.
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Implementation...
    
    def evaluate_per_class(self):
        """
        Evaluate the model performance per class.
        
        Returns:
            pd.DataFrame: DataFrame with per-class metrics
        """
        # Implementation...
    
    def confusion_matrix(self, normalize=False):
        """
        Generate confusion matrix for the model.
        
        Args:
            normalize (bool): Whether to normalize the confusion matrix
            
        Returns:
            np.ndarray: Confusion matrix
        """
        # Implementation...
    
    def plot_confusion_matrix(self, normalize=False):
        """
        Plot confusion matrix for the model.
        
        Args:
            normalize (bool): Whether to normalize the confusion matrix
            
        Returns:
            matplotlib.figure.Figure: Confusion matrix plot
        """
        # Implementation...
    
    def misclassified_examples(self, num_examples=10):
        """
        Get examples of misclassified images.
        
        Args:
            num_examples (int): Number of examples to return
            
        Returns:
            tuple: (misclassified_images, true_labels, predicted_labels)
        """
        # Implementation...
```

### 5.2 Custom Metrics (`metrics.py`)

Custom metrics for model evaluation:

```python
def precision_recall_f1(y_true, y_pred):
    """
    Calculate precision, recall, and F1 score for each class.
    
    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        
    Returns:
        tuple: (precision, recall, f1)
    """
    # Implementation...


def top_k_accuracy(y_true, y_pred, k=3):
    """
    Calculate top-k accuracy.
    
    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted probabilities
        k (int): Number of top predictions to consider
        
    Returns:
        float: Top-k accuracy
    """
    # Implementation...
```

## 6. Usage Examples

### 6.1 Creating and Training a Model

```python
from models.model_factory import create_model
from models.configs.simple_cnn_config import SimpleCNNConfig
from models.training.trainer import Trainer
from utils.data.loader import load_mnist_data, create_tf_dataset

# Load data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist_data()

# Create TensorFlow datasets
train_dataset = create_tf_dataset(X_train, y_train, batch_size=128, shuffle=True, augment=True)
val_dataset = create_tf_dataset(X_val, y_val, batch_size=128, shuffle=False)

# Create model
config = SimpleCNNConfig()
model = create_model('simple_cnn', input_shape=(28, 28, 1), num_classes=10)

# Create trainer
trainer = Trainer(model, config)

# Train model
history = trainer.train(train_dataset, val_dataset, epochs=15)
```

### 6.2 Evaluating a Model

```python
from models.evaluation.evaluator import Evaluator
from utils.visualization.plotting import plot_training_history

# Plot training history
plot_training_history(history)

# Create evaluator
evaluator = Evaluator(model, (X_test, y_test))

# Evaluate model
metrics = evaluator.evaluate()
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Loss: {metrics['loss']:.4f}")

# Plot confusion matrix
evaluator.plot_confusion_matrix(normalize=True)

# Get per-class metrics
class_metrics = evaluator.evaluate_per_class()
print(class_metrics)

# Examine misclassified examples
misclassified = evaluator.misclassified_examples(num_examples=10)
```

### 6.3 Using the Model Registry

```python
from models.model_registry import ModelRegistry

# List available models
available_models = ModelRegistry.list_models()
print(f"Available models: {available_models}")

# Get model info
model_info = ModelRegistry.get_model_info('lenet5')
print(f"LeNet-5 model info: {model_info}")

# Create a registered model
model = ModelRegistry.create_model('lenet5', input_shape=(28, 28, 1), num_classes=10)
```

### 6.4 Loading a Pretrained Model

```python
from models.model_factory import load_pretrained_model

# Load pretrained model
model = load_pretrained_model('simple_cnn', 'models/pretrained/simple_cnn/best_model.h5')

# Make predictions
predictions = model.predict(X_test)
```
