<div style="font-size:2em; font-weight:bold; text-align:center; margin-top:20px;">Notebooks Directory</div>

## Table of Contents 
<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#1-overview"><i><b>1. Overview</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#2-directory-contents"><i><b>2. Directory Contents</b></i></a>
</div>
&nbsp;

<details>
  <summary><a href="#3-notebook-descriptions"><i><b>3. Notebook Descriptions</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#31-01_data_explorationipynb">3.1. 01_data_exploration.ipynb</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#32-02_model_trainingipynb">3.2. 02_model_training.ipynb</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#4-using-the-notebooks"><i><b>4. Using the Notebooks</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#5-learning-objectives"><i><b>5. Learning Objectives</b></i></a>
</div>
&nbsp;

## 1. Overview

This directory contains Jupyter notebooks for interactive exploration, learning, and implementation of CNN-based MNIST digit recognition.

## 2. Directory Contents

```
Folder PATH listing
.
+---01_data_exploration.ipynb  <-- Data exploration and visualization notebook
+---02_model_training.ipynb    <-- Model training and evaluation notebook
+---figures/                   <-- Generated visualizations
+---models/                    <-- Saved models and model outputs
+---README.md                  <-- This documentation file
```

## 3. Notebook Descriptions

### 3.1 01_data_exploration.ipynb

A comprehensive notebook dedicated to exploring and understanding the MNIST dataset:

- Loading and inspecting the MNIST dataset
- Visualizing sample images from each digit class
- Analyzing class distributions and balance
- Exploring image characteristics (pixel distributions, average digits)
- Examining digit variations and morphology
- Performing dimensionality reduction (t-SNE, PCA)
- Preprocessing the data for CNN modeling (normalization, reshaping)
- Creating and visualizing training/validation/test splits

This notebook provides a thorough foundation for understanding the data before model development.

### 3.2 02_model_training.ipynb

The model implementation and training notebook covering:

- Building a CNN model for digit recognition
- Training the model with appropriate techniques
- Evaluating model performance
- Visualizing results and learned features
- Comparing CNN with traditional neural networks
- Experimenting with different architectures and hyperparameters

This notebook demonstrates the complete modeling workflow from architecture design to final evaluation.

## 4. Using the Notebooks

1. Make sure the Docker container is running:
   ```bash
   docker-compose up
   ```

2. Access Jupyter Lab in your browser at http://localhost:8889

3. Navigate to the notebooks directory and open the desired notebook

4. Execute cells sequentially (Shift+Enter) to follow the workflow

## 5. Learning Objectives

Through these notebooks, you will learn:
- How to properly explore and visualize image datasets
- Data preprocessing techniques for deep learning
- How to implement a CNN using TensorFlow/Keras
- The advantages of CNNs over traditional neural networks for image tasks
- Techniques for training and evaluating deep learning models
- Methods for visualizing and interpreting CNN feature maps
- Best practices for deep learning model development
