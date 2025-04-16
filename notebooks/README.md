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
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#31-cnn_mnist-dataipynb">3.1. CNN_MNIST-data.ipynb</a><br>
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

This directory contains Jupyter notebooks for interactive exploration and learning about CNN-based MNIST digit recognition.

## 2. Directory Contents

```
Folder PATH listing
.
+---CNN_MNIST-data.ipynb       <-- Main notebook for the project
+---README.md                  <-- This documentation file
```

## 3. Notebook Descriptions

### 3.1 CNN_MNIST-data.ipynb

The main notebook for the project that covers the complete workflow:

- Loading and preprocessing the MNIST dataset
- Building a CNN model for digit recognition
- Training the model with appropriate techniques
- Evaluating model performance
- Visualizing results and learned features
- Comparing CNN with traditional neural networks

This notebook is designed to be self-contained and educational, with detailed explanations of each step.

## 4. Using the Notebooks

1. Make sure the Docker container is running:
   ```bash
   docker-compose up
   ```

2. Access Jupyter Lab in your browser at http://localhost:8888

3. Navigate to the notebooks directory and open the desired notebook

4. Execute cells sequentially (Shift+Enter) to follow the workflow

## 5. Learning Objectives

Through these notebooks, you will learn:
- How to implement a CNN using TensorFlow/Keras
- The advantages of CNNs over traditional neural networks for image tasks
- Techniques for training and evaluating deep learning models
- Methods for visualizing and interpreting CNN feature maps
- Best practices for deep learning model development
