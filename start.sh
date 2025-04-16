#!/bin/bash

# Use relative paths for directories since we're using volume mounting
# Check if data directories exist
echo "Creating data directories if they don't exist..."
mkdir -p ./data/mnist
mkdir -p ./data/mnist_samples

# Check if models directory exists
echo "Creating models directory if it doesn't exist..."
mkdir -p ./models

# Check if figures directory exists
echo "Creating figures directory if it doesn't exist..."
mkdir -p ./figures

# Run Jupyter lab by default
echo "Starting Jupyter Lab..."
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
