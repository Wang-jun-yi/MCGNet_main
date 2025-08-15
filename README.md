# MCGNet_main
MCGNet: Multi-scale feature fusion architecture based on convolutional neural network and graph neural network for hyperspectral image classification
Repository Description:
This repository contains the implementation of MCGNet, a multi-scale feature fusion architecture designed for hyperspectral image (HSI) classification. MCGNet combines convolutional neural networks (CNNs) and graph convolutional networks (GCNs) to address the challenges in HSI classification, specifically the inability of traditional CNNs to capture long-range dependencies and the computational overhead of using GCNs directly on pixel graphs.

Key Features:
SNS (Spectral Noise Suppression): Enhances the signal-to-noise ratio of spectral features, improving feature quality for downstream processing.

LSE (Local Spectral Feature Extraction): Uses deep separable convolutions to extract local spectral-spatial features, providing a detailed understanding of local image characteristics.

SGC (Superpixel-level Graph Convolution): Captures dependencies between object regions using graph convolutions on superpixel maps, reducing computational complexity while preserving structural information.

PGC (Pixel-level Graph Convolution): Constructs an adaptive sparse pixel map based on spectral and spatial similarity, enabling the capture of fine-grained non-local relationships and irregular object boundaries.

Installation Instructions:

To use the MCGNet model for hyperspectral image classification, clone the repository and install the necessary dependencies:

git clone https://github.com/yourusername/MCGNet.git

Usage:

The repository includes scripts to train and evaluate the MCGNet model on three public hyperspectral datasets: Indian Pines, Pavia University, and Salinas. 
Contributions:

Novel integration of CNNs and GCNs for multi-scale feature modeling.

Introduction of new modules for spectral noise suppression, local spectral feature extraction, and efficient graph convolutions.

Comprehensive evaluation on multiple public hyperspectral image datasets.
