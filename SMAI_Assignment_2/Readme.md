# SMAI Assignment 2 - IIIT Hyderabad

This repository contains the complete implementation of **Assignment 2** for the course *Statistical Methods in AI* (SMAI), offered at IIIT-Hyderabad, under the guidance of **Prof. Vineet Gandhi**.

The assignment demonstrates a comprehensive understanding of core machine learning models including:
- Multi-Layer Perceptron (MLP)
- Gaussian Mixture Models (GMM)
- Principal Component Analysis (PCA)
- Autoencoder
- Variational Autoencoder (VAE)

## 📁 Structure

Each section of the assignment is implemented in a dedicated Jupyter Notebook, with code and visualizations integrated with markdown cells for clear documentation and results.

### 🔹 Multi-Layer Perceptron (MLP)
1. **Symbol Classification (Multi-class)**
   - Implemented MLP from scratch
   - 10-fold cross-validation
   - Various activation functions (Sigmoid, Tanh, ReLU) and optimizers (SGD, Batch, Mini-Batch)
   - Dataset: Handwritten historical symbols

2. **House Price Prediction in Bangalore (Regression)**
   - Data preprocessing (missing values, outliers)
   - MLP regression model from scratch
   - Evaluation metrics: MSE, RMSE, R²

3. **News Article Classification (Multi-label)**
   - Text preprocessing and TF-IDF implementation from scratch
   - Multi-label binarization
   - Custom MLP architecture for multi-label classification

---

### 🔹 Gaussian Mixture Model (GMM)
- GMM implemented from scratch
- Medical image segmentation (Gray Matter, White Matter, CSF)
- Analysis using ITK-SNAP
- Intensity vs Frequency and GMM distribution plots

---

### 🔹 Principal Component Analysis (PCA)
- PCA implementation from scratch (Eigen decomposition)
- Dimensionality reduction on MNIST
- Explained variance analysis and reconstruction
- MLP classification with and without PCA

---

### 🔹 Autoencoder
- PyTorch-based implementation
- Anomaly detection using reconstruction error
- Evaluation using Precision, Recall, F1-score
- Analysis of various bottleneck sizes using ROC-AUC

---

### 🔹 Variational Autoencoder (VAE)
- Implemented VAE using PyTorch
- Latent space visualization
- Experiments with and without reconstruction/KL loss
- Sampling from latent space grids (BCE and MSE losses)

---

## 🛠️ Technologies Used
- Python
- Jupyter Notebook
- NumPy, Matplotlib, scikit-learn
- PyTorch
- Torchvision
- ITK-SNAP (for image segmentation visualization)

## 📊 Results
- Extensive visualizations and metric tracking included within notebooks
- Plots: Training curves, accuracy trends, ROC curves, PCA projections, etc.
- Each model’s performance and tuning process is well-documented

---

## 📁 Submission Instructions Followed
- ✅ Implemented required parts from scratch
- ✅ Used PyTorch only where allowed (AE & VAE)
- ✅ Included all observations, plots, and analysis within the notebooks

---

## 🧠 Author
- Dr. Sam (Roleplayed)
- Implemented by: *[Your Name]* 
- Institute: IIIT Hyderabad 
- Course: Statistical Methods in AI (Spring 2025)

---

## 📌 Note
For educational use only. Do not reuse without proper citation.

