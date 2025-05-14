# SMAI Assignment 3 - IIIT Hyderabad

This repository contains the complete implementation of Assignment 3 for the course **Statistical Methods in AI (SMAI)**, offered at **IIIT-Hyderabad**, under the guidance of **Prof. Vineet Gandhi**.

This assignment focuses on **Age Prediction from Facial Images** using deep learning models. It involves training both a CNN from scratch and a fine-tuned ResNet-18 to perform regression on age labels extracted from face image filenames.

---

## 🧠 Tasks

The assignment is divided into two major tasks:

### 🔹 CNN from Scratch
- Implemented a Convolutional Neural Network from scratch using PyTorch
- Input: Facial images from the UTKFace dataset
- Target: Age regression
- Training-Test split: 80% training and 20% test
- Loss Function: Mean Squared Error (MSE)
- Evaluation metric: Final MSE on the test set

### 🔹 Fine-tuned ResNet-18
- Used a pretrained ResNet-18 model from PyTorch's torchvision library
- Modified the final fully connected layer to predict a **single continuous value (age)**
- Trained on the same dataset and configuration
- Reported the **MSE loss** on test data
- Compared performance with the CNN model

---

## 📁 Structure

Both models are implemented in a same Jupyter Notebook. Annotations and dataset-specific code are included in the repository for transparency and reproducibility.


---

## 🧪 Dataset

- Dataset: [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- Image naming format: `[age]_[gender]_[race]_[datetime].jpg`
- Used 55 unique images as per assignment instruction
- Converted filename labels into usable CSV format for model training

---

## 📊 Results

| Model                | Test MSE Loss |
|---------------------|---------------|
| CNN from Scratch     | *82.0894* |
| Fine-tuned ResNet-18 | *56.4122* |

### ✅ Best Model
**Model:** [CNN / ResNet-18]  
**Reason:** Achieved lower MSE loss on test data with better generalization.

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- NumPy, Matplotlib, Seaborn
- PyTorch
- Torchvision

---

## 📁 Submission Instructions Followed

✅ Implemented in Python using Jupyter Notebook  
✅ Used PyTorch as allowed for both models  
✅ Annotated and uploaded 55 **unique** images  
✅ Maintained the required format (image resolution, aspect ratio, and CSV layout)  
✅ Included observations and comparisons between models  
✅ Ensured no duplication or reuse of other students' images

---

## 👨‍💻 Author

**Implemented by:** P. Jakeer Hussain  
**Institute:** IIIT Hyderabad  
**Course:** Statistical Methods in AI (Spring 2025)

---

> Note: This assignment will contribute to the final mini-project dataset. All annotations were double-checked to ensure correctness.
