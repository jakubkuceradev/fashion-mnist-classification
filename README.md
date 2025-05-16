# Fashion MNIST Machine Learning & Neural Network Analysis

This repository contains two projects exploring different Machine Learning techniques applied to subsets of the **Fashion MNIST** dataset for classification tasks.

The goal is to showcase fundamental data science workflows, traditional ML models, dimensionality reduction, and deep learning using Python libraries like Scikit-learn and PyTorch.

The data (`train.csv`, `evaluate.csv`) consists of grayscale images (28x28 pixels for the first project, 32x32 pixels for the second) and corresponding labels.

## Projects Included

### 1. Binary Classification with Traditional ML & Dimensionality Reduction

*   **Notebook:** `fashion_mnist_traditional_analysis.ipynb`
*   **Description:** This project tackles a binary classification problem (distinguishing between two classes of clothing) using traditional Machine Learning models. It explores:
    *   Exploratory Data Analysis (EDA)
    *   Support Vector Machines (SVM)
    *   Naive Bayes Classifier
    *   Linear Discriminant Analysis (LDA)
    *   Hyperparameter tuning and model evaluation
    *   Dimensionality Reduction techniques (PCA, LLE) and their impact on model performance

### 2. Multi-Class Classification with Neural Networks

*   **Notebook:** `fashion_mnist_neural_networks_analysis.ipynb`
*   **Description:** This project addresses a multi-class classification problem (distinguishing between 10 classes of clothing) using Neural Networks implemented in PyTorch. It includes:
    *   Exploratory Data Analysis (EDA)
    *   Building and experimenting with Feedforward Neural Networks (FNNs)
    *   Building and experimenting with Convolutional Neural Networks (CNNs) - models specifically suited for image data
    *   Exploring different network architectures, preprocessing methods (Normalization, Standardization, Batch Normalization), optimizers (Adam, SGD), and regularization techniques (Dropout, L2)
    *   Model training, validation, and testing

---

**Note on Language:**

The detailed analysis and commentary within the Jupyter Notebook files (`.ipynb`) are written in **Czech**. The repository README and code comments are primarily in English.

---

## Technologies Used

*   **Python**
*   **Pandas** (Data handling)
*   **NumPy** (Numerical operations)
*   **Matplotlib** & **Seaborn** (Visualization)
*   **Scikit-learn** (Traditional ML models, preprocessing, evaluation, dimensionality reduction)
*   **PyTorch** (Deep Learning models)

## Repository Contents

*   `README.md`: This file.
*   `train.csv`: Training dataset.
*   `evaluate.csv`: Evaluation dataset (without labels).
*   `results.csv`: Prediction results.
*   `fashion_mnist_traditional_analysis.ipynb`: Notebook for the binary classification project.
*   `fashion_mnist_neural_networks_analysis.ipynb`: Notebook for the neural networks project.

## How to View and Run

1.  **View Notebooks:** Jupyter Notebooks (`.ipynb` files) can be viewed directly on GitHub.
2.  **Run Locally:**
    *   Clone this repository: `git clone [Repository URL]`
    *   Navigate to the repository directory.
    *   Ensure you have Python, Jupyter, and the required libraries installed (see Technologies Used section).
    *   Open the notebooks using JupyterLab or Jupyter Notebook: `jupyter lab` or `jupyter notebook`
    *   You can then run the code cells interactively.