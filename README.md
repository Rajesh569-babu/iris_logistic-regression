# Logistic Regression on Iris (Colab-ready)

A gentle, end-to-end notebook for **Logistic Regression** using the classic **Iris** dataset.

We will:
1. Load the data
2. Explore and visualize
3. Split into train/test
4. Train Logistic Regression
5. Evaluate with accuracy, confusion matrix, classification report
6. Try a few test-time predictions

**Tip:** Run cells from top to bottom. If you get a warning about convergence, we set a higher `max_iter` later.
# Logistic Regression on Iris Dataset

This notebook provides a step-by-step guide to performing Logistic Regression on the classic Iris dataset using Python and scikit-learn. It covers data loading, exploration, visualization, model training, evaluation, and making predictions.

## Table of Contents

1.  Setup
2.  Load the Iris dataset
3.  Quick EDA & Visualization
4.  Train/Test Split
5.  Feature Scaling (Optional)
6.  Train Logistic Regression
7.  Evaluate on Test Set
8.  Try Some Predictions
9.  Save Model (Optional)
10. Homework Ideas (for students)

## Requirements

-   Python 3.6+
-   `numpy`
-   `pandas`
-   `matplotlib`
-   `scikit-learn`
-   `joblib`

These libraries are commonly available in Colab environments.

## How to Run

1.  Open the notebook in Google Colab or any Jupyter-compatible environment.
2.  Run the cells sequentially from top to bottom.

## Notebook Overview

-   **Setup:** Imports necessary libraries and checks the Python version.
-   **Load Data:** Loads the Iris dataset using `sklearn.datasets.load_iris`.
-   **EDA & Visualization:** Provides basic statistical insights and visualizes feature distributions and relationships.
-   **Train/Test Split:** Splits the data into training and testing sets using stratified sampling.
-   **Feature Scaling:** Scales the features using `StandardScaler` (optional but recommended for Logistic Regression).
-   **Model Training:** Trains a `LogisticRegression` model on the scaled training data.
-   **Evaluation:** Evaluates the model's performance using accuracy, a confusion matrix, and a classification report.
-   **Predictions:** Demonstrates how to make predictions on new data samples.
-   **Save Model:** Shows how to save the trained model and scaler using `joblib`.
-   **Homework Ideas:** Suggests extensions and exercises for further learning.

## Dataset

The Iris dataset is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in 1936. It consists of 150 samples from three species of Iris (`setosa`, `versicolor`, `virginica`), with four features measured: sepal length, sepal width, petal length, and petal width.

## Model Details

-   **Model:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)
-   **Solver:** `lbfgs` (default for multinomial classification)
-   **`max_iter`:** Set to 1000 to ensure convergence.
-   **`multi_class`:** Set to `auto` (which selects `multinomial` for this dataset).

## Evaluation Metrics

-   **Accuracy:** Overall percentage of correctly classified instances.
-   **Confusion Matrix:** A table showing the counts of true positive, true negative, false positive, and false negative predictions.
-   **Classification Report:** Provides precision, recall, and f1-score for each class, as well as support.

## License

This notebook is provided under the [MIT License](https://opensource.org/licenses/MIT).
