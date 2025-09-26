# Task 4: Classification with Logistic Regression

This project is **ELEVATE LABS Internship – Task 4**  
It demonstrates how to implement a **binary classifier** using Logistic Regression on the **Breast Cancer Wisconsin dataset**.

# Files

- BreastCancer.csv -> Raw dataset (from sklearn.datasets)
- LogisticRegressionClassifier.py -> Python script implementing logistic regression

# Requirements

Install dependencies:
pip install pandas numpy matplotlib scikit-learn

# Steps Performed

- Load Breast Cancer Wisconsin dataset
- Split data into train-test sets
- Standardize features using StandardScaler
- Fit a Logistic Regression model using sklearn.linear_model
- Evaluate with:
    - Confusion Matrix
    - Precision, Recall, F1-score
    - ROC curve and ROC-AUC
- Tune classification threshold and observe changes
- Plot sigmoid function to explain probability mapping