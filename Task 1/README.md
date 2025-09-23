# Data Cleaning & Preprocessing

This project is ELEVATE LABS Internship – Task 1  
It demonstrates how to clean and preprocess the Titanic dataset to make it suitable for **Exploratory Data Analysis (EDA)** and **Machine Learning**.

# Files

- Titanic-Dataset.csv -> Raw dataset  
- titanic_cleaning.py -> Python script for preprocessing

# Requirements

Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn

# Steps Performed

* Load dataset (Titanic-Dataset.csv)
* Handle missing values
* Encode categorical features
* Scale numerical features using StandardScaler
* Outlier removal in Fare using IQR method
* Visualization → Boxplot of Fare before outlier removal
