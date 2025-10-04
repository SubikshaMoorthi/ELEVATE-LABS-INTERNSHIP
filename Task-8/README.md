## Task 8: Clustering with K-Means

This project is **ELEVATE LABS Internship – Task 8**
It demonstrates how to perform unsupervised clustering using the **K-Means algorithm** on the Mall Customer Segmentation dataset.

# Files

- Mall_Customers.csv → Raw dataset (uploaded CSV)
- KMeans.py → Python script implementing K-Means clustering
- Mall_Customers_Clustered.csv → Output file with assigned cluster labels

# Requirements

Install dependencies:
pip install pandas numpy matplotlib scikit-learn

# Steps Performed

- Load and inspect the Mall Customer dataset
- Extract relevant features (Annual Income, Spending Score)
- Standardize data using StandardScaler
- Apply Elbow Method to find the optimal number of clusters
- Train K-Means model with chosen K
- Visualize the clusters and centroids
- Evaluate model performance using Silhouette Score