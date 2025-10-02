# SVM Classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear SVM
linear_svm = SVC(kernel="linear", C=1)
linear_svm.fit(X_train, y_train)

y_pred_linear = linear_svm.predict(X_test)
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Classification Report (Linear SVM):\n", classification_report(y_test, y_pred_linear))

# Non-linear SVM
rbf_svm = SVC(kernel="rbf", C=1, gamma="scale")
rbf_svm.fit(X_train, y_train)

y_pred_rbf = rbf_svm.predict(X_test)
print("RBF SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("Classification Report (RBF SVM):\n", classification_report(y_test, y_pred_rbf))

param_grid = {"C": [0.1, 1, 10], "gamma": [0.001, 0.01, 0.1, 1]}
grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

print("Best Parameters (GridSearchCV):", grid.best_params_)
print("Best Cross-Validation Score:", grid.best_score_)

cv_scores = cross_val_score(rbf_svm, X, y, cv=5)
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

X_vis = X[:, :2]

X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y, test_size=0.2, random_state=42
)

svm_vis = SVC(kernel="rbf", C=1, gamma="scale")
svm_vis.fit(X_train_vis, y_train_vis)

x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, s=30, cmap=plt.cm.coolwarm, edgecolors="k")
plt.title("SVM Decision Boundary (RBF Kernel, 2 Features)")
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.show()