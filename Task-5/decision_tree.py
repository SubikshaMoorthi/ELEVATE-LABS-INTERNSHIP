import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

# 1. Load Dataset
data = pd.read_csv("heart.csv")
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

print("Decision Tree Accuracy:", dt.score(X_test, y_test))

export_graphviz(
    dt,
    out_file="heart_tree.dot",
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True
)
print("Tree exported as heart_tree.dot — run 'dot -Tpng heart_tree.dot -o heart_tree.png' to render.")

# 3. Overfitting Analysis (Varying Depth)
train_acc, test_acc = [], []
for depth in range(1, 11):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_acc.append(dt.score(X_train, y_train))
    test_acc.append(dt.score(X_test, y_test))

plt.plot(range(1, 11), train_acc, label="Train Accuracy")
plt.plot(range(1, 11), test_acc, label="Test Accuracy")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Overfitting vs Tree Depth (Heart Dataset)")
plt.savefig("heart_tree_depth.png")
plt.show()

# 4. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("Random Forest Accuracy:", rf.score(X_test, y_test))

# 5. Feature Importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), np.array(X.columns)[indices], rotation=90)
plt.title("Feature Importances (Random Forest - Heart Dataset)")
plt.tight_layout()
plt.savefig("heart_feature_importances.png")
plt.show()

# 6. Cross-validation
scores_dt = cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=5)
scores_rf = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=5)

print("Decision Tree CV Accuracy:", scores_dt.mean())
print("Random Forest CV Accuracy:", scores_rf.mean())
