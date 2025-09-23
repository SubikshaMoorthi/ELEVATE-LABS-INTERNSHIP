import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Basic info
print(df.shape)
print(df.info())
print(df.head())
print(df.isnull().sum())

# Summary statistics
print(df.describe())
print("Mean Age:", df['Age'].mean())
print("Median Fare:", df['Fare'].median())
print("Std of Age:", df['Age'].std())

# Histogram
plt.hist(df['Age'].dropna(), bins=20, edgecolor="black")
plt.title("Age Distribution")
plt.show()

plt.hist(df['Fare'].dropna(), bins=30, edgecolor="black")
plt.title("Fare Distribution")
plt.show()

# Boxplot
sns.boxplot(x="Pclass", y="Age", data=df)
plt.title("Age Distribution by Passenger Class")
plt.show()

sns.boxplot(x="Survived", y="Fare", data=df)
plt.title("Fare Distribution vs Survival")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Pairplot
sns.pairplot(df[["Age", "Fare", "Survived"]], hue="Survived")
plt.show()

# Survival Rate by Gender
sns.barplot(x="Sex", y="Survived", data=df)
plt.title("Survival Rate by Gender")
plt.show()

# Survival Rate by Passenger Class
sns.barplot(x="Pclass", y="Survived", data=df)
plt.title("Survival Rate by Class")
plt.show()

# Interactive histogram with Plotly
fig = px.histogram(df, x="Age", color="Survived", nbins=20, title="Age Distribution by Survival")
fig.show()

fig2 = px.scatter(df, x="Age", y="Fare", color="Survived", size="Fare", hover_data=["Pclass", "Sex"])
fig2.show()
