import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset from a CSV file

df = pd.read_csv('Titanic-Dataset.csv')

print("Initial Dataset Info")
df.info()

print("\nMissing values before cleaning")
print(df.isnull().sum())

# Handle missing values

imputer_age = SimpleImputer(strategy='median')
df['Age'] = imputer_age.fit_transform(df[['Age']])

imputer_embarked = SimpleImputer(strategy='most_frequent')
df['Embarked'] = imputer_embarked.fit_transform(df[['Embarked']]).ravel()

df = df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1)

# Verify that all missing values have been handled.
print("\n--- Missing values after imputation and dropping columns ---")
print(df.isnull().sum())

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nDataset after categorical encoding")
print(df.head())

numerical_features = ['Age', 'Fare']

# Use StandardScaler to standardize the features (mean=0, std=1).
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\n--- Dataset after standardizing numerical features ---")
print(df.head())

# Visualize outliers in 'Fare' using a boxplot.
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.show() 

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

print(f"\nOriginal shape: {df.shape}")
print(f"Shape after outlier removal: {df_cleaned.shape}")

print("\nFinal cleaned dataset info")
df_cleaned.info()