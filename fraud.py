import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# For handling imbalanced datasets
from imblearn.over_sampling import SMOTE

# Warnings filter
import warnings
warnings.filterwarnings("ignore")

# Load dataset (replace 'creditcard.csv' with your dataset path)
df = pd.read_csv('creditcard.csv')

# Check for missing values
print(df.isnull().sum())

# Drop any missing or irrelevant values (if any)
df = df.dropna()

# Split the data into features and target
X = df.drop('Class', axis=1)  # Features (e.g., transaction amount, timestamp, etc.)
y = df['Class']  # Target (Fraud or not)

# Balance the dataset using SMOTE to handle imbalanced classes
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Feature Scaling
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)

# Plot correlation matrix to detect feature relationships
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.show()

# # Distribution of the classes (fraud and non-fraud)
# sns.countplot(df['Class'])
# plt.title('Class Distribution')
# plt.show()

# Example of feature visualization
plt.figure(figsize=(10,6))
sns.distplot(df['Amount'], color='blue')
plt.title('Distribution of Transaction Amount')
plt.show()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res_scaled, y_res, test_size=0.2, random_state=42)

# Choose RandomForest as the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.show()

import joblib

# Save model
joblib.dump(model, 'fraud_detection_model.pkl')

# To load the model later
# model = joblib.load('fraud_detection_model.pkl')

