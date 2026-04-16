import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# 1. Load the dataset
df = pd.read_csv('heart.csv')

# 2. Preprocessing: Encoding and Scaling
# Identifying categorical columns that need One-Hot Encoding
# (Note: In this dataset, categorical features like 'cp', 'restecg', 'slope', etc. are already numeric)
categorical_features = ['cp', 'restecg', 'slope', 'ca', 'thal']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Separate features and target
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Scaling (Crucial for SVM, Logistic Regression, and PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Building (Before PCA)
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "SVM": SVC(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

print("--- Accuracies Before PCA ---")
results_before = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results_before[name] = acc
    print(f"{name}: {acc:.4f}")

# 4. Applying PCA for Dimensionality Reduction
# Retaining 95% of the variance
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nOriginal dimensions: {X_train_scaled.shape[1]}")
print(f"Dimensions after PCA: {X_train_pca.shape[1]}")

# 5. Retraining Models (After PCA)
print("\n--- Accuracies After PCA ---")
results_after = {}
for name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    results_after[name] = acc
    print(f"{name}: {acc:.4f}")
