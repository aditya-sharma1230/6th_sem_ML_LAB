import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# 1. Load the dataset
df = pd.read_csv('heart.csv')

# 2. Preprocessing: Encoding and Scaling
# identifying categorical columns for one-hot encoding
categorical_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features and target
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Scaling (Standardization is crucial for SVM, Logistic Regression, and PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Helper function to evaluate multiple models
def evaluate_models(X_tr, X_te, y_tr, y_te):
    models = {
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    accuracies = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        accuracies[name] = accuracy_score(y_te, y_pred)
    return accuracies

# 3. Model Building (Before PCA)
results_initial = evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
print("--- Accuracies Without PCA ---")
for model, acc in results_initial.items():
    print(f"{model}: {acc:.4f}")

# 4. Dimensionality Reduction using PCA
# Retaining 95% of the variance
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nOriginal dimensions: {X_train_scaled.shape[1]}")
print(f"Dimensions after PCA: {X_train_pca.shape[1]}")

# 5. Retraining Models (With PCA)
results_pca = evaluate_models(X_train_pca, X_test_pca, y_train, y_test)
print("\n--- Accuracies With PCA ---")
for model, acc in results_pca.items():
    print(f"{model}: {acc:.4f}")
