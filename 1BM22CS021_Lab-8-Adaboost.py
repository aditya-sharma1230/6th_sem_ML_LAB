import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
df = pd.read_csv('income.csv')

# 2. Prepare Data
# Separate features (X) and target (y)
X = df.drop('income_level', axis=1)
y = df['income_level']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build AdaBoost Classifier with 10 estimators
ada_10 = AdaBoostClassifier(n_estimators=10, random_state=42)
ada_10.fit(X_train, y_train)
y_pred_10 = ada_10.predict(X_test)
score_10 = accuracy_score(y_test, y_pred_10)

print(f"Prediction score with 10 trees: {score_10:.4f}")

# 4. Fine-tuning the model
# Testing various numbers of trees to identify the best performance
tree_counts = [5, 10, 25, 50, 100, 150, 200, 300, 400, 500]
scores = []

print("\nFine-tuning results:")
for n in tree_counts:
    model = AdaBoostClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    scores.append(acc)
    print(f"Trees: {n:3} | Accuracy: {acc:.4f}")

# Identify the best configuration
best_acc = max(scores)
best_n = tree_counts[scores.index(best_acc)]

print(f"\nOptimal performance: {best_acc:.4f} accuracy achieved with {best_n} trees.")

# 5. Visualization (Optional but helpful)
plt.figure(figsize=(8, 5))
plt.plot(tree_counts, scores, marker='o', linestyle='-', color='darkgreen')
plt.title('AdaBoost Accuracy vs. Number of Trees')
plt.xlabel('n_estimators')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.show()
