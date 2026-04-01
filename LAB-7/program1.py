
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
# Ensure the filename matches your local file (e.g., 'iris.csv')
df = pd.read_csv('iris.csv')

# Preprocessing: Separate features and target
X = df.drop('species', axis=1)
y = df['species']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Task 1: Build RF classifier with n_estimators=10 ---
rf_default = RandomForestClassifier(n_estimators=10, random_state=42)
rf_default.fit(X_train, y_train)
y_pred_default = rf_default.predict(X_test)
score_10 = accuracy_score(y_test, y_pred_default)

print(f"Accuracy with n_estimators=10: {score_10 * 100:.2f}%")

# --- Task 2: Fine-tune the model by changing number of trees ---
best_score = 0
best_n = 0

# Iterating from 1 to 100 trees to find the best score
for n in range(1, 101):
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    
    if score > best_score:
        best_score = score
        best_n = n

print(f"Best Accuracy Score: {best_score * 100:.2f}%")
print(f"Number of trees for best score: {best_n}")
