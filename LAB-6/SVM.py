
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load the dataset
# Note: Ensure the file name matches your local file path
df = pd.read_csv('iris.csv')
X = df.drop('species', axis=1)
y = df['species']

# 2. Split data: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build and evaluate SVM with Linear Kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

print("--- Linear Kernel Results ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_linear)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_linear))

# 4. Build and evaluate SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

print("\n--- RBF Kernel Results ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_rbf)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rbf))
