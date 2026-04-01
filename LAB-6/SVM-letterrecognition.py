import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize, LabelEncoder

# 1. Load the dataset
df_letter = pd.read_csv('letter-recognition.csv')
X_l = df_letter.drop('letter', axis=1)
y_l = df_letter['letter']

# 2. Encode labels and split data (80% Train, 20% Test)
le = LabelEncoder()
y_enc = le.fit_transform(y_l)
n_classes = len(le.classes_)

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l, y_enc, test_size=0.2, random_state=42)

# 3. Build SVM Classifier (using probability=True for ROC/AUC)
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_l, y_train_l)

# 4. Predictions and Scores
y_pred_l = svm_model.predict(X_test_l)
y_score_l = svm_model.predict_proba(X_test_l)

print(f"Accuracy Score: {accuracy_score(y_test_l, y_pred_l)}")
print("Confusion Matrix (Partial view):")
print(confusion_matrix(y_test_l, y_pred_l))

# 5. ROC Curve and AUC Score (Micro-average for Multiclass)
y_test_bin = label_binarize(y_test_l, classes=np.arange(n_classes))
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score_l.ravel())
roc_auc = auc(fpr, tpr)

print(f"Micro-average AUC Score: {roc_auc}")

# 6. Plotting
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Letter Recognition')
plt.legend(loc="lower right")
plt.show()
