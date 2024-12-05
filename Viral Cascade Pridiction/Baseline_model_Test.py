import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# Load the test dataset
df = pd.read_csv('cascade_features.csv')

# Load the trained model and testing data
best_rf = load('rf_model_simple.joblib')  
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')['virality'] 

# Make predictions
y_pred = best_rf.predict(X_test)

# Precision, Recall, and F1-Score for the "viral" class
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print(f"Precision (Viral Class): {precision:.4f}")
print(f"Recall (Viral Class): {recall:.4f}")
print(f"F1-Score (Viral Class): {f1:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Viral', 'Viral'], yticklabels=['Non-Viral', 'Viral'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Detailed Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))


'''Precision (Viral Class): 0.5652
Recall (Viral Class): 0.5200
F1-Score (Viral Class): 0.5417
Confusion Matrix:
 [[48 20]
 [24 26]]

Detailed Classification Report:
              precision    recall  f1-score   support

           0       0.67      0.71      0.69        68
           1       0.57      0.52      0.54        50

    accuracy                           0.63       118
'''