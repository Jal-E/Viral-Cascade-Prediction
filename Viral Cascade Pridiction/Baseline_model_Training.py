import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump

# Load the features dataset
df = pd.read_csv('cascade_features.csv')

# Features and target split
X = df.drop(columns=['virality'])
y = df['virality']

# Split into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE on the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the Random Forest model
rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1)
rf.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = rf.predict(X_test)

# Model Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))



# Save the trained model
dump(rf, 'rf_model_simple.joblib')

# Save test data for further evaluation
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
