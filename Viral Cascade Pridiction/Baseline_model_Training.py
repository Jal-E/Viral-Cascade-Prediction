import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from joblib import dump

# Load the features dataset
df = pd.read_csv('cascade_features.csv')

# Drop 'topic_id' and 'delta_t' since they are not predictive features
cols_to_drop = []
if 'topic_id' in df.columns:
    cols_to_drop.append('topic_id')
if 'delta_t' in df.columns:
    cols_to_drop.append('delta_t')

if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)

# Features and target split
X = df.drop(columns=['virality'])
y = df['virality']

# Split into training and testing sets (70% training, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE on the training data to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the Random Forest model
rf = RandomForestClassifier(
    random_state=42, 
    n_estimators=100, 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1
)
rf.fit(X_train_resampled, y_train_resampled)

# Predictions on the training data
y_train_pred = rf.predict(X_train_resampled)

# Predictions on the test data
y_test_pred = rf.predict(X_test)

# Model Evaluation on Training Data
print("=== Training Data Performance ===")
print("Confusion Matrix (Train):\n", confusion_matrix(y_train_resampled, y_train_pred))
print("Classification Report (Train):\n", classification_report(y_train_resampled, y_train_pred))
print("Accuracy Score (Train):", accuracy_score(y_train_resampled, y_train_pred))

# Model Evaluation on Test Data
print("\n=== Test Data Performance ===")
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report (Test):\n", classification_report(y_test, y_test_pred))
print("Accuracy Score (Test):", accuracy_score(y_test, y_test_pred))

# Save the trained model
dump(rf, 'rf_model_simple.joblib')


'''Confusion Matrix (Train):
 [[156   0]
 [  0 156]]
Classification Report (Train):
               precision    recall  f1-score   support

           0       1.00      1.00      1.00       156
           1       1.00      1.00      1.00       156

    accuracy                           1.00       312
   macro avg       1.00      1.00      1.00       312
weighted avg       1.00      1.00      1.00       312

Accuracy Score (Train): 1.0

=== Test Data Performance ===
Confusion Matrix (Test):
 [[45 18]
 [18 37]]
Classification Report (Test):
               precision    recall  f1-score   support

           0       0.71      0.71      0.71        63
           1       0.67      0.67      0.67        55

    accuracy                           0.69       118
   macro avg       0.69      0.69      0.69       118
weighted avg       0.69      0.69      0.69       118

Accuracy Score (Test): 0.6949152542372882
'''
