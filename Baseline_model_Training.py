import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from joblib import dump

# Load the features dataset
df_original = pd.read_csv('cascade_features.csv')

# Delta_t thresholds in days, converted to hours
delta_t_thresholds = [60, 90, 120, 150, 180]  # in days
delta_t_thresholds_hours = [threshold * 24 for threshold in delta_t_thresholds]

# Create a results DataFrame to save the evaluation metrics
results = []

# Create a directory to save the models
models_dir = "rf_models"
os.makedirs(models_dir, exist_ok=True)

# Iterate over delta_t thresholds
for threshold in delta_t_thresholds_hours:
    print(f"=== Processing for delta_t threshold: {threshold / 24} days ===")
    
    # Create a copy of the original dataset for this threshold
    df = df_original.copy()

    # Adjust class labels based on delta_t
    for index, row in df.iterrows():
        if pd.notna(row['delta_t']):  # If delta_t is not NaN
            if row['delta_t'] > threshold and row['virality'] == 1:
                df.at[index, 'virality'] = 0  # Change positive to negative
        else:  # If delta_t is NaN, treat as non-viral
            df.at[index, 'virality'] = 0

    # Drop non-predictive features
    cols_to_drop = ['topic_id', 'delta_t']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    # Features (X) and target (y) split
    X = df.drop(columns=['virality'])
    y = df['virality']

    # Stratified train/test split (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.3, 
        random_state=42,
        stratify=y
    )

    print("Class distribution before SMOTE:\n", y_train.value_counts())

    # Handle imbalance with SMOTE only if there are enough minority samples
    if y_train.value_counts().get(1, 0) > 1:  # Proceed with SMOTE only if >1 minority sample
        smote = SMOTE(random_state=42, k_neighbors=1)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:  # Skip SMOTE and use class weighting
        print("Skipping SMOTE due to insufficient minority samples.")
        X_train_resampled, y_train_resampled = X_train, y_train

    # Train the Random Forest model with class weighting
    rf = RandomForestClassifier(random_state=42, class_weight='balanced_subsample')
    rf.fit(X_train_resampled, y_train_resampled)

    # Predictions on the training data
    y_train_pred = rf.predict(X_train_resampled)
    # Predictions on the test data
    y_test_pred = rf.predict(X_test)

    # Evaluate performance
    train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Force classification_report to include both classes (0 and 1):
    train_report = classification_report(
        y_train_resampled,
        y_train_pred,
        labels=[0, 1],
        zero_division=0,
        output_dict=True
    )
    test_report = classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1],
        zero_division=0,
        output_dict=True
    )

    # Save results in a dictionary
    results.append({
        "delta_t_days": threshold / 24,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision_0": train_report["0"]["precision"],
        "train_recall_0": train_report["0"]["recall"],
        "train_f1_0": train_report["0"]["f1-score"],
        "test_precision_0": test_report["0"]["precision"],
        "test_recall_0": test_report["0"]["recall"],
        "test_f1_0": test_report["0"]["f1-score"],
        "train_precision_1": train_report["1"]["precision"],
        "train_recall_1": train_report["1"]["recall"],
        "train_f1_1": train_report["1"]["f1-score"],
        "test_precision_1": test_report["1"]["precision"],
        "test_recall_1": test_report["1"]["recall"],
        "test_f1_1": test_report["1"]["f1-score"],
    })

    # Save the trained model for this threshold
    model_filename = os.path.join(models_dir, f'rf_model_delta_t_{threshold / 24}_days.joblib')
    dump(rf, model_filename)
    print(f"Model saved as {model_filename}")

# Save all results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("rf_model_results.csv", index=False)
print("Results saved to rf_model_results.csv")
