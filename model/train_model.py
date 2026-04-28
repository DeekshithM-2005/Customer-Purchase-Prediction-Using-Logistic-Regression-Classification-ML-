"""
Train a Logistic Regression model for Customer Purchase Prediction.
Uses the Online Shoppers Purchasing Intention Dataset.
Saves the trained model, scaler, and performance metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import joblib
import json
import os

# --- Load Dataset ---
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'online_shoppers_intention.csv')
df = pd.read_csv(data_path)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# --- Preprocessing ---
# Encode Month
month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
df['Month'] = df['Month'].map(month_map).fillna(0).astype(int)

# Encode VisitorType
visitor_map = {'Returning_Visitor': 0, 'New_Visitor': 1, 'Other': 2}
df['VisitorType'] = df['VisitorType'].map(visitor_map).fillna(2).astype(int)

# Convert Weekend and Revenue to int
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)

# Drop rows with missing values
df = df.dropna()
print(f"After cleaning: {df.shape[0]} rows")

# --- Feature Engineering ---
FEATURE_COLUMNS = [
    'Administrative', 'Administrative_Duration',
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
    'Month', 'OperatingSystems', 'Browser', 'Region',
    'TrafficType', 'VisitorType', 'Weekend'
]

X = df[FEATURE_COLUMNS]
y = df['Revenue']

print(f"Target distribution: {dict(y.value_counts())}")

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train Logistic Regression ---
model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver='lbfgs',
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

# Cross-validation
cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')

print("\n" + "=" * 50)
print("MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"  Accuracy:       {accuracy:.4f}")
print(f"  Precision:      {precision:.4f}")
print(f"  Recall:         {recall:.4f}")
print(f"  F1-Score:       {f1:.4f}")
print(f"  ROC-AUC:        {roc_auc:.4f}")
print(f"  CV Accuracy:    {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"\n  Confusion Matrix:")
print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
print("\n" + classification_report(y_test, y_pred, target_names=['No Purchase', 'Purchase']))

# --- Feature Importance ---
feature_importance = dict(zip(FEATURE_COLUMNS, model.coef_[0].tolist()))
sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True))
print("\nFeature Importance (by coefficient magnitude):")
for feat, coef in sorted_importance.items():
    bar = "#" * int(abs(coef) * 10)
    direction = "+" if coef > 0 else "-"
    print(f"  {direction} {feat:30s} {coef:+.4f}  {bar}")

# --- Save Model Artifacts ---
model_dir = os.path.dirname(__file__)
joblib.dump(model, os.path.join(model_dir, 'logistic_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

# Save metrics for the API
metrics = {
    'accuracy': round(accuracy, 4),
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1_score': round(f1, 4),
    'roc_auc': round(roc_auc, 4),
    'cv_accuracy_mean': round(cv_scores.mean(), 4),
    'cv_accuracy_std': round(cv_scores.std(), 4),
    'confusion_matrix': cm.tolist(),
    'feature_importance': {k: round(v, 4) for k, v in sorted_importance.items()},
    'feature_columns': FEATURE_COLUMNS,
    'train_size': int(X_train.shape[0]),
    'test_size': int(X_test.shape[0]),
    'total_samples': int(df.shape[0]),
}

with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nModel saved to: {os.path.join(model_dir, 'logistic_model.pkl')}")
print(f"Scaler saved to: {os.path.join(model_dir, 'scaler.pkl')}")
print(f"Metrics saved to: {os.path.join(model_dir, 'metrics.json')}")
