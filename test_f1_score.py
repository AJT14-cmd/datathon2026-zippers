"""
============================================================================
F1 SCORE TESTING SCRIPT - For Accessibility Prediction Model
============================================================================

This script evaluates your trained models with detailed F1 scores and metrics.

REQUIREMENTS:
- accessibility_regression_model.pkl
- accessibility_classification_model.pkl
- label_encoder.pkl
- everydayLife_cleaned_dataset.csv

RUN:
python test_f1_score.py

============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, 
    mean_squared_error, r2_score, mean_absolute_error
)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("F1 SCORE TESTING - ACCESSIBILITY PREDICTION MODEL")
print("="*80)

# ============================================================================
# LOAD DATA AND PREPARE
# ============================================================================

print("\n[1/4] Loading data and models...")

# Load dataset
df = pd.read_csv('everydayLife_cleaned_dataset.csv')
print(f"   ✓ Loaded {len(df):,} accessibility issues")

# Create spatial grid
grid_size = 0.005
df['grid_lon'] = (df['geometry/coordinates/0'] // grid_size) * grid_size
df['grid_lat'] = (df['geometry/coordinates/1'] // grid_size) * grid_size
df['grid_id'] = df['grid_lon'].astype(str) + '_' + df['grid_lat'].astype(str)

# Aggregate features
grid_features = df.groupby('grid_id').agg({
    'geometry/coordinates/0': 'mean',
    'geometry/coordinates/1': 'mean',
    'properties/attribute_id': 'count',
    'properties/severity': ['mean', 'max', 'std'],
    'properties/is_temporary': 'sum',
    'properties/label_type': lambda x: x.nunique(),
}).reset_index()

grid_features.columns = ['grid_id', 'center_lon', 'center_lat', 'issue_count', 
                         'avg_severity', 'max_severity', 'std_severity', 
                         'temp_count', 'issue_type_diversity']
grid_features['std_severity'] = grid_features['std_severity'].fillna(0)

# Add issue types
issue_types = pd.get_dummies(df['properties/label_type'], prefix='type')
issue_types['grid_id'] = df['grid_id']
type_counts = issue_types.groupby('grid_id').sum()
grid_features = grid_features.merge(type_counts, on='grid_id', how='left').fillna(0)

# Add neighborhoods
neighborhood_mode = df.groupby('grid_id')['properties/neighborhood'].agg(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
)
grid_features = grid_features.merge(neighborhood_mode.rename('neighborhood'), on='grid_id', how='left')

# Encode neighborhoods
le = LabelEncoder()
grid_features['neighborhood_encoded'] = le.fit_transform(grid_features['neighborhood'])

# Define features
feature_cols = ['center_lon', 'center_lat', 'avg_severity', 'max_severity', 
                'std_severity', 'temp_count', 'issue_type_diversity',
                'type_CurbRamp', 'type_NoCurbRamp', 'type_NoSidewalk', 
                'type_Obstacle', 'type_Occlusion', 'type_Other', 
                'type_SurfaceProblem', 'neighborhood_encoded']

# Load models
try:
    with open('accessibility_regression_model.pkl', 'rb') as f:
        reg_model = pickle.load(f)
    with open('accessibility_classification_model.pkl', 'rb') as f:
        clf_model = pickle.load(f)
    print(f"   ✓ Loaded trained models")
except FileNotFoundError:
    print("\n   ❌ ERROR: Model files not found!")
    print("   Please run 'accessibility_prediction_local.py' first to train the models.")
    exit()

# ============================================================================
# TEST CLASSIFICATION MODEL (F1 SCORES)
# ============================================================================

print("\n[2/4] Testing Classification Model...")

# Create risk categories
def categorize_risk(count):
    if count < 30:
        return 'Low'
    elif count < 100:
        return 'Medium'
    else:
        return 'High'

grid_features['risk_level'] = grid_features['issue_count'].apply(categorize_risk)

X = grid_features[feature_cols]
y = grid_features['risk_level']

# Split data (same as training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predictions
y_pred = clf_model.predict(X_test)

print("\n" + "="*80)
print("CLASSIFICATION MODEL - DETAILED METRICS")
print("="*80)

# Overall metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# F1 Scores (different averaging methods)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_micro = f1_score(y_test, y_pred, average='micro')

print(f"\n🎯 F1 SCORES:")
print(f"   • Macro F1:    {f1_macro:.4f}  (unweighted average across classes)")
print(f"   • Weighted F1: {f1_weighted:.4f}  (weighted by class size)")
print(f"   • Micro F1:    {f1_micro:.4f}  (overall across all predictions)")

# Per-class metrics
print(f"\n📋 PER-CLASS METRICS:")
classes = ['High', 'Low', 'Medium']
for cls in classes:
    # Binary classification for this class
    y_test_binary = (y_test == cls)
    y_pred_binary = (y_pred == cls)
    
    if y_test_binary.sum() > 0:  # Only if class exists in test set
        f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
        precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
        support = y_test_binary.sum()
        
        print(f"\n   {cls} Risk:")
        print(f"      F1 Score:  {f1:.4f}")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall:    {recall:.4f}")
        print(f"      Support:   {support} samples")

# Detailed classification report
print(f"\n📈 DETAILED CLASSIFICATION REPORT:")
print("-" * 80)
print(classification_report(y_test, y_pred, target_names=classes, digits=4))

# Confusion Matrix
print(f"\n🔢 CONFUSION MATRIX:")
print("-" * 80)
cm = confusion_matrix(y_test, y_pred, labels=classes)
cm_df = pd.DataFrame(cm, index=[f'Actual {c}' for c in classes], 
                     columns=[f'Pred {c}' for c in classes])
print(cm_df)

# Calculate per-class accuracy from confusion matrix
print(f"\n✅ PER-CLASS ACCURACY:")
for i, cls in enumerate(classes):
    if cm[i].sum() > 0:
        class_accuracy = cm[i, i] / cm[i].sum()
        print(f"   {cls}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")

# ============================================================================
# TEST REGRESSION MODEL
# ============================================================================

print("\n[3/4] Testing Regression Model...")

X = grid_features[feature_cols]
y_reg = grid_features['issue_count']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

y_pred_reg = reg_model.predict(X_test_r)

print("\n" + "="*80)
print("REGRESSION MODEL - DETAILED METRICS")
print("="*80)

# Standard regression metrics
r2 = r2_score(y_test_r, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_reg))
mae = mean_absolute_error(y_test_r, y_pred_reg)
mse = mean_squared_error(y_test_r, y_pred_reg)

print(f"\n📊 PERFORMANCE METRICS:")
print(f"   • R² Score:  {r2:.4f}  ({r2*100:.2f}% variance explained)")
print(f"   • RMSE:      {rmse:.2f} issues")
print(f"   • MAE:       {mae:.2f} issues")
print(f"   • MSE:       {mse:.2f}")

# Additional metrics
mape = np.mean(np.abs((y_test_r - y_pred_reg) / y_test_r)) * 100
max_error = np.max(np.abs(y_test_r - y_pred_reg))

print(f"   • MAPE:      {mape:.2f}% (mean absolute percentage error)")
print(f"   • Max Error: {max_error:.2f} issues")

# Prediction accuracy within ranges
within_5 = np.sum(np.abs(y_test_r - y_pred_reg) <= 5) / len(y_test_r)
within_10 = np.sum(np.abs(y_test_r - y_pred_reg) <= 10) / len(y_test_r)
within_20 = np.sum(np.abs(y_test_r - y_pred_reg) <= 20) / len(y_test_r)

print(f"\n🎯 PREDICTION ACCURACY:")
print(f"   • Within ±5 issues:  {within_5:.2%}")
print(f"   • Within ±10 issues: {within_10:.2%}")
print(f"   • Within ±20 issues: {within_20:.2%}")

# ============================================================================
# CROSS-VALIDATION (BONUS)
# ============================================================================

print("\n[4/4] Running Cross-Validation...")

from sklearn.model_selection import cross_val_score

# Classification cross-validation
print(f"\n🔄 CLASSIFICATION - 5-Fold Cross-Validation:")
cv_scores_clf = cross_val_score(clf_model, X, y, cv=5, scoring='f1_weighted')
print(f"   F1 Scores per fold: {[f'{s:.4f}' for s in cv_scores_clf]}")
print(f"   Mean F1: {cv_scores_clf.mean():.4f} (±{cv_scores_clf.std():.4f})")

# Regression cross-validation
print(f"\n🔄 REGRESSION - 5-Fold Cross-Validation:")
cv_scores_reg = cross_val_score(reg_model, X, y_reg, cv=5, scoring='r2')
print(f"   R² Scores per fold: {[f'{s:.4f}' for s in cv_scores_reg]}")
print(f"   Mean R²: {cv_scores_reg.mean():.4f} (±{cv_scores_reg.std():.4f})")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY - KEY METRICS")
print("="*80)

print(f"\n🎯 CLASSIFICATION MODEL:")
print(f"   ✓ Overall Accuracy:    {accuracy*100:.2f}%")
print(f"   ✓ Weighted F1 Score:   {f1_weighted:.4f}")
print(f"   ✓ Macro F1 Score:      {f1_macro:.4f}")
print(f"   ✓ Cross-Val Mean F1:   {cv_scores_clf.mean():.4f}")

print(f"\n📈 REGRESSION MODEL:")
print(f"   ✓ R² Score:            {r2:.4f}")
print(f"   ✓ RMSE:                {rmse:.2f} issues")
print(f"   ✓ MAE:                 {mae:.2f} issues")
print(f"   ✓ Within ±10 issues:   {within_10:.1%}")
print(f"   ✓ Cross-Val Mean R²:   {cv_scores_reg.mean():.4f}")

print("\n" + "="*80)
print("✅ F1 SCORE TESTING COMPLETE!")
print("="*80)

# Save results to CSV
results_summary = pd.DataFrame({
    'Metric': [
        'Classification Accuracy',
        'Weighted F1 Score',
        'Macro F1 Score',
        'High Risk F1',
        'Medium Risk F1',
        'Low Risk F1',
        'Regression R²',
        'RMSE',
        'MAE',
        'Predictions within ±10'
    ],
    'Value': [
        f"{accuracy:.4f}",
        f"{f1_weighted:.4f}",
        f"{f1_macro:.4f}",
        f"{f1_score((y_test == 'High'), (y_pred == 'High')):.4f}",
        f"{f1_score((y_test == 'Medium'), (y_pred == 'Medium')):.4f}",
        f"{f1_score((y_test == 'Low'), (y_pred == 'Low')):.4f}",
        f"{r2:.4f}",
        f"{rmse:.2f}",
        f"{mae:.2f}",
        f"{within_10:.2%}"
    ]
})

results_summary.to_csv('f1_test_results.csv', index=False)
print(f"\n💾 Saved detailed results to 'f1_test_results.csv'")
