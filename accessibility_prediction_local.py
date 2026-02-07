"""
============================================================================
ACCESSIBILITY PREDICTION MODEL - LOCAL SETUP
============================================================================

INSTALLATION INSTRUCTIONS:
--------------------------
1. Make sure you have Python 3.7+ installed
2. Install required packages:
   pip install pandas numpy scikit-learn matplotlib seaborn

3. Place this script in the same folder as your dataset:
   - everydayLife_cleaned_dataset.csv

4. Run the script:
   python accessibility_prediction_local.py

OUTPUTS:
--------
- accessibility_regression_model.pkl (trained regression model)
- accessibility_classification_model.pkl (trained classification model)  
- label_encoder.pkl (for neighborhood encoding)
- results.png (visualizations)
- Console output with performance metrics

============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ACCESSIBILITY PREDICTION MODEL - STARTING...")
print("="*70)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n[1/5] Loading and preparing data...")

# Load data
df = pd.read_csv('everydayLife_cleaned_dataset.csv')
print(f"   ✓ Loaded {len(df):,} accessibility issues")

# Create spatial grid (0.005 degrees ≈ 500 meters)
grid_size = 0.005
df['grid_lon'] = (df['geometry/coordinates/0'] // grid_size) * grid_size
df['grid_lat'] = (df['geometry/coordinates/1'] // grid_size) * grid_size
df['grid_id'] = df['grid_lon'].astype(str) + '_' + df['grid_lat'].astype(str)

print(f"   ✓ Created {df['grid_id'].nunique()} spatial grid cells")

# Aggregate features by grid cell
grid_features = df.groupby('grid_id').agg({
    'geometry/coordinates/0': 'mean',  # center longitude
    'geometry/coordinates/1': 'mean',  # center latitude
    'properties/attribute_id': 'count',  # total issues
    'properties/severity': ['mean', 'max', 'std'],  # severity statistics
    'properties/is_temporary': 'sum',  # count of temporary issues
    'properties/label_type': lambda x: x.nunique(),  # issue type diversity
}).reset_index()

# Flatten column names
grid_features.columns = ['grid_id', 'center_lon', 'center_lat', 'issue_count', 
                         'avg_severity', 'max_severity', 'std_severity', 
                         'temp_count', 'issue_type_diversity']

grid_features['std_severity'] = grid_features['std_severity'].fillna(0)

# Add issue type distribution
issue_types = pd.get_dummies(df['properties/label_type'], prefix='type')
issue_types['grid_id'] = df['grid_id']
type_counts = issue_types.groupby('grid_id').sum()
grid_features = grid_features.merge(type_counts, on='grid_id', how='left').fillna(0)

# Add neighborhood
neighborhood_mode = df.groupby('grid_id')['properties/neighborhood'].agg(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
)
grid_features = grid_features.merge(
    neighborhood_mode.rename('neighborhood'), on='grid_id', how='left'
)

print(f"   ✓ Created {len(grid_features.columns)} features per grid cell")

# ============================================================================
# STEP 2: PREPARE FEATURES
# ============================================================================

print("\n[2/5] Preparing features...")

# Encode neighborhood
le = LabelEncoder()
grid_features['neighborhood_encoded'] = le.fit_transform(grid_features['neighborhood'])

# Define feature columns
feature_cols = ['center_lon', 'center_lat', 'avg_severity', 'max_severity', 
                'std_severity', 'temp_count', 'issue_type_diversity',
                'type_CurbRamp', 'type_NoCurbRamp', 'type_NoSidewalk', 
                'type_Obstacle', 'type_Occlusion', 'type_Other', 
                'type_SurfaceProblem', 'neighborhood_encoded']

X = grid_features[feature_cols]
y = grid_features['issue_count']

print(f"   ✓ Using {len(feature_cols)} features")
print(f"   ✓ Target: issue_count (min={y.min():.0f}, max={y.max():.0f}, mean={y.mean():.1f})")

# ============================================================================
# STEP 3: TRAIN REGRESSION MODEL
# ============================================================================

print("\n[3/5] Training regression model (predicting # of issues)...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   ✓ Train: {len(X_train)} cells | Test: {len(X_test)} cells")

# Train model
reg_model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=10, 
    random_state=42, 
    n_jobs=-1,
    verbose=0
)
reg_model.fit(X_train, y_train)
print(f"   ✓ Model trained!")

# Evaluate
y_pred_reg = reg_model.predict(X_test)
r2 = r2_score(y_test, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
mae = mean_absolute_error(y_test, y_pred_reg)

print(f"\n   📊 REGRESSION PERFORMANCE:")
print(f"      R² Score: {r2:.4f} ({r2*100:.1f}% variance explained)")
print(f"      RMSE: {rmse:.2f} issues")
print(f"      MAE: {mae:.2f} issues")

# Feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': reg_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   🎯 TOP 5 MOST IMPORTANT FEATURES:")
for idx, row in importance_df.head(5).iterrows():
    print(f"      {row['feature']:.<30} {row['importance']:.3f}")

# ============================================================================
# STEP 4: TRAIN CLASSIFICATION MODEL
# ============================================================================

print("\n[4/5] Training classification model (predicting risk level)...")

# Create risk categories
def categorize_risk(count):
    if count < 30:
        return 'Low'
    elif count < 100:
        return 'Medium'
    else:
        return 'High'

grid_features['risk_level'] = grid_features['issue_count'].apply(categorize_risk)
y_class = grid_features['risk_level']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

# Train
clf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42, 
    n_jobs=-1,
    verbose=0
)
clf_model.fit(X_train_c, y_train_c)
print(f"   ✓ Model trained!")

# Evaluate
y_pred_clf = clf_model.predict(X_test_c)
accuracy = (y_pred_clf == y_test_c).mean()

print(f"\n   📊 CLASSIFICATION PERFORMANCE:")
print(f"      Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"\n{classification_report(y_test_c, y_pred_clf, target_names=['High', 'Low', 'Medium'])}")

# ============================================================================
# STEP 5: VISUALIZE AND SAVE
# ============================================================================

print("\n[5/5] Creating visualizations...")

# Predict on all data for visualization
grid_features['predicted_issues'] = reg_model.predict(X)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Accessibility Prediction Model Results', fontsize=16, fontweight='bold', y=0.995)

# 1. Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred_reg, alpha=0.6, s=60, color='steelblue', edgecolors='black', linewidth=0.5)
max_val = max(y_test.max(), y_pred_reg.max())
axes[0, 0].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Issue Count', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Predicted Issue Count', fontsize=12, fontweight='bold')
axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2:.3f})', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)

# 2. Feature Importance
top_features = importance_df.head(10).sort_values('importance', ascending=True)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
axes[0, 1].barh(top_features['feature'], top_features['importance'], color=colors, edgecolor='black', linewidth=0.5)
axes[0, 1].set_xlabel('Importance Score', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Top 10 Feature Importance', fontsize=13, fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# 3. Spatial Heatmap
scatter = axes[1, 0].scatter(
    grid_features['center_lon'], 
    grid_features['center_lat'], 
    c=grid_features['predicted_issues'], 
    cmap='YlOrRd', 
    s=100, 
    alpha=0.7, 
    edgecolors='black', 
    linewidth=0.5
)
axes[1, 0].set_xlabel('Longitude', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Latitude', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Predicted Risk Heatmap (Darker = Higher Risk)', fontsize=13, fontweight='bold')
cbar = plt.colorbar(scatter, ax=axes[1, 0])
cbar.set_label('Predicted Issues', fontsize=11, fontweight='bold')

# 4. Distribution Comparison
bins = np.linspace(0, max(grid_features['issue_count'].max(), grid_features['predicted_issues'].max()), 40)
axes[1, 1].hist(grid_features['issue_count'], bins=bins, alpha=0.6, 
               label='Actual', color='dodgerblue', edgecolor='black', linewidth=0.8)
axes[1, 1].hist(grid_features['predicted_issues'], bins=bins, alpha=0.6, 
               label='Predicted', color='orangered', edgecolor='black', linewidth=0.8)
axes[1, 1].set_xlabel('Issue Count', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Frequency (# of Grid Cells)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Distribution: Actual vs Predicted', fontsize=13, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results.png', dpi=200, bbox_inches='tight')
print(f"   ✓ Saved visualization to 'results.png'")

# Save models
with open('accessibility_regression_model.pkl', 'wb') as f:
    pickle.dump(reg_model, f)
print(f"   ✓ Saved regression model")

with open('accessibility_classification_model.pkl', 'wb') as f:
    pickle.dump(clf_model, f)
print(f"   ✓ Saved classification model")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print(f"   ✓ Saved label encoder")

# ============================================================================
# EXAMPLE: HOW TO USE THE MODEL
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE: HOW TO USE THE MODEL FOR PREDICTIONS")
print("="*70)

example_code = """
# Load the trained model
import pickle
with open('accessibility_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Predict for a new location
import pandas as pd

# Example: Capitol Hill area with some existing issues
new_location = pd.DataFrame([{
    'center_lon': -122.33,
    'center_lat': 47.61,
    'avg_severity': 3.0,
    'max_severity': 4.0,
    'std_severity': 1.0,
    'temp_count': 2,
    'issue_type_diversity': 3,
    'type_CurbRamp': 5,
    'type_NoCurbRamp': 15,
    'type_NoSidewalk': 3,
    'type_Obstacle': 2,
    'type_Occlusion': 1,
    'type_Other': 0,
    'type_SurfaceProblem': 8,
    'neighborhood_encoded': le.transform(['Capitol Hill'])[0]
}])

predicted_issues = model.predict(new_location)[0]
print(f"Predicted issues: {predicted_issues:.0f}")
"""

print(example_code)

print("\n" + "="*70)
print("✓ ALL DONE! Your models are ready to use.")
print("="*70)
print("\nGenerated files:")
print("  1. accessibility_regression_model.pkl    - Predicts # of issues")
print("  2. accessibility_classification_model.pkl - Predicts High/Med/Low risk")
print("  3. label_encoder.pkl                     - Encodes neighborhoods")
print("  4. results.png                           - Visual performance report")
print("\nModel Performance Summary:")
print(f"  • Regression R²: {r2:.3f} (explains {r2*100:.1f}% of variance)")
print(f"  • Classification Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"  • Average prediction error: ±{mae:.1f} issues")
print("="*70)
