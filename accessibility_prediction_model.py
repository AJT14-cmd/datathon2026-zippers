"""
Accessibility Problem Prediction Model
Predicts where future accessibility issues are likely to occur in a city

Author: Built with Claude
Dataset: Seattle Accessibility Data (81,973 issues across 965 grid cells)
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

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

def load_and_preprocess(filepath, grid_size=0.005):
    """
    Load accessibility data and create spatial grid
    
    Args:
        filepath: Path to cleaned CSV dataset
        grid_size: Size of grid cells in degrees (~0.005 = 500m)
    
    Returns:
        DataFrame with grid-aggregated features
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} accessibility issues")
    
    # Create spatial grid
    df['grid_lon'] = (df['geometry/coordinates/0'] // grid_size) * grid_size
    df['grid_lat'] = (df['geometry/coordinates/1'] // grid_size) * grid_size
    df['grid_id'] = df['grid_lon'].astype(str) + '_' + df['grid_lat'].astype(str)
    
    print(f"Created {df['grid_id'].nunique()} spatial grid cells")
    
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
    
    return grid_features


# ============================================================================
# STEP 2: TRAIN REGRESSION MODEL (Predict # of Issues)
# ============================================================================

def train_regression_model(grid_features, test_size=0.2):
    """
    Train Random Forest to predict issue count per grid cell
    """
    print("\n" + "="*70)
    print("TRAINING REGRESSION MODEL")
    print("="*70)
    
    # Encode neighborhood
    le = LabelEncoder()
    grid_features['neighborhood_encoded'] = le.fit_transform(grid_features['neighborhood'])
    
    # Define features
    feature_cols = ['center_lon', 'center_lat', 'avg_severity', 'max_severity', 
                    'std_severity', 'temp_count', 'issue_type_diversity',
                    'type_CurbRamp', 'type_NoCurbRamp', 'type_NoSidewalk', 
                    'type_Obstacle', 'type_Occlusion', 'type_Other', 
                    'type_SurfaceProblem', 'neighborhood_encoded']
    
    X = grid_features[feature_cols]
    y = grid_features['issue_count']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train model
    print(f"Training on {len(X_train)} cells, testing on {len(X_test)} cells...")
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nPerformance Metrics:")
    print(f"  R² Score: {r2:.4f} ({r2*100:.1f}% variance explained)")
    print(f"  RMSE: {rmse:.2f} issues")
    print(f"  MAE: {mae:.2f} issues")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return model, (X_test, y_test, y_pred), feature_cols, le


# ============================================================================
# STEP 3: TRAIN CLASSIFICATION MODEL (Predict Risk Level)
# ============================================================================

def train_classification_model(grid_features, test_size=0.2):
    """
    Train Random Forest to classify grid cells as High/Medium/Low risk
    """
    print("\n" + "="*70)
    print("TRAINING CLASSIFICATION MODEL")
    print("="*70)
    
    # Create risk categories
    def categorize_risk(count):
        if count < 30:
            return 'Low'
        elif count < 100:
            return 'Medium'
        else:
            return 'High'
    
    grid_features['risk_level'] = grid_features['issue_count'].apply(categorize_risk)
    
    print("\nRisk Level Distribution:")
    print(grid_features['risk_level'].value_counts())
    
    # Encode neighborhood
    le = LabelEncoder()
    grid_features['neighborhood_encoded'] = le.fit_transform(grid_features['neighborhood'])
    
    feature_cols = ['center_lon', 'center_lat', 'avg_severity', 'max_severity', 
                    'std_severity', 'temp_count', 'issue_type_diversity',
                    'type_CurbRamp', 'type_NoCurbRamp', 'type_NoSidewalk', 
                    'type_Obstacle', 'type_Occlusion', 'type_Other', 
                    'type_SurfaceProblem', 'neighborhood_encoded']
    
    X = grid_features[feature_cols]
    y = grid_features['risk_level']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))
    
    return model, (X_test, y_test, y_pred)


# ============================================================================
# STEP 4: VISUALIZE RESULTS
# ============================================================================

def visualize_results(grid_features, regression_results, model, feature_cols, output_path='results.png'):
    """
    Create comprehensive visualization of model performance
    """
    X_test, y_test, y_pred = regression_results
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=50, color='steelblue')
    max_val = max(y_test.max(), y_pred.max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Issue Count', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Issue Count', fontsize=11)
    axes[0, 0].set_title('Actual vs Predicted Issue Counts', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Feature Importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(10)
    
    axes[0, 1].barh(importance_df['feature'], importance_df['importance'], color='coral')
    axes[0, 1].set_xlabel('Importance Score', fontsize=11)
    axes[0, 1].set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # 3. Spatial Heatmap
    X_all = grid_features[feature_cols]
    grid_features['predicted_issues'] = model.predict(X_all)
    
    scatter = axes[1, 0].scatter(
        grid_features['center_lon'], 
        grid_features['center_lat'], 
        c=grid_features['predicted_issues'], 
        cmap='YlOrRd', 
        s=80, 
        alpha=0.7, 
        edgecolors='black', 
        linewidth=0.5
    )
    axes[1, 0].set_xlabel('Longitude', fontsize=11)
    axes[1, 0].set_ylabel('Latitude', fontsize=11)
    axes[1, 0].set_title('Predicted Risk Heatmap (Red = High Risk)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=axes[1, 0], label='Predicted Issues')
    
    # 4. Distribution Comparison
    axes[1, 1].hist(grid_features['issue_count'], bins=30, alpha=0.6, 
                   label='Actual', color='blue', edgecolor='black')
    axes[1, 1].hist(grid_features['predicted_issues'], bins=30, alpha=0.6, 
                   label='Predicted', color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Issue Count', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Distribution: Actual vs Predicted', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")


# ============================================================================
# STEP 5: PREDICT NEW LOCATIONS
# ============================================================================

def predict_risk(model, feature_cols, le, lon, lat, neighborhood='Unknown', 
                 avg_severity=3.0, existing_issues=None):
    """
    Predict accessibility risk for a new location
    
    Args:
        model: Trained regression model
        feature_cols: List of feature names
        le: LabelEncoder for neighborhoods
        lon, lat: Coordinates
        neighborhood: Neighborhood name
        avg_severity: Average severity of existing issues
        existing_issues: Dict of issue type counts (e.g., {'CurbRamp': 5, 'SurfaceProblem': 10})
    
    Returns:
        Predicted issue count
    """
    if existing_issues is None:
        existing_issues = {}
    
    # Encode neighborhood
    try:
        neighborhood_encoded = le.transform([neighborhood])[0]
    except:
        neighborhood_encoded = 0  # Unknown neighborhood
    
    # Create feature vector
    features = {
        'center_lon': lon,
        'center_lat': lat,
        'avg_severity': avg_severity,
        'max_severity': avg_severity,
        'std_severity': 0,
        'temp_count': 0,
        'issue_type_diversity': len(existing_issues),
        'type_CurbRamp': existing_issues.get('CurbRamp', 0),
        'type_NoCurbRamp': existing_issues.get('NoCurbRamp', 0),
        'type_NoSidewalk': existing_issues.get('NoSidewalk', 0),
        'type_Obstacle': existing_issues.get('Obstacle', 0),
        'type_Occlusion': existing_issues.get('Occlusion', 0),
        'type_Other': existing_issues.get('Other', 0),
        'type_SurfaceProblem': existing_issues.get('SurfaceProblem', 0),
        'neighborhood_encoded': neighborhood_encoded
    }
    
    X_new = pd.DataFrame([features])[feature_cols]
    prediction = model.predict(X_new)[0]
    
    return prediction


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load and prepare data
    grid_features = load_and_preprocess('everydayLife_cleaned_dataset.csv')
    
    # Train regression model
    reg_model, reg_results, feature_cols, label_encoder = train_regression_model(grid_features)
    
    # Train classification model
    clf_model, clf_results = train_classification_model(grid_features)
    
    # Visualize
    visualize_results(grid_features, reg_results, reg_model, feature_cols)
    
    # Save models
    with open('accessibility_regression_model.pkl', 'wb') as f:
        pickle.dump(reg_model, f)
    with open('accessibility_classification_model.pkl', 'wb') as f:
        pickle.dump(clf_model, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("\n" + "="*70)
    print("✓ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("="*70)
    print("\nSaved files:")
    print("  • accessibility_regression_model.pkl")
    print("  • accessibility_classification_model.pkl")
    print("  • label_encoder.pkl")
    print("  • results.png")
    
    # Example prediction
    print("\n" + "="*70)
    print("EXAMPLE PREDICTION")
    print("="*70)
    example_pred = predict_risk(
        reg_model, feature_cols, label_encoder,
        lon=-122.33, lat=47.61, 
        neighborhood='Capitol Hill',
        existing_issues={'NoCurbRamp': 15, 'SurfaceProblem': 8}
    )
    print(f"Predicted issues for Capitol Hill location: {example_pred:.0f}")
    print("="*70)
