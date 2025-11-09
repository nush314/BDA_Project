import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DAY 2 - ML MODEL TRAINING: PRODUCT SUCCESS PREDICTION")
print("="*70)

# =====================================================
# STEP 1: LOAD ML DATASET
# =====================================================
print("\n" + "="*70)
print("STEP 1: LOADING ML DATASET")
print("="*70)

print("\nLoading product ML dataset...")
df = pd.read_csv('product_ml_dataset.csv')

print(f"✓ Loaded {len(df):,} products")
print(f"  Successful: {df['is_success'].sum():,} ({df['is_success'].mean()*100:.1f}%)")
print(f"  Not successful: {(1-df['is_success']).sum():,} ({(1-df['is_success'].mean())*100:.1f}%)")

# =====================================================
# STEP 2: PREPARE FEATURES
# =====================================================
print("\n" + "="*70)
print("STEP 2: PREPARING FEATURES")
print("="*70)

# Feature columns (early period + brand)
feature_columns = [
    'early_review_count',
    'early_avg_rating',
    'early_sentiment',
    'early_positive_ratio',
    'early_verified_ratio',
    'early_helpful_votes',
    'early_review_velocity',
    'brand_reputation_score',
    'total_customer_images'
]

print("\nFeatures for prediction:")
for i, feat in enumerate(feature_columns, 1):
    print(f"  {i}. {feat}")

# Prepare X and y
X = df[feature_columns]
y = df['is_success']

print(f"\n✓ Feature matrix: {X.shape}")
print(f"✓ Target vector: {y.shape}")

# =====================================================
# STEP 3: TRAIN-TEST SPLIT
# =====================================================
print("\n" + "="*70)
print("STEP 3: SPLITTING DATA")
print("="*70)

print("\nSplitting into train (70%), validation (15%), test (15%)...")

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Second split: 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"✓ Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"✓ Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# =====================================================
# STEP 4: FEATURE SCALING
# =====================================================
print("\n" + "="*70)
print("STEP 4: SCALING FEATURES")
print("="*70)

print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Features standardized (mean=0, std=1)")

# =====================================================
# STEP 5: TRAIN MODELS
# =====================================================
print("\n" + "="*70)
print("STEP 5: TRAINING ML MODELS")
print("="*70)

models = {}
results = []

# Model 1: Logistic Regression
print("\n1. TRAINING LOGISTIC REGRESSION (Baseline)...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr_model
print("✓ Logistic Regression trained")

# Model 2: Random Forest
print("\n2. TRAINING RANDOM FOREST...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
models['Random Forest'] = rf_model
print("✓ Random Forest trained")

# Model 3: Gradient Boosting
print("\n3. TRAINING GRADIENT BOOSTING (XGBoost-like)...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
models['Gradient Boosting'] = gb_model
print("✓ Gradient Boosting trained")

# =====================================================
# STEP 6: EVALUATE MODELS
# =====================================================
print("\n" + "="*70)
print("STEP 6: EVALUATING MODELS ON VALIDATION SET")
print("="*70)

for model_name, model in models.items():
    print(f"\n{'='*70}")
    print(f"{model_name.upper()}")
    print(f"{'='*70}")
    
    # Predictions
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(f"  True Negatives:  {cm[0,0]:,}")
    print(f"  False Positives: {cm[0,1]:,}")
    print(f"  False Negatives: {cm[1,0]:,}")
    print(f"  True Positives:  {cm[1,1]:,}")
    
    # Store results
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })

# =====================================================
# STEP 7: SELECT BEST MODEL
# =====================================================
print("\n" + "="*70)
print("STEP 7: MODEL COMPARISON & SELECTION")
print("="*70)

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
best_model = models[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   F1-Score: {results_df.loc[results_df['Model'] == best_model_name, 'F1-Score'].values[0]:.4f}")

# =====================================================
# STEP 8: FINAL TEST SET EVALUATION
# =====================================================
print("\n" + "="*70)
print("STEP 8: FINAL EVALUATION ON TEST SET")
print("="*70)

print(f"\nEvaluating {best_model_name} on held-out test set...")

y_test_pred = best_model.predict(X_test_scaled)
y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)

print(f"\n{'='*70}")
print(f"FINAL TEST SET PERFORMANCE - {best_model_name.upper()}")
print(f"{'='*70}")

print(f"\nPerformance Metrics:")
print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  ROC-AUC:   {test_roc_auc:.4f}")

print(f"\nConfusion Matrix:")
test_cm = confusion_matrix(y_test, y_test_pred)
print(f"  True Negatives:  {test_cm[0,0]:,}")
print(f"  False Positives: {test_cm[0,1]:,}")
print(f"  False Negatives: {test_cm[1,0]:,}")
print(f"  True Positives:  {test_cm[1,1]:,}")

# =====================================================
# STEP 9: FEATURE IMPORTANCE
# =====================================================
print("\n" + "="*70)
print("STEP 9: FEATURE IMPORTANCE ANALYSIS")
print("="*70)

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance Rankings:")
    for i, row in enumerate(feature_importance_df.itertuples(), 1):
        print(f"  {i}. {row.Feature:30s}: {row.Importance:.4f}")
    
    top_feature = feature_importance_df.iloc[0]['Feature']
    print(f"\n🔑 Most Important Feature: {top_feature}")

elif hasattr(best_model, 'coef_'):
    coefficients = best_model.coef_[0]
    feature_coef_df = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\nFeature Coefficients (Logistic Regression):")
    for i, row in enumerate(feature_coef_df.itertuples(), 1):
        sign = "+" if row.Coefficient > 0 else ""
        print(f"  {i}. {row.Feature:30s}: {sign}{row.Coefficient:.4f}")

# =====================================================
# STEP 10: SAVE MODEL & RESULTS
# =====================================================
print("\n" + "="*70)
print("STEP 10: SAVING MODEL & RESULTS")
print("="*70)

# Save predictions on full dataset
print("\nGenerating predictions for all products...")
X_all_scaled = scaler.transform(X)
df['success_probability'] = best_model.predict_proba(X_all_scaled)[:, 1]
df['predicted_success'] = best_model.predict(X_all_scaled)

predictions_file = 'product_success_predictions.csv'
print(f"\nSaving {predictions_file}...")
df.to_csv(predictions_file, index=False)
print(f"✓ Saved predictions for {len(df):,} products")

# Save model results
results_file = 'ml_model_results.csv'
print(f"\nSaving {results_file}...")
results_df.to_csv(results_file, index=False)
print(f"✓ Saved model comparison results")

import pickle
model_file = 'best_model.pkl'
print(f"\nSaving {model_file}...")
with open(model_file, 'wb') as f:
    pickle.dump({'model': best_model, 'scaler': scaler, 'features': feature_columns}, f)
print(f"✓ Saved {best_model_name} model")

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*70)
print("✅ DAY 2 COMPLETE! ML MODEL TRAINED SUCCESSFULLY!")
print("="*70)

print("\nTODAY'S ACHIEVEMENTS:")
print("  ✓ Feature engineering complete (9 predictive features)")
print("  ✓ 14,765 products prepared for ML")
print("  ✓ 3 models trained and evaluated")
print(f"  ✓ Best model: {best_model_name}")
print(f"  ✓ Test accuracy: {test_accuracy*100:.2f}%")
print(f"  ✓ Test F1-Score: {test_f1:.4f}")

print("\nDELIVERABLES:")
print("  ✓ product_success_predictions.csv")
print("  ✓ ml_model_results.csv")
print("  ✓ best_model.pkl")

print("\nKEY INSIGHTS:")
print(f"  • Model can predict product success with {test_accuracy*100:.1f}% accuracy")
print(f"  • Using only first 30 days of data + brand reputation")
print(f"  • {test_roc_auc:.1%} ROC-AUC score (excellent discrimination)")

print("\nDAY 3 PREVIEW:")
print("  → Create visualizations & dashboards")
print("  → Upload data to Cloudera for Big Data proof")
print("  → Generate business insights report")

print("\n" + "="*70)
print("🎉 EXCELLENT PROGRESS! 2 DAYS DOWN, 3 TO GO!")
print("="*70)