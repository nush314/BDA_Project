import pandas as pd
import pickle

print("="*70)
print("EXTRACTING FEATURE IMPORTANCE FROM TRAINED MODEL")
print("="*70)

# Load the saved model
print("\nLoading saved model...")
with open('best_model.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    
best_model = saved_data['model']
feature_columns = saved_data['features']

print(f"✓ Loaded model: {type(best_model).__name__}")
print(f"✓ Features: {len(feature_columns)}")

# Extract feature importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE RANKINGS")
    print("="*70)
    
    for i, row in enumerate(feature_importance_df.itertuples(), 1):
        print(f"  {i}. {row.Feature:30s}: {row.Importance:.4f}")
    
    print("\n" + "="*70)
    print("VALUES FOR POWER BI TABLE (Copy this!):")
    print("="*70)
    print("\nFeature                        | Importance")
    print("-" * 50)
    for row in feature_importance_df.itertuples():
        print(f"{row.Feature:30s} | {row.Importance:.4f}")
    
    # Save to CSV
    feature_importance_df.to_csv('feature_importance_for_powerbi.csv', index=False)
    print("\n✓ Saved to: feature_importance_for_powerbi.csv")
    
else:
    print("\n⚠️ This model doesn't have feature_importances_ attribute")
    print("(This happens with Logistic Regression - it uses coefficients instead)")