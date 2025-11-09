import pandas as pd
import os

print("="*70)
print("CHECKING DAY 2 OUTPUTS")
print("="*70)

# Check if files exist
files_to_check = [
    'product_features_complete.csv',
    'product_ml_dataset.csv'
]

print("\n1. FILE EXISTENCE CHECK:")
for file in files_to_check:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024**2)
        print(f"  ✓ {file} - {size_mb:.1f} MB")
    else:
        print(f"  ✗ {file} - NOT FOUND!")

# Load and check ML dataset
print("\n" + "="*70)
print("2. ML DATASET VALIDATION")
print("="*70)

try:
    ml_data = pd.read_csv('product_ml_dataset.csv')
    
    print(f"\n✓ Loaded ML dataset successfully")
    print(f"  Rows: {len(ml_data):,}")
    print(f"  Columns: {len(ml_data.columns)}")
    
    print("\n3. FEATURE COLUMNS:")
    for i, col in enumerate(ml_data.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print("\n4. SUCCESS LABEL DISTRIBUTION:")
    if 'is_success' in ml_data.columns:
        success_dist = ml_data['is_success'].value_counts()
        print(f"  Successful (1): {success_dist.get(1, 0):,} ({success_dist.get(1, 0)/len(ml_data)*100:.1f}%)")
        print(f"  Not successful (0): {success_dist.get(0, 0):,} ({success_dist.get(0, 0)/len(ml_data)*100:.1f}%)")
    
    print("\n5. SAMPLE DATA (first 3 rows):")
    print(ml_data.head(3))
    
    print("\n6. MISSING VALUES CHECK:")
    missing = ml_data.isnull().sum()
    if missing.sum() > 0:
        print("\nColumns with missing values:")
        print(missing[missing > 0])
    else:
        print("  ✓ No missing values!")
    
    print("\n" + "="*70)
    print("✅ DATA VALIDATION COMPLETE - READY FOR ML TRAINING!")
    print("="*70)
    
except FileNotFoundError:
    print("\n✗ ERROR: product_ml_dataset.csv not found!")
    print("  Please re-run: python day2_feature_engineering.py")

except Exception as e:
    print(f"\n✗ ERROR: {e}")