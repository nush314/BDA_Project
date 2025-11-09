import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("DAY 2 - FEATURE ENGINEERING FOR PRODUCT SUCCESS PREDICTION")
print("="*70)

# =====================================================
# STEP 1: LOAD DATA
# =====================================================
print("\n" + "="*70)
print("STEP 1: LOADING DATA")
print("="*70)

print("\nLoading reviews with sentiment...")
reviews_df = pd.read_csv('clothing_1.5M_with_sentiment.csv', parse_dates=['date', 'year_month'])

print("\nLoading brand reputation scores...")
brand_scores = pd.read_csv('brand_reputation_scores.csv')

print(f"✓ Loaded {len(reviews_df):,} reviews")
print(f"✓ Loaded {len(brand_scores):,} brand reputation scores")

# =====================================================
# STEP 2: PRODUCT-LEVEL AGGREGATION
# =====================================================
print("\n" + "="*70)
print("STEP 2: CREATING PRODUCT-LEVEL FEATURES")
print("="*70)

print("\nAggregating reviews by product...")

# Group by product
product_data = reviews_df.groupby('parent_asin').agg({
    'rating': ['mean', 'std', 'count'],
    'verified_purchase': 'mean',
    'helpful_vote': ['mean', 'sum'],
    'sentiment_score': 'mean',
    'sentiment_label': lambda x: (x == 'positive').sum() / len(x),
    'text_length': 'mean',
    'num_customer_images': 'sum',
    'date': ['min', 'max'],
    'brand': 'first'
}).reset_index()

# Flatten column names
product_data.columns = [
    'parent_asin', 
    'avg_rating', 'rating_std', 'review_count',
    'verified_ratio',
    'avg_helpful_votes', 'total_helpful_votes',
    'avg_sentiment',
    'positive_ratio',
    'avg_text_length',
    'total_customer_images',
    'first_review_date',
    'last_review_date',
    'brand'
]

# Calculate product lifespan
product_data['product_lifespan_days'] = (
    product_data['last_review_date'] - product_data['first_review_date']
).dt.days

# Review velocity (reviews per day)
product_data['review_velocity'] = product_data['review_count'] / (product_data['product_lifespan_days'] + 1)

print(f"✓ Created features for {len(product_data):,} products")

# =====================================================
# STEP 3: EARLY PERIOD FEATURES
# =====================================================
print("\n" + "="*70)
print("STEP 3: CALCULATING EARLY PERIOD FEATURES (First 30 Days)")
print("="*70)

print("\nCalculating early metrics for each product...")

early_features_list = []

for product_id in product_data['parent_asin']:
    product_reviews = reviews_df[reviews_df['parent_asin'] == product_id].sort_values('date')
    
    if len(product_reviews) == 0:
        continue
    
    first_date = product_reviews['date'].min()
    early_cutoff = first_date + timedelta(days=30)
    
    early_reviews = product_reviews[product_reviews['date'] <= early_cutoff]
    
    if len(early_reviews) > 0:
        early_features_list.append({
            'parent_asin': product_id,
            'early_review_count': len(early_reviews),
            'early_avg_rating': early_reviews['rating'].mean(),
            'early_sentiment': early_reviews['sentiment_score'].mean(),
            'early_positive_ratio': (early_reviews['sentiment_label'] == 'positive').sum() / len(early_reviews),
            'early_verified_ratio': early_reviews['verified_purchase'].mean(),
            'early_helpful_votes': early_reviews['helpful_vote'].mean(),
            'early_review_velocity': len(early_reviews) / 30
        })
    
    # Progress update
    if len(early_features_list) % 10000 == 0:
        print(f"  Processed {len(early_features_list):,} products...")

early_features_df = pd.DataFrame(early_features_list)

print(f"✓ Calculated early features for {len(early_features_df):,} products")

# Merge with product data
product_data = product_data.merge(early_features_df, on='parent_asin', how='left')

# =====================================================
# STEP 4: ADD BRAND REPUTATION SCORES
# =====================================================
print("\n" + "="*70)
print("STEP 4: INTEGRATING BRAND REPUTATION SCORES")
print("="*70)

print("\nMerging brand reputation scores...")

# Merge brand reputation
product_data = product_data.merge(
    brand_scores[['brand', 'brand_reputation_score', 'avg_rating']],
    on='brand',
    how='left',
    suffixes=('', '_brand')
)

print(f"✓ Added brand reputation for {product_data['brand_reputation_score'].notna().sum():,} products")

# =====================================================
# STEP 5: DEFINE SUCCESS LABEL
# =====================================================
print("\n" + "="*70)
print("STEP 5: DEFINING PRODUCT SUCCESS")
print("="*70)

print("\nDefining success criteria...")
print("  Success = Product in top 20% by (rating × review_count)")

# Calculate success metric
product_data['success_metric'] = product_data['avg_rating'] * product_data['review_count']

# Filter products with sufficient data (at least 10 reviews)
ml_data = product_data[product_data['review_count'] >= 10].copy()

print(f"\nFiltered to {len(ml_data):,} products with 10+ reviews")

# Define success (top 20%)
success_threshold = ml_data['success_metric'].quantile(0.80)
ml_data['is_success'] = (ml_data['success_metric'] >= success_threshold).astype(int)

success_count = ml_data['is_success'].sum()
success_rate = (success_count / len(ml_data)) * 100

print(f"\nSuccess distribution:")
print(f"  Successful products: {success_count:,} ({success_rate:.1f}%)")
print(f"  Non-successful products: {len(ml_data) - success_count:,} ({100-success_rate:.1f}%)")

# =====================================================
# STEP 6: FEATURE SELECTION
# =====================================================
print("\n" + "="*70)
print("STEP 6: SELECTING FEATURES FOR ML MODEL")
print("="*70)

# Features to use for prediction
feature_columns = [
    # Early period features (what we know in first 30 days)
    'early_review_count',
    'early_avg_rating',
    'early_sentiment',
    'early_positive_ratio',
    'early_verified_ratio',
    'early_helpful_votes',
    'early_review_velocity',
    
    # Brand features
    'brand_reputation_score',
    
    # Product metadata
    'total_customer_images'
]

# Remove rows with missing values in key features
ml_data_clean = ml_data.dropna(subset=feature_columns + ['is_success'])

print(f"\nML dataset size: {len(ml_data_clean):,} products")
print(f"\nFeatures for prediction:")
for i, feat in enumerate(feature_columns, 1):
    print(f"  {i}. {feat}")

# =====================================================
# STEP 7: SAVE PREPARED DATA
# =====================================================
print("\n" + "="*70)
print("STEP 7: SAVING PREPARED DATA")
print("="*70)

# Save full product data
product_file = 'product_features_complete.csv'
print(f"\nSaving {product_file}...")
product_data.to_csv(product_file, index=False)
print(f"✓ Saved {len(product_data):,} products")

# Save ML-ready data
ml_file = 'product_ml_dataset.csv'
print(f"\nSaving {ml_file}...")
ml_data_clean.to_csv(ml_file, index=False)
print(f"✓ Saved {len(ml_data_clean):,} products for ML")

import os
total_size = (
    os.path.getsize(product_file) +
    os.path.getsize(ml_file)
) / (1024**2)

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*70)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("="*70)

print("\nDELIVERABLES:")
print(f"  ✓ {product_file}")
print(f"  ✓ {ml_file}")
print(f"  Total size: {total_size:.1f} MB")

print("\nML DATASET SUMMARY:")
print(f"  • {len(ml_data_clean):,} products ready for training")
print(f"  • {len(feature_columns)} features")
print(f"  • {success_count:,} successful products ({success_rate:.1f}%)")

print("\nNEXT STEP:")
print("  Run: python day2_ml_training.py")

print("\n" + "="*70)