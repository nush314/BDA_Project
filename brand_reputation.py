import pandas as pd
import numpy as np
from datetime import datetime

print("="*70)
print("DAY 1 - TASK 3: BRAND REPUTATION ANALYSIS")
print("="*70)

# =====================================================
# STEP 1: LOAD DATA WITH SENTIMENT
# =====================================================
print("\n" + "="*70)
print("STEP 1: LOADING DATA WITH SENTIMENT")
print("="*70)

print("\nLoading data...")
df = pd.read_csv('clothing_1.5M_with_sentiment.csv')

# Convert date columns
df['date'] = pd.to_datetime(df['date'])
df['year_month'] = pd.to_datetime(df['year_month'])

print(f"✓ Loaded {len(df):,} reviews")
print(f"  Unique brands: {df['brand'].nunique():,}")

# =====================================================
# STEP 2: CALCULATE BRAND-LEVEL METRICS
# =====================================================
print("\n" + "="*70)
print("STEP 2: CALCULATING BRAND METRICS")
print("="*70)

print("\nCalculating metrics for each brand...")

# Group by brand and calculate metrics
brand_metrics = df.groupby('brand').agg({
    'rating': ['mean', 'count', 'std'],
    'verified_purchase': 'mean',
    'helpful_vote': 'mean',
    'sentiment_score': 'mean',
    'sentiment_label': lambda x: (x == 'positive').sum() / len(x),
    'text_length': 'mean',
    'date': ['min', 'max']
}).reset_index()

# Flatten column names
brand_metrics.columns = [
    'brand', 
    'avg_rating', 'review_count', 'rating_std',
    'verified_ratio', 
    'avg_helpful_votes',
    'avg_sentiment_score',
    'positive_ratio',
    'avg_text_length',
    'first_review_date',
    'last_review_date'
]

# Calculate brand age in days
brand_metrics['brand_age_days'] = (
    brand_metrics['last_review_date'] - brand_metrics['first_review_date']
).dt.days

print(f"✓ Calculated metrics for {len(brand_metrics):,} brands")

# =====================================================
# STEP 3: CALCULATE BRAND REPUTATION SCORE (BRS)
# =====================================================
print("\n" + "="*70)
print("STEP 3: CALCULATING BRAND REPUTATION SCORE")
print("="*70)

print("\nBrand Reputation Score (BRS) Formula:")
print("  BRS = (0.25 × normalized_avg_rating) +")
print("        (0.25 × verified_ratio) +")
print("        (0.25 × positive_ratio) +")
print("        (0.25 × normalized_helpful_votes)")
print("  Scale: 0-100")

# Normalize ratings (1-5 scale to 0-1)
brand_metrics['norm_rating'] = (brand_metrics['avg_rating'] - 1) / 4

# Normalize helpful votes (cap at 10 for normalization)
brand_metrics['norm_helpful'] = brand_metrics['avg_helpful_votes'].clip(0, 10) / 10

# Calculate BRS (0-100 scale)
brand_metrics['brand_reputation_score'] = (
    brand_metrics['norm_rating'] * 0.25 +
    brand_metrics['verified_ratio'] * 0.25 +
    brand_metrics['positive_ratio'] * 0.25 +
    brand_metrics['norm_helpful'] * 0.25
) * 100

# Round to 2 decimals
brand_metrics['brand_reputation_score'] = brand_metrics['brand_reputation_score'].round(2)

print("✓ Brand Reputation Score calculated!")

# =====================================================
# STEP 4: BRAND REPUTATION ANALYSIS
# =====================================================
print("\n" + "="*70)
print("STEP 4: BRAND REPUTATION INSIGHTS")
print("="*70)

# Filter brands with significant data (50+ reviews)
significant_brands = brand_metrics[brand_metrics['review_count'] >= 50].copy()
print(f"\nBrands with 50+ reviews: {len(significant_brands):,}")

print("\n1. TOP 30 BRANDS BY REPUTATION SCORE:")
top_brands = significant_brands.nlargest(30, 'brand_reputation_score')
for i, row in enumerate(top_brands.itertuples(), 1):
    print(f"  {i:2d}. {row.brand:35s} BRS: {row.brand_reputation_score:5.2f} "
          f"({row.review_count:,} reviews, {row.avg_rating:.2f}★)")

print("\n2. TOP 30 BRANDS BY REVIEW VOLUME:")
top_volume = significant_brands.nlargest(30, 'review_count')
for i, row in enumerate(top_volume.itertuples(), 1):
    print(f"  {i:2d}. {row.brand:35s} {row.review_count:6,} reviews "
          f"(BRS: {row.brand_reputation_score:.2f}, {row.avg_rating:.2f}★)")

print("\n3. BRAND REPUTATION DISTRIBUTION:")
brs_bins = [0, 50, 60, 70, 80, 90, 100]
brs_labels = ['Poor (<50)', 'Fair (50-60)', 'Good (60-70)', 
              'Very Good (70-80)', 'Excellent (80-90)', 'Outstanding (90-100)']
significant_brands['brs_category'] = pd.cut(
    significant_brands['brand_reputation_score'], 
    bins=brs_bins, 
    labels=brs_labels
)

brs_dist = significant_brands['brs_category'].value_counts().sort_index()
for category, count in brs_dist.items():
    pct = count / len(significant_brands) * 100
    print(f"  {category:25s}: {count:4,} brands ({pct:.1f}%)")

print("\n4. CORRELATION ANALYSIS:")
print("\nCorrelation between BRS and other metrics:")
correlations = significant_brands[[
    'brand_reputation_score', 'avg_rating', 'review_count', 
    'verified_ratio', 'positive_ratio', 'avg_helpful_votes'
]].corr()['brand_reputation_score'].sort_values(ascending=False)

for metric, corr in correlations.items():
    if metric != 'brand_reputation_score':
        print(f"  {metric:25s}: {corr:+.3f}")

# =====================================================
# STEP 5: MAJOR BRAND DEEP DIVE
# =====================================================
print("\n" + "="*70)
print("STEP 5: MAJOR FASHION BRANDS ANALYSIS")
print("="*70)

major_brands = [
    'Nike', 'adidas', 'Skechers', 'New Balance', 'ASICS',
    'Under Armour', 'PUMA', 'Reebok', 
    "Levi's", 'Wrangler', 'Lee', 'Carhartt',
    'Hanes', 'Fruit of the Loom', 'Gildan',
    'Calvin Klein', 'Tommy Hilfiger',
    'Columbia', 'The North Face',
    'Crocs', 'Clarks'
]

major_brand_data = brand_metrics[brand_metrics['brand'].isin(major_brands)].copy()
major_brand_data = major_brand_data.sort_values('brand_reputation_score', ascending=False)

print(f"\nFound {len(major_brand_data)} major brands in dataset:")
print(f"\n{'Brand':<25} {'BRS':>6} {'Rating':>7} {'Reviews':>8} {'Verified':>9} {'Sentiment':>10}")
print("-" * 75)
for row in major_brand_data.itertuples():
    print(f"{row.brand:<25} {row.brand_reputation_score:6.2f} "
          f"{row.avg_rating:7.2f} {row.review_count:8,} "
          f"{row.verified_ratio*100:8.1f}% {row.avg_sentiment_score:9.3f}")

# =====================================================
# STEP 6: TIME-SERIES BRAND REPUTATION
# =====================================================
print("\n" + "="*70)
print("STEP 6: MONTHLY BRAND REPUTATION TRENDS")
print("="*70)

print("\nCalculating monthly BRS for top brands...")

# Focus on top 20 brands by volume
top_20_brands = brand_metrics.nlargest(20, 'review_count')['brand'].tolist()

# Filter data for top brands
top_brands_df = df[df['brand'].isin(top_20_brands)].copy()

# Calculate monthly metrics
monthly_brand_metrics = top_brands_df.groupby(['brand', 'year_month']).agg({
    'rating': 'mean',
    'verified_purchase': 'mean',
    'sentiment_label': lambda x: (x == 'positive').sum() / len(x),
    'helpful_vote': 'mean',
    'asin': 'count'  # review count
}).reset_index()

monthly_brand_metrics.columns = [
    'brand', 'year_month', 'avg_rating', 'verified_ratio', 
    'positive_ratio', 'avg_helpful', 'review_count'
]

# Calculate monthly BRS
monthly_brand_metrics['norm_rating'] = (monthly_brand_metrics['avg_rating'] - 1) / 4
monthly_brand_metrics['norm_helpful'] = monthly_brand_metrics['avg_helpful'].clip(0, 10) / 10

monthly_brand_metrics['monthly_brs'] = (
    monthly_brand_metrics['norm_rating'] * 0.25 +
    monthly_brand_metrics['verified_ratio'] * 0.25 +
    monthly_brand_metrics['positive_ratio'] * 0.25 +
    monthly_brand_metrics['norm_helpful'] * 0.25
) * 100

print(f"✓ Calculated monthly BRS for {len(top_20_brands)} brands")
print(f"  Total monthly data points: {len(monthly_brand_metrics):,}")

# =====================================================
# STEP 7: SAVE ALL RESULTS
# =====================================================
print("\n" + "="*70)
print("STEP 7: SAVING RESULTS")
print("="*70)

# Save brand-level metrics
brand_file = 'brand_reputation_scores.csv'
print(f"\nSaving {brand_file}...")
brand_metrics.to_csv(brand_file, index=False)
print(f"✓ Saved {len(brand_metrics):,} brands")

# Save monthly trends
monthly_file = 'brand_reputation_monthly.csv'
print(f"\nSaving {monthly_file}...")
monthly_brand_metrics.to_csv(monthly_file, index=False)
print(f"✓ Saved {len(monthly_brand_metrics):,} monthly data points")

# Save significant brands only
significant_file = 'brand_reputation_significant.csv'
print(f"\nSaving {significant_file}...")
significant_brands.to_csv(significant_file, index=False)
print(f"✓ Saved {len(significant_brands):,} significant brands (50+ reviews)")

import os
total_size = (
    os.path.getsize(brand_file) + 
    os.path.getsize(monthly_file) + 
    os.path.getsize(significant_file)
) / (1024**2)

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*70)
print("✅ DAY 1 COMPLETE! AMAZING PROGRESS!")
print("="*70)

print("\nTODAY'S DELIVERABLES:")
print("  ✓ clothing_1.5M_processed.csv (580 MB)")
print("  ✓ clothing_1.5M_with_sentiment.csv (289 MB)")
print("  ✓ brand_reputation_scores.csv")
print("  ✓ brand_reputation_monthly.csv")
print("  ✓ brand_reputation_significant.csv")
print(f"  TOTAL SIZE: ~{(580 + 289 + total_size):.0f} MB")

print("\nKEY ACHIEVEMENTS:")
print(f"  • Processed 1.5M reviews")
print(f"  • Analyzed 704,742 branded reviews")
print(f"  • Calculated sentiment for all reviews")
print(f"  • Computed BRS for {len(brand_metrics):,} brands")
print(f"  • Created monthly trends for top 20 brands")

print("\nTOP 3 BRANDS BY REPUTATION:")
for i, row in enumerate(top_brands.head(3).itertuples(), 1):
    print(f"  {i}. {row.brand} - BRS: {row.brand_reputation_score:.2f}")

print("\nDAY 2 PREVIEW:")
print("  → Feature engineering for ML model")
print("  → Product success label creation")
print("  → Initial model training")

print("\n" + "="*70)
print("🎉 GREAT WORK! REST AND CONTINUE TOMORROW!")
print("="*70)