import pandas as pd
import json
import gzip
from datetime import datetime
import os

print("="*70)
print("DAY 1: LOADING 1.5 MILLION REVIEWS")
print("="*70)

def load_jsonl_large(file_path, num_lines=1500000):
    """
    Load large JSONL file efficiently
    """
    data = []
    
    try:
        if file_path.endswith('.gz'):
            f = gzip.open(file_path, 'rt', encoding='utf-8')
        else:
            f = open(file_path, 'r', encoding='utf-8')
        
        print(f"\nLoading {num_lines:,} lines from {os.path.basename(file_path)}...")
        
        with f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                try:
                    data.append(json.loads(line))
                except:
                    continue
                
                # Progress update every 100K
                if (i + 1) % 100000 == 0:
                    print(f"  Loaded {i+1:,} lines... ({(i+1)/num_lines*100:.1f}%)")
        
        print(f"✓ Completed loading {len(data):,} records")
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error: {e}")
        return None

# =====================================================
# STEP 1: LOAD REVIEWS
# =====================================================
print("\n" + "="*70)
print("STEP 1: LOADING REVIEWS")
print("="*70)

reviews_file = r"C:\Users\ajkan\Desktop\bda_amazon\Clothing_Shoes_and_Jewelry.jsonl"
print(f"File: {reviews_file}")

# Load 1.5M reviews
reviews_df = load_jsonl_large(reviews_file, num_lines=1500000)

if reviews_df is None:
    print("ERROR: Failed to load reviews!")
    exit()

print(f"\n✓ Loaded {len(reviews_df):,} reviews")
print(f"  Columns: {reviews_df.columns.tolist()}")
print(f"  Memory usage: {reviews_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# =====================================================
# STEP 2: LOAD METADATA
# =====================================================
print("\n" + "="*70)
print("STEP 2: LOADING METADATA")
print("="*70)

metadata_file = r"C:\Users\ajkan\Desktop\bda_amazon\meta_Clothing_Shoes_and_Jewelry.jsonl"
print(f"File: {metadata_file}")

# Load 500K metadata (should cover most products)
metadata_df = load_jsonl_large(metadata_file, num_lines=500000)

if metadata_df is None:
    print("ERROR: Failed to load metadata!")
    exit()

print(f"\n✓ Loaded {len(metadata_df):,} metadata records")
print(f"  Columns: {metadata_df.columns.tolist()}")
print(f"  Memory usage: {metadata_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# =====================================================
# STEP 3: BASIC PREPROCESSING
# =====================================================
print("\n" + "="*70)
print("STEP 3: BASIC PREPROCESSING")
print("="*70)

# Convert timestamp to datetime
print("\nConverting timestamps...")
reviews_df['date'] = pd.to_datetime(reviews_df['timestamp'], unit='ms')
reviews_df['year'] = reviews_df['date'].dt.year
reviews_df['month'] = reviews_df['date'].dt.month
reviews_df['year_month'] = reviews_df['date'].dt.to_period('M')

# Calculate text length
print("Calculating text lengths...")
reviews_df['text_length'] = reviews_df['text'].fillna('').str.len()

# Count customer images
print("Counting customer images...")
reviews_df['num_customer_images'] = reviews_df['images'].apply(
    lambda x: len(x) if isinstance(x, list) else 0
)

print("✓ Preprocessing complete")

# =====================================================
# STEP 4: MERGE REVIEWS WITH METADATA
# =====================================================
print("\n" + "="*70)
print("STEP 4: MERGING REVIEWS WITH METADATA")
print("="*70)

print("\nMerging on parent_asin...")
merged_df = reviews_df.merge(
    metadata_df[['parent_asin', 'store', 'average_rating', 'rating_number', 
                 'price', 'main_category']],
    on='parent_asin',
    how='left',
    suffixes=('', '_meta')
)

print(f"✓ Merge complete")
print(f"  Total records: {len(merged_df):,}")
print(f"  Records with brand: {merged_df['store'].notna().sum():,} ({merged_df['store'].notna().sum()/len(merged_df)*100:.1f}%)")

# Add brand column
merged_df['brand'] = merged_df['store']

# =====================================================
# STEP 5: DATA QUALITY SUMMARY
# =====================================================
print("\n" + "="*70)
print("STEP 5: DATA QUALITY SUMMARY")
print("="*70)

print("\n1. DATASET SIZE:")
print(f"  Total reviews: {len(merged_df):,}")
print(f"  Reviews with brands: {merged_df['brand'].notna().sum():,}")
print(f"  Unique products: {merged_df['parent_asin'].nunique():,}")
print(f"  Unique users: {merged_df['user_id'].nunique():,}")
print(f"  Unique brands: {merged_df['brand'].nunique():,}")

print("\n2. DATE RANGE:")
print(f"  Earliest: {merged_df['date'].min()}")
print(f"  Latest: {merged_df['date'].max()}")
print(f"  Years: {(merged_df['date'].max() - merged_df['date'].min()).days / 365:.1f}")

print("\n3. RATING DISTRIBUTION:")
rating_dist = merged_df['rating'].value_counts().sort_index()
for rating, count in rating_dist.items():
    pct = count / len(merged_df) * 100
    print(f"  {rating} stars: {count:,} ({pct:.1f}%)")

print("\n4. VERIFIED PURCHASES:")
verified_pct = merged_df['verified_purchase'].sum() / len(merged_df) * 100
print(f"  Verified: {merged_df['verified_purchase'].sum():,} ({verified_pct:.1f}%)")

print("\n5. TOP 20 BRANDS:")
brand_counts = merged_df[merged_df['brand'].notna()].groupby('brand').size().sort_values(ascending=False)
for i, (brand, count) in enumerate(brand_counts.head(20).items(), 1):
    print(f"  {i:2d}. {brand:30s}: {count:5,} reviews")

# =====================================================
# STEP 6: SAVE PROCESSED DATA
# =====================================================
print("\n" + "="*70)
print("STEP 6: SAVING PROCESSED DATA")
print("="*70)

# Save full merged dataset
output_file = 'clothing_1.5M_processed.csv'
print(f"\nSaving to {output_file}...")
merged_df.to_csv(output_file, index=False)

file_size_mb = os.path.getsize(output_file) / (1024**2)
file_size_gb = file_size_mb / 1024

print(f"✓ Saved successfully!")
print(f"  File: {output_file}")
print(f"  Size: {file_size_mb:.1f} MB ({file_size_gb:.2f} GB)")
print(f"  Records: {len(merged_df):,}")

# Also save just branded reviews (smaller, faster to work with)
branded_only = merged_df[merged_df['brand'].notna()].copy()
branded_file = 'clothing_1.5M_branded_only.csv'
print(f"\nSaving branded reviews only to {branded_file}...")
branded_only.to_csv(branded_file, index=False)

branded_size_mb = os.path.getsize(branded_file) / (1024**2)
print(f"✓ Saved successfully!")
print(f"  File: {branded_file}")
print(f"  Size: {branded_size_mb:.1f} MB")
print(f"  Records: {len(branded_only):,}")

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*70)
print("✅ DAY 1 TASK 1 COMPLETE!")
print("="*70)

print("\nDELIVERABLES:")
print(f"  ✓ {output_file} ({file_size_gb:.2f} GB)")
print(f"  ✓ {branded_file} ({branded_size_mb:.1f} MB)")

print("\nDATA SUMMARY:")
print(f"  • {len(merged_df):,} total reviews")
print(f"  • {branded_only.shape[0]:,} branded reviews")
print(f"  • {merged_df['brand'].nunique():,} unique brands")
print(f"  • {(merged_df['date'].max() - merged_df['date'].min()).days / 365:.1f} years of data")

print("\nNEXT STEP:")
print("  Run: python day1_sentiment.py")

print("\n" + "="*70)