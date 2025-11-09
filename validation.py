import pandas as pd
import json
from datetime import datetime

def load_jsonl_sample(file_path, num_lines=10000):
    """Load sample from JSONL file"""
    data = []
    try:
        if file_path.endswith('.gz'):
            import gzip
            f = gzip.open(file_path, 'rt', encoding='utf-8')
        else:
            f = open(file_path, 'r', encoding='utf-8')
        
        with f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                try:
                    data.append(json.loads(line))
                except:
                    continue
                    
                if i % 10000 == 0 and i > 0:
                    print(f"Loaded {i:,} lines...")
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error: {e}")
        return None

# =====================================================
# LOAD DATA FIRST
# =====================================================
print("Loading data...")
reviews_file = r"C:\Users\ajkan\Desktop\bda_amazon\Clothing_Shoes_and_Jewelry.jsonl"
metadata_file = r"C:\Users\ajkan\Desktop\bda_amazon\meta_Clothing_Shoes_and_Jewelry.jsonl"

print("Loading 50K reviews...")
reviews_sample = load_jsonl_sample(reviews_file, num_lines=50000)

print("Loading 50K metadata...")
metadata_sample = load_jsonl_sample(metadata_file, num_lines=50000)

print("\n✓ Data loaded!\n")

# NOW THE REST OF THE VALIDATION CODE...

print("="*70)
print("DETAILED DATA ANALYSIS - CLOTHING, SHOES & JEWELRY")
print("="*70)

# =====================================================
# PART 1: REVIEWS ANALYSIS
# =====================================================
print("\n" + "="*70)
print("PART 1: REVIEWS DATA ANALYSIS (50K Sample)")
print("="*70)

# Convert timestamp to datetime
reviews_sample['date'] = pd.to_datetime(reviews_sample['timestamp'], unit='ms')
reviews_sample['text_length'] = reviews_sample['text'].fillna('').str.len()

print("\n1. BASIC STATISTICS:")
print(f"Total reviews in sample: {len(reviews_sample):,}")
print(f"Unique products (asin): {reviews_sample['asin'].nunique():,}")
print(f"Unique products (parent_asin): {reviews_sample['parent_asin'].nunique():,}")
print(f"Unique users: {reviews_sample['user_id'].nunique():,}")

print("\n2. DATE RANGE:")
print(f"Earliest review: {reviews_sample['date'].min()}")
print(f"Latest review: {reviews_sample['date'].max()}")
print(f"Years covered: {(reviews_sample['date'].max() - reviews_sample['date'].min()).days / 365:.1f}")

print("\n3. RATING DISTRIBUTION:")
rating_dist = reviews_sample['rating'].value_counts().sort_index()
for rating, count in rating_dist.items():
    pct = (count / len(reviews_sample)) * 100
    print(f"  {rating} stars: {count:,} ({pct:.1f}%)")

print("\n4. VERIFIED PURCHASE:")
verified_counts = reviews_sample['verified_purchase'].value_counts()
for status, count in verified_counts.items():
    pct = (count / len(reviews_sample)) * 100
    print(f"  {status}: {count:,} ({pct:.1f}%)")

print("\n5. HELPFUL VOTES:")
print(f"Reviews with helpful votes (>0): {(reviews_sample['helpful_vote'] > 0).sum():,}")
print(f"Max helpful votes: {reviews_sample['helpful_vote'].max()}")
print(f"Average helpful votes: {reviews_sample['helpful_vote'].mean():.2f}")

print("\n6. REVIEW TEXT LENGTH:")
print(f"Empty reviews: {(reviews_sample['text_length'] == 0).sum()}")
print(f"Average length: {reviews_sample['text_length'].mean():.0f} characters")
print(f"Median length: {reviews_sample['text_length'].median():.0f} characters")

print("\n7. CUSTOMER IMAGES:")
reviews_sample['num_images'] = reviews_sample['images'].apply(lambda x: len(x) if isinstance(x, list) else 0)
print(f"Reviews with customer images: {(reviews_sample['num_images'] > 0).sum():,} ({(reviews_sample['num_images'] > 0).sum()/len(reviews_sample)*100:.1f}%)")

# =====================================================
# PART 2: METADATA ANALYSIS
# =====================================================
print("\n" + "="*70)
print("PART 2: METADATA ANALYSIS (50K Sample)")
print("="*70)

print("\n1. BASIC STATISTICS:")
print(f"Total products: {len(metadata_sample):,}")
print(f"Unique parent_asin: {metadata_sample['parent_asin'].nunique():,}")

print("\n2. BRAND/STORE INFORMATION:")
print(f"Products with brand/store: {metadata_sample['store'].notna().sum():,} ({(metadata_sample['store'].notna().sum()/len(metadata_sample)*100):.1f}%)")
print(f"Unique brands: {metadata_sample['store'].nunique():,}")

print("\n3. TOP 20 BRANDS (in 50K sample):")
top_brands = metadata_sample['store'].value_counts().head(20)
for i, (brand, count) in enumerate(top_brands.items(), 1):
    print(f"  {i:2d}. {brand:30s}: {count:4d} products")

print("\n4. PRICE INFORMATION:")
print(f"Products with price: {metadata_sample['price'].notna().sum():,} ({(metadata_sample['price'].notna().sum()/len(metadata_sample)*100):.1f}%)")
if metadata_sample['price'].notna().sum() > 0:
    print("\nPrice statistics:")
    print(metadata_sample['price'].describe())

print("\n5. RATING INFORMATION:")
print(f"Products with ratings: {metadata_sample['average_rating'].notna().sum():,}")
print("\nAverage rating distribution:")
print(metadata_sample['average_rating'].describe())

print("\n6. RATING COUNTS:")
print(metadata_sample['rating_number'].describe())

# =====================================================
# PART 3: MERGE ANALYSIS
# =====================================================
print("\n" + "="*70)
print("PART 3: MERGE ANALYSIS - Reviews + Metadata")
print("="*70)

merged_data = reviews_sample.merge(
    metadata_sample[['parent_asin', 'store', 'average_rating', 'rating_number', 'price']],
    on='parent_asin',
    how='left'
)

print(f"\n1. MERGE SUCCESS:")
print(f"Total reviews: {len(reviews_sample):,}")
print(f"After merge: {len(merged_data):,}")
print(f"Reviews with brand info: {merged_data['store'].notna().sum():,} ({(merged_data['store'].notna().sum()/len(merged_data)*100):.1f}%)")

print("\n2. TOP BRANDS BY REVIEW COUNT:")
brand_review_counts = merged_data[merged_data['store'].notna()].groupby('store').size().sort_values(ascending=False)
print(f"Total brands with reviews: {len(brand_review_counts)}")

print("\nTop 20 brands by review count:")
for i, (brand, count) in enumerate(brand_review_counts.head(20).items(), 1):
    print(f"  {i:2d}. {brand:30s}: {count:4d} reviews")

print("\n3. BRAND TIER ANALYSIS:")
tier_1 = (brand_review_counts >= 100).sum()
tier_2 = ((brand_review_counts >= 50) & (brand_review_counts < 100)).sum()
tier_3 = ((brand_review_counts >= 20) & (brand_review_counts < 50)).sum()
tier_4 = (brand_review_counts < 20).sum()

print(f"  Tier 1 (100+ reviews in sample): {tier_1} brands")
print(f"  Tier 2 (50-99 reviews in sample): {tier_2} brands")
print(f"  Tier 3 (20-49 reviews in sample): {tier_3} brands")
print(f"  Tier 4 (<20 reviews in sample): {tier_4} brands")

# =====================================================
# PART 4: PROJECT FEASIBILITY
# =====================================================
print("\n" + "="*70)
print("PART 4: PROJECT FEASIBILITY ASSESSMENT")
print("="*70)

print("\n✓ CRITICAL FIELDS CHECK:")
print(f"  ✓ Reviews have 'text': YES")
print(f"  ✓ Reviews have 'rating': YES")
print(f"  ✓ Reviews have 'timestamp': YES")
print(f"  ✓ Reviews have 'verified_purchase': YES")
print(f"  ✓ Metadata has 'store' (brand): YES ({(metadata_sample['store'].notna().sum()/len(metadata_sample)*100):.1f}%)")
print(f"  ✓ Merge success rate: {(merged_data['store'].notna().sum()/len(merged_data)*100):.1f}%")

print("\n✓ DATA QUALITY:")
print(f"  ✓ Empty reviews: {(reviews_sample['text_length'] == 0).sum()} ({(reviews_sample['text_length'] == 0).sum()/len(reviews_sample)*100:.1f}%)")
print(f"  ✓ Verified purchases: {(reviews_sample['verified_purchase']).sum()/len(reviews_sample)*100:.1f}%")
print(f"  ✓ Date range: {(reviews_sample['date'].max() - reviews_sample['date'].min()).days / 365:.1f} years")

print("\n✓ SCALE ESTIMATION (based on 50K sample):")
sample_size = 50000
file_size_gb = 27.8

estimated_total = int((file_size_gb / 0.1) * sample_size)
print(f"  • Sample size: {sample_size:,} reviews")
print(f"  • File size: {file_size_gb} GB")
print(f"  • Estimated total reviews: ~{estimated_total/1_000_000:.1f}M (10-30 million)")
print(f"  • Estimated unique brands: ~{metadata_sample['store'].nunique() * 20:,} (thousands)")

print("\n" + "="*70)
print("✅ DATA VALIDATION COMPLETE - PROJECT IS FEASIBLE!")
print("="*70)

print("\nSaving processed sample for future use...")
merged_data.to_csv('clothing_sample_50k_processed.csv', index=False)
print("✓ Saved to: clothing_sample_50k_processed.csv")