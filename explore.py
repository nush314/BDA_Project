import pandas as pd
import json
import gzip

def load_jsonl_sample(file_path, num_lines=10000):
    """
    Load sample from JSONL file (works with both .gz and plain .jsonl)
    """
    data = []
    
    # Check if gzipped
    try:
        if file_path.endswith('.gz'):
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
                    
                if i % 1000 == 0 and i > 0:
                    print(f"Loaded {i:,} lines...")
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error: {e}")
        return None

# =====================================================
# LOAD CLOTHING REVIEWS SAMPLE
# =====================================================
print("="*60)
print("LOADING CLOTHING_SHOES_AND_JEWELRY REVIEWS")
print("="*60)

# REPLACE WITH YOUR ACTUAL FILE PATH
reviews_file = r"C:\Users\ajkan\Desktop\bda_amazon\Clothing_Shoes_and_Jewelry.jsonl"  # or .jsonl.gz if compressed
reviews_sample = load_jsonl_sample(reviews_file, num_lines=50000)

if reviews_sample is not None:
    print(f"\n✓ Loaded {len(reviews_sample):,} reviews")
    print(f"\nColumns: {reviews_sample.columns.tolist()}")
    print(f"\nSample Review:")
    print(reviews_sample.head(1).to_dict('records'))
else:
    print("Failed to load reviews!")

# =====================================================
# LOAD METADATA SAMPLE
# =====================================================
print("\n" + "="*60)
print("LOADING CLOTHING_SHOES_AND_JEWELRY METADATA")
print("="*60)

metadata_file = r"C:\Users\ajkan\Desktop\bda_amazon\meta_Clothing_Shoes_and_Jewelry.jsonl"  # or .jsonl.gz
metadata_sample = load_jsonl_sample(metadata_file, num_lines=50000)

if metadata_sample is not None:
    print(f"\n✓ Loaded {len(metadata_sample):,} metadata records")
    print(f"\nColumns: {metadata_sample.columns.tolist()}")
    print(f"\nSample Metadata:")
    print(metadata_sample.head(1).to_dict('records'))
else:
    print("Failed to load metadata!")