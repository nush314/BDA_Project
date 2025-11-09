import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from datetime import datetime

print("="*70)
print("DAY 1 - TASK 2: SENTIMENT ANALYSIS")
print("="*70)

# =====================================================
# STEP 1: LOAD PROCESSED DATA
# =====================================================
print("\n" + "="*70)
print("STEP 1: LOADING PROCESSED DATA")
print("="*70)

# Load the branded reviews (smaller, faster)
print("\nLoading branded reviews...")
df = pd.read_csv('clothing_1.5M_branded_only.csv')

print(f"✓ Loaded {len(df):,} branded reviews")
print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# =====================================================
# STEP 2: INITIALIZE VADER SENTIMENT ANALYZER
# =====================================================
print("\n" + "="*70)
print("STEP 2: INITIALIZING SENTIMENT ANALYZER")
print("="*70)

analyzer = SentimentIntensityAnalyzer()
print("✓ VADER Sentiment Analyzer initialized")

# =====================================================
# STEP 3: APPLY SENTIMENT ANALYSIS
# =====================================================
print("\n" + "="*70)
print("STEP 3: ANALYZING SENTIMENT (This will take 15-20 minutes)")
print("="*70)

def get_sentiment_scores(text):
    """
    Get sentiment scores for text using VADER
    Returns compound score (-1 to +1)
    """
    if pd.isna(text) or text == '':
        return 0.0
    
    try:
        scores = analyzer.polarity_scores(str(text))
        return scores['compound']
    except:
        return 0.0

def get_sentiment_label(score):
    """
    Convert compound score to label
    """
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

print("\nProcessing sentiment for all reviews...")
print("Progress updates every 100K reviews:\n")

# Process in chunks for progress tracking
chunk_size = 100000
total_chunks = len(df) // chunk_size + 1

sentiment_scores = []

for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunk_scores = chunk['text'].apply(get_sentiment_scores)
    sentiment_scores.extend(chunk_scores)
    
    current = min(i + chunk_size, len(df))
    pct = (current / len(df)) * 100
    print(f"  Processed {current:,} / {len(df):,} ({pct:.1f}%)")

print("\n✓ Sentiment analysis complete!")

# Add sentiment columns
df['sentiment_score'] = sentiment_scores
df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)

# =====================================================
# STEP 4: SENTIMENT ANALYSIS SUMMARY
# =====================================================
print("\n" + "="*70)
print("STEP 4: SENTIMENT ANALYSIS RESULTS")
print("="*70)

print("\n1. SENTIMENT DISTRIBUTION:")
sentiment_dist = df['sentiment_label'].value_counts()
for label, count in sentiment_dist.items():
    pct = count / len(df) * 100
    print(f"  {label.capitalize():10s}: {count:,} ({pct:.1f}%)")

print("\n2. SENTIMENT SCORE STATISTICS:")
print(df['sentiment_score'].describe())

print("\n3. SENTIMENT BY RATING:")
sentiment_by_rating = df.groupby('rating')['sentiment_score'].mean()
print("\nAverage sentiment score by star rating:")
for rating, score in sentiment_by_rating.items():
    print(f"  {rating} stars: {score:+.3f}")

print("\n4. SAMPLE REVIEWS WITH SENTIMENT:")
print("\nMost Positive Review:")
most_positive = df.loc[df['sentiment_score'].idxmax()]
print(f"  Rating: {most_positive['rating']}")
print(f"  Sentiment: {most_positive['sentiment_score']:+.3f}")
print(f"  Text: {most_positive['text'][:200]}...")

print("\nMost Negative Review:")
most_negative = df.loc[df['sentiment_score'].idxmin()]
print(f"  Rating: {most_negative['rating']}")
print(f"  Sentiment: {most_negative['sentiment_score']:+.3f}")
print(f"  Text: {most_negative['text'][:200]}...")

print("\n5. BRAND SENTIMENT PREVIEW (Top 10 Brands):")
brand_sentiment = df.groupby('brand').agg({
    'sentiment_score': 'mean',
    'sentiment_label': lambda x: (x == 'positive').sum() / len(x) * 100
}).round(3)
brand_sentiment.columns = ['avg_sentiment_score', 'positive_pct']
brand_sentiment = brand_sentiment.sort_values('avg_sentiment_score', ascending=False)

print("\nBrands with highest sentiment:")
print(brand_sentiment.head(10))

# =====================================================
# STEP 5: SAVE DATA WITH SENTIMENT
# =====================================================
print("\n" + "="*70)
print("STEP 5: SAVING DATA WITH SENTIMENT")
print("="*70)

output_file = 'clothing_1.5M_with_sentiment.csv'
print(f"\nSaving to {output_file}...")
df.to_csv(output_file, index=False)

import os
file_size_mb = os.path.getsize(output_file) / (1024**2)
file_size_gb = file_size_mb / 1024

print(f"✓ Saved successfully!")
print(f"  File: {output_file}")
print(f"  Size: {file_size_mb:.1f} MB ({file_size_gb:.2f} GB)")
print(f"  Records: {len(df):,}")

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*70)
print("✅ DAY 1 TASK 2 COMPLETE!")
print("="*70)

print("\nDELIVERABLES:")
print(f"  ✓ {output_file} ({file_size_gb:.2f} GB)")

print("\nSENTIMENT SUMMARY:")
print(f"  • {len(df):,} reviews analyzed")
print(f"  • Positive: {(df['sentiment_label'] == 'positive').sum():,} ({(df['sentiment_label'] == 'positive').sum()/len(df)*100:.1f}%)")
print(f"  • Neutral: {(df['sentiment_label'] == 'neutral').sum():,} ({(df['sentiment_label'] == 'neutral').sum()/len(df)*100:.1f}%)")
print(f"  • Negative: {(df['sentiment_label'] == 'negative').sum():,} ({(df['sentiment_label'] == 'negative').sum()/len(df)*100:.1f}%)")

print("\nNEXT STEP:")
print("  Run: python day1_brand_reputation.py")

print("\n" + "="*70)