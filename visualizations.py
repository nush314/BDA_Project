import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DAY 3 - CREATING VISUALIZATIONS & DASHBOARDS")
print("="*70)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =====================================================
# STEP 1: LOAD DATA
# =====================================================
print("\n" + "="*70)
print("STEP 1: LOADING DATA")
print("="*70)

print("\nLoading datasets...")
brand_scores = pd.read_csv('brand_reputation_scores.csv')
brand_monthly = pd.read_csv('brand_reputation_monthly.csv')
ml_predictions = pd.read_csv('product_success_predictions.csv')
ml_results = pd.read_csv('ml_model_results.csv')

print(f"✓ Loaded {len(brand_scores):,} brand reputation scores")
print(f"✓ Loaded {len(brand_monthly):,} monthly data points")
print(f"✓ Loaded {len(ml_predictions):,} product predictions")
print(f"✓ Loaded {len(ml_results)} model results")

# =====================================================
# VISUALIZATION 1: TOP BRANDS BY REPUTATION
# =====================================================
print("\n" + "="*70)
print("VISUALIZATION 1: TOP 20 BRANDS BY REPUTATION")
print("="*70)

# Filter brands with 50+ reviews
significant_brands = brand_scores[brand_scores['review_count'] >= 50].copy()
top_20_brands = significant_brands.nlargest(20, 'brand_reputation_score')

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(top_20_brands)), top_20_brands['brand_reputation_score'])

# Color bars by score
colors = plt.cm.RdYlGn(top_20_brands['brand_reputation_score'] / 100)
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_yticks(range(len(top_20_brands)))
ax.set_yticklabels(top_20_brands['brand'].values)
ax.set_xlabel('Brand Reputation Score (0-100)', fontsize=12)
ax.set_title('Top 20 Brands by Reputation Score (50+ reviews)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(top_20_brands.iterrows()):
    ax.text(row['brand_reputation_score'] + 0.5, i, 
            f"{row['brand_reputation_score']:.1f}", 
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig('viz1_top_brands_reputation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz1_top_brands_reputation.png")
plt.close()

# =====================================================
# VISUALIZATION 2: BRAND REPUTATION DISTRIBUTION
# =====================================================
print("\n" + "="*70)
print("VISUALIZATION 2: BRAND REPUTATION DISTRIBUTION")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram
axes[0].hist(significant_brands['brand_reputation_score'], bins=30, 
             color='skyblue', edgecolor='black', alpha=0.7)
axes[0].axvline(significant_brands['brand_reputation_score'].mean(), 
                color='red', linestyle='--', linewidth=2, label=f'Mean: {significant_brands["brand_reputation_score"].mean():.1f}')
axes[0].set_xlabel('Brand Reputation Score', fontsize=11)
axes[0].set_ylabel('Number of Brands', fontsize=11)
axes[0].set_title('Distribution of Brand Reputation Scores', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot by rating category
significant_brands['rating_category'] = pd.cut(
    significant_brands['avg_rating'],
    bins=[0, 3.5, 4.0, 4.5, 5.0],
    labels=['Low\n(<3.5)', 'Medium\n(3.5-4.0)', 'High\n(4.0-4.5)', 'Very High\n(4.5+)']
)

significant_brands.boxplot(column='brand_reputation_score', by='rating_category', ax=axes[1])
axes[1].set_xlabel('Average Rating Category', fontsize=11)
axes[1].set_ylabel('Brand Reputation Score', fontsize=11)
axes[1].set_title('BRS by Rating Category', fontsize=12, fontweight='bold')
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig('viz2_brs_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz2_brs_distribution.png")
plt.close()

# =====================================================
# VISUALIZATION 3: SENTIMENT TRENDS OVER TIME
# =====================================================
print("\n" + "="*70)
print("VISUALIZATION 3: MONTHLY BRAND REPUTATION TRENDS")
print("="*70)

# Convert year_month to datetime
brand_monthly['year_month'] = pd.to_datetime(brand_monthly['year_month'])

# Get top 5 brands by total reviews
top_5_brands = brand_scores.nlargest(5, 'review_count')['brand'].tolist()

# Filter monthly data for top 5 brands
top_brands_monthly = brand_monthly[brand_monthly['brand'].isin(top_5_brands)].copy()
top_brands_monthly = top_brands_monthly.sort_values('year_month')

# Plot
fig, ax = plt.subplots(figsize=(14, 7))

for brand in top_5_brands:
    brand_data = top_brands_monthly[top_brands_monthly['brand'] == brand]
    ax.plot(brand_data['year_month'], brand_data['monthly_brs'], 
            marker='o', linewidth=2, label=brand, markersize=4)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Monthly Brand Reputation Score', fontsize=12)
ax.set_title('Brand Reputation Score Trends Over Time (Top 5 Brands)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('viz3_brs_trends.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz3_brs_trends.png")
plt.close()

# =====================================================
# VISUALIZATION 4: ML MODEL COMPARISON
# =====================================================
print("\n" + "="*70)
print("VISUALIZATION 4: ML MODEL PERFORMANCE COMPARISON")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25

for i, model in enumerate(ml_results['Model']):
    values = ml_results.loc[ml_results['Model'] == model, metrics].values[0]
    axes[0].bar(x + i*width, values, width, label=model)

axes[0].set_xlabel('Metrics', fontsize=11)
axes[0].set_ylabel('Score', fontsize=11)
axes[0].set_title('ML Model Performance Comparison', fontsize=12, fontweight='bold')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metrics, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim(0, 1)

# F1-Score comparison (main metric)
colors = ['#ff9999', '#66b3ff', '#99ff99']
bars = axes[1].bar(ml_results['Model'], ml_results['F1-Score'], color=colors, edgecolor='black')
axes[1].set_ylabel('F1-Score', fontsize=11)
axes[1].set_title('F1-Score by Model (Higher is Better)', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 0.4)
axes[1].grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('viz4_ml_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz4_ml_model_comparison.png")
plt.close()

# =====================================================
# VISUALIZATION 5: FEATURE IMPORTANCE
# =====================================================
print("\n" + "="*70)
print("VISUALIZATION 5: FEATURE IMPORTANCE")
print("="*70)

# Feature importance data (from Gradient Boosting)
features = [
    'Customer Images',
    'Brand Reputation',
    'Early Sentiment',
    'Early Helpful Votes',
    'Early Avg Rating',
    'Early Review Velocity',
    'Early Verified Ratio',
    'Early Review Count',
    'Early Positive Ratio'
]

importance = [0.3875, 0.2748, 0.1292, 0.0779, 0.0359, 0.0338, 0.0299, 0.0242, 0.0068]

fig, ax = plt.subplots(figsize=(10, 7))
colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
bars = ax.barh(features, importance, color=colors_gradient, edgecolor='black')

ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance for Product Success Prediction\n(Gradient Boosting Model)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar, imp in zip(bars, importance):
    ax.text(imp + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{imp:.4f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('viz5_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz5_feature_importance.png")
plt.close()

# =====================================================
# VISUALIZATION 6: SUCCESS PREDICTION ANALYSIS
# =====================================================
print("\n" + "="*70)
print("VISUALIZATION 6: PRODUCT SUCCESS PREDICTIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 6a: Success probability distribution
axes[0, 0].hist(ml_predictions['success_probability'], bins=50, 
                color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
axes[0, 0].set_xlabel('Success Probability', fontsize=11)
axes[0, 0].set_ylabel('Number of Products', fontsize=11)
axes[0, 0].set_title('Distribution of Success Probabilities', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 6b: Success by brand reputation score
ml_predictions_with_brand = ml_predictions[ml_predictions['brand_reputation_score'].notna()].copy()
ml_predictions_with_brand['brs_category'] = pd.cut(
    ml_predictions_with_brand['brand_reputation_score'],
    bins=[0, 60, 70, 80, 100],
    labels=['Fair\n(0-60)', 'Good\n(60-70)', 'Very Good\n(70-80)', 'Excellent\n(80+)']
)

success_by_brs = ml_predictions_with_brand.groupby('brs_category')['is_success'].mean()
bars = axes[0, 1].bar(success_by_brs.index, success_by_brs.values, 
                      color=['#ff9999', '#ffcc99', '#99ccff', '#99ff99'], edgecolor='black')
axes[0, 1].set_ylabel('Success Rate', fontsize=11)
axes[0, 1].set_title('Product Success Rate by Brand Reputation', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 6c: Success by early review count
ml_predictions['early_review_category'] = pd.cut(
    ml_predictions['early_review_count'],
    bins=[0, 5, 10, 20, 1000],
    labels=['1-5', '6-10', '11-20', '20+']
)

success_by_reviews = ml_predictions.groupby('early_review_category')['is_success'].mean()
axes[1, 0].bar(success_by_reviews.index, success_by_reviews.values, 
               color='coral', edgecolor='black')
axes[1, 0].set_xlabel('Early Review Count (First 30 Days)', fontsize=11)
axes[1, 0].set_ylabel('Success Rate', fontsize=11)
axes[1, 0].set_title('Success Rate by Early Review Volume', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# 6d: Actual vs Predicted success
confusion_data = pd.DataFrame({
    'Actual': ['Not Successful', 'Not Successful', 'Successful', 'Successful'],
    'Predicted': ['Not Successful', 'Successful', 'Not Successful', 'Successful'],
    'Count': [
        ((ml_predictions['is_success'] == 0) & (ml_predictions['predicted_success'] == 0)).sum(),
        ((ml_predictions['is_success'] == 0) & (ml_predictions['predicted_success'] == 1)).sum(),
        ((ml_predictions['is_success'] == 1) & (ml_predictions['predicted_success'] == 0)).sum(),
        ((ml_predictions['is_success'] == 1) & (ml_predictions['predicted_success'] == 1)).sum()
    ]
})

pivot_confusion = confusion_data.pivot(index='Actual', columns='Predicted', values='Count')
sns.heatmap(pivot_confusion, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], 
            cbar_kws={'label': 'Count'})
axes[1, 1].set_title('Confusion Matrix: Actual vs Predicted', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('viz6_success_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz6_success_predictions.png")
plt.close()

# =====================================================
# VISUALIZATION 7: MAJOR BRANDS DEEP DIVE
# =====================================================
print("\n" + "="*70)
print("VISUALIZATION 7: MAJOR FASHION BRANDS ANALYSIS")
print("="*70)

major_brands = [
    'Nike', 'adidas', 'Skechers', 'New Balance', 'ASICS',
    'Under Armour', 'Levi\'s', 'Columbia', 'Hanes', 'Carhartt'
]

major_brand_data = brand_scores[brand_scores['brand'].isin(major_brands)].copy()
major_brand_data = major_brand_data.sort_values('brand_reputation_score', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BRS comparison
colors_major = plt.cm.RdYlGn(major_brand_data['brand_reputation_score'] / 100)
bars = axes[0].barh(major_brand_data['brand'], major_brand_data['brand_reputation_score'], 
                    color=colors_major, edgecolor='black')
axes[0].set_xlabel('Brand Reputation Score', fontsize=11)
axes[0].set_title('Major Fashion Brands - Reputation Comparison', fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Review count vs BRS scatter
axes[1].scatter(major_brand_data['review_count'], major_brand_data['brand_reputation_score'],
                s=200, alpha=0.6, c=major_brand_data['brand_reputation_score'], 
                cmap='RdYlGn', edgecolor='black', linewidth=2)

for _, row in major_brand_data.iterrows():
    axes[1].annotate(row['brand'], 
                    (row['review_count'], row['brand_reputation_score']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

axes[1].set_xlabel('Number of Reviews', fontsize=11)
axes[1].set_ylabel('Brand Reputation Score', fontsize=11)
axes[1].set_title('Review Volume vs Reputation (Major Brands)', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('viz7_major_brands.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz7_major_brands.png")
plt.close()

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*70)
print("✅ ALL VISUALIZATIONS CREATED!")
print("="*70)

print("\nCreated 7 visualizations:")
print("  ✓ viz1_top_brands_reputation.png")
print("  ✓ viz2_brs_distribution.png")
print("  ✓ viz3_brs_trends.png")
print("  ✓ viz4_ml_model_comparison.png")
print("  ✓ viz5_feature_importance.png")
print("  ✓ viz6_success_predictions.png")
print("  ✓ viz7_major_brands.png")

print("\nNEXT STEP:")
print("  Review visualizations in your folder")
print("  Then: Cloudera upload & screenshots")

print("\n" + "="*70)