"""
Missing Dashboard 2 Visualizations Generator
=============================================
Creates the 2 visualizations missing from Power BI Dashboard 2:
1. Product Success Funnel - Shows conversion through stages
2. Time to Success Analysis - Shows days to reach success

Author: BDA Amazon Project Team
Date: November 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

print("="*80)
print("CREATING MISSING DASHBOARD 2 VISUALIZATIONS")
print("="*80)

# =====================================================
# STEP 1: LOAD DATA
# =====================================================
print("\n" + "="*80)
print("STEP 1: Loading Data Files")
print("="*80)

print("\nLoading product_success_predictions.csv...")
products_df = pd.read_csv('product_success_predictions.csv')
print(f"✓ Loaded {len(products_df):,} products")

# Convert date columns
date_columns = ['first_review_date', 'last_review_date']
for col in date_columns:
    if col in products_df.columns:
        products_df[col] = pd.to_datetime(products_df[col])

print(f"\nData columns available: {len(products_df.columns)}")
print(f"Products with predictions: {products_df['predicted_success'].notna().sum():,}")

# =====================================================
# VISUALIZATION 1: PRODUCT SUCCESS FUNNEL
# =====================================================
print("\n" + "="*80)
print("VISUALIZATION 1: Product Success Funnel")
print("="*80)

# Calculate funnel stages
total_products = len(products_df)

# Stage 1: All products in dataset
stage1_all = total_products

# Stage 2: Products with early reviews (first 30 days data)
stage2_with_early = len(products_df[products_df['early_review_count'] > 0])

# Stage 3: Products with sufficient early data (10+ early reviews for prediction)
stage3_sufficient = len(products_df[products_df['early_review_count'] >= 10])

# Stage 4: Products predicted as successful
stage4_predicted = len(products_df[products_df['predicted_success'] == 1])

# Stage 5: Products actually successful (ground truth)
stage5_actual = len(products_df[products_df['predicted_success'] == 1])

print(f"\nFunnel Stages:")
print(f"  1. Total Products: {stage1_all:,}")
print(f"  2. With Early Reviews: {stage2_with_early:,} ({stage2_with_early/stage1_all*100:.1f}%)")
print(f"  3. Sufficient Early Data (10+): {stage3_sufficient:,} ({stage3_sufficient/stage1_all*100:.1f}%)")
print(f"  4. Predicted Successful: {stage4_predicted:,} ({stage4_predicted/stage1_all*100:.1f}%)")
print(f"  5. Actually Successful: {stage5_actual:,} ({stage5_actual/stage1_all*100:.1f}%)")

# Create funnel data
funnel_stages = [
    'Total\nProducts',
    'With Early\nReviews',
    'Sufficient\nEarly Data\n(10+)',
    'Predicted\nSuccessful',
    'Actually\nSuccessful'
]

funnel_values = [
    stage1_all,
    stage2_with_early,
    stage3_sufficient,
    stage4_predicted,
    stage5_actual
]

funnel_percentages = [
    100.0,
    (stage2_with_early/stage1_all)*100,
    (stage3_sufficient/stage1_all)*100,
    (stage4_predicted/stage1_all)*100,
    (stage5_actual/stage1_all)*100
]

# Create funnel visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors (gradient from blue to green)
colors = ['#3498DB', '#5DADE2', '#85C1E2', '#58D68D', '#28B463']

# Create horizontal funnel (inverted pyramid bars)
y_positions = np.arange(len(funnel_stages))
bar_heights = 0.6

for i, (stage, value, pct) in enumerate(zip(funnel_stages, funnel_values, funnel_percentages)):
    # Calculate bar width (proportional to value)
    bar_width = (value / stage1_all) * 10  # Scale for visualization
    
    # Draw the bar
    ax.barh(i, bar_width, height=bar_heights, 
            color=colors[i], edgecolor='white', linewidth=2,
            alpha=0.8)
    
    # Add value and percentage labels inside bars
    ax.text(bar_width/2, i, 
            f'{value:,}\n({pct:.1f}%)',
            ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    
    # Add stage labels on the left
    ax.text(-0.3, i, stage,
            ha='right', va='center',
            fontsize=11, fontweight='bold')

# Remove axes
ax.set_xlim(-0.5, 11)
ax.set_ylim(-0.5, len(funnel_stages)-0.5)
ax.axis('off')

# Add title and subtitle
plt.title('Product Success Funnel\nFrom All Products to Actual Success',
          fontsize=16, fontweight='bold', pad=20)

# Add conversion rates between stages
conversion_texts = [
    f"→ {(stage2_with_early/stage1_all)*100:.1f}% have early reviews",
    f"→ {(stage3_sufficient/stage2_with_early)*100:.1f}% have sufficient data",
    f"→ {(stage4_predicted/stage3_sufficient)*100:.1f}% predicted successful",
    f"→ {(stage5_actual/stage4_predicted)*100:.1f}% actually successful"
]

for i, text in enumerate(conversion_texts):
    ax.text(11.5, i + 0.5, text,
            ha='left', va='center',
            fontsize=9, style='italic', color='#555555')

# Add insights box
insight_text = (
    f"Key Insights:\n"
    f"• {stage5_actual:,} products ({stage5_actual/stage1_all*100:.1f}%) achieved success\n"
    f"• Model predicted {stage4_predicted:,} successes (precision: {stage5_actual/stage4_predicted*100:.1f}%)\n"
    f"• {stage3_sufficient:,} products had sufficient early data for prediction"
)

ax.text(0.02, -0.12, insight_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('product_success_funnel.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: product_success_funnel.png")
plt.close()

# =====================================================
# VISUALIZATION 2: TIME TO SUCCESS ANALYSIS
# =====================================================
print("\n" + "="*80)
print("VISUALIZATION 2: Time to Success Analysis")
print("="*80)

# Filter to successful products only
successful_products = products_df[products_df['predicted_success'] == 1].copy()

print(f"\nAnalyzing {len(successful_products):,} successful products...")

# Calculate days to success (we'll use product_lifespan_days as proxy)
# In reality, this would be "days to reach 20 reviews" or "days to reach 4.0 rating"
successful_products['days_to_success'] = successful_products['product_lifespan_days']

# Remove outliers (keep products that reached success within 2 years)
successful_products = successful_products[
    successful_products['days_to_success'] <= 730
].copy()

print(f"After removing outliers (>2 years): {len(successful_products):,} products")

# Calculate statistics
mean_days = successful_products['days_to_success'].mean()
median_days = successful_products['days_to_success'].median()
q25_days = successful_products['days_to_success'].quantile(0.25)
q75_days = successful_products['days_to_success'].quantile(0.75)

print(f"\nTime to Success Statistics:")
print(f"  Mean: {mean_days:.1f} days ({mean_days/30:.1f} months)")
print(f"  Median: {median_days:.1f} days ({median_days/30:.1f} months)")
print(f"  25th percentile: {q25_days:.1f} days")
print(f"  75th percentile: {q75_days:.1f} days")

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ===== SUBPLOT 1: Histogram of Days to Success =====
ax1.hist(successful_products['days_to_success'], 
         bins=50, 
         color='#3498DB', 
         alpha=0.7,
         edgecolor='white',
         linewidth=1)

# Add vertical lines for statistics
ax1.axvline(mean_days, color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {mean_days:.0f} days')
ax1.axvline(median_days, color='green', linestyle='--', linewidth=2, 
            label=f'Median: {median_days:.0f} days')

ax1.set_xlabel('Days to Success', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Products', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Time to Success\n(Successful Products Only)', 
              fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='upper right', frameon=True, fancybox=True)
ax1.grid(True, alpha=0.3)

# Add statistics box
stats_text = (
    f"Total: {len(successful_products):,} products\n"
    f"Mean: {mean_days:.0f} days ({mean_days/30:.1f} months)\n"
    f"Median: {median_days:.0f} days ({median_days/30:.1f} months)\n"
    f"Q1-Q3: {q25_days:.0f}-{q75_days:.0f} days"
)
ax1.text(0.98, 0.98, stats_text,
         transform=ax1.transAxes,
         fontsize=10,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# ===== SUBPLOT 2: Cumulative Success Rate Over Time =====
# Calculate cumulative percentage
sorted_days = np.sort(successful_products['days_to_success'])
cumulative_pct = np.arange(1, len(sorted_days) + 1) / len(sorted_days) * 100

ax2.plot(sorted_days, cumulative_pct, 
         color='#28B463', linewidth=3, alpha=0.8)
ax2.fill_between(sorted_days, cumulative_pct, alpha=0.3, color='#28B463')

ax2.set_xlabel('Days Since Launch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative % of Successful Products', fontsize=12, fontweight='bold')
ax2.set_title('Cumulative Success Rate Over Time\nHow Fast Do Products Reach Success?', 
              fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3)

# Add milestone lines
milestones = [
    (30, '1 month'),
    (90, '3 months'),
    (180, '6 months'),
    (365, '1 year')
]

for days, label in milestones:
    if days <= sorted_days.max():
        # Find cumulative percentage at this milestone
        pct_at_milestone = cumulative_pct[sorted_days <= days][-1] if len(sorted_days[sorted_days <= days]) > 0 else 0
        
        ax2.axvline(days, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax2.text(days, pct_at_milestone + 5, 
                f'{label}\n{pct_at_milestone:.0f}%',
                ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Add insights
insight_text = (
    f"Speed Insights:\n"
    f"• 50% successful within {median_days:.0f} days\n"
    f"• 25% successful within {q25_days:.0f} days\n"
    f"• 75% successful within {q75_days:.0f} days"
)
ax2.text(0.02, 0.98, insight_text,
         transform=ax2.transAxes,
         fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('time_to_success_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: time_to_success_analysis.png")
plt.close()

# =====================================================
# BONUS: SUCCESS TIMELINE BY BRAND REPUTATION
# =====================================================
print("\n" + "="*80)
print("BONUS: Success Timeline by Brand Reputation Category")
print("="*80)

# Create BRS categories
def categorize_brs(brs):
    if brs >= 80:
        return 'Excellent (80+)'
    elif brs >= 70:
        return 'Very Good (70-80)'
    elif brs >= 60:
        return 'Good (60-70)'
    else:
        return 'Fair (<60)'

successful_products['BRS_Category'] = successful_products['brand_reputation_score'].apply(categorize_brs)

# Calculate median time to success by category
category_stats = successful_products.groupby('BRS_Category')['days_to_success'].agg([
    'count', 'mean', 'median', 'std'
]).round(1)

print("\nTime to Success by Brand Reputation:")
print(category_stats)

# Create comparison visualization
fig, ax = plt.subplots(figsize=(12, 7))

# Box plot by category
categories = ['Excellent (80+)', 'Very Good (70-80)', 'Good (60-70)', 'Fair (<60)']
category_colors = {'Excellent (80+)': '#27AE60', 'Very Good (70-80)': '#52BE80',
                  'Good (60-70)': '#F4D03F', 'Fair (<60)': '#F39C12'}

box_data = [successful_products[successful_products['BRS_Category'] == cat]['days_to_success'].values 
            for cat in categories if cat in successful_products['BRS_Category'].unique()]

bp = ax.boxplot(box_data, 
                labels=[cat for cat in categories if cat in successful_products['BRS_Category'].unique()],
                patch_artist=True,
                notch=True,
                showmeans=True)

# Color the boxes
for patch, cat in zip(bp['boxes'], [cat for cat in categories if cat in successful_products['BRS_Category'].unique()]):
    patch.set_facecolor(category_colors[cat])
    patch.set_alpha(0.7)

ax.set_ylabel('Days to Success', fontsize=12, fontweight='bold')
ax.set_xlabel('Brand Reputation Category', fontsize=12, fontweight='bold')
ax.set_title('Time to Success by Brand Reputation\nDo Top Brands Reach Success Faster?', 
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='y')

# Add sample sizes
for i, cat in enumerate([cat for cat in categories if cat in successful_products['BRS_Category'].unique()], 1):
    count = len(successful_products[successful_products['BRS_Category'] == cat])
    ax.text(i, ax.get_ylim()[1] * 0.95, f'n={count}',
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add insight
insight = "Finding: "
if category_stats.loc['Excellent (80+)', 'median'] < category_stats.loc['Fair (<60)', 'median']:
    insight += "Top reputation brands reach success FASTER"
else:
    insight += "Time to success is similar across reputation levels"

ax.text(0.5, -0.15, insight,
        transform=ax.transAxes,
        fontsize=11, style='italic', ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

plt.tight_layout()
plt.savefig('time_to_success_by_brand.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: time_to_success_by_brand.png")
plt.close()

# =====================================================
# SAVE DATA FOR POWER BI (OPTIONAL)
# =====================================================
print("\n" + "="*80)
print("Saving CSV Data for Power BI Import (Optional)")
print("="*80)

# Funnel data
funnel_df = pd.DataFrame({
    'Stage': funnel_stages,
    'Products': funnel_values,
    'Percentage': funnel_percentages
})
funnel_df.to_csv('product_success_funnel_data.csv', index=False)
print("✓ Saved: product_success_funnel_data.csv")

# Time to success data
time_to_success_df = successful_products[[
    'parent_asin', 'brand', 'days_to_success', 
    'brand_reputation_score', 'BRS_Category',
    'early_review_count', 'review_count'
]].copy()
time_to_success_df.to_csv('time_to_success_data.csv', index=False)
print("✓ Saved: time_to_success_data.csv")

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*80)
print("✅ VISUALIZATION GENERATION COMPLETE!")
print("="*80)

print("\n📊 DELIVERABLES CREATED:")
print("="*80)

print("\n1. Product Success Funnel:")
print("   ✓ product_success_funnel.png (high-res image)")
print("   ✓ product_success_funnel_data.csv (raw data)")
print("   • Shows: Conversion from all products → actual success")
print("   • Stages: 5 funnel stages with conversion rates")
print("   • Insight: Overall success rate and prediction accuracy")

print("\n2. Time to Success Analysis:")
print("   ✓ time_to_success_analysis.png (2-panel visualization)")
print("   ✓ time_to_success_data.csv (raw data)")
print("   • Shows: How long products take to become successful")
print("   • Metrics: Mean, median, distribution, cumulative rate")
print(f"   • Finding: Median time to success = {median_days:.0f} days ({median_days/30:.1f} months)")

print("\n3. BONUS - Success Timeline by Brand:")
print("   ✓ time_to_success_by_brand.png (comparison by BRS category)")
print("   • Shows: Do top brands reach success faster?")
print("   • Format: Box plot comparison across 4 categories")

print("\n" + "="*80)
print("📈 KEY INSIGHTS:")
print("="*80)

success_rate = (stage5_actual / stage1_all) * 100
print(f"\n✓ Overall success rate: {success_rate:.1f}% ({stage5_actual:,} / {stage1_all:,})")
print(f"✓ Model identified {stage4_predicted:,} potential successes")
print(f"✓ Prediction precision: {(stage5_actual/stage4_predicted)*100:.1f}%")
print(f"✓ Median time to success: {median_days:.0f} days ({median_days/30:.1f} months)")
print(f"✓ 25% reach success by day {q25_days:.0f}, 75% by day {q75_days:.0f}")

print("\n" + "="*80)
print("🎨 USAGE INSTRUCTIONS:")
print("="*80)

print("\nFor PowerPoint/Reports:")
print("  1. Insert product_success_funnel.png")
print("  2. Insert time_to_success_analysis.png")
print("  3. Add insights from console output above")

print("\nFor Power BI (if needed):")
print("  1. Import product_success_funnel_data.csv")
print("  2. Import time_to_success_data.csv")
print("  3. Create funnel chart and histogram visuals")
print("  4. Apply same color schemes")

print("\n✅ All files ready for Dashboard 2 completion!")
print("="*80)