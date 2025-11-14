"""
Missing Power BI Visualizations Generator
==========================================
Creates the two visualizations not yet in Power BI Dashboard 1:
1. Brand Performance Metrics Table (Top 30 brands)
2. Review Volume vs Reputation Scatter Plot

Author: BDA Amazon Project Team
Date: November 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CREATING MISSING POWER BI VISUALIZATIONS")
print("="*80)

# =====================================================
# STEP 1: LOAD DATA
# =====================================================
print("\n" + "="*80)
print("STEP 1: Loading Data Files")
print("="*80)

# Load brand reputation data
print("\nLoading brand_reputation_significant.csv...")
brands_df = pd.read_csv('brand_reputation_significant.csv')
print(f"✓ Loaded {len(brands_df):,} brands")

# =====================================================
# VISUALIZATION 1: BRAND PERFORMANCE METRICS TABLE
# =====================================================
print("\n" + "="*80)
print("VISUALIZATION 1: Brand Performance Metrics Table")
print("="*80)

# Select top 30 brands by BRS
top_30 = brands_df.nlargest(30, 'brand_reputation_score').copy()

# Prepare data for table
table_data = top_30[[
    'brand', 
    'brand_reputation_score', 
    'avg_rating', 
    'review_count',
    'verified_ratio',
    'positive_ratio'
]].copy()

# Rename columns for display
table_data.columns = [
    'Brand', 
    'BRS', 
    'Avg Rating', 
    'Reviews',
    'Verified %',
    'Positive %'
]

# Format percentages
table_data['Verified %'] = (table_data['Verified %'] * 100).round(1)
table_data['Positive %'] = (table_data['Positive %'] * 100).round(1)

# Format other columns
table_data['BRS'] = table_data['BRS'].round(0).astype(int)
table_data['Avg Rating'] = table_data['Avg Rating'].round(1)
table_data['Reviews'] = table_data['Reviews'].apply(lambda x: f"{x:,}")

print(f"\n✓ Prepared data for top 30 brands")

# Create figure
fig, ax = plt.subplots(figsize=(14, 16))
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    cellLoc='left',
    loc='center',
    bbox=[0, 0, 1, 1]
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color code header
for i in range(len(table_data.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(weight='bold', color='white')

# Color code BRS column (column index 1)
brs_values = top_30['brand_reputation_score'].values
for i in range(len(table_data)):
    brs = brs_values[i]
    cell = table[(i+1, 1)]  # BRS column
    
    # Green gradient based on BRS value
    if brs >= 85:
        color = '#27AE60'  # Dark green - Excellent
    elif brs >= 70:
        color = '#52BE80'  # Light green - Very Good
    elif brs >= 60:
        color = '#F4D03F'  # Yellow - Good
    elif brs >= 50:
        color = '#F39C12'  # Orange - Fair
    else:
        color = '#E74C3C'  # Red - Poor
    
    cell.set_facecolor(color)
    cell.set_text_props(weight='bold', color='white')

# Color code Rating column (column index 2)
rating_values = top_30['avg_rating'].values
for i in range(len(table_data)):
    rating = rating_values[i]
    cell = table[(i+1, 2)]  # Rating column
    
    # Color scale from red (low) to green (high)
    if rating >= 4.5:
        color = '#D5F4E6'  # Light green
    elif rating >= 4.0:
        color = '#ABEBC6'  # Medium green
    elif rating >= 3.5:
        color = '#FCF3CF'  # Light yellow
    elif rating >= 3.0:
        color = '#F9E79F'  # Yellow
    else:
        color = '#F5B7B1'  # Light red
    
    cell.set_facecolor(color)

# Alternate row colors for readability
for i in range(len(table_data)):
    if i % 2 == 0:
        for j in range(len(table_data.columns)):
            if j not in [1, 2]:  # Skip BRS and Rating (already colored)
                cell = table[(i+1, j)]
                cell.set_facecolor('#F8F9F9')

# Add title
plt.title('Brand Performance Metrics - Top 30 Brands by Reputation\n', 
          fontsize=16, fontweight='bold', pad=20)

# Add subtitle with explanation
subtitle_text = ('BRS: Brand Reputation Score (0-100) | '
                'Color coding: Green (High) → Yellow (Medium) → Red (Low)')
plt.text(0.5, 0.98, subtitle_text, 
         ha='center', va='top', fontsize=10, style='italic',
         transform=fig.transFigure, color='#555555')

plt.tight_layout()
plt.savefig('brand_performance_table.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: brand_performance_table.png")
plt.close()

# Also save as CSV for easy reference
table_data.to_csv('brand_performance_top30.csv', index=False)
print("✓ Saved: brand_performance_top30.csv")

# =====================================================
# VISUALIZATION 2: SCATTER PLOT - VOLUME VS REPUTATION
# =====================================================
print("\n" + "="*80)
print("VISUALIZATION 2: Review Volume vs Reputation Scatter Plot")
print("="*80)

# Use all significant brands (2,202 brands with 50+ reviews)
scatter_data = brands_df.copy()

# Create BRS categories for coloring
def get_brs_category(brs):
    if brs >= 80:
        return 'Excellent (80-100)'
    elif brs >= 70:
        return 'Very Good (70-80)'
    elif brs >= 60:
        return 'Good (60-70)'
    elif brs >= 50:
        return 'Fair (50-60)'
    else:
        return 'Poor (<50)'

scatter_data['BRS_Category'] = scatter_data['brand_reputation_score'].apply(get_brs_category)

# Define colors for categories
category_colors = {
    'Excellent (80-100)': '#27AE60',   # Dark green
    'Very Good (70-80)': '#52BE80',    # Light green
    'Good (60-70)': '#F4D03F',         # Yellow
    'Fair (50-60)': '#F39C12',         # Orange
    'Poor (<50)': '#E74C3C'            # Red
}

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Plot each category separately for proper legend
for category in ['Excellent (80-100)', 'Very Good (70-80)', 'Good (60-70)', 
                 'Fair (50-60)', 'Poor (<50)']:
    category_data = scatter_data[scatter_data['BRS_Category'] == category]
    
    if len(category_data) > 0:
        # Bubble size based on helpful votes (scaled)
        sizes = (category_data['avg_helpful_votes'] * 20).clip(20, 500)
        
        ax.scatter(
            category_data['review_count'],
            category_data['brand_reputation_score'],
            s=sizes,
            alpha=0.6,
            c=category_colors[category],
            label=f"{category} ({len(category_data)} brands)",
            edgecolors='white',
            linewidth=0.5
        )

# Set log scale for x-axis (better for wide range of review counts)
ax.set_xscale('log')

# Labels and title
ax.set_xlabel('Review Count (Log Scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Brand Reputation Score', fontsize=12, fontweight='bold')
ax.set_title('Review Volume vs Brand Reputation Score\n'
             'Bubble Size = Average Helpful Votes', 
             fontsize=16, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Legend
legend = ax.legend(
    loc='lower right', 
    frameon=True, 
    fancybox=True, 
    shadow=True,
    fontsize=10,
    title='BRS Category'
)
legend.get_title().set_fontweight('bold')

# Add annotations for notable brands
# Top 5 brands by BRS
top_5_brs = scatter_data.nlargest(5, 'brand_reputation_score')
for idx, row in top_5_brs.iterrows():
    ax.annotate(
        row['brand'],
        xy=(row['review_count'], row['brand_reputation_score']),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1)
    )

# Add insight text boxes
insight_text = (
    f"Total Brands: {len(scatter_data):,}\n"
    f"Avg BRS: {scatter_data['brand_reputation_score'].mean():.1f}\n"
    f"Avg Reviews: {scatter_data['review_count'].mean():.0f}"
)

ax.text(0.02, 0.98, insight_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Add correlation insight
correlation = scatter_data['review_count'].corr(scatter_data['brand_reputation_score'])
corr_text = f"Correlation: {correlation:.3f}"
corr_color = 'green' if abs(correlation) > 0.3 else 'orange' if abs(correlation) > 0.1 else 'red'

ax.text(0.98, 0.02, corr_text,
        transform=ax.transAxes,
        fontsize=10,
        horizontalalignment='right',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor=corr_color, alpha=0.3))

plt.tight_layout()
plt.savefig('volume_vs_reputation_scatter.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: volume_vs_reputation_scatter.png")
plt.close()

# =====================================================
# BONUS: SAVE SCATTER PLOT DATA
# =====================================================
# Save data for interactive version (if needed for web dashboard)
scatter_export = scatter_data[[
    'brand', 
    'brand_reputation_score', 
    'review_count',
    'avg_helpful_votes',
    'avg_rating',
    'BRS_Category'
]].copy()

scatter_export.to_csv('scatter_plot_data.csv', index=False)
print("✓ Saved: scatter_plot_data.csv (for interactive dashboards)")

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*80)
print("✅ VISUALIZATION GENERATION COMPLETE!")
print("="*80)

print("\n📊 DELIVERABLES CREATED:")
print("="*80)

print("\n1. Brand Performance Metrics Table:")
print("   ✓ brand_performance_table.png (high-res image)")
print("   ✓ brand_performance_top30.csv (raw data)")
print("   • Shows: Top 30 brands with BRS, Rating, Reviews, Verified %, Positive %")
print("   • Color coding: Green gradient for high BRS, rating color scale")
print("   • Use: Insert into presentations or reports")

print("\n2. Review Volume vs Reputation Scatter Plot:")
print("   ✓ volume_vs_reputation_scatter.png (high-res image)")
print("   ✓ scatter_plot_data.csv (raw data)")
print("   • Shows: Relationship between review volume and brand reputation")
print("   • Bubble size: Helpful votes (engagement metric)")
print("   • Color: By BRS category (Excellent to Poor)")
print("   • Insight: Correlation coefficient displayed")
print(f"   • Finding: Correlation = {correlation:.3f}")

print("\n" + "="*80)
print("📈 KEY INSIGHTS:")
print("="*80)

# Calculate insights
high_volume_high_rep = len(scatter_data[
    (scatter_data['review_count'] > 1000) & 
    (scatter_data['brand_reputation_score'] >= 70)
])

high_volume_low_rep = len(scatter_data[
    (scatter_data['review_count'] > 1000) & 
    (scatter_data['brand_reputation_score'] < 70)
])

print(f"\n✓ {high_volume_high_rep} high-volume brands (1K+ reviews) maintain high reputation (BRS ≥ 70)")
print(f"✓ {high_volume_low_rep} high-volume brands struggle with reputation (BRS < 70)")
print(f"✓ Volume-Reputation Correlation: {correlation:.3f}")

if abs(correlation) < 0.1:
    print("  → Insight: Review volume has WEAK correlation with reputation")
    print("  → Implication: More reviews ≠ better reputation. Quality matters!")
elif correlation > 0.1:
    print("  → Insight: Slight POSITIVE correlation - popular brands maintain quality")
else:
    print("  → Insight: Slight NEGATIVE correlation - some popular brands decline")

print("\n✓ Top performer: " + top_5_brs.iloc[0]['brand'] + 
      f" (BRS: {top_5_brs.iloc[0]['brand_reputation_score']:.1f})")

print("\n" + "="*80)
print("🎨 USAGE INSTRUCTIONS:")
print("="*80)

print("\nFor PowerPoint/Reports:")
print("  1. Insert brand_performance_table.png")
print("  2. Insert volume_vs_reputation_scatter.png")
print("  3. Add insights from console output above")

print("\nFor Power BI (if needed):")
print("  1. Use brand_performance_top30.csv to create table visual")
print("  2. Use scatter_plot_data.csv to create scatter plot")
print("  3. Apply same color schemes as in images")

print("\nFor Interactive Dashboards:")
print("  1. Load scatter_plot_data.csv into Plotly/Tableau")
print("  2. Add tooltips showing brand details")
print("  3. Enable zoom and pan for exploration")

print("\n" + "="*80)
print("🚀 ALL FILES READY FOR DASHBOARD INTEGRATION!")
print("="*80)

print("\n💡 Next Steps:")
print("  1. Review generated images")
print("  2. Share with team for feedback")
print("  3. Insert into final presentation")
print("  4. Optional: Create interactive versions if needed")

print("\n✅ Script Complete! Check your output folder for files.")
print("="*80)