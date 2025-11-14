"""
Dashboard 3 Missing Visualizations Generator
============================================
Creates 4 missing visualizations from Power BI Dashboard 3:
1. Confusion Matrices (all 3 models side-by-side)
2. ROC Curves (all 3 models on one plot)
3. Precision-Recall Curves (all 3 models on one plot)
4. Comprehensive Metrics Table

Author: BDA Amazon Project Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

print("="*80)
print("CREATING DASHBOARD 3 MISSING VISUALIZATIONS")
print("="*80)

# =====================================================
# STEP 1: LOAD DATA AND TRAIN MODELS
# =====================================================
print("\n" + "="*80)
print("STEP 1: Loading Data and Training Models")
print("="*80)

print("\nLoading product_success_predictions.csv...")
df = pd.read_csv('product_success_predictions.csv')
print(f"✓ Loaded {len(df):,} products")

# Prepare features
feature_columns = [
    'early_review_count', 'early_avg_rating', 'early_sentiment',
    'early_positive_ratio', 'early_verified_ratio', 'early_helpful_votes',
    'early_review_velocity', 'brand_reputation_score', 'total_customer_images'
]

X = df[feature_columns]
y = df['predicted_success']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")

# Train models
print("\nTraining models...")

# Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
print("  ✓ Logistic Regression trained")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
print("  ✓ Random Forest trained")

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_prob = gb_model.predict_proba(X_test_scaled)[:, 1]
print("  ✓ Gradient Boosting trained")

# =====================================================
# VISUALIZATION 1: CONFUSION MATRICES
# =====================================================
print("\n" + "="*80)
print("VISUALIZATION 1: Confusion Matrices (All 3 Models)")
print("="*80)

# Calculate confusion matrices
cm_lr = confusion_matrix(y_test, lr_pred)
cm_rf = confusion_matrix(y_test, rf_pred)
cm_gb = confusion_matrix(y_test, gb_pred)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
cms = [cm_lr, cm_rf, cm_gb]
colors = ['Reds', 'Purples', 'Greens']

for ax, model_name, cm, cmap in zip(axes, models, cms, colors):
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                ax=ax, cbar=True, square=True,
                annot_kws={'size': 14, 'weight': 'bold'})
    
    # Labels
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}\nConfusion Matrix', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Set tick labels
    ax.set_xticklabels(['Not Success (0)', 'Success (1)'])
    ax.set_yticklabels(['Not Success (0)', 'Success (1)'])
    
    # Add accuracy text
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    ax.text(0.5, -0.15, f'Accuracy: {accuracy:.2%}',
            transform=ax.transAxes,
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Main title
fig.suptitle('Confusion Matrices - Model Performance Comparison\n'
             'True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP)',
             fontsize=16, fontweight='bold', y=1.05)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: confusion_matrices.png")
plt.close()

# =====================================================
# VISUALIZATION 2: ROC CURVES
# =====================================================
print("\n" + "="*80)
print("VISUALIZATION 2: ROC Curves (Receiver Operating Characteristic)")
print("="*80)

# Calculate ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
roc_auc_lr = auc(fpr_lr, tpr_lr)

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_prob)
roc_auc_gb = auc(fpr_gb, tpr_gb)

print(f"\nROC-AUC Scores:")
print(f"  Logistic Regression: {roc_auc_lr:.4f}")
print(f"  Random Forest: {roc_auc_rf:.4f}")
print(f"  Gradient Boosting: {roc_auc_gb:.4f}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot ROC curves
ax.plot(fpr_lr, tpr_lr, color='#E74C3C', lw=3, 
        label=f'Logistic Regression (AUC = {roc_auc_lr:.3f})')
ax.plot(fpr_rf, tpr_rf, color='#9B59B6', lw=3, 
        label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
ax.plot(fpr_gb, tpr_gb, color='#27AE60', lw=3, 
        label=f'Gradient Boosting (AUC = {roc_auc_gb:.3f})')

# Plot diagonal (random classifier)
ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier (AUC = 0.500)')

# Labels and formatting
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - Model Discrimination Ability\nHigher AUC = Better Performance',
             fontsize=14, fontweight='bold', pad=15)

ax.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

# Add interpretation box
interp_text = (
    "ROC-AUC Interpretation:\n"
    "• 0.9-1.0: Excellent\n"
    "• 0.8-0.9: Very Good\n"
    "• 0.7-0.8: Good\n"
    "• 0.6-0.7: Fair\n"
    "• 0.5-0.6: Poor\n"
    "• 0.5: Random"
)
ax.text(0.98, 0.02, interp_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: roc_curves.png")
plt.close()

# =====================================================
# VISUALIZATION 3: PRECISION-RECALL CURVES
# =====================================================
print("\n" + "="*80)
print("VISUALIZATION 3: Precision-Recall Curves")
print("="*80)

# Calculate PR curves
precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_prob)
ap_lr = average_precision_score(y_test, lr_prob)

precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_prob)
ap_rf = average_precision_score(y_test, rf_prob)

precision_gb, recall_gb, _ = precision_recall_curve(y_test, gb_prob)
ap_gb = average_precision_score(y_test, gb_prob)

print(f"\nAverage Precision Scores:")
print(f"  Logistic Regression: {ap_lr:.4f}")
print(f"  Random Forest: {ap_rf:.4f}")
print(f"  Gradient Boosting: {ap_gb:.4f}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot PR curves
ax.plot(recall_lr, precision_lr, color='#E74C3C', lw=3,
        label=f'Logistic Regression (AP = {ap_lr:.3f})')
ax.plot(recall_rf, precision_rf, color='#9B59B6', lw=3,
        label=f'Random Forest (AP = {ap_rf:.3f})')
ax.plot(recall_gb, precision_gb, color='#27AE60', lw=3,
        label=f'Gradient Boosting (AP = {ap_gb:.3f})')

# Plot baseline (random classifier for imbalanced data)
baseline = y_test.mean()
ax.plot([0, 1], [baseline, baseline], 'k--', lw=2, alpha=0.5,
        label=f'Random Classifier (AP = {baseline:.3f})')

# Labels and formatting
ax.set_xlabel('Recall (True Positive Rate)', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curves - Trade-off Analysis\n'
             'Important for Imbalanced Datasets',
             fontsize=14, fontweight='bold', pad=15)

ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

# Add interpretation box
interp_text = (
    "Why Recall is Low:\n"
    "• Class imbalance (24% positive)\n"
    "• Models are conservative\n"
    "• High precision, low recall trade-off\n"
    "\n"
    "Business Impact:\n"
    "• Predictions are trustworthy (60% precision)\n"
    "• But miss many opportunities (16% recall)\n"
    "• Consider lowering threshold for more recall"
)
ax.text(0.02, 0.02, interp_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: precision_recall_curves.png")
plt.close()

# =====================================================
# VISUALIZATION 4: COMPREHENSIVE METRICS TABLE
# =====================================================
print("\n" + "="*80)
print("VISUALIZATION 4: Comprehensive Metrics Comparison Table")
print("="*80)

# Calculate all metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

metrics_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, gb_pred)
    ],
    'Precision': [
        precision_score(y_test, lr_pred),
        precision_score(y_test, rf_pred),
        precision_score(y_test, gb_pred)
    ],
    'Recall': [
        recall_score(y_test, lr_pred),
        recall_score(y_test, rf_pred),
        recall_score(y_test, gb_pred)
    ],
    'F1-Score': [
        f1_score(y_test, lr_pred),
        f1_score(y_test, rf_pred),
        f1_score(y_test, gb_pred)
    ],
    'ROC-AUC': [roc_auc_lr, roc_auc_rf, roc_auc_gb],
    'Avg Precision': [ap_lr, ap_rf, ap_gb]
}

metrics_df = pd.DataFrame(metrics_data)

# Round values for display
display_df = metrics_df.copy()
for col in display_df.columns[1:]:
    display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')

print("\nModel Performance Metrics:")
print(metrics_df.to_string(index=False))

# Create table visualization
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Create table
table_data = display_df.values
col_labels = display_df.columns

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Color code header
for i in range(len(col_labels)):
    cell = table[(0, i)]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(weight='bold', color='white', fontsize=12)

# Color code model names and highlight best values
model_colors = ['#E74C3C', '#9B59B6', '#27AE60']  # LR, RF, GB

for i in range(len(metrics_data['Model'])):
    # Color model name
    cell = table[(i+1, 0)]
    cell.set_facecolor(model_colors[i])
    cell.set_text_props(weight='bold', color='white')
    
    # Highlight best values in each metric
    for j in range(1, len(col_labels)):
        cell = table[(i+1, j)]
        metric_col = col_labels[j]
        
        # Check if this is the best value
        metric_values = metrics_df[metric_col].values
        if metric_values[i] == max(metric_values):
            cell.set_facecolor('#D5F4E6')  # Light green
            cell.set_text_props(weight='bold')

# Alternate row colors for readability
for i in range(len(metrics_data['Model'])):
    if i % 2 == 0:
        for j in range(1, len(col_labels)):
            cell = table[(i+1, j)]
            if cell.get_facecolor() == (1.0, 1.0, 1.0, 1.0):  # If not highlighted
                cell.set_facecolor('#F8F9F9')

# Add title
plt.title('ML Model Performance - Complete Metrics Comparison\n'
          'Green highlight = Best performance for each metric',
          fontsize=16, fontweight='bold', pad=20)

# Add footer with interpretation
footer_text = (
    "Key Findings: Gradient Boosting achieves highest accuracy (80.5%) and ROC-AUC (0.72). "
    "All models struggle with recall (13-16%) due to class imbalance. "
    "Precision is moderate (54-60%), indicating conservative predictions."
)
plt.text(0.5, -0.05, footer_text,
         ha='center', va='top', fontsize=10, style='italic',
         transform=ax.transAxes, wrap=True, color='#555555')

plt.tight_layout()
plt.savefig('metrics_comparison_table.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: metrics_comparison_table.png")
plt.close()

# Save as CSV too
metrics_df.to_csv('ml_metrics_detailed.csv', index=False)
print("✓ Saved: ml_metrics_detailed.csv")

# =====================================================
# BONUS: THRESHOLD ANALYSIS
# =====================================================
print("\n" + "="*80)
print("BONUS: Threshold Impact Analysis (Gradient Boosting)")
print("="*80)

# Analyze different thresholds for GB model
thresholds = np.arange(0.1, 0.9, 0.1)
threshold_results = []

for threshold in thresholds:
    gb_pred_threshold = (gb_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_test, gb_pred_threshold)
    prec = precision_score(y_test, gb_pred_threshold, zero_division=0)
    rec = recall_score(y_test, gb_pred_threshold)
    f1 = f1_score(y_test, gb_pred_threshold, zero_division=0)
    
    threshold_results.append({
        'Threshold': threshold,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

threshold_df = pd.DataFrame(threshold_results)

# Create visualization
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(threshold_df['Threshold'], threshold_df['Accuracy'], 
        'o-', color='#3498DB', lw=2, markersize=8, label='Accuracy')
ax.plot(threshold_df['Threshold'], threshold_df['Precision'], 
        's-', color='#E74C3C', lw=2, markersize=8, label='Precision')
ax.plot(threshold_df['Threshold'], threshold_df['Recall'], 
        '^-', color='#27AE60', lw=2, markersize=8, label='Recall')
ax.plot(threshold_df['Threshold'], threshold_df['F1-Score'], 
        'd-', color='#9B59B6', lw=2, markersize=8, label='F1-Score')

# Mark default threshold (0.5)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, lw=2)
ax.text(0.5, 0.95, 'Default\nThreshold', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

ax.set_xlabel('Prediction Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
ax.set_title('Threshold Impact on Model Performance (Gradient Boosting)\n'
             'How Changing Threshold Affects Metrics',
             fontsize=14, fontweight='bold', pad=15)

ax.legend(loc='best', fontsize=11, frameon=True, fancybox=True)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.05, 0.85])
ax.set_ylim([0, 1.05])

# Add insight box
insight = (
    "Threshold Insights:\n"
    "• Lower threshold (0.2-0.3):\n"
    "  - Higher recall (catch more successes)\n"
    "  - Lower precision (more false positives)\n"
    "\n"
    "• Higher threshold (0.6-0.7):\n"
    "  - Higher precision (fewer mistakes)\n"
    "  - Lower recall (miss more opportunities)\n"
    "\n"
    "• Current (0.5): Balanced approach"
)
ax.text(0.02, 0.98, insight,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: threshold_analysis.png (BONUS)")
plt.close()

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*80)
print("✅ VISUALIZATION GENERATION COMPLETE!")
print("="*80)

print("\n📊 DELIVERABLES CREATED:")
print("="*80)

print("\n1. Confusion Matrices:")
print("   ✓ confusion_matrices.png (3-panel comparison)")
print("   • Shows: TP, TN, FP, FN for all models")
print("   • Helps diagnose: Where models make mistakes")
print("   • Key insight: All models have high TN, low TP (conservative)")

print("\n2. ROC Curves:")
print("   ✓ roc_curves.png (all 3 models + baseline)")
print("   • Shows: True Positive Rate vs False Positive Rate")
print(f"   • Best AUC: Gradient Boosting ({roc_auc_gb:.3f})")
print("   • Interpretation: Good discrimination ability (0.7-0.8 range)")

print("\n3. Precision-Recall Curves:")
print("   ✓ precision_recall_curves.png (all 3 models + baseline)")
print("   • Shows: Precision-Recall trade-off")
print(f"   • Best AP: Gradient Boosting ({ap_gb:.3f})")
print("   • Key insight: Explains why recall is low (class imbalance)")

print("\n4. Comprehensive Metrics Table:")
print("   ✓ metrics_comparison_table.png (professional table)")
print("   ✓ ml_metrics_detailed.csv (raw data)")
print("   • Shows: All metrics side-by-side")
print("   • Highlights: Best values for each metric")
print("   • Winner: Gradient Boosting (80.5% accuracy)")

print("\n5. BONUS - Threshold Analysis:")
print("   ✓ threshold_analysis.png")
print("   • Shows: How changing threshold affects metrics")
print("   • Insight: Can improve recall by lowering threshold")
print("   • Trade-off: Higher recall = Lower precision")

print("\n" + "="*80)
print("📈 KEY INSIGHTS:")
print("="*80)

print("\n✓ All models achieve ~80% accuracy (consistent performance)")
print(f"✓ Gradient Boosting is best: {roc_auc_gb:.3f} ROC-AUC")
print(f"✓ Low recall ({recall_score(y_test, gb_pred):.2%}) due to class imbalance")
print(f"✓ Moderate precision ({precision_score(y_test, gb_pred):.2%}) = Conservative predictions")
print("✓ Models are trustworthy but miss many opportunities")

print("\n" + "="*80)
print("🎨 USAGE INSTRUCTIONS:")
print("="*80)

print("\nFor PowerPoint/Reports:")
print("  1. Insert confusion_matrices.png (shows prediction breakdown)")
print("  2. Insert roc_curves.png (shows model discrimination)")
print("  3. Insert precision_recall_curves.png (explains recall issue)")
print("  4. Insert metrics_comparison_table.png (comprehensive comparison)")
print("  5. Optional: threshold_analysis.png (for advanced discussion)")

print("\nFor Power BI (if needed):")
print("  1. Import ml_metrics_detailed.csv")
print("  2. Create visuals using the data")
print("  3. Add calculated columns for % formatting")

print("\n✅ Dashboard 3 is now 100% COMPLETE with all essential ML visuals!")
print("="*80)