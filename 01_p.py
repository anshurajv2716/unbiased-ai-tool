# ============================================================
# UNBIASED AI DECISION TOOL
# Google Solution Challenge 2026
# Team: Anjani + Anshu
# Day 1 — Core Bias Detection + Visualization
# ============================================================
#
# CHALANE KA TARIKA:
# Terminal mein likho: python 01_p.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

# ============================================================
# STEP 1 — SAMPLE DATASET BANANA
# ============================================================

print("\n" + "="*60)
print("   UNBIASED AI DECISION TOOL")
print("   Google Solution Challenge 2026")
print("="*60)

print("\n[1/5] Dataset taiyar ho raha hai...")

np.random.seed(42)
n = 600

# Hiring dataset banao
data = pd.DataFrame({
    'gender':     np.random.choice([1, 0], n, p=[0.55, 0.45]),  # 1=Male 0=Female
    'age':        np.random.randint(21, 50, n),
    'experience': np.random.randint(0, 20, n),
    'education':  np.random.choice([0, 1, 2], n),               # 0=Basic 1=Grad 2=PostGrad
    'location':   np.random.choice([1, 0], n, p=[0.65, 0.35]),  # 1=Urban 0=Rural
    'caste_proxy':np.random.choice([1, 0], n, p=[0.70, 0.30]),  # Indian context bias
})

# Biased outcome — yahi real duniya mein hota hai
hired = []
for _, row in data.iterrows():
    prob = 0.25 + (row['experience'] * 0.025) + (row['education'] * 0.08)
    if row['gender'] == 0:           # Female ko unfair penalty
        prob -= 0.22
    if row['location'] == 0:         # Rural ko unfair penalty
        prob -= 0.12
    if row['caste_proxy'] == 0:      # Indian context bias
        prob -= 0.10
    prob = max(0.05, min(0.92, prob))
    hired.append(np.random.binomial(1, prob))

data['hired'] = hired

print(f"   Total candidates : {len(data)}")
print(f"   Male             : {sum(data['gender']==1)}")
print(f"   Female           : {sum(data['gender']==0)}")
print(f"   Urban            : {sum(data['location']==1)}")
print(f"   Rural            : {sum(data['location']==0)}")


# ============================================================
# STEP 2 — BIAS MEASURE KARO — GENDER
# ============================================================

print("\n[2/5] Gender bias measure kar raha hai...")

dataset_gender = BinaryLabelDataset(
    df=data[['gender', 'age', 'experience',
             'education', 'location', 'hired']],
    label_names=['hired'],
    protected_attribute_names=['gender'],
    favorable_label=1,
    unfavorable_label=0
)

priv_gender   = [{'gender': 1}]
unpriv_gender = [{'gender': 0}]

metric_gender_before = BinaryLabelDatasetMetric(
    dataset_gender,
    unprivileged_groups=unpriv_gender,
    privileged_groups=priv_gender
)

di_gender_before  = metric_gender_before.disparate_impact()
spd_gender_before = metric_gender_before.statistical_parity_difference()

male_rate   = data[data['gender']==1]['hired'].mean() * 100
female_rate = data[data['gender']==0]['hired'].mean() * 100

print(f"   Male hire rate   : {male_rate:.1f}%")
print(f"   Female hire rate : {female_rate:.1f}%")
print(f"   Disparate Impact : {di_gender_before:.3f}")
print(f"   Bias Level       : {'HIGH BIAS' if di_gender_before < 0.8 else 'MEDIUM' if di_gender_before < 0.9 else 'LOW'}")


# ============================================================
# STEP 3 — BIAS FIX KARO — REWEIGHING
# ============================================================

print("\n[3/5] Bias fix kar raha hai (Reweighing)...")

RW = Reweighing(
    unprivileged_groups=unpriv_gender,
    privileged_groups=priv_gender
)
dataset_gender_fixed = RW.fit_transform(dataset_gender)

metric_gender_after = BinaryLabelDatasetMetric(
    dataset_gender_fixed,
    unprivileged_groups=unpriv_gender,
    privileged_groups=priv_gender
)

di_gender_after  = metric_gender_after.disparate_impact()
spd_gender_after = metric_gender_after.statistical_parity_difference()

weights      = dataset_gender_fixed.instance_weights
male_mask    = data['gender'] == 1
female_mask  = data['gender'] == 0
male_fixed   = np.average(data[male_mask]['hired'],   weights=weights[male_mask])   * 100
female_fixed = np.average(data[female_mask]['hired'], weights=weights[female_mask]) * 100

print(f"   Male hire rate   : {male_fixed:.1f}%  (was {male_rate:.1f}%)")
print(f"   Female hire rate : {female_fixed:.1f}%  (was {female_rate:.1f}%)")
print(f"   Disparate Impact : {di_gender_after:.3f}  (was {di_gender_before:.3f})")
improvement = ((di_gender_after - di_gender_before) / di_gender_before) * 100
print(f"   Improvement      : +{improvement:.1f}%")


# ============================================================
# STEP 4 — LOCATION BIAS BHI CHECK KARO
# ============================================================

print("\n[4/5] Location bias check kar raha hai...")

dataset_loc = BinaryLabelDataset(
    df=data[['location', 'age', 'experience',
             'education', 'gender', 'hired']],
    label_names=['hired'],
    protected_attribute_names=['location'],
    favorable_label=1,
    unfavorable_label=0
)

priv_loc   = [{'location': 1}]
unpriv_loc = [{'location': 0}]

metric_loc = BinaryLabelDatasetMetric(
    dataset_loc,
    unprivileged_groups=unpriv_loc,
    privileged_groups=priv_loc
)

di_loc    = metric_loc.disparate_impact()
urban_rate = data[data['location']==1]['hired'].mean() * 100
rural_rate = data[data['location']==0]['hired'].mean() * 100

print(f"   Urban hire rate  : {urban_rate:.1f}%")
print(f"   Rural hire rate  : {rural_rate:.1f}%")
print(f"   Disparate Impact : {di_loc:.3f}")
print(f"   Bias Level       : {'HIGH BIAS' if di_loc < 0.8 else 'MEDIUM' if di_loc < 0.9 else 'LOW'}")


# ============================================================
# STEP 5 — CHARTS BANAO
# ============================================================

print("\n[5/5] Charts bana raha hai...")

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    'UNBIASED AI DECISION TOOL\nGoogle Solution Challenge 2026 — Bias Detection Report',
    fontsize=15, fontweight='bold', y=0.98
)

# --- Chart 1: Gender Hire Rate Before vs After ---
ax1 = fig.add_subplot(2, 3, 1)
cats   = ['Male\nBefore', 'Female\nBefore', 'Male\nAfter', 'Female\nAfter']
vals   = [male_rate, female_rate, male_fixed, female_fixed]
colors = ['#4A90D9', '#E05C5C', '#4A90D9', '#5CB85C']
bars   = ax1.bar(cats, vals, color=colors, width=0.55,
                  edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center',
             fontweight='bold', fontsize=9)
ax1.set_title('Gender Bias — Before vs After', fontweight='bold', fontsize=10)
ax1.set_ylabel('Hire Rate (%)')
ax1.set_ylim(0, max(vals) * 1.3)
ax1.set_facecolor('#F8F9FA')
ax1.grid(axis='y', alpha=0.3)


# --- Chart 2: Disparate Impact Score ---
ax2 = fig.add_subplot(2, 3, 2)
di_vals   = [di_gender_before, di_gender_after]
di_labels = ['Before\n(Biased)', 'After\n(Fixed)']
di_colors = ['#E05C5C' if v < 0.8 else
             '#F0A500' if v < 0.9 else '#5CB85C'
             for v in di_vals]
bars2 = ax2.bar(di_labels, di_vals, color=di_colors,
                 width=0.45, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, di_vals):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f'{val:.3f}', ha='center',
             fontweight='bold', fontsize=12)
ax2.axhline(y=0.8, color='orange', linestyle='--',
             linewidth=2, label='Min Fair (0.8)')
ax2.axhline(y=1.0, color='green', linestyle='--',
             linewidth=2, label='Perfect (1.0)')
ax2.set_title('Disparate Impact Score', fontweight='bold', fontsize=10)
ax2.set_ylabel('Score (1.0 = Perfect Fair)')
ax2.set_ylim(0, 1.25)
ax2.legend(fontsize=8)
ax2.set_facecolor('#F8F9FA')
ax2.grid(axis='y', alpha=0.3)


# --- Chart 3: Location Bias ---
ax3 = fig.add_subplot(2, 3, 3)
loc_vals   = [urban_rate, rural_rate]
loc_labels = ['Urban', 'Rural']
loc_colors = ['#4A90D9', '#E05C5C' if di_loc < 0.8 else '#F0A500']
bars3 = ax3.bar(loc_labels, loc_vals, color=loc_colors,
                 width=0.45, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars3, loc_vals):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center',
             fontweight='bold', fontsize=11)
ax3.axhline(y=(urban_rate+rural_rate)/2, color='gray',
             linestyle='--', alpha=0.6, label='Average')
ax3.set_title('Location Bias (Urban vs Rural)', fontweight='bold', fontsize=10)
ax3.set_ylabel('Hire Rate (%)')
ax3.set_ylim(0, max(loc_vals) * 1.3)
ax3.legend(fontsize=8)
ax3.set_facecolor('#F8F9FA')
ax3.grid(axis='y', alpha=0.3)


# --- Chart 4: Overall Bias Summary Bar ---
ax4 = fig.add_subplot(2, 3, 4)
metrics_names  = ['Gender DI\n(Before)', 'Gender DI\n(After)', 'Location DI']
metrics_values = [di_gender_before, di_gender_after, di_loc]
m_colors = ['#E05C5C' if v < 0.8 else
            '#F0A500' if v < 0.9 else '#5CB85C'
            for v in metrics_values]
bars4 = ax4.barh(metrics_names, metrics_values,
                  color=m_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars4, metrics_values):
    ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontweight='bold', fontsize=10)
ax4.axvline(x=0.8, color='orange', linestyle='--',
             linewidth=2, label='Min Fair (0.8)')
ax4.axvline(x=1.0, color='green', linestyle='--',
             linewidth=2, label='Perfect (1.0)')
ax4.set_title('All Bias Metrics Overview', fontweight='bold', fontsize=10)
ax4.set_xlabel('Disparate Impact Score')
ax4.set_xlim(0, 1.3)
ax4.legend(fontsize=8)
ax4.set_facecolor('#F8F9FA')
ax4.grid(axis='x', alpha=0.3)


# --- Chart 5: Traffic Light Summary ---
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.axis('off')
ax5.set_facecolor('#F8F9FA')
ax5.set_title('Bias Summary Card', fontweight='bold', fontsize=10)

bg = plt.Rectangle((0.3, 0.3), 9.4, 9.4,
                     fill=True, facecolor='white',
                     edgecolor='#DDDDDD', linewidth=2)
ax5.add_patch(bg)

# Gender bias indicator
g_color = '#E05C5C' if di_gender_before < 0.8 else '#F0A500'
g_label = 'HIGH BIAS' if di_gender_before < 0.8 else 'MEDIUM BIAS'
ax5.text(5, 9.0, 'GENDER BIAS', ha='center',
          fontweight='bold', fontsize=10, color='#333')
c1 = plt.Circle((5, 7.8), 0.7, color=g_color)
ax5.add_patch(c1)
ax5.text(5, 7.8, g_label, ha='center', va='center',
          fontweight='bold', fontsize=7, color='white')
ax5.text(5, 6.9, f'Score: {di_gender_before:.3f} → {di_gender_after:.3f}',
          ha='center', fontsize=9, color='#555')

# Location bias indicator
l_color = '#E05C5C' if di_loc < 0.8 else '#F0A500'
l_label = 'HIGH BIAS' if di_loc < 0.8 else 'MEDIUM BIAS'
ax5.text(5, 6.1, 'LOCATION BIAS', ha='center',
          fontweight='bold', fontsize=10, color='#333')
c2 = plt.Circle((5, 4.9), 0.7, color=l_color)
ax5.add_patch(c2)
ax5.text(5, 4.9, l_label, ha='center', va='center',
          fontweight='bold', fontsize=7, color='white')
ax5.text(5, 4.0, f'Score: {di_loc:.3f}',
          ha='center', fontsize=9, color='#555')

# Improvement
ax5.text(5, 3.0, f'Improvement: +{improvement:.1f}%',
          ha='center', fontweight='bold', fontsize=11, color='#2E7D32')
ax5.text(5, 2.2, 'SDG 5  |  SDG 8  |  SDG 10  |  SDG 16',
          ha='center', fontsize=8, color='#1565C0')
ax5.text(5, 1.4, 'Tool: AIF360 + Python',
          ha='center', fontsize=8, color='#888')


# --- Chart 6: Improvement Progress ---
ax6 = fig.add_subplot(2, 3, 6)
categories_imp = ['Bias\nReduced', 'Fairness\nGained']
before_vals    = [abs(spd_gender_before) * 100, 0]
after_vals     = [0, (di_gender_after - di_gender_before) * 100]
x = np.arange(len(categories_imp))
w = 0.35
b1 = ax6.bar(x - w/2, [abs(spd_gender_before)*100,
             di_gender_before*100],
             w, label='Before', color='#E05C5C',
             edgecolor='white', linewidth=1.5)
b2 = ax6.bar(x + w/2, [abs(spd_gender_after)*100,
             di_gender_after*100],
             w, label='After', color='#5CB85C',
             edgecolor='white', linewidth=1.5)
ax6.set_title('Before vs After — Key Metrics', fontweight='bold', fontsize=10)
ax6.set_ylabel('Score (%)')
ax6.set_xticks(x)
ax6.set_xticklabels(['Stat Parity\nDifference (%)', 'Disparate\nImpact (%)'])
ax6.legend(fontsize=9)
ax6.set_facecolor('#F8F9FA')
ax6.grid(axis='y', alpha=0.3)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('bias_report_day1.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()


# ============================================================
# FINAL SUMMARY PRINT
# ============================================================

print("\n" + "="*60)
print("   FINAL REPORT")
print("="*60)
print(f"\n   GENDER BIAS:")
print(f"   Before — DI: {di_gender_before:.3f} | Female: {female_rate:.1f}% | Male: {male_rate:.1f}%")
print(f"   After  — DI: {di_gender_after:.3f} | Female: {female_fixed:.1f}% | Male: {male_fixed:.1f}%")
print(f"   Improvement : +{improvement:.1f}%")
print(f"\n   LOCATION BIAS:")
print(f"   Urban: {urban_rate:.1f}% | Rural: {rural_rate:.1f}%")
print(f"   DI Score: {di_loc:.3f} ({'HIGH BIAS' if di_loc < 0.8 else 'MEDIUM'})")
print(f"\n   Chart saved : bias_report_day1.png")
print(f"\n   SDG Goals   : SDG 5 | SDG 8 | SDG 10 | SDG 16")
print("\n" + "="*60)
print("   Day 1 Complete! Kal: Streamlit Web App banayenge!")
print("="*60 + "\n")