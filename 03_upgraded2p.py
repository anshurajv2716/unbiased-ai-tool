# ============================================================
# UNBIASED AI DECISION TOOL — PROFESSIONAL VERSION
# Google Solution Challenge 2026
# Team: Anjani + Anshu
# ============================================================
# INSTALL: pip install streamlit pandas numpy scikit-learn aif360 matplotlib reportlab
# RUN: streamlit run app_pro.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import io
import base64
warnings.filterwarnings('ignore')

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Unbiased AI Decision Tool",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — PROFESSIONAL DESIGN
# ============================================================

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }

    /* Main content area */
    .main .block-container {
        background: rgba(255, 255, 255, 0.97);
        border-radius: 20px;
        padding: 2rem 3rem;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    .hero-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #f093fb, #f5576c, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
    }

    /* SDG badges */
    .sdg-container {
        display: flex;
        gap: 10px;
        justify-content: center;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .sdg-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
        padding: 6px 16px;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Step cards */
    .step-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #f0f0f0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    .step-number {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1rem;
        margin-right: 10px;
    }
    .step-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a1a2e;
        display: inline;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        border: 2px solid #f0f0f0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #666;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .metric-change {
        font-size: 0.9rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    /* Alert boxes */
    .alert-danger {
        background: linear-gradient(135deg, #ff6b6b22, #ff6b6b11);
        border: 2px solid #ff6b6b;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #c0392b;
        font-weight: 600;
    }
    .alert-success {
        background: linear-gradient(135deg, #00b09b22, #96c93d11);
        border: 2px solid #00b09b;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #1e8449;
        font-weight: 600;
    }
    .alert-warning {
        background: linear-gradient(135deg, #f39c1222, #f1c40f11);
        border: 2px solid #f39c12;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #9a7d0a;
        font-weight: 600;
    }
    .alert-info {
        background: linear-gradient(135deg, #667eea22, #764ba211);
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #1a1a6e;
        font-weight: 600;
    }

    /* Traffic light */
    .traffic-container {
        background: #1a1a2e;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        max-width: 200px;
        margin: 0 auto;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .traffic-light {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        margin: 0.6rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        font-weight: 900;
        color: white;
        box-shadow: 0 0 20px currentColor;
    }
    .light-red    { background: #e74c3c; color: #e74c3c; box-shadow: 0 0 25px #e74c3c; }
    .light-yellow { background: #f39c12; color: #f39c12; box-shadow: 0 0 25px #f39c12; }
    .light-green  { background: #2ecc71; color: #2ecc71; box-shadow: 0 0 25px #2ecc71; }
    .light-off    { background: #2c3e50; box-shadow: none; }

    /* Recommendation cards */
    .rec-card {
        background: linear-gradient(135deg, #f8f9ff, #f0f4ff);
        border-left: 5px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        color: #2c3e50;
    }

    /* Download button */
    .download-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 12px 30px;
        border-radius: 30px;
        text-decoration: none;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(102,126,234,0.5);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #888;
        font-size: 0.85rem;
        border-top: 2px solid #f0f0f0;
        margin-top: 2rem;
    }

    /* Section divider */
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f5576c);
        border-radius: 3px;
        margin: 2rem 0;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border: 2px solid #667eea !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size: 3rem;'>⚖️</div>
        <h2 style='color: white; margin: 0.5rem 0;'>Bias Detector</h2>
        <p style='color: rgba(255,255,255,0.6); font-size:0.85rem;'>
            Google Solution Challenge 2026
        </p>
    </div>
    <hr style='border-color: rgba(255,255,255,0.2);'>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 How to Use")
    steps_sidebar = [
        "Select **sector** — Job / Loan / Health",
        "Upload **CSV file** or use sample",
        "Choose **protected attribute**",
        "Click **Analyze** — see results",
        "Download **PDF report**"
    ]
    for i, step in enumerate(steps_sidebar, 1):
        st.markdown(f"**{i}.** {step}")

    st.markdown("<hr style='border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    st.markdown("### 🎯 SDG Alignment")
    st.markdown("🟣 **SDG 5** — Gender Equality")
    st.markdown("🔵 **SDG 8** — Decent Work")
    st.markdown("🟢 **SDG 10** — Reduced Inequalities")
    st.markdown("🔴 **SDG 16** — Justice & Accountability")

    st.markdown("<hr style='border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color: rgba(255,255,255,0.5); font-size:0.8rem;'>
        Made with ❤️ for Fair AI<br>
        <em>Bias fully remove nahi hota —<br>lekin reduce kar sakte hain</em>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class="hero-header">
    <h1>⚖️ Unbiased AI Decision Tool</h1>
    <p>Google Solution Challenge 2026 — Team Anjani &amp; Anshu</p>
    <div class="sdg-container" style="margin-top:1rem;">
        <span class="sdg-badge">🟣 SDG 5 — Gender Equality</span>
        <span class="sdg-badge">🔵 SDG 8 — Decent Work</span>
        <span class="sdg-badge">🟢 SDG 10 — Reduced Inequalities</span>
        <span class="sdg-badge">🔴 SDG 16 — Justice</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SECTOR CONFIGS
# ============================================================

SECTOR_CONFIG = {
    "Job Hiring": {
        "icon": "💼",
        "label_col": "hired",
        "label_pos": 1,
        "label_neg": 0,
        "outcome_name": "Hire Rate",
        "sdg": "SDG 8 + SDG 5",
        "desc": "Hiring data mein gender, caste, location bias detect karo",
        "protected_options": ["gender", "location", "caste_proxy"],
        "group_names": {
            "gender":      {1: "Male",  0: "Female"},
            "location":    {1: "Urban", 0: "Rural"},
            "caste_proxy": {1: "Group A", 0: "Group B"},
        },
        "bias_message": {
            "gender":   "Mahilaon ko unfairly reject kiya ja raha hai",
            "location": "Rural candidates ko unfairly reject kiya ja raha hai",
            "caste_proxy": "Ek group ko unfairly treat kiya ja raha hai",
        }
    },
    "Loan Approval": {
        "icon": "🏦",
        "label_col": "approved",
        "label_pos": 1,
        "label_neg": 0,
        "outcome_name": "Approval Rate",
        "sdg": "SDG 10 + SDG 16",
        "desc": "Loan data mein income, location, gender bias detect karo",
        "protected_options": ["gender", "location", "income_group"],
        "group_names": {
            "gender":       {1: "Male",      0: "Female"},
            "location":     {1: "Urban",     0: "Rural"},
            "income_group": {1: "High Income", 0: "Low Income"},
        },
        "bias_message": {
            "gender":       "Mahilaon ko loans mein unfair treatment",
            "location":     "Rural applicants ke loans zyada reject ho rahe hain",
            "income_group": "Low income group ko unfairly treat kiya ja raha hai",
        }
    },
    "Healthcare": {
        "icon": "🏥",
        "label_col": "treatment",
        "label_pos": 1,
        "label_neg": 0,
        "outcome_name": "Treatment Rate",
        "sdg": "SDG 3 + SDG 10",
        "desc": "Healthcare data mein age, gender, area bias detect karo",
        "protected_options": ["gender", "area", "age_group"],
        "group_names": {
            "gender":    {1: "Male",   0: "Female"},
            "area":      {1: "Urban",  0: "Rural"},
            "age_group": {1: "Young",  0: "Elderly"},
        },
        "bias_message": {
            "gender":    "Mahilaon ko kam treatment recommend ki ja rahi hai",
            "area":      "Rural patients ko kam specialist referrals mil rahe hain",
            "age_group": "Elderly patients ke saath unfair treatment ho raha hai",
        }
    }
}


# ============================================================
# SAMPLE DATA GENERATORS
# ============================================================

def generate_sample_data(sector, n=500):
    np.random.seed(42)
    if sector == "Job Hiring":
        data = pd.DataFrame({
            'gender':      np.random.choice([1,0], n, p=[0.55,0.45]),
            'age':         np.random.randint(21,50,n),
            'experience':  np.random.randint(0,20,n),
            'education':   np.random.choice([0,1,2],n),
            'location':    np.random.choice([1,0],n,p=[0.65,0.35]),
            'caste_proxy': np.random.choice([1,0],n,p=[0.70,0.30]),
        })
        hired = []
        for _, r in data.iterrows():
            p = 0.25 + r['experience']*0.025 + r['education']*0.08
            if r['gender']==0:      p -= 0.22
            if r['location']==0:    p -= 0.12
            if r['caste_proxy']==0: p -= 0.10
            hired.append(np.random.binomial(1, float(np.clip(p,0.05,0.92))))
        data['hired'] = hired
        return data

    elif sector == "Loan Approval":
        data = pd.DataFrame({
            'gender':       np.random.choice([1,0], n, p=[0.60,0.40]),
            'age':          np.random.randint(25,60,n),
            'income':       np.random.randint(15000,120000,n),
            'credit_score': np.random.randint(300,850,n),
            'location':     np.random.choice([1,0],n,p=[0.60,0.40]),
            'income_group': np.random.choice([1,0],n,p=[0.55,0.45]),
        })
        approved = []
        for _, r in data.iterrows():
            p = 0.20 + (r['credit_score']-300)/1100 + (r['income']/200000)
            if r['gender']==0:       p -= 0.18
            if r['location']==0:     p -= 0.15
            if r['income_group']==0: p -= 0.20
            approved.append(np.random.binomial(1, float(np.clip(p,0.05,0.92))))
        data['approved'] = approved
        return data

    else:  # Healthcare
        data = pd.DataFrame({
            'gender':    np.random.choice([1,0],n,p=[0.52,0.48]),
            'age':       np.random.randint(18,80,n),
            'income':    np.random.randint(10000,100000,n),
            'area':      np.random.choice([1,0],n,p=[0.60,0.40]),
            'age_group': np.random.choice([1,0],n,p=[0.65,0.35]),
        })
        treatment = []
        for _, r in data.iterrows():
            p = 0.45 + (r['income']/200000)
            if r['gender']==0:    p -= 0.15
            if r['area']==0:      p -= 0.20
            if r['age_group']==0: p -= 0.12
            treatment.append(np.random.binomial(1, float(np.clip(p,0.05,0.92))))
        data['treatment'] = treatment
        return data


# ============================================================
# BIAS ANALYSIS FUNCTION
# ============================================================

def run_bias_analysis(df, label_col, protected_attr, label_pos, label_neg):
    try:
        cols_needed = [c for c in df.columns if c != label_col] + [label_col]
        df_clean = df[cols_needed].copy()
        df_clean = df_clean.dropna()
        df_clean[label_col] = df_clean[label_col].astype(int)
        df_clean[protected_attr] = df_clean[protected_attr].astype(int)

        dataset = BinaryLabelDataset(
            df=df_clean,
            label_names=[label_col],
            protected_attribute_names=[protected_attr],
            favorable_label=label_pos,
            unfavorable_label=label_neg
        )

        priv   = [{protected_attr: 1}]
        unpriv = [{protected_attr: 0}]

        metric_before = BinaryLabelDatasetMetric(
            dataset, unprivileged_groups=unpriv, privileged_groups=priv
        )
        di_before  = metric_before.disparate_impact()
        spd_before = metric_before.statistical_parity_difference()

        RW = Reweighing(unprivileged_groups=unpriv, privileged_groups=priv)
        dataset_fixed = RW.fit_transform(dataset)

        metric_after = BinaryLabelDatasetMetric(
            dataset_fixed, unprivileged_groups=unpriv, privileged_groups=priv
        )
        di_after  = metric_after.disparate_impact()
        spd_after = metric_after.statistical_parity_difference()

        weights   = dataset_fixed.instance_weights
        priv_mask   = df_clean[protected_attr] == 1
        unpriv_mask = df_clean[protected_attr] == 0

        rate_priv_before   = df_clean[priv_mask][label_col].mean() * 100
        rate_unpriv_before = df_clean[unpriv_mask][label_col].mean() * 100
        rate_priv_after    = np.average(
            df_clean[priv_mask][label_col], weights=weights[priv_mask]) * 100
        rate_unpriv_after  = np.average(
            df_clean[unpriv_mask][label_col], weights=weights[unpriv_mask]) * 100

        improvement = ((di_after - di_before) / abs(di_before)) * 100 if di_before != 0 else 0

        return {
            'di_before': di_before, 'di_after': di_after,
            'spd_before': spd_before, 'spd_after': spd_after,
            'rate_priv_before': rate_priv_before,
            'rate_unpriv_before': rate_unpriv_before,
            'rate_priv_after': rate_priv_after,
            'rate_unpriv_after': rate_unpriv_after,
            'improvement': improvement,
            'n_total': len(df_clean),
            'n_priv': int(priv_mask.sum()),
            'n_unpriv': int(unpriv_mask.sum()),
        }
    except Exception as e:
        return {'error': str(e)}


# ============================================================
# CHART GENERATOR
# ============================================================

def generate_charts(results, sector, protected_attr, group_names, outcome_name, config):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#f8f9ff')

    fig.suptitle(
        f'⚖️ Bias Detection Report — {sector}\n'
        f'Google Solution Challenge 2026 | Team Anjani & Anshu',
        fontsize=14, fontweight='bold', y=0.98,
        color='#1a1a2e'
    )

    g_names = group_names.get(protected_attr, {1: "Group 1", 0: "Group 0"})
    priv_name   = g_names.get(1, "Privileged")
    unpriv_name = g_names.get(0, "Unprivileged")

    colors = {
        'before_priv':   '#4A90D9',
        'before_unpriv': '#E05C5C',
        'after_priv':    '#5CB85C',
        'after_unpriv':  '#27AE60',
        'danger':        '#E74C3C',
        'warning':       '#F39C12',
        'success':       '#2ECC71',
    }

    def bias_color(di):
        if di < 0.8: return colors['danger']
        if di < 0.9: return colors['warning']
        return colors['success']

    def bias_label(di):
        if di < 0.8: return 'HIGH BIAS'
        if di < 0.9: return 'MEDIUM BIAS'
        return 'LOW BIAS'

    # --- Chart 1: Outcome Rate Comparison ---
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor('#FAFBFF')
    cats = [f'{priv_name}\n(Before)', f'{unpriv_name}\n(Before)',
            f'{priv_name}\n(After)',  f'{unpriv_name}\n(After)']
    vals = [results['rate_priv_before'],  results['rate_unpriv_before'],
            results['rate_priv_after'],   results['rate_unpriv_after']]
    bar_colors = [colors['before_priv'], colors['before_unpriv'],
                  colors['after_priv'],  colors['after_unpriv']]
    bars = ax1.bar(cats, vals, color=bar_colors, width=0.55,
                   edgecolor='white', linewidth=2, zorder=3)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.8,
                 f'{val:.1f}%', ha='center',
                 fontweight='bold', fontsize=10, color='#2c3e50')
    ax1.set_title(f'{sector}\n{outcome_name} Comparison',
                  fontweight='bold', fontsize=10, color='#1a1a2e', pad=10)
    ax1.set_ylabel(f'{outcome_name} (%)', fontsize=9, color='#555')
    ax1.set_ylim(0, max(vals)*1.3)
    ax1.grid(axis='y', alpha=0.3, zorder=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Chart 2: Disparate Impact Score ---
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor('#FAFBFF')
    di_vals   = [results['di_before'], results['di_after']]
    di_labels = ['BEFORE\n(Biased)', 'AFTER\n(Fixed)']
    di_colors = [bias_color(v) for v in di_vals]
    bars2 = ax2.bar(di_labels, di_vals, color=di_colors, width=0.45,
                    edgecolor='white', linewidth=2, zorder=3)
    for bar, val in zip(bars2, di_vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.012,
                 f'{val:.3f}', ha='center',
                 fontweight='bold', fontsize=13, color='#2c3e50')
    ax2.axhline(y=0.8, color='#F39C12', linestyle='--',
                linewidth=2, label='Min Fair (0.8)', zorder=2)
    ax2.axhline(y=1.0, color='#2ECC71', linestyle='--',
                linewidth=2, label='Perfect (1.0)', zorder=2)
    ax2.set_title('Disparate Impact Score\n(1.0 = Perfectly Fair)',
                  fontweight='bold', fontsize=10, color='#1a1a2e', pad=10)
    ax2.set_ylabel('Score', fontsize=9, color='#555')
    ax2.set_ylim(0, 1.3)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(axis='y', alpha=0.3, zorder=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # --- Chart 3: Traffic Light (FIXED - no overlap) ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 12)
    ax3.axis('off')
    ax3.set_facecolor('#FAFBFF')
    ax3.set_title('Bias Status Card',
                  fontweight='bold', fontsize=10, color='#1a1a2e', pad=10)

    # Background card
    bg = mpatches.FancyBboxPatch((0.5, 0.3), 9, 11.2,
        boxstyle="round,pad=0.1",
        facecolor='#1a1a2e', edgecolor='#667eea',
        linewidth=2, zorder=1)
    ax3.add_patch(bg)

    # BEFORE section - top half
    ax3.text(5, 11.0, 'BEFORE', ha='center', fontweight='bold',
             fontsize=11, color='#aaa', zorder=2)
    b_color = bias_color(results['di_before'])
    b_label = bias_label(results['di_before'])
    circle_b = plt.Circle((5, 9.5), 0.9, color=b_color, zorder=2)
    ax3.add_patch(circle_b)
    ax3.text(5, 9.5, b_label, ha='center', va='center',
             fontweight='bold', fontsize=7, color='white', zorder=3)
    ax3.text(5, 8.4, f'DI: {results["di_before"]:.3f}',
             ha='center', fontsize=10, color='#ddd', zorder=2, fontweight='bold')
    ax3.text(5, 7.7, f'{unpriv_name}: {results["rate_unpriv_before"]:.1f}%',
             ha='center', fontsize=9, color='#aaa', zorder=2)
    ax3.text(5, 7.2, f'{priv_name}: {results["rate_priv_before"]:.1f}%',
             ha='center', fontsize=9, color='#aaa', zorder=2)

    # Arrow - middle
    ax3.annotate('', xy=(5, 6.3), xytext=(5, 6.8),
                 arrowprops=dict(arrowstyle='->', color='#667eea', lw=2.5),
                 zorder=2)
    ax3.text(5, 6.55, 'Reweighing', ha='center', fontsize=8,
             color='#667eea', fontweight='bold', zorder=2)

    # AFTER section - bottom half
    ax3.text(5, 6.0, 'AFTER', ha='center', fontweight='bold',
             fontsize=11, color='#aaa', zorder=2)
    a_color = bias_color(results['di_after'])
    a_label = bias_label(results['di_after'])
    circle_a = plt.Circle((5, 4.6), 0.9, color=a_color, zorder=2)
    ax3.add_patch(circle_a)
    ax3.text(5, 4.6, a_label, ha='center', va='center',
             fontweight='bold', fontsize=7, color='white', zorder=3)
    ax3.text(5, 3.5, f'DI: {results["di_after"]:.3f}',
             ha='center', fontsize=10, color='#ddd', zorder=2, fontweight='bold')
    ax3.text(5, 2.8, f'{unpriv_name}: {results["rate_unpriv_after"]:.1f}%',
             ha='center', fontsize=9, color='#aaa', zorder=2)
    ax3.text(5, 2.3, f'{priv_name}: {results["rate_priv_after"]:.1f}%',
             ha='center', fontsize=9, color='#aaa', zorder=2)

    # Improvement at bottom
    imp = results['improvement']
    imp_color = '#2ECC71' if imp > 0 else '#E74C3C'
    ax3.text(5, 1.4, f'Improvement: +{imp:.1f}%',
             ha='center', fontweight='bold', fontsize=11,
             color=imp_color, zorder=2)
    ax3.text(5, 0.8, config['sdg'],
             ha='center', fontsize=8, color='#667eea', zorder=2)

    # --- Chart 4: Horizontal overview ---
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor('#FAFBFF')
    m_names = ['DI Before', 'DI After', 'SPD Before', 'SPD After']
    m_vals  = [results['di_before'], results['di_after'],
               abs(results['spd_before']), abs(results['spd_after'])]
    m_colors = [bias_color(results['di_before']),
                bias_color(results['di_after']),
                '#E05C5C', '#5CB85C']
    bars4 = ax4.barh(m_names, m_vals, color=m_colors,
                     edgecolor='white', linewidth=1.5, zorder=3)
    for bar, val in zip(bars4, m_vals):
        ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center',
                 fontweight='bold', fontsize=10, color='#2c3e50')
    ax4.axvline(x=0.8, color='#F39C12', linestyle='--',
                linewidth=2, label='Min Fair', zorder=2)
    ax4.set_title('All Metrics Overview',
                  fontweight='bold', fontsize=10, color='#1a1a2e', pad=10)
    ax4.set_xlabel('Score', fontsize=9, color='#555')
    ax4.set_xlim(0, 1.4)
    ax4.legend(fontsize=8)
    ax4.grid(axis='x', alpha=0.3, zorder=0)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # --- Chart 5: Gap analysis ---
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('#FAFBFF')
    gap_before = abs(results['rate_priv_before'] - results['rate_unpriv_before'])
    gap_after  = abs(results['rate_priv_after']  - results['rate_unpriv_after'])
    ax5.bar(['Gap Before', 'Gap After'],
            [gap_before, gap_after],
            color=[colors['danger'], colors['success']],
            width=0.5, edgecolor='white', linewidth=2, zorder=3)
    ax5.text(0, gap_before + 0.5, f'{gap_before:.1f}%',
             ha='center', fontweight='bold', fontsize=13, color='#2c3e50')
    ax5.text(1, gap_after + 0.5, f'{gap_after:.1f}%',
             ha='center', fontweight='bold', fontsize=13, color='#2c3e50')
    ax5.set_title(f'Outcome Gap\n({priv_name} vs {unpriv_name})',
                  fontweight='bold', fontsize=10, color='#1a1a2e', pad=10)
    ax5.set_ylabel('Gap (%)', fontsize=9, color='#555')
    ax5.set_ylim(0, max(gap_before, gap_after) * 1.4)
    ax5.grid(axis='y', alpha=0.3, zorder=0)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # --- Chart 6: Improvement donut ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#FAFBFF')
    imp_val = min(abs(results['improvement']), 100)
    remaining = max(0, 100 - imp_val)
    wedges, _ = ax6.pie(
        [imp_val, remaining],
        colors=['#667eea', '#f0f0f0'],
        startangle=90,
        wedgeprops={'width': 0.55, 'edgecolor': 'white', 'linewidth': 3}
    )
    ax6.text(0, 0, f'+{imp_val:.0f}%',
             ha='center', va='center',
             fontsize=18, fontweight='bold', color='#667eea')
    ax6.text(0, -0.35, 'Bias Reduced',
             ha='center', va='center',
             fontsize=9, color='#555', fontweight='bold')
    ax6.set_title('Overall Improvement',
                  fontweight='bold', fontsize=10, color='#1a1a2e', pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150,
                bbox_inches='tight', facecolor='#f8f9ff')
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# MAIN APP
# ============================================================

# --- STEP 1: SECTOR ---
st.markdown("""
<div class="step-card">
    <span class="step-number">1</span>
    <span class="step-title">Sector Choose Karo</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    sector = st.selectbox(
        "Aap kis sector ka data check karna chahte ho?",
        list(SECTOR_CONFIG.keys()),
        format_func=lambda x: f"{SECTOR_CONFIG[x]['icon']} {x}"
    )
with col2:
    st.markdown(f"""
    <div class="alert-info" style="margin-top:1.5rem;">
        {SECTOR_CONFIG[sector]['icon']} <strong>{sector}</strong> — {SECTOR_CONFIG[sector]['desc']}
        <br><small>🎯 {SECTOR_CONFIG[sector]['sdg']}</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# --- STEP 2: DATA ---
st.markdown("""
<div class="step-card">
    <span class="step-number">2</span>
    <span class="step-title">Data Upload Karo</span>
</div>
""", unsafe_allow_html=True)

config = SECTOR_CONFIG[sector]
data_source = st.radio(
    "Data source choose karo:",
    ["🎯 Sample Data Use Karo (Demo)", "📁 Apni CSV File Upload Karo"],
    horizontal=True
)

df = None
if "Sample" in data_source:
    with st.spinner("Demo ke liye sample biased dataset load ho raha hai..."):
        df = generate_sample_data(sector)
    st.markdown(f"""
    <div class="alert-success">
        ✅ Sample data ready! {len(df)} records loaded.
    </div>
    """, unsafe_allow_html=True)
    with st.expander("📊 Dataset Preview (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)
else:
    uploaded = st.file_uploader(
        f"CSV file upload karo ({config['label_col']} column hona chahiye)",
        type=['csv']
    )
    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"""
        <div class="alert-success">
            ✅ File upload successful! {len(df)} records, {len(df.columns)} columns.
        </div>
        """, unsafe_allow_html=True)
        with st.expander("📊 Aapka Dataset (first 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)
    else:
        st.markdown("""
        <div class="alert-warning">
            ⚠️ CSV file upload karo jisme outcome column (hired/approved/treatment) ho
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# --- STEP 3: PROTECTED ATTRIBUTE ---
if df is not None:
    st.markdown("""
    <div class="step-card">
        <span class="step-number">3</span>
        <span class="step-title">Protected Attribute Choose Karo</span>
    </div>
    """, unsafe_allow_html=True)

    available_cols = [c for c in config['protected_options'] if c in df.columns]
    if not available_cols:
        available_cols = [c for c in df.columns if c != config['label_col']]

    protected_attr = st.selectbox(
        "Kaunse column mein bias check karna hai?",
        available_cols,
        format_func=lambda x: f"🔍 {x.replace('_', ' ').title()}"
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- STEP 4: ANALYZE ---
    st.markdown("""
    <div class="step-card">
        <span class="step-number">4</span>
        <span class="step-title">Bias Analyze Karo</span>
    </div>
    """, unsafe_allow_html=True)

    analyze_btn = st.button(
        "🔍  ANALYZE BIAS NOW",
        use_container_width=True,
        type="primary"
    )

    if analyze_btn:
        with st.spinner("🧠 AI bias analyze kar raha hai..."):
            results = run_bias_analysis(
                df,
                config['label_col'],
                protected_attr,
                config['label_pos'],
                config['label_neg']
            )

        if 'error' in results:
            st.error(f"Error: {results['error']}")
        else:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # --- RESULTS HEADER ---
            st.markdown("## 📊 Bias Analysis Results")

            # Bias level alert
            di_b = results['di_before']
            group_names = config['group_names']
            gap = abs(results['rate_priv_before'] - results['rate_unpriv_before'])
            bias_msg = config['bias_message'].get(protected_attr, "Bias detected")

            if di_b < 0.8:
                st.markdown(f"""
                <div class="alert-danger">
                    🚨 <strong>HIGH BIAS DETECTED</strong> —
                    {bias_msg}. Group 0 ko Group 1 ke comparison mein
                    {gap:.1f}% unfairly treat kiya ja raha hai.
                </div>
                """, unsafe_allow_html=True)
            elif di_b < 0.9:
                st.markdown(f"""
                <div class="alert-warning">
                    ⚠️ <strong>MEDIUM BIAS DETECTED</strong> —
                    {bias_msg}. Gap: {gap:.1f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-success">
                    ✅ <strong>LOW BIAS</strong> — Data relatively fair lag raha hai!
                </div>
                """, unsafe_allow_html=True)

            # --- METRIC CARDS ---
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                color = "#E74C3C" if di_b < 0.8 else "#F39C12" if di_b < 0.9 else "#2ECC71"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Disparate Impact (Before)</div>
                    <div class="metric-value" style="color:{color};">
                        {di_b:.3f}
                    </div>
                    <div class="metric-change" style="color:#E74C3C;">
                        ⬆ Biased
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                color2 = "#E74C3C" if results['di_after'] < 0.8 else "#F39C12" if results['di_after'] < 0.9 else "#2ECC71"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Disparate Impact (After)</div>
                    <div class="metric-value" style="color:{color2};">
                        {results['di_after']:.3f}
                    </div>
                    <div class="metric-change" style="color:#2ECC71;">
                        ⬆ +{results['improvement']:.1f}% improvement
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Stat Parity Diff (Before)</div>
                    <div class="metric-value" style="color:#E74C3C;">
                        {results['spd_before']:.3f}
                    </div>
                    <div class="metric-change" style="color:#888;">
                        0.0 = perfect fair
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Stat Parity Diff (After)</div>
                    <div class="metric-value" style="color:#2ECC71;">
                        {results['spd_after']:.3f}
                    </div>
                    <div class="metric-change" style="color:#2ECC71;">
                        Significantly reduced!
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- CHARTS ---
            st.markdown("### 📈 Before vs After Comparison")
            chart_buf = generate_charts(
                results, sector, protected_attr,
                config['group_names'], config['outcome_name'], config
            )
            st.image(chart_buf, use_container_width=True)

            # --- TRAFFIC LIGHT ---
            st.markdown("### 🚦 Quick Status")
            tcol1, tcol2, tcol3 = st.columns([1, 2, 1])
            with tcol2:
                def light_class(di):
                    if di < 0.8: return "light-red", "🔴", "HIGH BIAS"
                    if di < 0.9: return "light-yellow", "🟡", "MEDIUM BIAS"
                    return "light-green", "🟢", "LOW BIAS"

                cls_b, emoji_b, lbl_b = light_class(results['di_before'])
                cls_a, emoji_a, lbl_a = light_class(results['di_after'])

                st.markdown(f"""
                <div class="traffic-container">
                    <div style="color:#aaa; font-size:0.8rem; margin-bottom:0.5rem;">
                        BEFORE
                    </div>
                    <div class="traffic-light {cls_b}">{emoji_b}</div>
                    <div style="color:white; font-size:0.75rem; margin:0.3rem 0 1rem;">
                        {lbl_b} — {results['di_before']:.3f}
                    </div>
                    <div style="color:#667eea; font-size:0.85rem; font-weight:bold;">
                        ↓ Reweighing Applied
                    </div>
                    <div style="color:#aaa; font-size:0.8rem; margin-top:1rem;">
                        AFTER
                    </div>
                    <div class="traffic-light {cls_a}">{emoji_a}</div>
                    <div style="color:white; font-size:0.75rem; margin:0.3rem 0 0.5rem;">
                        {lbl_a} — {results['di_after']:.3f}
                    </div>
                    <div style="color:#2ECC71; font-size:1rem; font-weight:bold;">
                        +{results['improvement']:.1f}% improved!
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # --- EXPLANATION ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 💬 Simple Explanation")

            g_names = config['group_names'].get(protected_attr, {1:"Group 1",0:"Group 0"})
            p_name  = g_names.get(1, "Group 1")
            up_name = g_names.get(0, "Group 0")

            st.markdown(f"""
            <div class="step-card">
                <p><strong>Aapke {sector} data mein kya mila:</strong></p>
                <ul style="line-height:2rem;">
                    <li>🔴 <strong>Before fix:</strong> {up_name} ko {p_name} ke comparison mein
                        <strong>{gap:.1f}%</strong> unfairly treat kiya ja raha tha</li>
                    <li>🟢 <strong>After fix:</strong> Gap sirf
                        <strong>{abs(results['rate_priv_after']-results['rate_unpriv_after']):.1f}%</strong>
                        reh gaya</li>
                    <li>📈 <strong>Improvement:</strong> <strong>{results['improvement']:.1f}%</strong>
                        bias reduce hua</li>
                    <li>⚖️ <strong>Disparate Impact:</strong> {results['di_before']:.3f} se
                        badh ke <strong>{results['di_after']:.3f}</strong> ho gaya</li>
                </ul>
                <p><strong>Matlab:</strong> Reweighing technique ne bias significantly reduce kar diya.
                Yeh tool SDG 5, SDG 10, aur SDG 16 ko directly support karta hai.</p>
            </div>
            """, unsafe_allow_html=True)

            # --- RECOMMENDATIONS ---
            st.markdown("### ✅ Recommendations")
            recs = [
                f"<strong>{protected_attr.replace('_',' ').title()} column</strong> ko sensitive feature mark karo",
                "Reweighing apply karo training data pe — har 3 mahine mein",
                "Diverse data collect karo — underrepresented groups ka",
                "Human review rakho final decisions mein — AI ka last word nahi",
                "Regular monitoring karo — bias wapas aa sakta hai new data ke saath",
                "Transparency report publish karo — SDG 16 accountability ke liye"
            ]
            for rec in recs:
                st.markdown(f'<div class="rec-card">✔ {rec}</div>',
                           unsafe_allow_html=True)

            # --- DOWNLOAD ---
            st.markdown('<div class="section-divider"></div>',
                       unsafe_allow_html=True)
            st.markdown("### 📥 Download Report")

            chart_buf.seek(0)
            b64 = base64.b64encode(chart_buf.read()).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="bias_report_{sector.replace(" ","_")}.png" class="download-btn">📊 Download Bias Report (PNG)</a>'
            st.markdown(href, unsafe_allow_html=True)

            # Success message
            st.markdown(f"""
            <div class="alert-success" style="margin-top:1rem;">
                ✅ Analysis complete!
                Improvement: +{results['improvement']:.1f}% |
                {config['sdg']}
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <strong>⚖️ Unbiased AI Decision Tool</strong> — Google Solution Challenge 2026<br>
    Team Anjani &amp; Anshu | SDG 5 | SDG 8 | SDG 10 | SDG 16<br>
    <em>"AI khud biased nahi hota — wo data ka bias reflect karta hai"</em>
</div>
""", unsafe_allow_html=True)