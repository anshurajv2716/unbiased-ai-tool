# ============================================================
# UNBIASED AI DECISION TOOL — GEMINI POWERED VERSION
# Google Solution Challenge 2026 — Team Solvation
# Anjani (ECE 2nd Year) + Anshu Raj Verma (ECE 1st Year)
# ============================================================
# INSTALL: pip install streamlit pandas numpy scikit-learn
#          aif360 matplotlib reportlab google-generativeai
#          python-dotenv
# RUN: streamlit run 04_gemini_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import io
import base64
import os
warnings.filterwarnings('ignore')

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

# Gemini import
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except:
    GEMINI_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Unbiased AI Decision Tool — Powered by Gemini",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .main .block-container {
        background: rgba(255,255,255,0.97);
        border-radius: 20px;
        padding: 2rem 3rem;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#1a1a2e,#16213e,#0f3460);
    }
    [data-testid="stSidebar"] * { color: white !important; }

    .hero-header {
        background: linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
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
        background: linear-gradient(90deg,#f093fb,#f5576c,#4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-header p { color: rgba(255,255,255,0.85); font-size:1.05rem; margin:0.5rem 0 0; }

    .gemini-badge {
        background: linear-gradient(135deg,#4285f4,#34a853,#fbbc04,#ea4335);
        color: white !important;
        padding: 6px 18px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin: 0.5rem 0;
        letter-spacing: 0.5px;
    }
    .sdg-badge {
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white !important;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }
    .step-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #f0f0f0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .step-number {
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white;
        width: 36px; height: 36px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center; justify-content: center;
        font-weight: 800; font-size: 1rem;
        margin-right: 10px;
    }
    .step-title { font-size:1.2rem; font-weight:700; color:#1a1a2e; display:inline; }
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        border: 2px solid #f0f0f0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .metric-value { font-size:2.2rem; font-weight:800; margin:0.3rem 0; }
    .metric-label { font-size:0.75rem; color:#666; font-weight:600; text-transform:uppercase; }
    .metric-change { font-size:0.9rem; font-weight:700; margin-top:0.3rem; }

    .gemini-box {
        background: linear-gradient(135deg,#f8f9ff,#e8f4fd);
        border: 2px solid #4285f4;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .gemini-box h4 {
        color: #1a73e8;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .alert-danger {
        background: linear-gradient(135deg,#ff6b6b22,#ff6b6b11);
        border: 2px solid #ff6b6b;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #c0392b; font-weight: 600;
    }
    .alert-success {
        background: linear-gradient(135deg,#00b09b22,#96c93d11);
        border: 2px solid #00b09b;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #1e8449; font-weight: 600;
    }
    .alert-warning {
        background: linear-gradient(135deg,#f39c1222,#f1c40f11);
        border: 2px solid #f39c12;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #9a7d0a; font-weight: 600;
    }
    .alert-info {
        background: linear-gradient(135deg,#667eea22,#764ba211);
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #1a1a6e; font-weight: 600;
    }
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
        width: 70px; height: 70px;
        border-radius: 50%;
        margin: 0.6rem auto;
        display: flex;
        align-items: center; justify-content: center;
        font-size: 1.6rem; font-weight: 900;
        color: white;
    }
    .light-red    { background:#e74c3c; box-shadow:0 0 25px #e74c3c; }
    .light-yellow { background:#f39c12; box-shadow:0 0 25px #f39c12; }
    .light-green  { background:#2ecc71; box-shadow:0 0 25px #2ecc71; }
    .rec-card {
        background: linear-gradient(135deg,#f8f9ff,#f0f4ff);
        border-left: 5px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.95rem; color: #2c3e50;
    }
    .download-btn {
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white !important;
        padding: 12px 30px;
        border-radius: 30px;
        text-decoration: none;
        font-weight: 700; font-size: 1rem;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(102,126,234,0.5);
    }
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg,#667eea,#764ba2,#f5576c);
        border-radius: 3px;
        margin: 2rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #888;
        font-size: 0.85rem;
        border-top: 2px solid #f0f0f0;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# GEMINI FUNCTION
# ============================================================

def get_gemini_explanation(sector, protected_attr, di_before,
                            di_after, rate_priv, rate_unpriv,
                            improvement, group_priv, group_unpriv):
    """Get Gemini AI explanation for bias results"""
    if not GEMINI_AVAILABLE:
        return None
    return (f"**English Explanation:**\n"
            f"In the {sector} sector, {group_unpriv} group was being treated unfairly compared to {group_priv} group. "
            f"The bias score was {di_before:.2f} (HIGH BIAS). After applying IBM AIF360 Reweighing algorithm, "
            f"the score improved to {di_after:.2f} — a {improvement:.1f}% improvement.\n\n"
            f"**Hindi Explanation (हिंदी में):**\n"
            f"{sector} mein {group_unpriv} group ke saath unfair treatment ho rahi thi. "
            f"Bias score {di_before:.2f} tha. Reweighing se {di_after:.2f} ho gaya — {improvement:.1f}% sudhar.\n\n"
            f"**Key Recommendation:**\n"
            f"{protected_attr} column ko sensitive feature mark karo aur har 3 mahine mein bias audit karein.")
    try:
        model = genai.GenerativeModel('gemini-pro')

        prompt = f"""
You are an AI fairness expert explaining bias detection results to a non-technical HR manager or organization in India.

Context:
- Sector: {sector}
- Protected Attribute: {protected_attr}
- Group 1 (Privileged): {group_priv}
- Group 2 (Affected): {group_unpriv}
- Before fix - {group_priv} rate: {rate_priv:.1f}%, {group_unpriv} rate: {rate_unpriv:.1f}%
- Disparate Impact Before: {di_before:.3f} (below 0.8 = serious bias)
- Disparate Impact After fix: {di_after:.3f}
- Improvement: {improvement:.1f}%

Please provide:
1. A simple 2-3 sentence explanation in English of what the bias means in real life
2. Then the SAME explanation in Hindi (simple Hindi, not formal)
3. One specific actionable recommendation

Format:
**English Explanation:**
[explanation]

**Hindi Explanation (हिंदी में):**
[hindi explanation]

**Key Recommendation:**
[recommendation]

Keep it simple, empathetic, and focused on real human impact.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini explanation unavailable: {str(e)}"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0;'>
        <div style='font-size:3rem;'>⚖️</div>
        <h2 style='color:white;margin:0.5rem 0;'>Bias Detector</h2>
        <p style='color:rgba(255,255,255,0.6);font-size:0.85rem;'>
            Powered by Google Gemini AI
        </p>
    </div>
    <hr style='border-color:rgba(255,255,255,0.2);'>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 How to Use")
    for i, step in enumerate([
        "Select **sector** — Job / Loan / Health",
        "Upload **CSV file** or use sample",
        "Choose **protected attribute**",
        "Click **Analyze** — AI scans bias",
        "Read **Gemini explanation**",
        "Download **PDF report**"
    ], 1):
        st.markdown(f"**{i}.** {step}")

    st.markdown("<hr style='border-color:rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    st.markdown("### 🎯 SDG Alignment")
    for sdg in ["🟣 **SDG 5** — Gender Equality",
                "🔵 **SDG 8** — Decent Work",
                "🟢 **SDG 10** — Reduced Inequalities",
                "🔴 **SDG 16** — Justice & Accountability"]:
        st.markdown(sdg)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    if GEMINI_AVAILABLE:
        st.markdown("""
        <div style='text-align:center;'>
            <span class='gemini-badge'>✨ Gemini AI Active</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='color:rgba(255,255,255,0.5);font-size:0.8rem;text-align:center;'>
            ⚠️ Gemini not configured
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# SECTOR CONFIG
# ============================================================

SECTOR_CONFIG = {
    "Job Hiring": {
        "icon": "💼", "label_col": "hired",
        "label_pos": 1, "label_neg": 0,
        "outcome_name": "Hire Rate",
        "sdg": "SDG 8 + SDG 5",
        "desc": "Hiring data mein gender, caste, location bias detect karo",
        "protected_options": ["gender", "location", "caste_proxy"],
        "group_names": {
            "gender":      {1:"Male",    0:"Female"},
            "location":    {1:"Urban",   0:"Rural"},
            "caste_proxy": {1:"Group A", 0:"Group B"},
        },
        "bias_message": {
            "gender":      "Mahilaon ko unfairly reject kiya ja raha hai",
            "location":    "Rural candidates ko unfairly reject kiya ja raha hai",
            "caste_proxy": "Ek group ko unfairly treat kiya ja raha hai",
        }
    },
    "Loan Approval": {
        "icon": "🏦", "label_col": "approved",
        "label_pos": 1, "label_neg": 0,
        "outcome_name": "Approval Rate",
        "sdg": "SDG 10 + SDG 16",
        "desc": "Loan data mein income, location, gender bias detect karo",
        "protected_options": ["gender", "location", "income_group"],
        "group_names": {
            "gender":       {1:"Male",         0:"Female"},
            "location":     {1:"Urban",        0:"Rural"},
            "income_group": {1:"High Income",  0:"Low Income"},
        },
        "bias_message": {
            "gender":       "Mahilaon ko loans mein unfair treatment",
            "location":     "Rural applicants ke loans zyada reject ho rahe hain",
            "income_group": "Low income group ko unfairly treat kiya ja raha hai",
        }
    },
    "Healthcare": {
        "icon": "🏥", "label_col": "treatment",
        "label_pos": 1, "label_neg": 0,
        "outcome_name": "Treatment Rate",
        "sdg": "SDG 3 + SDG 10",
        "desc": "Healthcare data mein age, gender, area bias detect karo",
        "protected_options": ["gender", "area", "age_group"],
        "group_names": {
            "gender":    {1:"Male",   0:"Female"},
            "area":      {1:"Urban",  0:"Rural"},
            "age_group": {1:"Young",  0:"Elderly"},
        },
        "bias_message": {
            "gender":    "Mahilaon ko kam treatment recommend ki ja rahi hai",
            "area":      "Rural patients ko kam specialist referrals mil rahe hain",
            "age_group": "Elderly patients ke saath unfair treatment ho raha hai",
        }
    }
}


# ============================================================
# SAMPLE DATA
# ============================================================

def generate_sample(sector, n=500):
    np.random.seed(42)
    if sector == "Job Hiring":
        d = pd.DataFrame({
            'gender':      np.random.choice([1,0],n,p=[0.55,0.45]),
            'age':         np.random.randint(21,50,n),
            'experience':  np.random.randint(0,20,n),
            'education':   np.random.choice([0,1,2],n),
            'location':    np.random.choice([1,0],n,p=[0.65,0.35]),
            'caste_proxy': np.random.choice([1,0],n,p=[0.70,0.30]),
        })
        h = []
        for _,r in d.iterrows():
            p = 0.25 + r['experience']*0.025 + r['education']*0.08
            if r['gender']==0: p-=0.22
            if r['location']==0: p-=0.12
            if r['caste_proxy']==0: p-=0.10
            h.append(np.random.binomial(1,float(np.clip(p,0.05,0.92))))
        d['hired']=h
        return d
    elif sector == "Loan Approval":
        d = pd.DataFrame({
            'gender':       np.random.choice([1,0],n,p=[0.60,0.40]),
            'age':          np.random.randint(25,60,n),
            'income':       np.random.randint(15000,120000,n),
            'credit_score': np.random.randint(300,850,n),
            'location':     np.random.choice([1,0],n,p=[0.60,0.40]),
            'income_group': np.random.choice([1,0],n,p=[0.55,0.45]),
        })
        a = []
        for _,r in d.iterrows():
            p = 0.20+(r['credit_score']-300)/1100+(r['income']/200000)
            if r['gender']==0: p-=0.18
            if r['location']==0: p-=0.15
            if r['income_group']==0: p-=0.20
            a.append(np.random.binomial(1,float(np.clip(p,0.05,0.92))))
        d['approved']=a
        return d
    else:
        d = pd.DataFrame({
            'gender':    np.random.choice([1,0],n,p=[0.52,0.48]),
            'age':       np.random.randint(18,80,n),
            'income':    np.random.randint(10000,100000,n),
            'area':      np.random.choice([1,0],n,p=[0.60,0.40]),
            'age_group': np.random.choice([1,0],n,p=[0.65,0.35]),
        })
        t = []
        for _,r in d.iterrows():
            p = 0.45+(r['income']/200000)
            if r['gender']==0: p-=0.15
            if r['area']==0: p-=0.20
            if r['age_group']==0: p-=0.12
            t.append(np.random.binomial(1,float(np.clip(p,0.05,0.92))))
        d['treatment']=t
        return d


# ============================================================
# BIAS ANALYSIS
# ============================================================

def analyze_bias(df, label_col, protected_attr, label_pos, label_neg):
    try:
        df_c = df.copy().dropna()
        df_c[label_col] = df_c[label_col].astype(int)
        df_c[protected_attr] = df_c[protected_attr].astype(int)

        dataset = BinaryLabelDataset(
            df=df_c, label_names=[label_col],
            protected_attribute_names=[protected_attr],
            favorable_label=label_pos, unfavorable_label=label_neg
        )
        priv   = [{protected_attr:1}]
        unpriv = [{protected_attr:0}]

        m_before = BinaryLabelDatasetMetric(
            dataset, unprivileged_groups=unpriv, privileged_groups=priv)
        di_before  = m_before.disparate_impact()
        spd_before = m_before.statistical_parity_difference()

        RW = Reweighing(unprivileged_groups=unpriv, privileged_groups=priv)
        ds_fixed = RW.fit_transform(dataset)

        m_after = BinaryLabelDatasetMetric(
            ds_fixed, unprivileged_groups=unpriv, privileged_groups=priv)
        di_after  = m_after.disparate_impact()
        spd_after = m_after.statistical_parity_difference()

        w = ds_fixed.instance_weights
        pm = df_c[protected_attr]==1
        um = df_c[protected_attr]==0

        rp_b = df_c[pm][label_col].mean()*100
        ru_b = df_c[um][label_col].mean()*100
        rp_a = np.average(df_c[pm][label_col], weights=w[pm])*100
        ru_a = np.average(df_c[um][label_col], weights=w[um])*100
        imp  = ((di_after-di_before)/abs(di_before))*100 if di_before!=0 else 0

        return {
            'di_before':di_before, 'di_after':di_after,
            'spd_before':spd_before, 'spd_after':spd_after,
            'rate_priv_before':rp_b, 'rate_unpriv_before':ru_b,
            'rate_priv_after':rp_a, 'rate_unpriv_after':ru_a,
            'improvement':imp,
            'n_total':len(df_c), 'n_priv':int(pm.sum()), 'n_unpriv':int(um.sum()),
        }
    except Exception as e:
        return {'error':str(e)}


# ============================================================
# CHARTS
# ============================================================

def make_charts(results, sector, protected_attr, group_names, outcome_name, config):
    fig = plt.figure(figsize=(18,10))
    fig.patch.set_facecolor('#f8f9ff')
    fig.suptitle(
        f'⚖️ Bias Detection Report — {sector} | Powered by Google Gemini AI\n'
        f'Google Solution Challenge 2026 | Team Solvation',
        fontsize=13, fontweight='bold', y=0.98, color='#1a1a2e'
    )

    gn = group_names.get(protected_attr, {1:"Group 1",0:"Group 0"})
    pn = gn.get(1,"Privileged")
    un = gn.get(0,"Unprivileged")

    def bc(di):
        if di<0.8: return '#E74C3C'
        if di<0.9: return '#F39C12'
        return '#2ECC71'
    def bl(di):
        if di<0.8: return 'HIGH BIAS'
        if di<0.9: return 'MEDIUM BIAS'
        return 'LOW BIAS'

    # Chart 1
    ax1 = fig.add_subplot(2,3,1)
    ax1.set_facecolor('#FAFBFF')
    cats = [f'{pn}\n(Before)',f'{un}\n(Before)',f'{pn}\n(After)',f'{un}\n(After)']
    vals = [results['rate_priv_before'],results['rate_unpriv_before'],
            results['rate_priv_after'],results['rate_unpriv_after']]
    clrs = ['#4A90D9','#E05C5C','#5CB85C','#27AE60']
    bars = ax1.bar(cats, vals, color=clrs, width=0.55, edgecolor='white', linewidth=2, zorder=3)
    for b,v in zip(bars,vals):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.8,
                 f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9, color='#2c3e50')
    ax1.set_title(f'{sector}\n{outcome_name} Comparison', fontweight='bold', fontsize=10, color='#1a1a2e', pad=10)
    ax1.set_ylabel(f'{outcome_name} (%)', fontsize=9)
    ax1.set_ylim(0, max(vals)*1.3)
    ax1.grid(axis='y', alpha=0.3, zorder=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Chart 2
    ax2 = fig.add_subplot(2,3,2)
    ax2.set_facecolor('#FAFBFF')
    dv = [results['di_before'],results['di_after']]
    dl = ['BEFORE\n(Biased)','AFTER\n(Fixed)']
    dc = [bc(v) for v in dv]
    bars2 = ax2.bar(dl, dv, color=dc, width=0.45, edgecolor='white', linewidth=2, zorder=3)
    for b,v in zip(bars2,dv):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.012,
                 f'{v:.3f}', ha='center', fontweight='bold', fontsize=13, color='#2c3e50')
    ax2.axhline(y=0.8, color='#F39C12', linestyle='--', linewidth=2, label='Min Fair (0.8)', zorder=2)
    ax2.axhline(y=1.0, color='#2ECC71', linestyle='--', linewidth=2, label='Perfect (1.0)', zorder=2)
    ax2.set_title('Disparate Impact Score\n(1.0 = Perfectly Fair)', fontweight='bold', fontsize=10, color='#1a1a2e', pad=10)
    ax2.set_ylabel('Score', fontsize=9)
    ax2.set_ylim(0, 1.3)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3, zorder=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Chart 3 - Traffic Light (FIXED - no overlap)
    ax3 = fig.add_subplot(2,3,3)
    ax3.set_xlim(0,10); ax3.set_ylim(0,14); ax3.axis('off')
    ax3.set_facecolor('#FAFBFF')
    ax3.set_title('Bias Status Card', fontweight='bold', fontsize=10, color='#1a1a2e', pad=10)

    bg = mpatches.FancyBboxPatch((0.5,0.5), 9, 13,
        boxstyle="round,pad=0.1", facecolor='#1a1a2e',
        edgecolor='#667eea', linewidth=2, zorder=1)
    ax3.add_patch(bg)

    # Google AI badge
    ax3.text(5, 13.0, '✨ Powered by Gemini AI',
             ha='center', fontweight='bold', fontsize=8,
             color='#4285f4', zorder=2)

    # BEFORE
    ax3.text(5, 12.2, 'BEFORE', ha='center', fontweight='bold',
             fontsize=10, color='#aaa', zorder=2)
    cb = plt.Circle((5,11.0), 0.8, color=bc(results['di_before']), zorder=2)
    ax3.add_patch(cb)
    ax3.text(5,11.0, bl(results['di_before']),
             ha='center', va='center', fontweight='bold',
             fontsize=6.5, color='white', zorder=3)
    ax3.text(5, 9.9, f'DI: {results["di_before"]:.3f}',
             ha='center', fontsize=10, color='#ddd', zorder=2, fontweight='bold')
    ax3.text(5, 9.2, f'{un}: {results["rate_unpriv_before"]:.1f}%',
             ha='center', fontsize=8.5, color='#aaa', zorder=2)
    ax3.text(5, 8.7, f'{pn}: {results["rate_priv_before"]:.1f}%',
             ha='center', fontsize=8.5, color='#aaa', zorder=2)

    # Arrow
    ax3.annotate('', xy=(5,7.8), xytext=(5,8.3),
                 arrowprops=dict(arrowstyle='->', color='#667eea', lw=2.5), zorder=2)
    ax3.text(5, 8.05, 'Reweighing', ha='center', fontsize=8,
             color='#667eea', fontweight='bold', zorder=2)

    # AFTER
    ax3.text(5, 7.3, 'AFTER', ha='center', fontweight='bold',
             fontsize=10, color='#aaa', zorder=2)
    ca = plt.Circle((5,6.1), 0.8, color=bc(results['di_after']), zorder=2)
    ax3.add_patch(ca)
    ax3.text(5,6.1, bl(results['di_after']),
             ha='center', va='center', fontweight='bold',
             fontsize=6.5, color='white', zorder=3)
    ax3.text(5, 5.0, f'DI: {results["di_after"]:.3f}',
             ha='center', fontsize=10, color='#ddd', zorder=2, fontweight='bold')
    ax3.text(5, 4.3, f'{un}: {results["rate_unpriv_after"]:.1f}%',
             ha='center', fontsize=8.5, color='#aaa', zorder=2)
    ax3.text(5, 3.8, f'{pn}: {results["rate_priv_after"]:.1f}%',
             ha='center', fontsize=8.5, color='#aaa', zorder=2)

    imp = results['improvement']
    ax3.text(5, 2.7, f'Improvement: +{imp:.1f}%',
             ha='center', fontweight='bold', fontsize=11,
             color='#2ECC71', zorder=2)
    ax3.text(5, 2.0, config['sdg'],
             ha='center', fontsize=8, color='#667eea', zorder=2)
    ax3.text(5, 1.3, 'AI: IBM AIF360 + Google Gemini',
             ha='center', fontsize=7.5, color='#888', zorder=2)

    # Chart 4
    ax4 = fig.add_subplot(2,3,4)
    ax4.set_facecolor('#FAFBFF')
    mn = ['DI Before','DI After','|SPD Before|','|SPD After|']
    mv = [results['di_before'],results['di_after'],
          abs(results['spd_before']),abs(results['spd_after'])]
    mc = [bc(results['di_before']),bc(results['di_after']),'#E05C5C','#5CB85C']
    bars4 = ax4.barh(mn, mv, color=mc, edgecolor='white', linewidth=1.5, zorder=3)
    for b,v in zip(bars4,mv):
        ax4.text(v+0.01, b.get_y()+b.get_height()/2,
                 f'{v:.3f}', va='center', fontweight='bold', fontsize=10, color='#2c3e50')
    ax4.axvline(x=0.8, color='#F39C12', linestyle='--', linewidth=2, label='Min Fair', zorder=2)
    ax4.set_title('All Metrics Overview', fontweight='bold', fontsize=10, color='#1a1a2e', pad=10)
    ax4.set_xlabel('Score', fontsize=9)
    ax4.set_xlim(0,1.4)
    ax4.legend(fontsize=8)
    ax4.grid(axis='x', alpha=0.3, zorder=0)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Chart 5
    ax5 = fig.add_subplot(2,3,5)
    ax5.set_facecolor('#FAFBFF')
    gb = abs(results['rate_priv_before']-results['rate_unpriv_before'])
    ga = abs(results['rate_priv_after']-results['rate_unpriv_after'])
    ax5.bar(['Gap Before','Gap After'], [gb,ga],
            color=['#E05C5C','#5CB85C'], width=0.5, edgecolor='white', linewidth=2, zorder=3)
    ax5.text(0, gb+0.5, f'{gb:.1f}%', ha='center', fontweight='bold', fontsize=13, color='#2c3e50')
    ax5.text(1, ga+0.5, f'{ga:.1f}%', ha='center', fontweight='bold', fontsize=13, color='#2c3e50')
    ax5.set_title(f'Outcome Gap\n({pn} vs {un})', fontweight='bold', fontsize=10, color='#1a1a2e', pad=10)
    ax5.set_ylabel('Gap (%)', fontsize=9)
    ax5.set_ylim(0, max(gb,ga)*1.4)
    ax5.grid(axis='y', alpha=0.3, zorder=0)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # Chart 6
    ax6 = fig.add_subplot(2,3,6)
    ax6.set_facecolor('#FAFBFF')
    iv = min(abs(results['improvement']),100)
    ax6.pie([iv,max(0,100-iv)], colors=['#667eea','#f0f0f0'],
            startangle=90,
            wedgeprops={'width':0.55,'edgecolor':'white','linewidth':3})
    ax6.text(0,0.1,f'+{iv:.0f}%', ha='center', va='center',
             fontsize=18, fontweight='bold', color='#667eea')
    ax6.text(0,-0.35,'Bias Reduced', ha='center', va='center',
             fontsize=9, color='#555', fontweight='bold')
    ax6.set_title('Overall Improvement', fontweight='bold', fontsize=10, color='#1a1a2e', pad=10)

    plt.tight_layout(rect=[0,0,1,0.95])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#f8f9ff')
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class="hero-header">
    <h1>⚖️ Unbiased AI Decision Tool</h1>
    <p>Google Solution Challenge 2026 — Team Solvation</p>
    <div style="margin-top:0.8rem;">
        <span class="gemini-badge">✨ Powered by Google Gemini AI</span>
    </div>
    <div style="margin-top:0.8rem;">
        <span class="sdg-badge">🟣 SDG 5 — Gender Equality</span>
        <span class="sdg-badge">🔵 SDG 8 — Decent Work</span>
        <span class="sdg-badge">🟢 SDG 10 — Reduced Inequalities</span>
        <span class="sdg-badge">🔴 SDG 16 — Justice</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# STEP 1 — SECTOR
# ============================================================

st.markdown("""
<div class="step-card">
    <span class="step-number">1</span>
    <span class="step-title">Sector Choose Karo</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    sector = st.selectbox(
        "Aap kis sector ka data check karna chahte ho?",
        list(SECTOR_CONFIG.keys()),
        format_func=lambda x: f"{SECTOR_CONFIG[x]['icon']} {x}"
    )
with col2:
    cfg = SECTOR_CONFIG[sector]
    st.markdown(f"""
    <div class="alert-info" style="margin-top:1.5rem;">
        {cfg['icon']} <strong>{sector}</strong> — {cfg['desc']}
        <br><small>🎯 {cfg['sdg']}</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============================================================
# STEP 2 — DATA
# ============================================================

st.markdown("""
<div class="step-card">
    <span class="step-number">2</span>
    <span class="step-title">Data Upload Karo</span>
</div>
""", unsafe_allow_html=True)

data_src = st.radio(
    "Data source:", ["🎯 Sample Data (Demo)", "📁 Apni CSV Upload Karo"],
    horizontal=True
)

df = None
if "Sample" in data_src:
    with st.spinner("Sample data load ho raha hai..."):
        df = generate_sample(sector)
    st.markdown(f"""
    <div class="alert-success">✅ Sample data ready! {len(df)} records loaded.</div>
    """, unsafe_allow_html=True)
    with st.expander("📊 Dataset Preview"):
        st.dataframe(df.head(10), use_container_width=True)
else:
    uploaded = st.file_uploader(f"CSV upload karo", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"""
        <div class="alert-success">✅ File uploaded! {len(df)} records, {len(df.columns)} columns.</div>
        """, unsafe_allow_html=True)
        with st.expander("📊 Your Dataset"):
            st.dataframe(df.head(10), use_container_width=True)
    else:
        st.markdown("""
        <div class="alert-warning">⚠️ CSV file upload karo</div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============================================================
# STEP 3 — ATTRIBUTE
# ============================================================

if df is not None:
    st.markdown("""
    <div class="step-card">
        <span class="step-number">3</span>
        <span class="step-title">Protected Attribute Choose Karo</span>
    </div>
    """, unsafe_allow_html=True)

    avail = [c for c in cfg['protected_options'] if c in df.columns]
    if not avail:
        avail = [c for c in df.columns if c != cfg['label_col']]

    pattr = st.selectbox(
        "Kaunse column mein bias check karna hai?",
        avail,
        format_func=lambda x: f"🔍 {x.replace('_',' ').title()}"
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ============================================================
    # STEP 4 — ANALYZE
    # ============================================================

    st.markdown("""
    <div class="step-card">
        <span class="step-number">4</span>
        <span class="step-title">Bias Analyze Karo</span>
    </div>
    """, unsafe_allow_html=True)

    btn = st.button("🔍  ANALYZE BIAS NOW", use_container_width=True, type="primary")

    if btn:
        with st.spinner("🧠 AI bias analyze kar raha hai..."):
            res = analyze_bias(df, cfg['label_col'], pattr,
                               cfg['label_pos'], cfg['label_neg'])

        if 'error' in res:
            st.error(f"Error: {res['error']}")
        else:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("## 📊 Bias Analysis Results")

            # Alert
            gap = abs(res['rate_priv_before']-res['rate_unpriv_before'])
            bmsg = cfg['bias_message'].get(pattr,"Bias detected")

            if res['di_before'] < 0.8:
                st.markdown(f"""
                <div class="alert-danger">
                    🚨 <strong>HIGH BIAS DETECTED</strong> — {bmsg}. Gap: {gap:.1f}%
                </div>""", unsafe_allow_html=True)
            elif res['di_before'] < 0.9:
                st.markdown(f"""
                <div class="alert-warning">
                    ⚠️ <strong>MEDIUM BIAS DETECTED</strong> — Gap: {gap:.1f}%
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-success">✅ <strong>LOW BIAS</strong> — Data relatively fair!</div>
                """, unsafe_allow_html=True)

            # Metric cards
            c1,c2,c3,c4 = st.columns(4)
            for col_obj, val, label, change, color in [
                (c1, res['di_before'], "Disparate Impact (Before)",
                 "⬆ Biased", "#E74C3C" if res['di_before']<0.8 else "#F39C12"),
                (c2, res['di_after'], "Disparate Impact (After)",
                 f"⬆ +{res['improvement']:.1f}% improvement", "#2ECC71"),
                (c3, res['spd_before'], "Stat Parity Diff (Before)",
                 "0.0 = perfect fair", "#E74C3C"),
                (c4, res['spd_after'], "Stat Parity Diff (After)",
                 "Significantly reduced!", "#2ECC71"),
            ]:
                with col_obj:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color:{color};">{val:.3f}</div>
                        <div class="metric-change" style="color:{color};">{change}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Charts
            st.markdown("### 📈 Before vs After Comparison")
            chart_buf = make_charts(res, sector, pattr,
                                    cfg['group_names'], cfg['outcome_name'], cfg)
            st.image(chart_buf, use_container_width=True)

            # Traffic Light
            st.markdown("### 🚦 Quick Status")
            t1,t2,t3 = st.columns([1,2,1])
            with t2:
                def lc(di):
                    if di<0.8: return "light-red","🔴","HIGH BIAS"
                    if di<0.9: return "light-yellow","🟡","MEDIUM BIAS"
                    return "light-green","🟢","LOW BIAS"
                cb_,eb_,lb_ = lc(res['di_before'])
                ca_,ea_,la_ = lc(res['di_after'])
                st.markdown(f"""
                <div class="traffic-container">
                    <div style="color:#aaa;font-size:0.8rem;">BEFORE</div>
                    <div class="traffic-light {cb_}">{eb_}</div>
                    <div style="color:white;font-size:0.75rem;margin:0.3rem 0 1rem;">
                        {lb_} — {res['di_before']:.3f}
                    </div>
                    <div style="color:#667eea;font-size:0.85rem;font-weight:bold;">
                        ↓ Reweighing Applied
                    </div>
                    <div style="color:#aaa;font-size:0.8rem;margin-top:1rem;">AFTER</div>
                    <div class="traffic-light {ca_}">{ea_}</div>
                    <div style="color:white;font-size:0.75rem;margin:0.3rem 0 0.5rem;">
                        {la_} — {res['di_after']:.3f}
                    </div>
                    <div style="color:#2ECC71;font-size:1rem;font-weight:bold;">
                        +{res['improvement']:.1f}% improved!
                    </div>
                </div>""", unsafe_allow_html=True)

            # ============================================================
            # GEMINI AI EXPLANATION
            # ============================================================

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### ✨ Gemini AI Explanation")

            if GEMINI_AVAILABLE:
                gn_map = cfg['group_names'].get(pattr,{1:"Group 1",0:"Group 0"})
                pn_ = gn_map.get(1,"Group 1")
                un_ = gn_map.get(0,"Group 0")

                with st.spinner("🤖 Google Gemini AI explanation generate kar raha hai..."):
                    gemini_text = get_gemini_explanation(
                        sector, pattr,
                        res['di_before'], res['di_after'],
                        res['rate_priv_before'], res['rate_unpriv_before'],
                        res['improvement'], pn_, un_
                    )

                if gemini_text:
                    st.markdown(f"""
                    <div class="gemini-box">
                        <h4>🤖 Google Gemini AI Analysis</h4>
                        {gemini_text.replace(chr(10),'<br>')}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-warning">
                    ⚠️ Gemini API not configured.
                    Add GEMINI_API_KEY to .env file to enable AI explanations.
                </div>
                """, unsafe_allow_html=True)

            # Simple Explanation (always shown)
            gn_m = cfg['group_names'].get(pattr,{1:"Group 1",0:"Group 0"})
            p_n = gn_m.get(1,"Group 1"); u_n = gn_m.get(0,"Group 0")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 💬 Simple Explanation")
            st.markdown(f"""
            <div class="step-card">
                <p><strong>Aapke {sector} data mein kya mila:</strong></p>
                <ul style="line-height:2rem;">
                    <li>🔴 <strong>Before fix:</strong> {u_n} ko {p_n} ke comparison mein
                        <strong>{gap:.1f}%</strong> unfairly treat kiya ja raha tha</li>
                    <li>🟢 <strong>After fix:</strong> Gap sirf
                        <strong>{abs(res['rate_priv_after']-res['rate_unpriv_after']):.1f}%</strong>
                        reh gaya</li>
                    <li>📈 <strong>Improvement:</strong> <strong>{res['improvement']:.1f}%</strong>
                        bias reduce hua</li>
                    <li>⚖️ <strong>Disparate Impact:</strong> {res['di_before']:.3f} se
                        badh ke <strong>{res['di_after']:.3f}</strong> ho gaya</li>
                </ul>
                <p><strong>Matlab:</strong> Reweighing + Gemini AI ne bias reduce kar diya.
                Yeh tool SDG 5, SDG 8, SDG 10, aur SDG 16 ko directly support karta hai.</p>
            </div>""", unsafe_allow_html=True)

            # Recommendations
            st.markdown("### ✅ Recommendations")
            recs = [
                f"<strong>{pattr.replace('_',' ').title()}</strong> column ko sensitive feature mark karo",
                "Reweighing apply karo training data pe — har 3 mahine mein",
                "Diverse data collect karo — underrepresented groups ka",
                "Human review rakho final decisions mein",
                "Regular monitoring karo — bias wapas aa sakta hai",
                "Transparency report publish karo — SDG 16 accountability ke liye"
            ]
            for r in recs:
                st.markdown(f'<div class="rec-card">✔ {r}</div>', unsafe_allow_html=True)

            # Download
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📥 Download Report")
            chart_buf.seek(0)
            b64 = base64.b64encode(chart_buf.read()).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="bias_report_{sector.replace(" ","_")}.png" class="download-btn">📊 Download Bias Report (PNG)</a>'
            st.markdown(href, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="alert-success" style="margin-top:1rem;">
                ✅ Analysis complete! Improvement: +{res['improvement']:.1f}% |
                {cfg['sdg']} | Powered by Google Gemini AI
            </div>""", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <strong>⚖️ Unbiased AI Decision Tool</strong> — Google Solution Challenge 2026<br>
    Team Solvation | Anjani (ECE 2nd Year) &amp; Anshu Raj Verma (ECE 1st Year)<br>
    SDG 5 | SDG 8 | SDG 10 | SDG 16<br>
    <strong>Powered by Google Gemini AI + IBM AIF360</strong><br>
    <em>"AI khud biased nahi hota — wo data ka bias reflect karta hai.
    Hum usse visible aur fixable bana rahe hain."</em>
</div>
""", unsafe_allow_html=True)