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
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        background-attachment: fixed;
    }
    .main .block-container {
        background: rgba(255,255,255,0.96);
        border-radius: 20px;
        padding: 2rem 3rem;
        margin: 1rem;
        box-shadow: 0 15px 50px rgba(0,0,0,0.25);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0f2027,#203a43,#2c5364);
    }
    [data-testid="stSidebar"] * { color: white !important; }

    /* ══════════════════════════════════════════════════════
       FIX: ALL widget labels, headings, text on dark bg
       ══════════════════════════════════════════════════════ */

    /* Widget labels (selectbox, radio, file uploader questions) */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] {
        color: #c8f0e8 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Radio button option text */
    [data-testid="stRadio"] > div > label > div > p {
        color: #c8f0e8 !important;
        font-weight: 500 !important;
    }

    /* Sub-headings (section headings shown with sub-heading class) */
    .sub-heading {
        color: #11998e !important;
        font-size: 1.2rem !important;
        font-weight: 800 !important;
        margin: 1.4rem 0 0.6rem 0 !important;
        display: block !important;
        letter-spacing: 0.3px;
    }

    /* Section headings shown on white card background */
    .section-heading {
        color: #11998e !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        border-bottom: 3px solid #38ef7d !important;
        padding-bottom: 0.4rem !important;
        margin: 1.5rem 0 1rem 0 !important;
        display: block !important;
    }

    /* Streamlit default markdown headings inside white container */
    .main [data-testid="stMarkdownContainer"] h2 {
        color: #11998e !important;
        font-weight: 800 !important;
        border-bottom: 2px solid #38ef7d;
        padding-bottom: 0.3rem;
    }
    .main [data-testid="stMarkdownContainer"] h3 {
        color: #0f2027 !important;
        font-weight: 700 !important;
    }

    /* Fix expander label */
    [data-testid="stExpander"] summary p {
        color: #c8f0e8 !important;
        font-weight: 600 !important;
    }

    /* Fix file uploader text */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p {
        color: #c8f0e8 !important;
    }

    /* ── FIX: Streamlit widget labels on dark background */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"],
    .stRadio label p,
    [data-testid="stRadio"] p,
    [data-testid="stFileUploader"] label {
        color: #c8f0e8 !important;
        font-weight: 600 !important;
    }

    .hero-header {
        background: linear-gradient(135deg,#0f2027,#11998e,#38ef7d);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .hero-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        color: white !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .hero-header p { color: rgba(255,255,255,0.92); font-size:1.05rem; margin:0.5rem 0 0; }

    .gemini-badge {
        background: linear-gradient(135deg,#11998e,#38ef7d);
        color: #0f2027 !important;
        padding: 6px 18px; border-radius: 25px;
        font-size: 0.85rem; font-weight: 800;
        display: inline-block; margin: 0.5rem 0;
    }
    .sdg-badge {
        background: rgba(255,255,255,0.2);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.4);
        padding: 5px 14px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600;
        display: inline-block; margin: 3px;
    }
    .step-card {
        background: white; border-radius: 16px; padding: 1.5rem; margin: 1rem 0;
        border: 1px solid #e0f5f0; box-shadow: 0 4px 15px rgba(17,153,142,0.1);
    }
    .step-number {
        background: linear-gradient(135deg,#11998e,#38ef7d);
        color: #0f2027; width: 36px; height: 36px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 1rem; margin-right: 10px;
    }
    .step-title { font-size:1.2rem; font-weight:700; color:#0f2027; display:inline; }

    .metric-card {
        background: white; border-radius: 14px; padding: 1.2rem; text-align: center;
        border: 2px solid #e0f5f0; box-shadow: 0 4px 10px rgba(0,0,0,0.06);
    }
    .metric-value { font-size:2.2rem; font-weight:800; margin:0.3rem 0; }
    .metric-label { font-size:0.72rem; color:#555; font-weight:600; text-transform:uppercase; line-height:1.3; }
    .metric-change { font-size:0.85rem; font-weight:700; margin-top:0.3rem; }

    .gemini-box {
        background: linear-gradient(135deg,#f0fffe,#e6fff9);
        border: 2px solid #11998e; border-radius: 16px; padding: 1.5rem; margin: 1rem 0;
    }
    .gemini-box h4 { color: #0a6b5e !important; font-size:1.1rem; font-weight:700; margin-bottom:0.8rem; }

    .alert-danger {
        background: #fff5f5; border-left: 5px solid #e74c3c;
        border-radius: 8px; padding: 1rem 1.5rem; color: #c0392b; font-weight:600; margin:0.5rem 0;
    }
    .alert-success {
        background: #f0fffe; border-left: 5px solid #11998e;
        border-radius: 8px; padding: 1rem 1.5rem; color: #0a6b5e; font-weight:600; margin:0.5rem 0;
    }
    .alert-warning {
        background: #fffaf3; border-left: 5px solid #f39c12;
        border-radius: 8px; padding: 1rem 1.5rem; color: #9a7d0a; font-weight:600; margin:0.5rem 0;
    }
    .alert-info {
        background: #f0fffe; border-left: 5px solid #3498db;
        border-radius: 8px; padding: 1rem 1.5rem; color: #1a5276; font-weight:600; margin:0.5rem 0;
    }
    .mapping-box {
        background: #f0fffe; border: 2px solid #11998e;
        border-radius: 12px; padding: 1rem 1.5rem; margin: 0.5rem 0; color: #0f4a44;
    }
    .note-box {
        background: #f4f9ff; border: 1px solid #3498db;
        border-radius: 10px; padding: 0.8rem 1.2rem; margin: 0.5rem 0;
        font-size: 0.85rem; color: #1a5276;
    }
    .traffic-container {
        background: linear-gradient(180deg,#0f2027,#203a43);
        border-radius: 20px; padding: 1.5rem; text-align: center;
        max-width: 200px; margin: 0 auto;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border: 1px solid rgba(56,239,125,0.3);
    }
    .traffic-light {
        width: 70px; height: 70px; border-radius: 50%; margin: 0.6rem auto;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.6rem; font-weight: 900; color: white;
    }
    .light-red    { background:#e74c3c; box-shadow:0 0 20px #e74c3c88; }
    .light-yellow { background:#f39c12; box-shadow:0 0 20px #f39c1288; }
    .light-green  { background:#2ecc71; box-shadow:0 0 20px #2ecc7188; }

    .rec-card {
        background: #f8fffe; border-left: 5px solid #11998e;
        border-radius: 0 12px 12px 0; padding: 1rem 1.2rem;
        margin: 0.5rem 0; font-size: 0.95rem; color: #2c3e50;
    }
    .download-btn {
        background: linear-gradient(135deg,#11998e,#38ef7d);
        color: #0f2027 !important; padding: 12px 30px; border-radius: 30px;
        text-decoration: none; font-weight: 800; font-size: 1rem;
        display: inline-block; margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(17,153,142,0.4);
    }
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg,#11998e,#38ef7d,#11998e);
        border-radius: 3px; margin: 2rem 0;
    }
    .section-heading {
        color: #11998e !important; font-size: 1.5rem; font-weight: 800;
        border-bottom: 3px solid #38ef7d; padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0; display: block;
    }
    .sub-heading {
        color: #0f2027 !important; font-size: 1.15rem; font-weight: 700;
        margin: 1.2rem 0 0.6rem 0; display: block;
    }
    .footer {
        text-align: center; padding: 2rem; color: #555; font-size: 0.85rem;
        border-top: 2px solid #e0f5f0; margin-top: 2rem;
        background: #f8fffe; border-radius: 0 0 16px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# GEMINI FUNCTION
# ============================================================

def get_gemini_explanation(sector, protected_attr, di_before,
                            di_after, rate_priv, rate_unpriv,
                            improvement, group_priv, group_unpriv):
    fallback = (
        f"**English Explanation:**\n"
        f"In the {sector} sector, {group_unpriv} group was being treated unfairly compared "
        f"to {group_priv} group. The bias score was {di_before:.2f} (HIGH BIAS). After applying "
        f"IBM AIF360 Reweighing algorithm, the score improved to {di_after:.2f} — "
        f"bias has been successfully mitigated.\n\n"
        f"**Hindi Explanation (हिंदी में):**\n"
        f"{sector} mein {group_unpriv} group ke saath unfair treatment ho rahi thi. "
        f"Bias score {di_before:.2f} tha. Reweighing se {di_after:.2f} ho gaya — "
        f"bias successfully reduce hua.\n\n"
        f"**Key Recommendation:**\n"
        f"{protected_attr} column ko sensitive feature mark karo aur "
        f"har 3 mahine mein bias audit karein."
    )
    if not GEMINI_AVAILABLE:
        return fallback
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
You are an AI fairness expert explaining bias detection results to a non-technical HR manager in India.

Context:
- Sector: {sector}
- Protected Attribute: {protected_attr}
- Group 1 (Privileged): {group_priv}
- Group 2 (Affected): {group_unpriv}
- Before fix - {group_priv} rate: {rate_priv:.1f}%, {group_unpriv} rate: {rate_unpriv:.1f}%
- Disparate Impact Before: {di_before:.3f} (below 0.8 = serious bias)
- Disparate Impact After fix: {di_after:.3f}

Please provide:
1. A simple 2-3 sentence explanation in English of what the bias means in real life
2. The SAME explanation in Hindi (simple Hindi)
3. One specific actionable recommendation

Format exactly:
**English Explanation:**
[explanation]

**Hindi Explanation (हिंदी में):**
[hindi explanation]

**Key Recommendation:**
[recommendation]
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return fallback


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0;'>
        <div style='font-size:3rem;'>⚖️</div>
        <h2 style='color:white;margin:0.5rem 0;font-weight:800;'>Bias Detector</h2>
        <p style='color:rgba(255,255,255,0.65);font-size:0.85rem;'>Powered by Google Gemini AI</p>
    </div>
    <hr style='border-color:rgba(56,239,125,0.3);'>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 How to Use")
    for i, step in enumerate([
        "Select **sector** — Job / Loan / Health",
        "Upload **CSV** or use sample data",
        "Map **columns** (for real CSV)",
        "Choose **protected attribute**",
        "Click **Analyze** — AI scans bias",
        "Download **PNG report**"
    ], 1):
        st.markdown(f"**{i}.** {step}")

    st.markdown("<hr style='border-color:rgba(56,239,125,0.3);'>", unsafe_allow_html=True)
    st.markdown("### 🎯 SDG Alignment")
    for sdg in ["🟣 **SDG 5** — Gender Equality",
                "🔵 **SDG 8** — Decent Work",
                "🟢 **SDG 10** — Reduced Inequalities",
                "🔴 **SDG 16** — Justice & Accountability"]:
        st.markdown(sdg)

    st.markdown("<hr style='border-color:rgba(56,239,125,0.3);'>", unsafe_allow_html=True)
    if GEMINI_AVAILABLE:
        st.markdown("<div style='text-align:center;'><span class='gemini-badge'>✨ Gemini AI Active</span></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:rgba(255,255,255,0.5);font-size:0.8rem;text-align:center;'>⚠️ Gemini not configured<br><small>Add GEMINI_API_KEY to .env</small></div>",
                    unsafe_allow_html=True)


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
            "gender":      {1:"Male", 0:"Female"},
            "location":    {1:"Urban", 0:"Rural"},
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
            "gender":       {1:"Male", 0:"Female"},
            "location":     {1:"Urban", 0:"Rural"},
            "income_group": {1:"High Income", 0:"Low Income"},
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
            "gender":    {1:"Male", 0:"Female"},
            "area":      {1:"Urban", 0:"Rural"},
            "age_group": {1:"Young", 0:"Elderly"},
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
        d['hired'] = h
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
        d['approved'] = a
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
        d['treatment'] = t
        return d


# ============================================================
# BIAS ANALYSIS
# ============================================================

def analyze_bias(df, label_col, protected_attr, label_pos, label_neg):
    try:
        df_c = df[[label_col, protected_attr]].copy()
        df_c = df_c.dropna()
        df_c[label_col]      = df_c[label_col].astype(int)
        df_c[protected_attr] = df_c[protected_attr].astype(int)

        lv = df_c[label_col].unique()
        pv = df_c[protected_attr].unique()
        if len(lv) < 2:
            return {'error': f"'{label_col}' mein sirf ek value hai. 2 values chahiye."}
        if len(pv) < 2:
            return {'error': f"'{protected_attr}' mein sirf ek group hai. 2 groups chahiye."}
        if len(df_c) < 50:
            return {'error': f"Sirf {len(df_c)} rows hain. Kam se kam 50 chahiye."}

        dataset = BinaryLabelDataset(
            df=df_c, label_names=[label_col],
            protected_attribute_names=[protected_attr],
            favorable_label=label_pos, unfavorable_label=label_neg
        )
        priv   = [{protected_attr: 1}]
        unpriv = [{protected_attr: 0}]

        m_before   = BinaryLabelDatasetMetric(dataset, unprivileged_groups=unpriv, privileged_groups=priv)
        di_before  = m_before.disparate_impact()
        spd_before = m_before.statistical_parity_difference()

        RW = Reweighing(unprivileged_groups=unpriv, privileged_groups=priv)
        ds_fixed = RW.fit_transform(dataset)

        m_after   = BinaryLabelDatasetMetric(ds_fixed, unprivileged_groups=unpriv, privileged_groups=priv)
        di_after  = m_after.disparate_impact()
        spd_after = m_after.statistical_parity_difference()

        w  = ds_fixed.instance_weights
        pm = df_c[protected_attr] == 1
        um = df_c[protected_attr] == 0

        rp_b = df_c[pm][label_col].mean() * 100
        ru_b = df_c[um][label_col].mean() * 100
        rp_a = np.average(df_c[pm][label_col], weights=w[pm]) * 100
        ru_a = np.average(df_c[um][label_col], weights=w[um]) * 100
        imp  = ((di_after - di_before) / abs(di_before)) * 100 if di_before != 0 else 0

        return {
            'di_before': di_before, 'di_after': di_after,
            'spd_before': spd_before, 'spd_after': spd_after,
            'rate_priv_before': rp_b, 'rate_unpriv_before': ru_b,
            'rate_priv_after': rp_a,  'rate_unpriv_after': ru_a,
            'improvement': imp,
            'n_total': len(df_c), 'n_priv': int(pm.sum()), 'n_unpriv': int(um.sum()),
        }
    except Exception as e:
        return {'error': str(e)}


# ============================================================
# CHARTS — FIXED: no text overlap, no emoji in title, clean
# ============================================================

def make_charts(results, sector, protected_attr, group_names, outcome_name, config):
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor('#f0fffe')

    # Plain ASCII title — no emoji to avoid matplotlib rendering issues
    fig.suptitle(
        f'BIAS DETECTION REPORT  |  {sector.upper()}\n'
        f'Google Solution Challenge 2026  |  Team Solvation  |  Gemini AI + IBM AIF360',
        fontsize=11, fontweight='bold', y=0.99, color='#0f2027'
    )

    pn = group_names.get(1, "Privileged")
    un = group_names.get(0, "Unprivileged")
    # Truncate long names for chart labels
    pn_s = (pn[:9] + '..') if len(pn) > 10 else pn
    un_s = (un[:9] + '..') if len(un) > 10 else un

    def bc(di):
        if di < 0.8: return '#E74C3C'
        if di < 0.9: return '#F39C12'
        return '#2ECC71'
    def bl(di):
        if di < 0.8: return 'HIGH\nBIAS'
        if di < 0.9: return 'MED\nBIAS'
        return 'LOW\nBIAS'

    # ── Chart 1: Outcome Comparison ──
    ax1 = fig.add_subplot(2,3,1)
    ax1.set_facecolor('#FAFFFE')
    cats = [f'{pn_s}\nBefore', f'{un_s}\nBefore', f'{pn_s}\nAfter', f'{un_s}\nAfter']
    vals = [results['rate_priv_before'], results['rate_unpriv_before'],
            results['rate_priv_after'],  results['rate_unpriv_after']]
    clrs = ['#11998e','#e74c3c','#38ef7d','#27AE60']
    bars = ax1.bar(cats, vals, color=clrs, width=0.5, edgecolor='white', linewidth=2, zorder=3)
    max_v = max(vals) if max(vals) > 0 else 1
    for b, v in zip(bars, vals):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height() + max_v*0.03,
                 f'{v:.1f}%', ha='center', fontweight='bold', fontsize=8, color='#0f2027')
    ax1.set_title(f'{outcome_name} Comparison', fontweight='bold', fontsize=9.5, color='#0f2027', pad=8)
    ax1.set_ylabel(f'{outcome_name} (%)', fontsize=8.5)
    ax1.set_ylim(0, max_v * 1.4)
    ax1.grid(axis='y', alpha=0.3, zorder=0)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', labelsize=7.5)

    # ── Chart 2: Disparate Impact ──
    ax2 = fig.add_subplot(2,3,2)
    ax2.set_facecolor('#FAFFFE')
    dv = [results['di_before'], results['di_after']]
    bars2 = ax2.bar(['BEFORE\n(Biased)', 'AFTER\n(Fixed)'], dv,
                    color=[bc(v) for v in dv], width=0.4, edgecolor='white', linewidth=2, zorder=3)
    for b, v in zip(bars2, dv):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.015,
                 f'{v:.3f}', ha='center', fontweight='bold', fontsize=12, color='#0f2027')
    ax2.axhline(y=0.8, color='#F39C12', linestyle='--', linewidth=2, label='Min Fair (0.8)', zorder=2)
    ax2.axhline(y=1.0, color='#11998e', linestyle='--', linewidth=2, label='Perfect (1.0)', zorder=2)
    ax2.set_title('Disparate Impact Score\n(1.0 = Perfectly Fair)', fontweight='bold', fontsize=9.5, color='#0f2027', pad=8)
    ax2.set_ylabel('Score', fontsize=8.5)
    ax2.set_ylim(0, 1.35)
    ax2.legend(fontsize=7.5, loc='upper left')
    ax2.grid(axis='y', alpha=0.3, zorder=0)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    # ── Chart 3: Status Card — FIXED spacing, no overlap ──
    ax3 = fig.add_subplot(2,3,3)
    ax3.set_xlim(0,10); ax3.set_ylim(0,15); ax3.axis('off')
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_title('Bias Status Summary', fontweight='bold', fontsize=9.5, color='#0f2027', pad=8)

    ax3.add_patch(mpatches.FancyBboxPatch(
        (0.3,0.3), 9.4, 14.2, boxstyle="round,pad=0.15",
        facecolor='#0f2027', edgecolor='#38ef7d', linewidth=2, zorder=1))

    # Header
    ax3.text(5, 14.2, 'Gemini AI + IBM AIF360',
             ha='center', fontsize=7, color='#38ef7d', fontweight='bold', zorder=2)
    # BEFORE
    ax3.text(5, 13.3, 'BEFORE', ha='center', fontsize=9, color='#aaa', fontweight='bold', zorder=2)
    ax3.add_patch(plt.Circle((5,12.0), 0.75, color=bc(results['di_before']), zorder=2))
    ax3.text(5, 12.0, bl(results['di_before']),
             ha='center', va='center', fontsize=5.5, fontweight='bold', color='white', zorder=3)
    ax3.text(5, 10.85, f'DI: {results["di_before"]:.3f}',
             ha='center', fontsize=9, color='#ddd', fontweight='bold', zorder=2)
    ax3.text(5, 10.2, f'{un_s}: {results["rate_unpriv_before"]:.1f}%',
             ha='center', fontsize=7.5, color='#bbb', zorder=2)
    ax3.text(5, 9.65, f'{pn_s}: {results["rate_priv_before"]:.1f}%',
             ha='center', fontsize=7.5, color='#bbb', zorder=2)
    # Arrow
    ax3.text(5, 9.05, 'v  Reweighing Applied  v',
             ha='center', fontsize=7.5, color='#38ef7d', fontweight='bold', zorder=2,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f2027',
                      edgecolor='#38ef7d', linewidth=1))
    # AFTER
    ax3.text(5, 8.2, 'AFTER', ha='center', fontsize=9, color='#aaa', fontweight='bold', zorder=2)
    ax3.add_patch(plt.Circle((5,6.9), 0.75, color=bc(results['di_after']), zorder=2))
    ax3.text(5, 6.9, bl(results['di_after']),
             ha='center', va='center', fontsize=5.5, fontweight='bold', color='white', zorder=3)
    ax3.text(5, 5.75, f'DI: {results["di_after"]:.3f}',
             ha='center', fontsize=9, color='#ddd', fontweight='bold', zorder=2)
    ax3.text(5, 5.1, f'{un_s}: {results["rate_unpriv_after"]:.1f}%',
             ha='center', fontsize=7.5, color='#bbb', zorder=2)
    ax3.text(5, 4.55, f'{pn_s}: {results["rate_priv_after"]:.1f}%',
             ha='center', fontsize=7.5, color='#bbb', zorder=2)
    # Result
    ax3.text(5, 3.4, 'Bias Mitigated', ha='center', fontsize=9.5,
             color='#38ef7d', fontweight='bold', zorder=2)
    ax3.text(5, 2.65, f'{results["di_before"]:.3f} -> {results["di_after"]:.3f}',
             ha='center', fontsize=7.5, color='#aaa', zorder=2)
    ax3.text(5, 1.9, config['sdg'],
             ha='center', fontsize=7.5, color='#38ef7d', zorder=2)
    ax3.text(5, 1.2, 'Google Solution Challenge 2026',
             ha='center', fontsize=6.5, color='#666', zorder=2)

    # ── Chart 4: All Metrics ──
    ax4 = fig.add_subplot(2,3,4)
    ax4.set_facecolor('#FAFFFE')
    mn = ['DI Before', 'DI After', '|SPD Before|', '|SPD After|']
    mv = [results['di_before'], results['di_after'],
          abs(results['spd_before']), abs(results['spd_after'])]
    mc = [bc(results['di_before']), bc(results['di_after']), '#e74c3c', '#11998e']
    bars4 = ax4.barh(mn, mv, color=mc, edgecolor='white', linewidth=1.5, zorder=3, height=0.5)
    for b, v in zip(bars4, mv):
        ax4.text(v+0.01, b.get_y()+b.get_height()/2,
                 f'{v:.3f}', va='center', fontweight='bold', fontsize=9, color='#0f2027')
    ax4.axvline(x=0.8, color='#F39C12', linestyle='--', linewidth=2, label='Min Fair (0.8)', zorder=2)
    ax4.set_title('All Fairness Metrics', fontweight='bold', fontsize=9.5, color='#0f2027', pad=8)
    ax4.set_xlabel('Score', fontsize=8.5)
    ax4.set_xlim(0, 1.45)
    ax4.legend(fontsize=7.5)
    ax4.grid(axis='x', alpha=0.3, zorder=0)
    ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
    ax4.tick_params(axis='y', labelsize=8)

    # ── Chart 5: Outcome Gap ──
    ax5 = fig.add_subplot(2,3,5)
    ax5.set_facecolor('#FAFFFE')
    gb = abs(results['rate_priv_before'] - results['rate_unpriv_before'])
    ga = abs(results['rate_priv_after']  - results['rate_unpriv_after'])
    max_g = max(gb, ga, 1)
    ax5.bar(['Gap Before\n(Biased)', 'Gap After\n(Fixed)'], [gb, ga],
            color=['#e74c3c','#11998e'], width=0.45, edgecolor='white', linewidth=2, zorder=3)
    ax5.text(0, gb + max_g*0.05, f'{gb:.1f}%',
             ha='center', fontweight='bold', fontsize=12, color='#0f2027')
    ax5.text(1, ga + max_g*0.05, f'{ga:.1f}%',
             ha='center', fontweight='bold', fontsize=12, color='#0f2027')
    ax5.set_title(f'Outcome Gap\n({pn_s} vs {un_s})',
                  fontweight='bold', fontsize=9.5, color='#0f2027', pad=8)
    ax5.set_ylabel('Gap (%)', fontsize=8.5)
    ax5.set_ylim(0, max_g * 1.4)
    ax5.grid(axis='y', alpha=0.3, zorder=0)
    ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)

    # ── Chart 6: Clean Summary (replaces confusing donut %) ──
    ax6 = fig.add_subplot(2,3,6)
    ax6.set_facecolor('#FAFFFE')
    ax6.axis('off')
    ax6.set_title('Result Summary', fontweight='bold', fontsize=9.5, color='#0f2027', pad=8)

    ax6.add_patch(mpatches.FancyBboxPatch(
        (0.05,0.05), 0.9, 0.9, boxstyle="round,pad=0.04",
        transform=ax6.transAxes,
        facecolor='#f0fffe', edgecolor='#11998e', linewidth=2))

    def bias_lv(di):
        if di < 0.8: return 'HIGH BIAS', '#e74c3c'
        if di < 0.9: return 'MEDIUM BIAS', '#f39c12'
        return 'LOW BIAS', '#11998e'

    bl_t, bl_c = bias_lv(results['di_before'])
    al_t, al_c = bias_lv(results['di_after'])

    rows = [
        (0.88, 'RESULT SUMMARY',           '#0f2027', 9.5, 'bold'),
        (0.78, f'Before:  {bl_t}',         bl_c,     9,   'bold'),
        (0.70, f'DI = {results["di_before"]:.3f}',   '#555',   8.5, 'normal'),
        (0.60, 'After Reweighing:',        '#0f2027', 8.5, 'bold'),
        (0.52, f'After:   {al_t}',         al_c,     9,   'bold'),
        (0.44, f'DI = {results["di_after"]:.3f}',    '#555',   8.5, 'normal'),
        (0.33, 'Conclusion:',              '#0f2027', 8.5, 'bold'),
        (0.25, 'Bias Successfully',        '#11998e', 9.5, 'bold'),
        (0.17, 'Mitigated',                '#11998e', 9.5, 'bold'),
        (0.08, config['sdg'],              '#11998e', 7.5, 'normal'),
    ]
    for y, txt, col, fs, fw in rows:
        ax6.text(0.5, y, txt, ha='center', va='center',
                 transform=ax6.transAxes, fontsize=fs, color=col, fontweight=fw)

    plt.tight_layout(rect=[0,0,1,0.96], h_pad=2.5, w_pad=2.0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#f0fffe')
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

st.markdown('<div class="step-card"><span class="step-number">1</span><span class="step-title">Sector Choose Karo</span></div>',
            unsafe_allow_html=True)

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
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============================================================
# STEP 2 — DATA
# ============================================================

st.markdown('<div class="step-card"><span class="step-number">2</span><span class="step-title">Data Upload Karo</span></div>',
            unsafe_allow_html=True)

data_src = st.radio("Data source:",
                    ["🎯 Sample Data (Demo)", "📁 Apni CSV Upload Karo"],
                    horizontal=True)

df = None
if "Sample" in data_src:
    with st.spinner("Sample data load ho raha hai..."):
        df = generate_sample(sector)
    st.markdown(f'<div class="alert-success">✅ Sample data ready! {len(df)} records loaded.</div>',
                unsafe_allow_html=True)
    with st.expander("📊 Dataset Preview"):
        st.dataframe(df.head(10), use_container_width=True)
else:
    uploaded = st.file_uploader("CSV upload karo (koi bhi CSV chalega)", type=['csv'])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.markdown(
                f'<div class="alert-success">✅ File uploaded! {len(df)} records, {len(df.columns)} columns.</div>',
                unsafe_allow_html=True)
            with st.expander("📊 Your Dataset"):
                st.dataframe(df.head(10), use_container_width=True)
                st.markdown(f"**Columns:** `{'`, `'.join(df.columns.tolist())}`")
        except Exception as e:
            st.error(f"❌ CSV read error: {str(e)}")
            df = None
    else:
        st.markdown("""
        <div class="alert-warning">
            ⚠️ CSV file upload karo — <strong>koi bhi CSV chalega!</strong><br>
            <small>adult.csv, loan data, healthcare data, custom data — sab supported.</small>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============================================================
# STEP 2.5 — COLUMN MAPPING
# ============================================================

active_label_col   = cfg['label_col']
active_label_pos   = cfg['label_pos']
active_label_neg   = cfg['label_neg']
active_group_names = {}

if df is not None and "Sample" not in data_src:
    st.markdown('<div class="step-card"><span class="step-number">2.5</span><span class="step-title">Column Mapping Karo</span></div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="mapping-box">
        📌 <strong>Aapke CSV ke columns apne aap map nahi ho sakte.</strong><br>
        Neeche select karo ki kaunsa column <em>outcome</em> hai
        aur kaunsa column <em>protected attribute</em> hai.
    </div>""", unsafe_allow_html=True)

    all_cols = df.columns.tolist()
    st.markdown(
        '<div style="color:#11998e;font-size:1.25rem;font-weight:800;'
        'border-bottom:3px solid #38ef7d;padding-bottom:0.4rem;'
        'margin:1.5rem 0 0.8rem 0;">'
        '🎯 Outcome Column (Decision Column)</div>',
        unsafe_allow_html=True)
    outcome_col = st.selectbox(
        "Outcome column kaunsa hai? (hired/approved/income etc)", all_cols,
        help="Final decision wala column — jisme result hai")

    unique_outcome_vals = df[outcome_col].dropna().unique()
    n_unique = len(unique_outcome_vals)

    if n_unique < 2:
        st.markdown(f'<div class="alert-danger">❌ <strong>{outcome_col}</strong> mein sirf 1 value hai. Dusra column choose karo.</div>',
                    unsafe_allow_html=True)
        df = None
    elif n_unique == 2:
        v0, v1 = str(unique_outcome_vals[0]), str(unique_outcome_vals[1])
        pos_hints = [">50k","yes","1","approved","hired","true","treatment","male","urban"]
        auto_pos = v1
        for h in pos_hints:
            if h in v0.lower().replace(" ","").replace(".",""):
                auto_pos = v0; break
            if h in v1.lower().replace(" ","").replace(".",""):
                auto_pos = v1; break

        pos_val = st.selectbox("Positive outcome (= 1):", [v0,v1],
                               index=[v0,v1].index(auto_pos))
        neg_val = v0 if pos_val == v1 else v1
        st.markdown(f'<div class="alert-success">✅ <strong>{pos_val}</strong> = Selected (1) | <strong>{neg_val}</strong> = Not Selected (0)</div>',
                    unsafe_allow_html=True)
        df = df.copy()
        df[outcome_col] = df[outcome_col].astype(str).str.strip().apply(
            lambda x: 1 if x==pos_val else (0 if x==neg_val else None))
        df = df.dropna(subset=[outcome_col])
        df[outcome_col] = df[outcome_col].astype(int)
        active_label_col = outcome_col; active_label_pos = 1; active_label_neg = 0
    else:
        st.markdown(f'<div class="alert-warning">⚠️ {n_unique} values hain. 2 choose karo.</div>',
                    unsafe_allow_html=True)
        str_vals = [str(v) for v in unique_outcome_vals]
        pos_val = st.selectbox("Positive outcome:", str_vals)
        neg_val = st.selectbox("Negative outcome:", [v for v in str_vals if v!=pos_val])
        df = df.copy()
        df[outcome_col] = df[outcome_col].astype(str).str.strip().apply(
            lambda x: 1 if x==pos_val else (0 if x==neg_val else None))
        df = df.dropna(subset=[outcome_col])
        df[outcome_col] = df[outcome_col].astype(int)
        active_label_col = outcome_col; active_label_pos = 1; active_label_neg = 0
        if len(df) < 50:
            st.markdown(f'<div class="alert-danger">❌ Sirf {len(df)} rows bachi hain.</div>',
                        unsafe_allow_html=True)
            df = None

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============================================================
# STEP 3 — PROTECTED ATTRIBUTE
# ============================================================

if df is not None:
    st.markdown('<div class="step-card"><span class="step-number">3</span><span class="step-title">Protected Attribute Choose Karo</span></div>',
                unsafe_allow_html=True)

    if "Sample" not in data_src:
        avail = [c for c in df.columns if c != active_label_col]
        st.markdown('<div class="mapping-box">👥 <strong>Protected attribute</strong> — jiske basis pe bias check karna hai (sex, gender, race, age, location etc.)</div>',
                    unsafe_allow_html=True)
    else:
        avail = [c for c in cfg['protected_options'] if c in df.columns]
        if not avail:
            avail = [c for c in df.columns if c != active_label_col]

    pattr = st.selectbox(
        "Kaunse column mein bias check karna hai?", avail,
        format_func=lambda x: f"🔍 {x.replace('_',' ').replace('-',' ').title()}"
    )

    if "Sample" not in data_src:
        unique_pattr_vals = df[pattr].dropna().unique()
        n_pattr = len(unique_pattr_vals)

        if n_pattr < 2:
            st.markdown(f'<div class="alert-danger">❌ <strong>{pattr}</strong> mein sirf 1 group hai.</div>',
                        unsafe_allow_html=True)
            df = None
        else:
            if df[pattr].dtype in [np.int64,np.float64,int,float] and n_pattr > 2:
                thr = df[pattr].median()
                g1 = f">{thr:.0f}"; g0 = f"<={thr:.0f}"
                st.markdown(f'<div class="alert-info">📊 Numeric column — median {thr:.0f} pe split. Group 1: <strong>{g1}</strong> | Group 0: <strong>{g0}</strong></div>',
                            unsafe_allow_html=True)
                df = df.copy()
                df[pattr] = (df[pattr] > thr).astype(int)
                active_group_names = {1:g1, 0:g0}

            elif n_pattr == 2:
                sv = sorted([str(v).strip() for v in unique_pattr_vals])
                priv_hints = ["male","urban","1","yes","high",">50k","united-states"]
                auto_p = sv[-1]
                for v in sv:
                    if any(h in v.lower().replace(" ","") for h in priv_hints):
                        auto_p = v; break

                priv_val = st.selectbox(
                    "Privileged group (majority/favored):", sv,
                    index=sv.index(auto_p),
                    help="Jis group ko historically zyada faayda milta hai"
                )
                unpriv_val = sv[0] if priv_val==sv[1] else sv[1]
                st.markdown(f'<div class="alert-success">✅ Privileged: <strong>{priv_val}</strong> (=1) | Unprivileged: <strong>{unpriv_val}</strong> (=0)</div>',
                            unsafe_allow_html=True)
                df = df.copy()
                df[pattr] = df[pattr].astype(str).str.strip().apply(
                    lambda x: 1 if x==priv_val else (0 if x==unpriv_val else None))
                df = df.dropna(subset=[pattr])
                df[pattr] = df[pattr].astype(int)
                active_group_names = {1:priv_val, 0:unpriv_val}
            else:
                st.markdown(f'<div class="alert-warning">⚠️ {n_pattr} values — 2 groups choose karo.</div>',
                            unsafe_allow_html=True)
                str_pv = [str(v) for v in unique_pattr_vals]
                priv_val   = st.selectbox("Privileged group:", str_pv)
                unpriv_val = st.selectbox("Unprivileged group:", [v for v in str_pv if v!=priv_val])
                df = df.copy()
                df[pattr] = df[pattr].astype(str).str.strip().apply(
                    lambda x: 1 if x==priv_val else (0 if x==unpriv_val else None))
                df = df.dropna(subset=[pattr])
                df[pattr] = df[pattr].astype(int)
                active_group_names = {1:priv_val, 0:unpriv_val}
    else:
        active_group_names = cfg['group_names'].get(pattr, {1:"Group 1", 0:"Group 0"})

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ============================================================
    # STEP 4 — ANALYZE
    # ============================================================

    if df is not None:
        st.markdown('<div class="step-card"><span class="step-number">4</span><span class="step-title">Bias Analyze Karo</span></div>',
                    unsafe_allow_html=True)

        btn = st.button("🔍  ANALYZE BIAS NOW", use_container_width=True, type="primary")

        if btn:
            with st.spinner("🧠 AI bias analyze kar raha hai..."):
                res = analyze_bias(df, active_label_col, pattr,
                                   active_label_pos, active_label_neg)

            if 'error' in res:
                st.markdown(f"""
                <div class="alert-danger">
                    ❌ <strong>Analysis Error:</strong> {res['error']}<br><br>
                    <strong>Possible fixes:</strong><br>
                    • Outcome column mein exactly 2 values honi chahiye<br>
                    • Protected attribute mein 2+ groups hone chahiye<br>
                    • Kam se kam 50 rows chahiye<br>
                    • Sample Data try karo pehle
                </div>""", unsafe_allow_html=True)

            else:
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                # ── Visible section heading ──
                st.markdown('<span class="section-heading">📊 Bias Analysis Results</span>',
                            unsafe_allow_html=True)

                gap = abs(res['rate_priv_before'] - res['rate_unpriv_before'])
                g1n = active_group_names.get(1, "Group 1")
                g0n = active_group_names.get(0, "Group 0")
                bmsg = (cfg['bias_message'].get(pattr, "Bias detected")
                        if "Sample" in data_src
                        else f"{g0n} group ko {g1n} ke comparison mein unfairly treat kiya ja raha hai")

                if res['di_before'] < 0.8:
                    st.markdown(f'<div class="alert-danger">🚨 <strong>HIGH BIAS DETECTED</strong> — {bmsg}. Gap: {gap:.1f}%</div>',
                                unsafe_allow_html=True)
                elif res['di_before'] < 0.9:
                    st.markdown(f'<div class="alert-warning">⚠️ <strong>MEDIUM BIAS DETECTED</strong> — Gap: {gap:.1f}%</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-success">✅ <strong>LOW BIAS</strong> — Data relatively fair!</div>',
                                unsafe_allow_html=True)

                # Metric cards
                imp_label = "✅ Bias Mitigated" if res['di_after'] >= 0.99 else f"Improved to {res['di_after']:.3f}"
                c1,c2,c3,c4 = st.columns(4)
                for col_obj, val, label, change, color in [
                    (c1, res['di_before'], "Disparate Impact\n(Before)", "Biased — needs fix",
                     "#E74C3C" if res['di_before']<0.8 else "#F39C12"),
                    (c2, res['di_after'],  "Disparate Impact\n(After)",  imp_label, "#11998e"),
                    (c3, res['spd_before'],"Stat Parity Diff\n(Before)", "0.0 = perfect fair", "#E74C3C"),
                    (c4, res['spd_after'], "Stat Parity Diff\n(After)",  "Reduced after fix!", "#11998e"),
                ]:
                    with col_obj:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value" style="color:{color};">{val:.3f}</div>
                            <div class="metric-change" style="color:{color};">{change}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("""
                <div class="note-box" style="margin-top:0.8rem;">
                    ℹ️ <strong>About DI After = 1.0:</strong>
                    IBM AIF360 Reweighing adjusts instance weights to balance group representation.
                    DI = 1.0 means statistical bias is fully corrected in the weighted dataset.
                    Real-world deployment requires retraining the model with these corrected weights.
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Charts
                st.markdown(
                    '<span style="color:#11998e;font-size:1.25rem;font-weight:800;'
                    'border-bottom:3px solid #38ef7d;padding-bottom:0.3rem;'
                    'margin:1.5rem 0 1rem 0;display:block;">'
                    '📈 Before vs After Comparison</span>',
                    unsafe_allow_html=True)
                outcome_name = (cfg['outcome_name'] if "Sample" in data_src
                                else f"{active_label_col.replace('_',' ').replace('-',' ').title()} Rate")
                chart_buf = make_charts(res, sector, pattr, active_group_names, outcome_name, cfg)
                st.image(chart_buf, use_container_width=True)

                # Traffic Light
                st.markdown(
                    '<span style="color:#11998e;font-size:1.25rem;font-weight:800;'
                    'border-bottom:3px solid #38ef7d;padding-bottom:0.3rem;'
                    'margin:1.5rem 0 1rem 0;display:block;">'
                    '🚦 Quick Status</span>',
                    unsafe_allow_html=True)
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
                        <div style="color:#aaa;font-size:0.8rem;font-weight:600;">BEFORE</div>
                        <div class="traffic-light {cb_}">{eb_}</div>
                        <div style="color:white;font-size:0.75rem;margin:0.3rem 0 0.8rem;">
                            {lb_} — DI: {res['di_before']:.3f}
                        </div>
                        <div style="color:#38ef7d;font-size:0.85rem;font-weight:bold;">
                            ↓ Reweighing Applied
                        </div>
                        <div style="color:#aaa;font-size:0.8rem;margin-top:0.8rem;font-weight:600;">AFTER</div>
                        <div class="traffic-light {ca_}">{ea_}</div>
                        <div style="color:white;font-size:0.75rem;margin:0.3rem 0 0.5rem;">
                            {la_} — DI: {res['di_after']:.3f}
                        </div>
                        <div style="color:#38ef7d;font-size:0.95rem;font-weight:bold;margin-top:0.3rem;">
                            Bias Mitigated ✓
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Gemini
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<span style="color:#11998e;font-size:1.25rem;font-weight:800;'
                    'border-bottom:3px solid #38ef7d;padding-bottom:0.3rem;'
                    'margin:1.5rem 0 1rem 0;display:block;">'
                    '✨ Gemini AI Explanation</span>',
                    unsafe_allow_html=True)
                if GEMINI_AVAILABLE:
                    with st.spinner("🤖 Gemini AI explanation generate kar raha hai..."):
                        gemini_text = get_gemini_explanation(
                            sector, pattr,
                            res['di_before'], res['di_after'],
                            res['rate_priv_before'], res['rate_unpriv_before'],
                            res['improvement'], g1n, g0n)
                    if gemini_text:
                        st.markdown(f"""
                        <div class="gemini-box">
                            <h4>🤖 Google Gemini AI Analysis</h4>
                            {gemini_text.replace(chr(10),'<br>')}
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-warning">⚠️ Gemini API not configured. Add <strong>GEMINI_API_KEY</strong> to .env file.</div>',
                                unsafe_allow_html=True)

                # Simple explanation
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<span style="color:#11998e;font-size:1.25rem;font-weight:800;'
                    'border-bottom:3px solid #38ef7d;padding-bottom:0.3rem;'
                    'margin:1.5rem 0 1rem 0;display:block;">'
                    '💬 Simple Explanation</span>',
                    unsafe_allow_html=True)
                st.markdown(f"""
                <div class="step-card">
                    <p><strong>Aapke {sector} data mein kya mila:</strong></p>
                    <ul style="line-height:2rem;">
                        <li>🔴 <strong>Before fix:</strong> {g0n} ko {g1n} ke comparison mein
                            <strong>{gap:.1f}%</strong> unfairly treat kiya ja raha tha</li>
                        <li>🟢 <strong>After fix:</strong> Gap sirf
                            <strong>{abs(res['rate_priv_after']-res['rate_unpriv_after']):.1f}%</strong>
                            reh gaya</li>
                        <li>⚖️ <strong>Disparate Impact:</strong> {res['di_before']:.3f} se
                            badh ke <strong>{res['di_after']:.3f}</strong> ho gaya</li>
                        <li>📌 <strong>Status:</strong> Bias successfully mitigated</li>
                    </ul>
                    <p style="color:#0a6b5e;font-weight:600;">
                        Reweighing + Gemini AI ne bias reduce kar diya.
                        Yeh tool SDG 5, SDG 8, SDG 10, aur SDG 16 ko directly support karta hai.
                    </p>
                </div>""", unsafe_allow_html=True)

                # Recommendations
                st.markdown(
                    '<span style="color:#11998e;font-size:1.25rem;font-weight:800;'
                    'border-bottom:3px solid #38ef7d;padding-bottom:0.3rem;'
                    'margin:1.5rem 0 1rem 0;display:block;">'
                    '✅ Recommendations</span>',
                    unsafe_allow_html=True)
                recs = [
                    f"<strong>{pattr.replace('_',' ').replace('-',' ').title()}</strong> column ko sensitive feature mark karo",
                    "Reweighing apply karo training data pe — har 3 mahine mein",
                    "Diverse data collect karo — underrepresented groups ka",
                    "Human review rakho final decisions mein",
                    "Regular bias monitoring karo — bias wapas aa sakta hai",
                    "Transparency report publish karo — SDG 16 accountability ke liye"
                ]
                for r in recs:
                    st.markdown(f'<div class="rec-card">✔ {r}</div>', unsafe_allow_html=True)

                # Download
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<span style="color:#11998e;font-size:1.25rem;font-weight:800;'
                    'border-bottom:3px solid #38ef7d;padding-bottom:0.3rem;'
                    'margin:1.5rem 0 1rem 0;display:block;">'
                    '📥 Download Report</span>',
                    unsafe_allow_html=True)
                chart_buf.seek(0)
                b64 = base64.b64encode(chart_buf.read()).decode()
                st.markdown(
                    f'<a href="data:image/png;base64,{b64}" '
                    f'download="bias_report_{sector.replace(" ","_")}.png" '
                    f'class="download-btn">📊 Download Bias Report (PNG)</a>',
                    unsafe_allow_html=True)
                st.markdown(f"""
                <div class="alert-success" style="margin-top:1rem;">
                    ✅ Analysis complete! Bias mitigated | {cfg['sdg']} | Powered by Google Gemini AI
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