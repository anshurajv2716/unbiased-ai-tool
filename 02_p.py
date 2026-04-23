# ============================================
# UNBIASED AI DECISION TOOL
# Google Solution Challenge 2026
# Team: Anjani + Anshu
# ============================================
# CHALANE KA TARIKA:
# Terminal mein likho: streamlit run app.py
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

# ============================================
# PAGE SETUP
# ============================================
st.set_page_config(
    page_title="Unbiased AI Decision Tool",
    page_icon="⚖️",
    layout="wide"
)

# ============================================
# HEADER
# ============================================
st.title("⚖️ Unbiased AI Decision Tool")
st.markdown("### Google Solution Challenge 2026 — Team Anjani & Anshu")
st.markdown("---")

st.markdown("""
**SDG 5** — Gender Equality &nbsp;|&nbsp;
**SDG 10** — Reduced Inequalities &nbsp;|&nbsp;
**SDG 16** — Justice & Accountability
""")

st.markdown("---")

# ============================================
# SIDEBAR — USER INSTRUCTIONS
# ============================================
with st.sidebar:
    st.header("📋 How to Use")
    st.markdown("""
    1. **Select sector** — Job / Loan / Health
    2. **Upload CSV file** — ya sample data use karo
    3. **Select columns** — protected attribute choose karo
    4. **Click Analyze** — bias report dekho
    5. **Download PDF** — report save karo
    """)
    st.markdown("---")
    st.markdown("**Made with ❤️ for fair AI**")
    st.markdown("*Bias fully remove nahi hota — lekin control aur reduce kar sakte hain*")

# ============================================
# STEP 1 — SECTOR SELECTION
# ============================================
st.header("Step 1 — Sector Choose Karo")

sector = st.selectbox(
    "Aap kis sector ka data check karna chahte ho?",
    ["Job Hiring", "Loan Approval", "Healthcare"]
)

sector_info = {
    "Job Hiring": {
        "desc": "Hiring data mein gender, caste, location bias detect karo",
        "sdg": "SDG 8 + SDG 5",
        "label": "hired",
        "protected": "gender"
    },
    "Loan Approval": {
        "desc": "Loan data mein income, zip code, religion bias detect karo",
        "sdg": "SDG 10 + SDG 16",
        "label": "approved",
        "protected": "gender"
    },
    "Healthcare": {
        "desc": "Healthcare data mein rural/urban, gender bias detect karo",
        "sdg": "SDG 3 + SDG 10",
        "label": "treatment",
        "protected": "gender"
    }
}

st.info(f"**{sector}** — {sector_info[sector]['desc']} ({sector_info[sector]['sdg']})")

# ============================================
# STEP 2 — DATA INPUT
# ============================================
st.header("Step 2 — Data Upload Karo")

data_choice = st.radio(
    "Data source choose karo:",
    ["Sample Data Use Karo (Demo)", "Apni CSV File Upload Karo"]
)

df = None

if data_choice == "Sample Data Use Karo (Demo)":
    st.info("Demo ke liye sample biased dataset load ho raha hai...")

    np.random.seed(42)
    n = 500

    if sector == "Job Hiring":
        data = pd.DataFrame({
            'gender':     np.random.choice([1, 0], n, p=[0.6, 0.4]),
            'age':        np.random.randint(22, 55, n),
            'experience': np.random.randint(0, 15, n),
            'education':  np.random.choice([0, 1, 2], n),
            'location':   np.random.choice([1, 0], n, p=[0.7, 0.3]),
        })
        hired = []
        for _, row in data.iterrows():
            prob = 0.3 + (row['experience'] * 0.03) + (row['education'] * 0.1)
            if row['gender'] == 0:
                prob -= 0.25
            if row['location'] == 0:
                prob -= 0.10
            prob = max(0.05, min(0.95, prob))
            hired.append(np.random.binomial(1, prob))
        data['hired'] = hired
        df = data
        label_col     = 'hired'
        protected_col = 'gender'

    elif sector == "Loan Approval":
        data = pd.DataFrame({
            'gender':  np.random.choice([1, 0], n, p=[0.55, 0.45]),
            'age':     np.random.randint(25, 60, n),
            'income':  np.random.randint(20000, 100000, n),
            'urban':   np.random.choice([1, 0], n, p=[0.65, 0.35]),
        })
        approved = []
        for _, row in data.iterrows():
            prob = 0.4 + (row['income'] / 200000)
            if row['gender'] == 0:
                prob -= 0.20
            if row['urban'] == 0:
                prob -= 0.15
            prob = max(0.05, min(0.95, prob))
            approved.append(np.random.binomial(1, prob))
        data['approved'] = approved
        df = data
        label_col     = 'approved'
        protected_col = 'gender'

    else:  # Healthcare
        data = pd.DataFrame({
            'gender': np.random.choice([1, 0], n, p=[0.5, 0.5]),
            'age':    np.random.randint(18, 80, n),
            'urban':  np.random.choice([1, 0], n, p=[0.6, 0.4]),
            'income': np.random.randint(10000, 80000, n),
        })
        treatment = []
        for _, row in data.iterrows():
            prob = 0.5
            if row['gender'] == 0:
                prob -= 0.18
            if row['urban'] == 0:
                prob -= 0.12
            prob = max(0.05, min(0.95, prob))
            treatment.append(np.random.binomial(1, prob))
        data['treatment'] = treatment
        df = data
        label_col     = 'treatment'
        protected_col = 'gender'

    st.success(f"✅ Sample data ready! {len(df)} records loaded.")
    st.dataframe(df.head(10))

else:
    uploaded_file = st.file_uploader(
        "CSV file upload karo",
        type=['csv']
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded! {len(df)} records, {len(df.columns)} columns.")
        st.dataframe(df.head(10))

        col1, col2 = st.columns(2)
        with col1:
            label_col = st.selectbox(
                "Outcome column kaunsa hai? (hired/approved/treatment)",
                df.columns.tolist()
            )
        with col2:
            protected_col = st.selectbox(
                "Protected attribute kaunsa hai? (gender/caste/location)",
                df.columns.tolist()
            )

# ============================================
# STEP 3 — ANALYZE BUTTON
# ============================================
if df is not None:
    st.header("Step 3 — Bias Analyze Karo")

    if st.button("🔍 ANALYZE BIAS NOW", type="primary", use_container_width=True):

        with st.spinner("Bias scan ho raha hai... please wait..."):

            try:
                # AIF360 dataset banana
                dataset = BinaryLabelDataset(
                    df=df[[protected_col, label_col] +
                          [c for c in df.columns if c not in [protected_col, label_col]]],
                    label_names=[label_col],
                    protected_attribute_names=[protected_col],
                    favorable_label=1,
                    unfavorable_label=0
                )

                privileged_groups   = [{protected_col: 1}]
                unprivileged_groups = [{protected_col: 0}]

                # BEFORE metrics
                metric_before = BinaryLabelDatasetMetric(
                    dataset,
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups
                )
                di_before  = metric_before.disparate_impact()
                spd_before = metric_before.statistical_parity_difference()

                # Group rates
                group1_rate = df[df[protected_col] == 1][label_col].mean() * 100
                group0_rate = df[df[protected_col] == 0][label_col].mean() * 100

                # Reweighing — AFTER
                RW = Reweighing(
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups
                )
                dataset_fixed = RW.fit_transform(dataset)

                metric_after = BinaryLabelDatasetMetric(
                    dataset_fixed,
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups
                )
                di_after  = metric_after.disparate_impact()
                spd_after = metric_after.statistical_parity_difference()

                weights = dataset_fixed.instance_weights
                mask1   = df[protected_col] == 1
                mask0   = df[protected_col] == 0
                group1_rate_fixed = np.average(df[mask1][label_col], weights=weights[mask1]) * 100
                group0_rate_fixed = np.average(df[mask0][label_col], weights=weights[mask0]) * 100

                # ============================================
                # RESULTS DISPLAY
                # ============================================
                st.markdown("---")
                st.header("📊 Bias Analysis Results")

                # Traffic Light
                if di_before < 0.8:
                    bias_level  = "🔴 HIGH BIAS DETECTED"
                    bias_color  = "error"
                    bias_msg    = f"Serious bias hai! Group 0 ko {abs(group1_rate - group0_rate):.1f}% unfairly treat kiya ja raha hai."
                elif di_before < 0.9:
                    bias_level  = "🟡 MEDIUM BIAS DETECTED"
                    bias_color  = "warning"
                    bias_msg    = "Moderate bias hai — fix karna recommended hai."
                else:
                    bias_level  = "🟢 LOW BIAS — ACCEPTABLE"
                    bias_color  = "success"
                    bias_msg    = "Data relatively fair hai!"

                if bias_color == "error":
                    st.error(f"**{bias_level}** — {bias_msg}")
                elif bias_color == "warning":
                    st.warning(f"**{bias_level}** — {bias_msg}")
                else:
                    st.success(f"**{bias_level}** — {bias_msg}")

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Disparate Impact (Before)", f"{di_before:.3f}",
                            delta="Biased" if di_before < 0.8 else "OK")
                col2.metric("Disparate Impact (After)",  f"{di_after:.3f}",
                            delta=f"+{di_after - di_before:.3f} improvement")
                col3.metric("Stat Parity Diff (Before)", f"{spd_before:.3f}")
                col4.metric("Stat Parity Diff (After)",  f"{spd_after:.3f}")

                # ============================================
                # CHARTS
                # ============================================
                st.subheader("📈 Before vs After Comparison")

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.patch.set_facecolor('white')

                # Chart 1 — Hire/Approval Rates
                ax1  = axes[0]
                cats = ['Group 1\n(Before)', 'Group 0\n(Before)',
                        'Group 1\n(After)',  'Group 0\n(After)']
                vals   = [group1_rate, group0_rate, group1_rate_fixed, group0_rate_fixed]
                colors = ['#4A90D9', '#E05C5C', '#4A90D9', '#5CB85C']
                bars   = ax1.bar(cats, vals, color=colors, edgecolor='white', linewidth=1.5)
                for bar, val in zip(bars, vals):
                    ax1.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.5,
                             f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)
                ax1.set_title(f'{sector} — Outcome Rate Comparison', fontweight='bold')
                ax1.set_ylabel('Rate (%)')
                ax1.set_ylim(0, max(vals) * 1.3)
                ax1.set_facecolor('#F8F9FA')
                ax1.grid(axis='y', alpha=0.3)

                # Chart 2 — Disparate Impact
                ax2    = axes[1]
                di_vals   = [di_before, di_after]
                di_labels = ['BEFORE', 'AFTER']
                di_colors = ['#E05C5C' if v < 0.8 else '#F0A500' if v < 0.9 else '#5CB85C'
                             for v in di_vals]
                bars2 = ax2.bar(di_labels, di_vals, color=di_colors,
                                width=0.4, edgecolor='white', linewidth=1.5)
                for bar, val in zip(bars2, di_vals):
                    ax2.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.01,
                             f'{val:.3f}', ha='center', fontweight='bold', fontsize=13)
                ax2.axhline(y=0.8, color='orange', linestyle='--',
                            linewidth=2, label='Min Fair (0.8)')
                ax2.axhline(y=1.0, color='green',  linestyle='--',
                            linewidth=2, label='Perfect (1.0)')
                ax2.set_title('Disparate Impact Score', fontweight='bold')
                ax2.set_ylabel('Score (1.0 = Perfect Fair)')
                ax2.set_ylim(0, 1.3)
                ax2.legend()
                ax2.set_facecolor('#F8F9FA')
                ax2.grid(axis='y', alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                # ============================================
                # PLAIN LANGUAGE EXPLANATION
                # ============================================
                st.subheader("💬 Simple Explanation")
                improvement = ((di_after - di_before) / abs(di_before)) * 100

                st.markdown(f"""
                **Aapke {sector} data mein kya mila:**

                - **Before fix:** Group 0 ko Group 1 ke comparison mein
                  **{abs(group1_rate - group0_rate):.1f}% unfairly treat** kiya ja raha tha
                - **After fix:** Gap reduce hokar sirf
                  **{abs(group1_rate_fixed - group0_rate_fixed):.1f}%** reh gaya
                - **Improvement: {improvement:.1f}%** bias reduce hua
                - **Disparate Impact** {di_before:.3f} se badh ke {di_after:.3f} ho gaya

                **Matlab:** Reweighing technique ne bias significantly reduce kar diya.
                Yeh tool SDG 5, SDG 10, aur SDG 16 ko directly support karta hai.
                """)

                # ============================================
                # RECOMMENDATIONS
                # ============================================
                st.subheader("✅ Recommendations")
                st.markdown(f"""
                1. **{protected_col} column** ko sensitive feature mark karo
                2. **Reweighing** apply karo training data pe
                3. **Regular monitoring** karo — har 3 mahine mein check karo
                4. **Diverse data** collect karo — underrepresented groups ka
                5. **Human review** rakho final decisions mein
                """)

                st.success(f"✅ Analysis complete! Improvement: +{improvement:.1f}% | SDG 5 | SDG 10 | SDG 16")

            except Exception as e:
                st.error(f"Error aaya: {str(e)}")
                st.info("Tip: Make sure columns mein sirf numbers hain (0 aur 1) — text nahi.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
Unbiased AI Decision Tool — Google Solution Challenge 2026<br>
Team: Anjani & Anshu | SDG 5 | SDG 10 | SDG 16<br>
"AI khud biased nahi hota — wo data ka bias reflect karta hai."
</div>
""", unsafe_allow_html=True)