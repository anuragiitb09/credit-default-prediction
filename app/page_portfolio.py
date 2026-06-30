import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import batch_score


REQUIRED_COLS = [
    'loan_amnt', 'int_rate', 'grade', 'sub_grade', 'emp_length',
    'home_ownership', 'annual_inc', 'dti', 'fico_range_low',
    'revol_util', 'open_acc', 'purpose', 'addr_state', 'earliest_cr_line'
]


def render(model, preprocessor, threshold):
    st.title("Portfolio batch scorer")
    st.caption("Upload a CSV of loan applications to score the full portfolio")

    with st.expander("Required CSV format", expanded=False):
        st.markdown(f"Your CSV must contain these {len(REQUIRED_COLS)} columns:")
        st.code(", ".join(REQUIRED_COLS))
        sample = pd.DataFrame([{
            'loan_amnt': 15000, 'int_rate': 13.5, 'grade': 'C', 'sub_grade': 'C3',
            'emp_length': '10+ years', 'home_ownership': 'MORTGAGE', 'annual_inc': 65000,
            'dti': 18.0, 'fico_range_low': 695, 'revol_util': 45.0, 'open_acc': 8,
            'purpose': 'debt_consolidation', 'addr_state': 'CA', 'earliest_cr_line': '2010-06-01'
        }])
        st.dataframe(sample, use_container_width=True)
        st.download_button(
            "Download sample template",
            sample.to_csv(index=False),
            file_name="portfolio_template.csv",
            mime="text/csv"
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV file to begin batch scoring")
        return

    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return

    st.success(f"Loaded {len(df_raw):,} applications")

    with st.spinner("Scoring portfolio..."):
        try:
            results = batch_score(df_raw, model, preprocessor, threshold)
        except Exception as e:
            st.error(f"Scoring failed: {e}")
            return

    st.markdown("---")
    st.subheader("Portfolio summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total applications", f"{len(results):,}")
    c2.metric("Approval rate", f"{(results['decision']=='APPROVE').mean():.1%}")
    c3.metric("Avg default probability", f"{results['default_probability'].mean():.1%}")
    c4.metric("Total expected loss", f"${results['expected_loss'].sum():,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(results['default_probability'], bins=30, color='#7F77DD', alpha=0.8)
        ax.axvline(threshold, color='#D85A30', linestyle='--', linewidth=1.5, label=f'Threshold: {threshold:.2f}')
        ax.set_xlabel('Predicted default probability')
        ax.set_ylabel('Number of applications')
        ax.set_title('Score distribution')
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        approval_by_grade = (
            results.groupby('grade')['decision']
            .apply(lambda x: (x == 'APPROVE').mean())
            .reindex(['A','B','C','D','E','F','G'])
        )
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(approval_by_grade.index, approval_by_grade.values, color='#1D9E75', alpha=0.85)
        ax.set_xlabel('Grade')
        ax.set_ylabel('Approval rate')
        ax.set_title('Approval rate by loan grade')
        ax.set_ylim(0, 1)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Scored applications")

    decision_filter = st.multiselect(
        "Filter by decision",
        options=['APPROVE', 'DECLINE'],
        default=['APPROVE', 'DECLINE']
    )
    filtered = results[results['decision'].isin(decision_filter)]

    st.dataframe(
        filtered.style.background_gradient(
            subset=['default_probability'], cmap='RdYlGn_r'
        ).format({
            'default_probability': '{:.2%}',
            'expected_loss': '${:,.0f}',
            'annual_inc': '${:,.0f}',
            'loan_amnt': '${:,.0f}',
        }),
        use_container_width=True,
        height=450
    )

    st.download_button(
        "Download scored portfolio (CSV)",
        filtered.to_csv(index=False),
        file_name="scored_portfolio.csv",
        mime="text/csv",
        type="primary"
    )