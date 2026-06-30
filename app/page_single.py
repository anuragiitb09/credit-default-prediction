import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import shap

from utils import build_raw_input_df, score_applicant, get_shap_explanation, clean_feature_name


GRADE_TO_SUBGRADES = {
    g: [f"{g}{i}" for i in range(1, 6)] for g in ['A','B','C','D','E','F','G']
}


def render(model, preprocessor, explainer, meta, threshold):
    st.title("Single applicant risk scorer")
    st.caption("Enter applicant details to generate a risk score and SHAP explanation")

    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Loan details")
            loan_amnt = st.number_input("Loan amount ($)", min_value=500, max_value=40000, value=15000, step=500)
            int_rate  = st.number_input("Interest rate (%)", min_value=5.0, max_value=31.0, value=13.5, step=0.1)
            grade     = st.selectbox("Grade", options=['A','B','C','D','E','F','G'], index=2)
            sub_grade = st.selectbox("Sub-grade", options=GRADE_TO_SUBGRADES[grade], index=2)
            purpose   = st.selectbox("Loan purpose", options=[
                'debt_consolidation', 'credit_card', 'home_improvement',
                'major_purchase', 'small_business', 'car', 'medical',
                'moving', 'vacation', 'house', 'wedding', 'renewable_energy', 'other'
            ])

        with col2:
            st.subheader("Borrower profile")
            annual_inc      = st.number_input("Annual income ($)", min_value=0, max_value=2_000_000, value=65000, step=1000)
            emp_length      = st.selectbox("Employment length", options=[
                '< 1 year', '1 year', '2 years', '3 years', '4 years',
                '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
            ], index=10)
            home_ownership  = st.selectbox("Home ownership", options=['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
            addr_state      = st.selectbox("State", options=[
                'CA','TX','NY','FL','IL','PA','OH','GA','NC','MI','NJ','VA','WA',
                'AZ','MA','TN','IN','MO','MD','WI','CO','MN','SC','AL','LA','KY',
                'OR','OK','CT','UT','IA','NV','AR','MS','KS','NM','NE','WV','ID',
                'HI','NH','ME','RI','MT','DE','SD','ND','AK','VT','WY','DC'
            ])

        with col3:
            st.subheader("Credit profile")
            dti             = st.number_input("Debt-to-income ratio (%)", min_value=0.0, max_value=60.0, value=18.0, step=0.5)
            fico_range_low  = st.number_input("FICO score", min_value=300, max_value=850, value=695, step=5)
            revol_util      = st.number_input("Revolving utilization (%)", min_value=0.0, max_value=150.0, value=45.0, step=1.0)
            open_acc        = st.number_input("Open credit accounts", min_value=0, max_value=60, value=8, step=1)
            earliest_cr_line = st.date_input(
                "Earliest credit line opened",
                value=datetime.date(2010, 6, 1),
                min_value=datetime.date(1960, 1, 1),
                max_value=datetime.date(2018, 12, 31)
            )

        submitted = st.form_submit_button("Score applicant", type="primary", use_container_width=True)

    if submitted:
        inputs = {
            'loan_amnt': loan_amnt, 'int_rate': int_rate, 'grade': grade,
            'sub_grade': sub_grade, 'emp_length': emp_length,
            'home_ownership': home_ownership, 'annual_inc': annual_inc,
            'dti': dti, 'fico_range_low': fico_range_low,
            'revol_util': revol_util, 'open_acc': open_acc,
            'purpose': purpose, 'addr_state': addr_state,
            'earliest_cr_line': earliest_cr_line.strftime('%Y-%m-%d'),
        }

        raw_df = build_raw_input_df(inputs)
        prob, decision, risk_score, X_processed = score_applicant(raw_df, model, preprocessor, threshold)

        st.markdown("---")
        st.subheader("Result")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk score", f"{risk_score}/100", help="100 = lowest risk, 0 = highest risk")
        c2.metric("Default probability", f"{prob:.1%}")
        c3.metric("Decision", decision)
        c4.metric("Expected loss", f"${prob * 0.45 * loan_amnt:,.0f}", help="PD × LGD(0.45) × EAD")

        if decision == "DECLINE":
            st.error(f"Application declined — predicted default probability ({prob:.1%}) exceeds threshold ({threshold:.0%})")
        else:
            st.success(f"Application approved — predicted default probability ({prob:.1%}) is below threshold ({threshold:.0%})")

        st.markdown("---")
        st.subheader("Why this decision — SHAP explanation")
        st.caption("Each bar shows how much a feature pushed the prediction above or below the average applicant")

        feature_names_clean = [clean_feature_name(c) for c in X_processed.columns]
        shap_exp = get_shap_explanation(X_processed, explainer, feature_names_clean)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_exp[0], max_display=10, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("View raw input data sent to model"):
            st.dataframe(raw_df, use_container_width=True)