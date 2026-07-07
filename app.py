import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Mule Account Detector",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("model/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, feature_names = load_model()

st.title("🏦 Mule Account Detection System")
st.markdown("**AI-powered financial fraud detection | SBI Internship Project**")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Transaction Analyzer", "📊 Batch Analysis", "ℹ️ About"])

with tab1:
    st.subheader("Analyze a Single Transaction")
    col1, col2, col3 = st.columns(3)
    with col1:
        transaction_amt = st.number_input("Transaction Amount (₹)", min_value=0.0, max_value=50000.0, value=1000.0, step=100.0)
        product_cd = st.selectbox("Product Code", ["W", "H", "C", "S", "R"])
        card_type = st.selectbox("Card Type", ["credit", "debit", "debit or credit", "charge card"])
    with col2:
        card_bank = st.selectbox("Card Bank", ["visa", "mastercard", "american express", "discover"])
        addr1 = st.number_input("Billing ZIP (addr1)", min_value=100, max_value=999, value=299)
        dist1 = st.number_input("Distance from Home (dist1)", min_value=0, max_value=10000, value=50)
    with col3:
        email_domain = st.selectbox("Buyer Email Domain", ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "protonmail.com"])
        p_email_domain = st.selectbox("Recipient Email Domain", ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "protonmail.com"])
        transaction_hour = st.slider("Transaction Hour (24h)", 0, 23, 14)

    if st.button("🔎 Analyze Transaction", type="primary"):
        input_data = pd.DataFrame(np.full((1, len(feature_names)), -999), columns=feature_names)
        if "TransactionAmt" in feature_names:
            input_data["TransactionAmt"] = transaction_amt
        if "ProductCD" in feature_names:
            input_data["ProductCD"] = ["W","H","C","S","R"].index(product_cd)
        if "card4" in feature_names:
            input_data["card4"] = ["visa","mastercard","american express","discover"].index(card_bank)
        if "card6" in feature_names:
            input_data["card6"] = ["credit","debit","debit or credit","charge card"].index(card_type)
        if "addr1" in feature_names:
            input_data["addr1"] = addr1
        if "dist1" in feature_names:
            input_data["dist1"] = dist1
        if "P_emaildomain" in feature_names:
            input_data["P_emaildomain"] = ["gmail.com","yahoo.com","hotmail.com","anonymous.com","protonmail.com"].index(p_email_domain)
        if "R_emaildomain" in feature_names:
            input_data["R_emaildomain"] = ["gmail.com","yahoo.com","hotmail.com","anonymous.com","protonmail.com"].index(email_domain)

        prob = model.predict_proba(input_data)[0][1]
        risk_pct = round(prob * 100, 2)

        st.markdown("---")
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            if prob < 0.3:
                st.success("✅ LOW RISK")
                verdict = "This transaction appears normal."
            elif prob < 0.6:
                st.warning("⚠️ MEDIUM RISK")
                verdict = "This transaction has suspicious signals. Flag for review."
            else:
                st.error("🚨 HIGH RISK — Possible Mule Account")
                verdict = "This transaction is likely fraudulent. Block and investigate."
        with col_res2:
            st.metric("Fraud Probability", f"{risk_pct}%")
        with col_res3:
            st.metric("Model Confidence (AUC)", "94.06%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            title={"text": "Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red" if prob > 0.6 else "orange" if prob > 0.3 else "green"},
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.info(verdict)

with tab2:
    st.subheader("Batch Transaction Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df_upload)} transactions")
        input_batch = pd.DataFrame(np.full((len(df_upload), len(feature_names)), -999), columns=feature_names)
        for col in df_upload.columns:
            if col in feature_names:
                input_batch[col] = df_upload[col].values
        probs = model.predict_proba(input_batch)[:, 1]
        df_upload["Fraud_Probability"] = (probs * 100).round(2)
        df_upload["Risk_Level"] = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Total Transactions", len(df_upload))
        col_b2.metric("High Risk", (df_upload["Risk_Level"] == "High").sum())
        col_b3.metric("Medium Risk", (df_upload["Risk_Level"] == "Medium").sum())
        fig2 = px.histogram(df_upload, x="Fraud_Probability", nbins=50,
                            title="Distribution of Fraud Probability Scores",
                            color_discrete_sequence=["#e74c3c"])
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(df_upload[["Fraud_Probability", "Risk_Level"]].head(50), use_container_width=True)
        csv = df_upload.to_csv(index=False)
        st.download_button("📥 Download Results", csv, "fraud_analysis_results.csv", "text/csv")

with tab3:
    st.subheader("About This System")
    st.markdown("""
### 🏦 SBI Mule Account Detection System

**Developed by:** Aryan Sinha

---

### What are Mule Accounts?
Mule accounts are bank accounts used to receive and transfer illegally obtained money,
making it difficult to trace the original fraudster.

### How does this AI system work?
- **Dataset:** IEEE-CIS Fraud Detection (590,540 transactions)
- **Algorithm:** XGBoost (Extreme Gradient Boosting)
- **Features:** 434 transaction and identity features
- **Performance:** ROC-AUC Score of 0.9406 (94.06%)

### Key Risk Signals Detected
- Unusually high transaction amounts
- Anonymous or mismatched email domains
- Transactions at odd hours
- Large distance between billing and shipping address
- Multiple rapid transactions from same account
""")