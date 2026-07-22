import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Mule Hunter", page_icon="🎯", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e, #16213e, #0f3460);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.big-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(90deg, #ff416c, #ff4b2b, #f7971e, #ffd200);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0;
}
.subtitle { font-size: 1rem; color: #aaa; margin-bottom: 2rem; }
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.85);
    border-right: 1px solid rgba(255,255,255,0.1);
}
.stButton > button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.5rem 2rem;
}
div[data-testid="stMetricValue"] { color: white !important; font-size: 1.8rem !important; font-weight: 700 !important; }
div[data-testid="stMetricLabel"] { color: #aaa !important; }
.stTabs [aria-selected="true"] { color: #ff416c !important; border-bottom: 2px solid #ff416c; }
h1, h2, h3, h4, p, label, .stMarkdown { color: white !important; }
.rank-badge {
    display: inline-block;
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white; border-radius: 50%; width: 28px; height: 28px;
    text-align: center; line-height: 28px; font-weight: 700;
    font-size: 0.85rem; margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("model/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, feature_names = load_model()

st.markdown('<p class="big-title">🎯 Mule Hunter</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered mule account detection — upload transactions, hunt fraudsters</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Hunter Settings")
    high_threshold = st.slider("🔴 High Risk threshold (%)", 20, 90, 40)
    med_threshold  = st.slider("🟡 Medium Risk threshold (%)", 10, 50, 25)
    min_transactions = st.number_input("Min transactions per account", 1, 20, 1)
    st.markdown("---")
    st.markdown("**🤖 Model Info**")
    st.markdown("Algorithm: `XGBoost`")
    st.markdown("ROC-AUC: `94.06%`")
    st.markdown("Trained on: `590,540 transactions`")
    st.markdown("---")
    st.markdown("**Developed by:** Aryan Sinha")

uploaded_file = st.file_uploader("📂 Upload transaction CSV", type=["csv"])

INDIAN_COL_MAP = {
    "Transaction_Amount":   "TransactionAmt",
    "Account_Age_Months":   "card2",
    "Daily_Transactions":   "C1",
    "Total_Credit":         "D1",
    "Total_Debit":          "D2",
    "Avg_Transaction":      "D15",
    "Unique_Beneficiaries": "C13",
}
TRANSACTION_TYPE_MAP = {"UPI": 0, "IMPS": 1, "NEFT": 2, "RTGS": 3, "SWIFT": 4}
DEVICE_MAP = {"Mobile": 0, "Laptop": 1, "Desktop": 2, "Tablet": 3}
CAT_MAP = {
    "ProductCD":     ["W","H","C","S","R"],
    "card4":         ["visa","mastercard","american express","discover"],
    "card6":         ["credit","debit","debit or credit","charge card"],
    "P_emaildomain": ["gmail.com","yahoo.com","hotmail.com","anonymous.com","protonmail.com"],
    "R_emaildomain": ["gmail.com","yahoo.com","hotmail.com","anonymous.com","protonmail.com"],
}

if uploaded_file is None:
    st.info("👆 Upload a CSV file to start hunting mule accounts")
    st.markdown("### 📋 Supported Formats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**IEEE-CIS Format**")
        sample1 = pd.DataFrame({
            "AccountID":     ["ACC001", "ACC002"],
            "TransactionAmt":[15000, 500],
            "ProductCD":     ["W", "H"],
            "card4":         ["visa", "mastercard"],
            "P_emaildomain": ["anonymous.com", "gmail.com"],
        })
        st.dataframe(sample1, use_container_width=True)
    with col2:
        st.markdown("**Indian Banking Format**")
        sample2 = pd.DataFrame({
            "Account_ID":           ["ACC001", "ACC002"],
            "Transaction_Amount":   [85000, 1800],
            "Transaction_Type":     ["IMPS", "UPI"],
            "Daily_Transactions":   [19, 2],
            "Unique_Beneficiaries": [18, 1],
        })
        st.dataframe(sample2, use_container_width=True)

else:
    df = pd.read_csv(uploaded_file)

    if "Account_ID" in df.columns and "AccountID" not in df.columns:
        df["AccountID"] = df["Account_ID"]
    elif "AccountID" not in df.columns:
        st.warning("⚠️ No AccountID column found — treating each row as a separate account")
        df["AccountID"] = [f"TXN_{i:05d}" for i in range(len(df))]

    st.success(f"✅ Loaded {len(df):,} transactions")

    with st.spinner("🔍 Hunting mule accounts..."):
        is_indian = "Transaction_Amount" in df.columns

        if is_indian:
            st.info("📌 Indian transaction format detected — mapping columns to model features")
            df_mapped = df.copy()
            for indian_col, ieee_col in INDIAN_COL_MAP.items():
                if indian_col in df_mapped.columns:
                    df_mapped[ieee_col] = pd.to_numeric(df_mapped[indian_col], errors="coerce").fillna(-999)
            if "Transaction_Type" in df_mapped.columns:
                df_mapped["ProductCD"] = df_mapped["Transaction_Type"].map(TRANSACTION_TYPE_MAP).fillna(-999)
            if "Device_Type" in df_mapped.columns:
                df_mapped["card4"] = df_mapped["Device_Type"].map(DEVICE_MAP).fillna(-999)
            if "Location" in df_mapped.columns:
                df_mapped["addr1"] = pd.Categorical(df_mapped["Location"]).codes
        else:
            df_mapped = df.copy()

        input_data = pd.DataFrame(np.full((len(df_mapped), len(feature_names)), -999.0), columns=feature_names)
        for col in df_mapped.columns:
            if col in feature_names:
                input_data[col] = pd.to_numeric(df_mapped[col], errors="coerce").fillna(-999).values
            elif col in CAT_MAP and col in feature_names:
                input_data[col] = df_mapped[col].apply(
                    lambda x: CAT_MAP[col].index(x) if x in CAT_MAP[col] else -999
                )

        probs = model.predict_proba(input_data)[:, 1]
        df["fraud_prob"] = probs
        df["fraud_pct"]  = (probs * 100).round(2)

    account_stats = df.groupby("AccountID").agg(
        total_transactions=("fraud_prob", "count"),
        avg_risk=("fraud_pct", "mean"),
        max_risk=("fraud_pct", "max"),
        high_risk_txns=("fraud_prob", lambda x: (x > high_threshold/100).sum())
    ).reset_index()

    if "Transaction_Amount" in df.columns:
        amt = df.groupby("AccountID")["Transaction_Amount"].sum().reset_index()
        amt.columns = ["AccountID", "total_amount"]
        account_stats = account_stats.merge(amt, on="AccountID", how="left")
    elif "TransactionAmt" in df.columns:
        amt = df.groupby("AccountID")["TransactionAmt"].sum().reset_index()
        amt.columns = ["AccountID", "total_amount"]
        account_stats = account_stats.merge(amt, on="AccountID", how="left")

    account_stats["avg_risk"] = account_stats["avg_risk"].round(2)
    account_stats["max_risk"] = account_stats["max_risk"].round(2)

    def classify(row):
        if row["avg_risk"] >= high_threshold or row["high_risk_txns"] >= 2:
            return "🔴 High Risk"
        elif row["avg_risk"] >= med_threshold:
            return "🟡 Medium Risk"
        else:
            return "🟢 Low Risk"

    account_stats = account_stats[account_stats["total_transactions"] >= min_transactions]
    account_stats["Risk Level"] = account_stats.apply(classify, axis=1)
    account_stats = account_stats.sort_values("avg_risk", ascending=False).reset_index(drop=True)
    account_stats.index += 1

    high_risk = (account_stats["Risk Level"] == "🔴 High Risk").sum()
    med_risk  = (account_stats["Risk Level"] == "🟡 Medium Risk").sum()
    low_risk  = (account_stats["Risk Level"] == "🟢 Low Risk").sum()

    st.markdown("---")
    st.markdown("## 🚨 Hunt Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Accounts", f"{len(account_stats):,}")
    m2.metric("🔴 High Risk", high_risk)
    m3.metric("🟡 Medium Risk", med_risk)
    m4.metric("🟢 Low Risk", low_risk)

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 📋 Flagged Account Leaderboard")
        display_df = account_stats[["AccountID", "Risk Level", "avg_risk", "max_risk", "total_transactions", "high_risk_txns"]].copy()
        display_df.columns = ["Account ID", "Risk Level", "Avg Risk %", "Max Risk %", "Total Txns", "High Risk Txns"]
        st.dataframe(display_df, use_container_width=True, height=400)

    with col_right:
        st.markdown("### 📊 Risk Distribution")
        pie_data = pd.DataFrame({
            "Risk":  ["High Risk", "Medium Risk", "Low Risk"],
            "Count": [high_risk, med_risk, low_risk]
        })
        fig_pie = px.pie(
            pie_data, values="Count", names="Risk", color="Risk",
            color_discrete_map={"High Risk":"#ff416c","Medium Risk":"#f7971e","Low Risk":"#2ecc71"},
            hole=0.45
        )
        fig_pie.update_layout(
            height=280, margin=dict(t=0,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", legend_font_color="white"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### 🏆 Top 5 Most Suspicious")
        for i, row in account_stats.head(5).iterrows():
            st.markdown(
                f'<span class="rank-badge">{i}</span> **{row["AccountID"]}** — `{row["avg_risk"]}%` {row["Risk Level"]}',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("### 🔬 Drill Into an Account")
    selected_account = st.selectbox(
        "Select an account to inspect",
        account_stats["AccountID"].tolist(),
        format_func=lambda x: f"{x} — {account_stats[account_stats['AccountID']==x]['Risk Level'].values[0]} ({account_stats[account_stats['AccountID']==x]['avg_risk'].values[0]}% avg risk)"
    )

    if selected_account:
        acct_txns = df[df["AccountID"] == selected_account].copy()
        acct_info = account_stats[account_stats["AccountID"] == selected_account].iloc[0]

        d1, d2, d3 = st.columns(3)
        d1.metric("Risk Level", acct_info["Risk Level"])
        d2.metric("Avg Fraud Risk", f"{acct_info['avg_risk']}%")
        d3.metric("Total Transactions", int(acct_info["total_transactions"]))

        fig_bar = px.bar(
            acct_txns.reset_index(),
            x=acct_txns.reset_index().index,
            y="fraud_pct",
            title=f"Transaction Risk Scores — {selected_account}",
            color="fraud_pct",
            color_continuous_scale=["#2ecc71","#f7971e","#ff416c"],
            labels={"fraud_pct":"Fraud Risk %","index":"Transaction #"}
        )
        fig_bar.add_hline(y=high_threshold, line_dash="dash", line_color="#ff416c", annotation_text="High Risk")
        fig_bar.add_hline(y=med_threshold,  line_dash="dash", line_color="#f7971e", annotation_text="Medium Risk")
        fig_bar.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", title_font_color="white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(acct_txns.drop(columns=["fraud_prob"]).reset_index(drop=True), use_container_width=True)

    st.markdown("---")

    try:
        fpr     = np.load("model/fpr.npy")
        tpr     = np.load("model/tpr.npy")
        roc_auc = np.load("model/roc_auc.npy")[0]
        st.markdown("### 📈 Model Performance — ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f'XGBoost (AUC = {roc_auc:.4f})',
            line=dict(color='#ff416c', width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash')
        ))
        fig_roc.update_layout(
            title='ROC Curve — Mule Account Detection Model',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', legend=dict(bgcolor='rgba(0,0,0,0)'), height=450
        )
        fig_roc.add_annotation(
            x=0.6, y=0.15, text=f"AUC = {roc_auc:.4f}",
            font=dict(size=18, color='#ff416c'), showarrow=False
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption("AUC of 0.94 means the model correctly distinguishes fraud from non-fraud 94% of the time")
    except FileNotFoundError:
        pass

    csv_out = account_stats.to_csv(index=False)
    st.download_button(
        "📥 Download Full Flagged Report",
        csv_out, "mule_hunter_report.csv", "text/csv", type="primary"
    )
