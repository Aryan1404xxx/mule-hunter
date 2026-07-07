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

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

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
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff416c, #ff4b2b, #f7971e, #ffd200);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    font-size: 1rem;
    color: #aaa;
    margin-bottom: 2rem;
}

.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
}

section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.85);
    border-right: 1px solid rgba(255,255,255,0.1);
}

.stButton > button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 2rem;
    transition: transform 0.2s;
}

.stButton > button:hover {
    transform: scale(1.03);
    color: white;
}

.stDataFrame { border-radius: 12px; }

div[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricLabel"] {
    color: #aaa !important;
}

.stTabs [data-baseweb="tab"] {
    color: #aaa;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #ff416c !important;
    border-bottom: 2px solid #ff416c;
}

h1, h2, h3, h4, p, label, .stMarkdown {
    color: white !important;
}

.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #ccc !important;
}

.rank-badge {
    display: inline-block;
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    text-align: center;
    line-height: 28px;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 8px;
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
    high_threshold = st.slider("🔴 High Risk threshold (%)", 50, 90, 60)
    med_threshold = st.slider("🟡 Medium Risk threshold (%)", 20, 50, 30)
    min_transactions = st.number_input("Min transactions per account", 1, 20, 1)
    st.markdown("---")
    st.markdown("**🤖 Model Info**")
    st.markdown("Algorithm: `XGBoost`")
    st.markdown("ROC-AUC: `94.06%`")
    st.markdown("Trained on: `590,540 transactions`")
    st.markdown("---")
    st.markdown("**Developed by:** Aryan Sinha")
    st.markdown("**SBI Internship, Patna**")

uploaded_file = st.file_uploader(
    "📂 Upload transaction CSV",
    type=["csv"],
    help="Upload a CSV with transaction data."
)

if uploaded_file is None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Expected CSV Format")
    sample = pd.DataFrame({
        "AccountID": ["ACC001", "ACC001", "ACC002", "ACC003"],
        "TransactionAmt": [15000, 23000, 500, 87000],
        "ProductCD": ["W", "W", "H", "C"],
        "card4": ["visa", "visa", "mastercard", "visa"],
        "card6": ["credit", "credit", "debit", "credit"],
        "addr1": [299, 299, 150, 400],
        "dist1": [500, 600, 10, 1200],
        "P_emaildomain": ["anonymous.com", "protonmail.com", "gmail.com", "anonymous.com"],
    })
    st.dataframe(sample, use_container_width=True)
    st.caption("AccountID groups transactions per account. Other columns feed the AI model.")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} transactions")

    if "AccountID" not in df.columns:
        st.warning("⚠️ No 'AccountID' column found — treating each row as a separate account")
        df["AccountID"] = [f"TXN_{i:05d}" for i in range(len(df))]

    with st.spinner("🔍 Hunting mule accounts..."):
        cat_map = {
            "ProductCD": ["W","H","C","S","R"],
            "card4": ["visa","mastercard","american express","discover"],
            "card6": ["credit","debit","debit or credit","charge card"],
            "P_emaildomain": ["gmail.com","yahoo.com","hotmail.com","anonymous.com","protonmail.com"],
            "R_emaildomain": ["gmail.com","yahoo.com","hotmail.com","anonymous.com","protonmail.com"],
        }

        input_data = pd.DataFrame(np.full((len(df), len(feature_names)), -999.0), columns=feature_names)
        for col in df.columns:
            if col in feature_names:
                input_data[col] = pd.to_numeric(df[col], errors="coerce").fillna(-999).values
            elif col in cat_map and col in feature_names:
                input_data[col] = df[col].apply(lambda x: cat_map[col].index(x) if x in cat_map[col] else -999)

        probs = model.predict_proba(input_data)[:, 1]
        df["fraud_prob"] = probs
        df["fraud_pct"] = (probs * 100).round(2)

    agg_dict = {
        "fraud_prob": ["count", "mean", "max"],
    }
    account_stats = df.groupby("AccountID").agg(
        total_transactions=("fraud_prob", "count"),
        avg_risk=("fraud_pct", "mean"),
        max_risk=("fraud_pct", "max"),
        high_risk_txns=("fraud_prob", lambda x: (x > high_threshold/100).sum())
    ).reset_index()

    if "TransactionAmt" in df.columns:
        amt_stats = df.groupby("AccountID")["TransactionAmt"].sum().reset_index()
        amt_stats.columns = ["AccountID", "total_amount"]
        account_stats = account_stats.merge(amt_stats, on="AccountID", how="left")
    else:
        account_stats["total_amount"] = 0

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
            "Risk": ["High Risk", "Medium Risk", "Low Risk"],
            "Count": [high_risk, med_risk, low_risk]
        })
        fig_pie = px.pie(
            pie_data, values="Count", names="Risk",
            color="Risk",
            color_discrete_map={"High Risk": "#ff416c", "Medium Risk": "#f7971e", "Low Risk": "#2ecc71"},
            hole=0.45
        )
        fig_pie.update_layout(
            height=280,
            margin=dict(t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            legend_font_color="white"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### 🏆 Top 5 Most Suspicious")
        for i, row in account_stats.head(5).iterrows():
            st.markdown(f'<span class="rank-badge">{i}</span> **{row["AccountID"]}** — `{row["avg_risk"]}%` {row["Risk Level"]}', unsafe_allow_html=True)

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
            color_continuous_scale=["#2ecc71", "#f7971e", "#ff416c"],
            labels={"fraud_pct": "Fraud Risk %", "index": "Transaction #"}
        )
        fig_bar.add_hline(y=high_threshold, line_dash="dash", line_color="#ff416c", annotation_text="High Risk")
        fig_bar.add_hline(y=med_threshold, line_dash="dash", line_color="#f7971e", annotation_text="Medium Risk")
        fig_bar.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            title_font_color="white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(acct_txns.drop(columns=["fraud_prob"]).reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    csv_out = account_stats.to_csv(index=False)
    st.download_button(
        "📥 Download Full Flagged Report",
        csv_out,
        "mule_hunter_report.csv",
        "text/csv",
        type="primary"
    )
