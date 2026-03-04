import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from generate_data import generate_churn_dataset

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="B2B SaaS Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Load & train model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = generate_churn_dataset(1000)

    le_contract = LabelEncoder()
    le_industry = LabelEncoder()
    df["contract_encoded"] = le_contract.fit_transform(df["contract_type"])
    df["industry_encoded"] = le_industry.fit_transform(df["industry"])

    features = [
        "tenure_months", "monthly_spend", "num_users",
        "support_tickets_last_90d", "product_modules_used",
        "last_login_days_ago", "nps_score", "contract_encoded",
        "onboarding_completed", "has_dedicated_csm", "industry_encoded"
    ]

    X = df[features]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    return model, df, features, le_contract, le_industry, acc

model, df, features, le_contract, le_industry, accuracy = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
st.sidebar.title("B2B SaaS\nChurn Predictor")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by [Shashank Tripathi](https://www.linkedin.com/in/shashank-tripathi-a46679190/)")
st.sidebar.markdown("Inspired by 2 years managing enterprise SaaS clients at **Shipsy** and **ZingHR**.")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Predict Churn Risk", "📈 Feature Insights"])

# ── Dashboard ─────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("📊 B2B SaaS Client Health Dashboard")
    st.caption("Understanding what drives churn across an enterprise SaaS portfolio")

    col1, col2, col3, col4 = st.columns(4)
    churn_rate = df["churned"].mean()
    avg_tenure = df["tenure_months"].mean()
    avg_spend = df["monthly_spend"].mean()
    avg_tickets = df["support_tickets_last_90d"].mean()

    col1.metric("Total Clients", f"{len(df):,}")
    col2.metric("Churn Rate", f"{churn_rate:.1%}", delta=f"{churn_rate - 0.25:.1%} vs benchmark")
    col3.metric("Avg Tenure", f"{avg_tenure:.0f} months")
    col4.metric("Model Accuracy", f"{accuracy:.1%}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Churn by Contract Type")
        churn_contract = df.groupby("contract_type")["churned"].mean().reset_index()
        churn_contract.columns = ["Contract Type", "Churn Rate"]
        fig1 = px.bar(
            churn_contract, x="Contract Type", y="Churn Rate",
            color="Churn Rate", color_continuous_scale="RdYlGn_r",
            text_auto=".1%"
        )
        fig1.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.subheader("Churn by Industry")
        churn_industry = df.groupby("industry")["churned"].mean().reset_index()
        churn_industry.columns = ["Industry", "Churn Rate"]
        churn_industry = churn_industry.sort_values("Churn Rate", ascending=True)
        fig2 = px.bar(
            churn_industry, x="Churn Rate", y="Industry",
            orientation="h", color="Churn Rate",
            color_continuous_scale="RdYlGn_r", text_auto=".1%"
        )
        fig2.update_layout(showlegend=False, xaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Tenure vs Churn Risk")
        fig3 = px.histogram(
            df, x="tenure_months", color=df["churned"].map({0: "Retained", 1: "Churned"}),
            barmode="overlay", opacity=0.7,
            color_discrete_map={"Retained": "#00cc96", "Churned": "#ff4b4b"},
            labels={"color": "Status"}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.subheader("NPS Score vs Churn")
        nps_churn = df.groupby("nps_score")["churned"].mean().reset_index()
        fig4 = px.line(
            nps_churn, x="nps_score", y="churned",
            markers=True, color_discrete_sequence=["#636efa"]
        )
        fig4.update_layout(yaxis_tickformat=".0%", xaxis_title="NPS Score", yaxis_title="Churn Rate")
        st.plotly_chart(fig4, use_container_width=True)

# ── Predict ───────────────────────────────────────────────────────────────────
elif page == "🔮 Predict Churn Risk":
    st.title("🔮 Predict Client Churn Risk")
    st.caption("Enter a client's profile to get their churn probability and risk level")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account Info")
        tenure = st.slider("Tenure (months)", 1, 60, 12)
        monthly_spend = st.number_input("Monthly Spend (€)", 500, 10000, 2000, step=100)
        num_users = st.slider("Number of Users", 5, 200, 30)
        contract_type = st.selectbox("Contract Type", ["Monthly", "Annual", "Multi-year"])
        industry = st.selectbox("Industry", ["Logistics", "HR-Tech", "Fintech", "Healthcare", "Retail"])

    with col2:
        st.subheader("Engagement Signals")
        support_tickets = st.slider("Support Tickets (last 90 days)", 0, 20, 3)
        modules_used = st.slider("Product Modules Used", 1, 8, 4)
        last_login = st.slider("Last Login (days ago)", 1, 120, 10)
        nps = st.slider("NPS Score", 0, 10, 7)

    with col3:
        st.subheader("Success Factors")
        onboarding = st.radio("Onboarding Completed?", ["Yes", "No"])
        csm = st.radio("Has Dedicated CSM?", ["Yes", "No"])

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary")

    if predict_btn:
        contract_enc = le_contract.transform([contract_type])[0]
        industry_enc = le_industry.transform([industry])[0]
        onboarding_val = 1 if onboarding == "Yes" else 0
        csm_val = 1 if csm == "Yes" else 0

        input_data = np.array([[
            tenure, monthly_spend, num_users, support_tickets,
            modules_used, last_login, nps, contract_enc,
            onboarding_val, csm_val, industry_enc
        ]])

        churn_prob = model.predict_proba(input_data)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            if churn_prob >= 0.7:
                risk_label = "🔴 HIGH RISK"
                color = "#ff4b4b"
            elif churn_prob >= 0.4:
                risk_label = "🟠 MEDIUM RISK"
                color = "#ffa500"
            else:
                risk_label = "🟢 LOW RISK"
                color = "#00cc96"

            st.markdown(f"<h2 style='color:{color}'>{risk_label}</h2>", unsafe_allow_html=True)
            st.metric("Churn Probability", f"{churn_prob:.1%}")

        with col_res2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Churn Risk %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#e8f5e9"},
                        {"range": [40, 70], "color": "#fff3e0"},
                        {"range": [70, 100], "color": "#ffebee"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 70}
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=40, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_res3:
            st.subheader("Recommended Actions")
            if churn_prob >= 0.7:
                st.error("Immediate intervention needed")
                st.markdown("""
                - 📞 Schedule executive business review
                - 🎁 Offer contract extension incentive
                - 🔧 Assign dedicated CSM immediately
                - 📋 Conduct satisfaction survey
                """)
            elif churn_prob >= 0.4:
                st.warning("Monitor closely")
                st.markdown("""
                - 📧 Send personalised check-in email
                - 📊 Share usage insights report
                - 🎓 Offer additional training session
                - 💡 Highlight unused product features
                """)
            else:
                st.success("Account is healthy")
                st.markdown("""
                - ⭐ Identify expansion opportunity
                - 🤝 Request case study / referral
                - 📈 Propose upsell conversation
                - 🔄 Schedule quarterly business review
                """)

# ── Feature Insights ──────────────────────────────────────────────────────────
elif page == "📈 Feature Insights":
    st.title("📈 What Drives Churn?")
    st.caption("Feature importance from the Random Forest model — understanding which signals matter most")

    feature_labels = [
        "Tenure (months)", "Monthly Spend", "Num Users",
        "Support Tickets", "Modules Used", "Last Login (days)",
        "NPS Score", "Contract Type", "Onboarding Done",
        "Has CSM", "Industry"
    ]

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_labels,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        fi_df, x="Importance", y="Feature",
        orientation="h", color="Importance",
        color_continuous_scale="Blues",
        title="Feature Importance in Churn Prediction"
    )
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Takeaways")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Last Login Days** is typically the strongest signal. Disengaged users churn. Track logins weekly.")
    with col2:
        st.warning("**NPS Score & Support Tickets** are early warning signs. Low NPS + high tickets = intervention needed.")
    with col3:
        st.success("**Tenure & Contract Type** reflect commitment. Annual/multi-year clients churn less. Push annual contracts.")
