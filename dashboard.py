import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("Customer Churn Prediction Dashboard")

# Load the raw dataset and the model comparison table
df = pd.read_csv("Data Folder/telco_churn.csv")
comparison = pd.read_csv("Data Folder/model_comparison.csv")

tab1, tab2, tab3 = st.tabs(["Data Insights", "Model Comparison", "Predict Churn"])

# ---------- Tab 1: Data insights ----------
with tab1:
    total = len(df)
    churned = (df["Churn"] == "Yes").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Churn Rate", f"{churned / total:.1%}")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(df, x="Contract", color="Churn", barmode="group",
                           title="Churn by Contract Type")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.histogram(df, x="tenure", color="Churn", nbins=30,
                           title="Churn by Tenure (months)")
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fig = px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                     title="Monthly Charges vs Churn")
        st.plotly_chart(fig, use_container_width=True)
    with col_d:
        fig = px.histogram(df, x="InternetService", color="Churn", barmode="group",
                           title="Churn by Internet Service")
        st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 2: Model comparison ----------
with tab2:
    st.subheader("Model Performance")
    st.dataframe(comparison, use_container_width=True)
    melted = comparison.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(melted, x="Metric", y="Score", color="Model", barmode="group",
                 title="Metrics by Model")
    st.plotly_chart(fig, use_container_width=True)
    best = comparison.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    st.success(f"Best model by ROC-AUC: {best}")

# ---------- Tab 3: Live prediction using the Flask API ----------
with tab3:
    st.subheader("Predict churn for a customer")
    st.caption("Make sure the Flask API (python app.py) is running.")

    c1, c2 = st.columns(2)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 3)
        monthly = st.number_input("Monthly Charges", 18.0, 120.0, 95.5)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    with c2:
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        security = st.selectbox("Online Security", ["Yes", "No"])
        support = st.selectbox("Tech Support", ["Yes", "No"])

    if st.button("Predict"):
        customer = {
            "tenure": tenure, "MonthlyCharges": monthly, "Contract": contract,
            "InternetService": internet, "PaymentMethod": payment,
            "PaperlessBilling": paperless, "SeniorCitizen": 0,
            "PhoneService": "Yes", "OnlineSecurity": security, "TechSupport": support,
        }
        try:
            result = requests.post("http://127.0.0.1:5000/predict", json=customer).json()
            prob = result["churn_probability"]
            if result["churn_prediction"] == 1:
                st.error(f"{result['result']} (probability: {prob:.1%})")
            else:
                st.success(f"{result['result']} (probability: {prob:.1%})")
            st.progress(prob)
        except Exception:
            st.warning("Could not reach the API. Start it with: python app.py")