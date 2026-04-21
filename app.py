import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))
    mean_values = joblib.load(os.path.join(BASE_DIR, "mean_values.pkl"))
    return model, scaler, columns, mean_values

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "Customer Churn.csv"))

model, scaler, columns, mean_values = load_model()
df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Churn Dashboard")
page = st.sidebar.radio("Navigation", ["Dashboard", "Prediction", "Insights"])

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.title("Customer Analytics")

    col1, col2, col3 = st.columns(3)

    churn_rate = (df["Churn"] == "Yes").mean()
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{churn_rate:.2%}")
    col3.metric("Avg Monthly Charges", f"{df['MonthlyCharges'].mean():.2f}")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            df.copy(),
            x="tenure",
            color="Churn",
            title="Tenure vs Churn",
            template="simple_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            df.groupby(["Contract", "Churn"]).size().reset_index(name="Count"),
            x="Contract",
            y="Count",
            color="Churn",
            barmode="group",
            template="simple_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.box(
            df.copy(),
            x="Churn",
            y="MonthlyCharges",
            title="Monthly Charges vs Churn",
            template="simple_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.pie(
            df.copy(),
            names="PaymentMethod",
            title="Payment Method Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.title("Churn Prediction")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure", 0, 72, 12)
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
        )

    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    payment_map = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer": 2,
        "Credit card": 3
    }

    if st.button("Predict"):

        input_dict = mean_values.to_dict()

        input_dict["tenure"] = tenure
        input_dict["MonthlyCharges"] = monthly
        input_dict["TotalCharges"] = monthly * max(tenure, 1)
        input_dict["Contract"] = contract_map[contract]
        input_dict["PaymentMethod"] = payment_map[payment]

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[columns]

        scaled = scaler.transform(input_df)
        prob = model.predict_proba(scaled)[0][1]

        # -------- Gauge --------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Churn Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0066FF"},
                "steps": [
                    {"range": [0, 40], "color": "#D1FAE5"},
                    {"range": [40, 70], "color": "#FEF3C7"},
                    {"range": [70, 100], "color": "#FEE2E2"},
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        if prob > 0.7:
            st.error(f"High Risk ({prob:.2f})")
        elif prob > 0.4:
            st.warning(f"Medium Risk ({prob:.2f})")
        else:
            st.success(f"Low Risk ({prob:.2f})")

# ---------------- INSIGHTS ----------------
elif page == "Insights":

    st.title("Model Insights")

    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=columns)
        top = importance.sort_values(ascending=False).head(10)

        fig = px.bar(
            top,
            orientation="h",
            title="Top Features Affecting Churn",
            template="simple_white"
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Customer Churn System | Final Project")