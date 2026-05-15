# Gender -> 1 Female 0 Male
# Chrun -> 1 Yes 0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of the X -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="centered"
)

def load_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Sidebar
st.sidebar.title("📌 About Project")
st.sidebar.write(
    "This app predicts whether a customer is likely to churn using a trained machine learning model."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Inputs")
st.sidebar.write("Age, Gender, Tenure, Monthly Charges")

st.sidebar.markdown("---")
st.sidebar.caption("Built using Streamlit + Scikit-learn")

# Header
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">Machine Learning App</div>
        <h1>📊 Customer Churn Prediction</h1>
        <p>Enter customer details below and predict whether the customer is likely to churn.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Info cards
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        """
        <div class="metric-card">
            <h3>👤 Customer</h3>
            <p>Demographic details</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_b:
    st.markdown(
        """
        <div class="metric-card">
            <h3>📅 Tenure</h3>
            <p>Customer duration</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_c:
    st.markdown(
        """
        <div class="metric-card">
            <h3>💳 Charges</h3>
            <p>Monthly billing info</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('<div class="section-title">Enter Customer Details</div>', unsafe_allow_html=True)

# Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Enter age", min_value=10, max_value=100, value=30)
    tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)

with col2:
    monthlycharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150)
    gender = st.selectbox("Enter the Gender", ["Male", "Female"])

predictbutton = st.button("🚀 Predict Churn", use_container_width=True)

if predictbutton:

    gender_selected = 1 if gender == "Female" else 0

    X = [age, gender_selected, tenure, monthlycharge]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    if predicted == "Yes":
        st.markdown(
            """
            <div class="result-card result-danger">
                <h2>⚠️ Predicted: Yes</h2>
                <p>This customer is likely to churn. Retention action may be needed.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.balloons()
        st.markdown(
            """
            <div class="result-card result-success">
                <h2>✅ Predicted: No</h2>
                <p>This customer is not likely to churn based on the entered details.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.markdown(
        """
        <div class="waiting-card">
            Please enter the values and click the Predict Churn button.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="footer">
        Built with Streamlit | Customer Churn Prediction ML Project
    </div>
    """,
    unsafe_allow_html=True
)