import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from groq import Groq

# Load environment variables from .env file
load_dotenv()

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="ClaimWatch AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ ClaimWatch AI – Insurance Fraud Detection Platform")
st.markdown("AI + Machine Learning system for detecting fraudulent insurance claims.")

# -------------------------------------------------
# Groq API Setup
# -------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None
    st.warning("⚠ GROQ API Key not found. AI Investigator will not work.")

# -------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------

menu = st.sidebar.radio(
    "Navigation",
    [
        "📂 Upload Dataset",
        "📊 Dataset Overview",
        "📈 EDA Dashboard",
        "🤖 Train Model",
        "🔎 Fraud Prediction",
        "📑 Bulk Prediction",
        "🧠 AI Fraud Investigator"
    ]
)

# -------------------------------------------------
# Dataset Upload
# -------------------------------------------------

if "df" not in st.session_state:
    st.session_state.df = None

if menu == "📂 Upload Dataset":

    st.subheader("Upload Insurance Claims Dataset")

    uploaded_file = st.file_uploader("Upload CSV file")

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.session_state.df = df

        st.success("Dataset uploaded successfully!")

        st.write(df.head())

# -------------------------------------------------
# Dataset Overview
# -------------------------------------------------

elif menu == "📊 Dataset Overview":

    if st.session_state.df is None:
        st.warning("Upload dataset first.")
    else:

        df = st.session_state.df

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.write("### Preview")
        st.dataframe(df.head())

        st.write("### Column Types")
        st.write(df.dtypes)

        st.write("### Missing Values")
        st.write(df.isnull().sum())

# -------------------------------------------------
# EDA Dashboard
# -------------------------------------------------

elif menu == "📈 EDA Dashboard":

    if st.session_state.df is None:
        st.warning("Upload dataset first.")
    else:

        df = st.session_state.df

        st.subheader("Exploratory Data Analysis")

        if "fraud_reported" in df.columns:

            fig, ax = plt.subplots()

            sns.countplot(x="fraud_reported", data=df, ax=ax)

            ax.set_title("Fraud vs Legitimate Claims")

            st.pyplot(fig)

        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:

            feature = st.selectbox("Select Numeric Feature", numeric_cols)

            fig2, ax2 = plt.subplots()

            sns.histplot(df[feature], kde=True, ax=ax2)

            ax2.set_title(f"{feature} Distribution")

            st.pyplot(fig2)

# -------------------------------------------------
# Train Model
# -------------------------------------------------

elif menu == "🤖 Train Model":

    if st.session_state.df is None:
        st.warning("Upload dataset first.")
    else:

        df = st.session_state.df.copy()

        st.subheader("Train Fraud Detection Model")

        if "fraud_reported" not in df.columns:
            st.error("Dataset must contain 'fraud_reported' column.")
        else:

            encoders = {}

            for col in df.select_dtypes(include="object").columns:

                le = LabelEncoder()

                df[col] = le.fit_transform(df[col])

                encoders[col] = le

            X = df.drop("fraud_reported", axis=1)
            y = df["fraud_reported"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            st.success(f"Model Accuracy: {round(acc*100,2)}%")

            st.session_state.model = model
            st.session_state.features = X.columns
            st.session_state.encoders = encoders

            st.write("### Confusion Matrix")

            cm = confusion_matrix(y_test, preds)

            fig3, ax3 = plt.subplots()

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)

            st.pyplot(fig3)

            st.write("### Feature Importance")

            importance = model.feature_importances_

            imp_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(imp_df.set_index("Feature"))

# -------------------------------------------------
# Fraud Prediction
# -------------------------------------------------

elif menu == "🔎 Fraud Prediction":

    st.subheader("Single Claim Prediction")

    if "model" not in st.session_state:
        st.warning("Train model first.")
    else:

        inputs = []

        for feature in st.session_state.features:

            val = st.number_input(feature, value=0.0)

            inputs.append(val)

        if st.button("Predict Fraud"):

            arr = np.array(inputs).reshape(1, -1)

            model = st.session_state.model

            pred = model.predict(arr)[0]

            prob = model.predict_proba(arr)[0][1]

            st.metric("Fraud Probability", f"{round(prob*100,2)}%")

            if pred == 1:
                st.error("⚠ Fraudulent Claim Detected")
            else:
                st.success("✅ Legitimate Claim")

# -------------------------------------------------
# Bulk Prediction
# -------------------------------------------------

elif menu == "📑 Bulk Prediction":

    st.subheader("Bulk Fraud Detection")

    if "model" not in st.session_state:
        st.warning("Train model first.")
    else:

        file = st.file_uploader("Upload Claims CSV")

        if file:

            data = pd.read_csv(file)

            model = st.session_state.model

            preds = model.predict(data)

            probs = model.predict_proba(data)[:,1]

            data["Fraud_Prediction"] = preds
            data["Fraud_Probability"] = probs

            st.dataframe(data.head())

            st.download_button(
                "Download Results",
                data.to_csv(index=False),
                file_name="fraud_predictions.csv"
            )

# -------------------------------------------------
# AI Fraud Investigator
# -------------------------------------------------

elif menu == "🧠 AI Fraud Investigator":

    st.subheader("AI Insurance Fraud Investigator")

    if client is None:
        st.error("Groq API key missing.")
    else:

        claim_text = st.text_area(
            "Enter Claim Description",
            height=200
        )

        if st.button("Analyze Claim"):

            if claim_text:

                prompt = f"""
You are an expert insurance fraud investigator.

Analyze the claim below and identify fraud risks.

Claim:
{claim_text}

Return:
1. Fraud Risk Level
2. Suspicious Indicators
3. Investigation Recommendation
"""

                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role":"user","content":prompt}]
                )

                st.write(response.choices[0].message.content)

            else:
                st.warning("Enter claim text.")

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.write("ClaimWatch AI")
st.sidebar.write("AI-Powered Fraud Detection")