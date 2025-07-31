import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model & scaler
with open("heart_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# Streamlit App Layout
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Heart Disease Prediction App")
st.markdown("Enter the following patient details to predict the presence of heart disease.")

# Input Fields
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=120)
chol = st.number_input("Serum Cholesterol in mg/dl (chol)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved (thalach)", value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"❌ **Heart Disease** with a probability of {probability:.2f}.")
    else:
        st.success(f"✅ **No Heart Disease** with a probability of {1 - probability:.2f}.")

# Feature Description (Centered)
with st.expander("📘 Click to view feature descriptions"):
    st.markdown("""
    <div style='text-align: center;'>
        <table style="margin: 0 auto; border-collapse: collapse; width: 90%;">
            <thead>
                <tr>
                    <th style="padding: 8px;">Feature Name</th>
                    <th style="padding: 8px;">Description</th>
                </tr>
            </thead>
            <tbody>
                <tr><td><strong>age</strong></td><td>Age of the person</td></tr>
                <tr><td><strong>sex</strong></td><td>Gender (1 = male, 0 = female)</td></tr>
                <tr><td><strong>cp</strong></td><td>Chest pain type (0–3)</td></tr>
                <tr><td><strong>trestbps</strong></td><td>Resting blood pressure</td></tr>
                <tr><td><strong>chol</strong></td><td>Serum cholesterol in mg/dl</td></tr>
                <tr><td><strong>fbs</strong></td><td>Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)</td></tr>
                <tr><td><strong>restecg</strong></td><td>Resting ECG results (0–2)</td></tr>
                <tr><td><strong>thalach</strong></td><td>Maximum heart rate achieved</td></tr>
                <tr><td><strong>exang</strong></td><td>Exercise induced angina (1 = yes; 0 = no)</td></tr>
                <tr><td><strong>oldpeak</strong></td><td>ST depression induced by exercise</td></tr>
                <tr><td><strong>slope</strong></td><td>Slope of the peak exercise ST segment (0–2)</td></tr>
                <tr><td><strong>ca</strong></td><td>Number of major vessels (0–3) colored by fluoroscopy</td></tr>
                <tr><td><strong>thal</strong></td><td>Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)</td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# Sidebar - Visualizations
st.sidebar.title("📊 Visualizations")

# Target distribution
if st.sidebar.checkbox("Show Target Distribution"):
    st.subheader("🎯 Target Variable Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='target', palette='Set2', ax=ax1)
    ax1.set_xticklabels(['No Disease (0)', 'Disease (1)'])
    st.pyplot(fig1)

# Correlation Heatmap
if st.sidebar.checkbox("Show Correlation Heatmap"):
    st.subheader("🔥 Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

# Feature Distribution (Dropdown Below Description)
selected_feature = st.sidebar.selectbox(
    "📈 Select a feature to view its distribution",
    options=['', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
)

if selected_feature:
    st.subheader(f"📊 Distribution of `{selected_feature}`")
    fig3, ax3 = plt.subplots()
    if df[selected_feature].nunique() <= 10:
        sns.countplot(x=selected_feature, data=df, palette='Set3', ax=ax3)
    else:
        sns.histplot(df[selected_feature], kde=True, color='teal', ax=ax3)
    st.pyplot(fig3)