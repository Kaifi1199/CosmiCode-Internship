# 💓 Heart Disease Prediction App

This Streamlit web app predicts the likelihood of a person having heart disease using a trained machine learning model. The app takes several medical features as input and provides a prediction along with visual insights.

---

## 🚀 Features

- 🔍 Input patient data such as age, cholesterol, chest pain type, etc.
- 🤖 Get a prediction (Heart Disease / No Heart Disease) with probability.
- 📊 Visualize target distribution, correlation heatmap, and individual feature distributions.
- 📘 View detailed descriptions of all features.

---

## 📂 Files

- `app.py`: Main Streamlit application code.
- `Week-6`: Implemented EDA, preprocessing, model training, evaluation, and saved the best model for deployment.
- `heart_model.pkl`: Pre-trained ML model.
- `scaler.pkl`: Standard scaler used for preprocessing.
- `heart.csv`: Dataset used for visualization and analysis.
- `README.md`: Project documentation.

---

## 📦 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Kaifi1199/Week-6.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

---

## 🧠 Model

- The model used is a binary classifier trained on the Heart Disease dataset.
- Features are scaled using `StandardScaler`.
- Model was serialized using `pickle`.

---

## 📝 Feature Descriptions

| Feature Name | Description                                                              |
| ------------ | ------------------------------------------------------------------------ |
| **age**      | Age of the person                                                        |
| **sex**      | Gender (1 = male, 0 = female)                                            |
| **cp**       | Chest pain type (0–3)                                                    |
| **trestbps** | Resting blood pressure                                                   |
| **chol**     | Serum cholesterol in mg/dl                                               |
| **fbs**      | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)                    |
| **restecg**  | Resting ECG results (0–2)                                                |
| **thalach**  | Maximum heart rate achieved                                              |
| **exang**    | Exercise induced angina (1 = yes; 0 = no)                                |
| **oldpeak**  | ST depression induced by exercise                                        |
| **slope**    | Slope of the peak exercise ST segment (0–2)                              |
| **ca**       | Number of major vessels (0–3) colored by fluoroscopy                     |
| **thal**     | Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)        |

---

## 👨‍⚕️ Disclaimer

This project is for **educational purposes only**. It should **not be used for real medical decisions**. Always consult a licensed healthcare provider.
