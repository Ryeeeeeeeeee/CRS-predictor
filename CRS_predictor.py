
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost.pkl')


# Define feature names
feature_names = [
    "Age (years)", "BUN (mg/dl)", "BNP (pg/mL)", "cTNI (ng/ml),
    "Hypertension"
]

# Streamlit user interface
st.title("CRS Predictor")


age = st.number_input("Age (years)：", min_value=1, max_value=120, value=50)


hypertension = st.selectbox("Hypertension (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')



bun = st.number_input("BUN (mg/dl):", min_value=0, max_value=200, value=5)


bnp = st.number_input("BNP (pg/mL):", min_value=10, max_value=50000, value=100)



cTNI = st.number_input("cTNI (ng/ml):", min_value=0, max_value=500, value=10)



# Process inputs and make predictions
feature_values = [age, hypertension, bun, bnp, cTNI]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of CRS. "
            f"The model predicts that your probability of having CRS is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult cardiologists and nephrologists as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of CRS. "
            f"The model predicts that your probability of not having CRS is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart and kidney health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")