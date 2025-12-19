import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/best_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first by running Final.py")
        return None

model = load_model()

# Title and description
st.title("🏥 Stroke Risk Prediction System")
st.markdown("""
This application predicts the risk of stroke based on various health and demographic factors.
Fill in the information below to get a prediction.
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    patient_id = st.text_input("Patient ID", value="", placeholder="e.g., P001")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])

with col2:
    st.subheader("Health & Lifestyle")
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Predict button
if st.button("🔍 Predict Stroke Risk", type="primary"):
    if model is None:
        st.error("Cannot make prediction. Model not loaded.")
    else:
        # Create input dataframe
        input_data = pd.DataFrame({
            'id': [patient_id if patient_id else "Unknown"],
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status]
        })
        
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            if patient_id:
                st.info(f"**Patient ID:** {patient_id}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "High Risk" if prediction == 1 else "Low Risk")
            
            with col2:
                st.metric("Stroke Probability", f"{prediction_proba[1]:.2%}")
            
            with col3:
                st.metric("No Stroke Probability", f"{prediction_proba[0]:.2%}")
            
            # Risk level indicator
            risk_level = prediction_proba[1]
            if risk_level < 0.3:
                st.success("✅ Low risk of stroke. Continue maintaining a healthy lifestyle.")
            elif risk_level < 0.7:
                st.warning("⚠️ Moderate risk of stroke. Consider consulting with a healthcare provider.")
            else:
                st.error("🚨 High risk of stroke. Please consult with a healthcare provider immediately.")
            
            # Display probability bar chart
            st.markdown("### Probability Distribution")
            prob_df = pd.DataFrame({
                'Class': ['No Stroke', 'Stroke'],
                'Probability': [prediction_proba[0], prediction_proba[1]]
            })
            st.bar_chart(prob_df.set_index('Class'))
            
            # SHAP explanation (optional - only if SHAP is available)
            try:
                st.markdown("---")
                st.subheader("Feature Importance (SHAP)")
                
                # Get preprocessed data
                preprocessor = model.named_steps['preprocess']
                model_step = model.named_steps['model']
                X_pre = preprocessor.transform(input_data)
                
                # Create explainer
                explainer = shap.TreeExplainer(model_step)
                shap_values = explainer.shap_values(X_pre)
                
                # Plot SHAP waterfall
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value,
                        data=X_pre[0]
                    ),
                    show=False
                )
                st.pyplot(fig)
                
            except Exception as e:
                st.info("SHAP explanation not available for this prediction.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure all fields are filled correctly and the model is properly trained.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Stroke Risk Prediction System**
    
    This system uses machine learning (XGBoost) to predict stroke risk based on:
    - Demographics (age, gender)
    - Medical history (hypertension, heart disease)
    - Lifestyle factors (smoking, BMI)
    - Other health indicators
    
    **Note:** This is a prediction tool and should not replace professional medical advice.
    """)
    
    st.markdown("---")
    st.header("📊 Model Information")
    if model is not None:
        st.success("✅ Model loaded successfully")
        st.info("Model: XGBoost Classifier")
    else:
        st.error("❌ Model not loaded")
    
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** 
    This tool is for educational and informational purposes only. 
    Always consult healthcare professionals for medical decisions.
    """)