import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Prediksi Risiko Stroke",
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
st.title("🏥 Aplikasi Prediksi Risiko Stroke")
st.markdown("""
Aplikasi ini memprediksi risiko penyakit stroke berdasarkan berbagai informasi kesehatan dan faktor demogarfi.
Isi informasi dibawah untuk mendapatkan prediksi.
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Infomasi Pribadi")
    gender = st.selectbox("Jenis Kelamin", ["Laki-Laki", "Perempuan"])
    age = st.number_input("Umur", min_value=0, max_value=120, value=45)
    hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    heart_disease = st.selectbox("Penyakit Jantung", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    ever_married = st.selectbox("Status Menikah", ["Ya", "Tidak"])

with col2:
    st.subheader("Kesehatan dan Gaya Hidup")
    work_type = st.selectbox("Tipe Pekerjaan", ["Pribadi", "WiraSwasta", "Pemerintahan", "Anak-Anak", "Belum Pernah Bekerja"])
    residence_type = st.selectbox("Tipe Tempat Tinggal", ["Perkotaan", "Pedesaan"])
    avg_glucose_level = st.number_input("Rata-Rata Kadar Gula Darah", min_value=0.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    smoking_status = st.selectbox("Status Merokok", ["pernah merokok", "tidak pernah merokok", "merokok", "Unknown"])

# Predict button
if st.button("🔍 Prediksi Risiko", type="primary"):
    if model is None:
        st.error("Tidak bisa membuat prediksi. Model belum diload.")
    else:
        # Create input dataframe
        input_data = pd.DataFrame({
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
            st.subheader("Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediksi", "Risiko Tinggi" if prediction == 1 else "Risiko Rendah")
            
            with col2:
                st.metric("Probabilitas Stroke", f"{prediction_proba[1]:.2%}")
            
            with col3:
                st.metric("tidak Ada Probabilitas Stroke", f"{prediction_proba[0]:.2%}")
            
            # Risk level indicator
            risk_level = prediction_proba[1]
            if risk_level < 0.3:
                st.success("✅ Resiko Terkena Stroke Rendah")
            elif risk_level < 0.7:
                st.warning("⚠️ Risiko Terkena Stroke Sedang")
            else:
                st.error("🚨 Risiko Terkena Stroke Tinggi")
            
            # Display probability bar chart
            st.markdown("### Distribusi Probabilitas")
            prob_df = pd.DataFrame({
                'Class': ['Tidak Stroke', 'Stroke'],
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
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    **Aplikasi Prediksi Risiko Stroke**
    
    Sistem ini menggunakan model machine learning dengan (XGBoost) untuk memprediksi risiko stroke berdasarkan:
    - Demografi (age, gender)
    - Riwayat Penyakit (hypertension, heart disease)
    - Gaya Hidup (smoking, BMI)
    - Indikasi Penyakit Lainnya
    
    **Catatan:** Sistem ini merupakan aplikasi prediksi dan tidak bisa digunakan untuk menggantikan diagnosa sesungguhnya.
    """)
    
    st.markdown("---")
    st.header("📊 Informasi Model")
    if model is not None:
        st.success("✅ Model Sudah Diterapkan")
        st.info("Model: XGBoost Classifier")
    else:
        st.error("❌ Model Belum Diterapkan")
    
    st.markdown("---")
    st.markdown("""
    **Nama Kelompok:**
    1. Devinka Marta Legawa (23081010142)
    2. Azzahra Asti Khairunnissa (23081010157)
    3. Dinda Ayu Puspaningrum Marhono (23081010175) 
    """)