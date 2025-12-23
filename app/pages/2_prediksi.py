import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================
# Load model & data
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model/best_model.pkl")

@st.cache_resource
def load_test_data():
    return joblib.load("model/X_test.pkl"), joblib.load("model/y_test.pkl")

model = load_model()
X_test, y_test = load_test_data()

st.title("Halaman Prediksi")

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Prediksi Data CSV", "Prediksi Input Manual", "Evaluasi Model"])

# =========================
# TAB 1 : Prediksi CSV
# =========================
with tab1:
    st.subheader("Prediksi dari Data CSV")

    if "uploaded_data" not in st.session_state:
        st.warning("Belum ada data CSV yang diupload. Silakan upload di halaman Upload CSV.")
    else:
        df = st.session_state["uploaded_data"]

        if st.button("Prediksi Data CSV"):
            preds = model.predict(df)
            probas = model.predict_proba(df)[:, 1]

            df_result = df.copy()
            df_result["prediksi"] = np.where(preds == 1, "Stroke", "Tidak Stroke")
            df_result["probabilitas_stroke"] = probas

            st.success("Prediksi selesai")
            st.dataframe(df_result, use_container_width=True)

            st.session_state["has_prediction"] = True
            st.session_state["prediction_source"] = "csv"
            st.session_state["prediction_result"] = df_result

# =========================
# TAB 2 : Prediksi Input Manual
# =========================
with tab2:
    st.subheader("Input Data Pasien")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Infomasi Pribadi")
        gender = st.selectbox("Jenis Kelamin", ["Laki-Laki", "Perempuan"])
        age = st.number_input("Usia", min_value=0, max_value=120, value=45)
        hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        heart_disease = st.selectbox("Penyakit Jantung", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        ever_married = st.selectbox("Menikah", ["Ya", "Tidak"])

    with col2:
        st.subheader("Kesehatan dan Gaya Hidup")
        work_type = st.selectbox("Tipe Pekerjaan", ["Pribadi", "WiraSwasta", "Pemerintahan", "Anak-Anak", "Belum Pernah Bekerja"])
        residence_type = st.selectbox("Tempat Tinggal", ["Perkotaan", "Pedesaan"])
        avg_glucose_level = st.number_input("Glukosa", min_value=0.0, max_value=300.0, value=100.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
        smoking_status = st.selectbox("Merokok", ["pernah merokok", "tidak pernah merokok", "merokok"])

    if st.button("Prediksi"):
        input_df = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status
        }])

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.success(f"**Hasil Prediksi:** {'Stroke' if pred==1 else 'Tidak Stroke'}")
        st.metric("Probabilitas Stroke", f"{proba:.2%}")

        df_result = input_df.copy()
        df_result["prediksi"] = "Stroke" if pred else "Tidak Stroke"
        df_result["probabilitas_stroke"] = proba

        st.session_state["has_prediction"] = True
        st.session_state["prediction_source"] = "manual"
        st.session_state["prediction_result"] = df_result

# =========================
# TAB 3 : Evaluasi Model
# =========================
with tab3:
    st.subheader("Confusion Matrix & Performance")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Performance Test")
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    roc = roc_auc_score(y_test, y_proba) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.2f}%")
    col2.metric("Precision", f"{prec:.2f}%")
    col3.metric("Recall", f"{rec:.2f}%")

    col1b, col2b, col3b = st.columns(3)
    col1b.metric("F1-Score", f"{f1:.2f}%")
    col2b.metric("ROC-AUC", f"{roc:.2f}%")
    col3b.empty()

    st.info("""
    **Penjelasan Hasil:**
    - **Accuracy**: ketepatan keseluruhan model dalam mengklasifikasikan data.
    - **Precision**: tingkat ketepatan model saat memprediksi pasien mengalami stroke.
    - **Recall**: kemampuan model dalam mendeteksi seluruh kasus stroke yang sebenarnya.
    - **F1-Score**: keseimbangan antara precision dan recall.
    - **ROC-AUC**: kemampuan model dalam membedakan antara pasien stroke dan tidak stroke pada berbagai nilai threshold.
    """)