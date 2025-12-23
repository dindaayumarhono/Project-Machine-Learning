import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# Load model & data
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model/best_model.pkl")

@st.cache_resource
def load_test_data():
    return (joblib.load("model/x_test.pkl"), 
            joblib.load("model/y_test.pkl")
    )

model = load_model()
X_test, y_test = load_test_data()

st.title("Halaman Grafik SHAP")

# =========================
# SHAP
# =========================
try:
    st.subheader("Feature Importance (SHAP)")

    # Ambil preprocessed data
    preprocessor = model.named_steps['preprocess']
    model_step = model.named_steps['model']

    # Transform 
    X_test_pre = preprocessor.transform(X_test)

    # Buat explainer
    explainer = shap.TreeExplainer(model_step)
    shap_values = explainer.shap_values(X_test_pre)
        
    # Ambil nama fitur setelah preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # Hitung SHAP untuk data test
    shap_values_test = explainer.shap_values(X_test_pre)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values_test,
        X_test_pre,
        feature_names=feature_names,
        show=False
    )
    st.pyplot(fig)

except Exception as e:
    st.info("SHAP explanation not available for this prediction.")


st.markdown("---")
st.subheader("Top Feature Berdasarkan SHAP")

# Jika klasifikasi biner → shap_values biasanya list
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # kelas positif (Stroke)
else:
    shap_vals = shap_values

# Hitung mean absolute SHAP
mean_shap = np.abs(shap_vals).mean(axis=0)

top_features = pd.DataFrame({
    "Feature": feature_names,
    "Mean |SHAP|": mean_shap
}).sort_values("Mean |SHAP|", ascending=False)

st.bar_chart(
    top_features.set_index("Feature").head(10)
)

# Pastikan X_test berbentuk DataFrame
if not hasattr(X_test, "iloc"):
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
else:
    X_test_df = X_test

# Ambil 1 contoh data SETELAH preprocessing
sample_idx = 0
sample_shap = shap_vals[sample_idx]
sample_feature_values = X_test_pre[sample_idx]

# Urutkan kontribusi SHAP
sorted_idx = np.argsort(sample_shap)

# Fitur paling MENURUNKAN risiko (biru)
top_negative = [feature_names[i] for i in sorted_idx[:3]]

# Fitur paling MENINGKATKAN risiko (merah)
top_positive = [feature_names[i] for i in sorted_idx[-3:]]

st.markdown("---")
st.subheader("Penjelasan Prediksi Berdasarkan SHAP")

st.write(f"""
Berdasarkan visualisasi SHAP di atas:

- Fitur **{', '.join(top_positive)}** ditunjukkan dengan **warna merah**
  dan merupakan faktor yang **paling meningkatkan risiko stroke**.
- Sebaliknya, fitur **{', '.join(top_negative)}** ditunjukkan dengan
  **warna biru** dan berkontribusi dalam **menurunkan risiko stroke**.

Semakin besar jarak suatu fitur dari titik nol pada grafik SHAP,
semakin besar pula pengaruhnya terhadap hasil prediksi model.
""")


