import streamlit as st
import pandas as pd

st.title("**Upload & Preview Data**")

st.markdown("""
Halaman ini digunakan untuk mengunggah data dalam format CSV
yang berisi fitur numerik dan kategorikal untuk dilakukan
preview dan prediksi menggunakan model.
""")

# =========================
# Upload CSV
# =========================
uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=";")
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
        st.stop()
    
    st.session_state["uploaded_data"] = df
    st.session_state["has_upload"] = True

    # =========================
    # PREVIEW
    # =========================
    st.subheader("Preview Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Informasi Data")
    col1, col2 = st.columns(2)
    col1.write("Jumlah baris:")
    col1.metric("", df.shape[0])
    col2.write("Jumlah kolom:")
    col2.metric("", df.shape[1])

    st.write("Tipe data:")
    st.dataframe(df.dtypes.astype(str), use_container_width=True)

    # =========================
    # VALIDASI KOLOM
    # =========================
    required_columns = [
        "gender", "age", "hypertension", "heart_disease",
        "ever_married", "work_type", "Residence_type",
        "avg_glucose_level", "bmi", "smoking_status"
    ]

    missing_cols = set(required_columns) - set(df.columns)

    if missing_cols:
        st.error(
            "Struktur kolom tidak sesuai.\n\n"
            f"Kolom yang hilang: {', '.join(missing_cols)}"
        )
        st.stop()