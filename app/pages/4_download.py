import streamlit as st
import pandas as pd

st.title("Download Hasil Prediksi")

# =========================
# CEK APAKAH ADA HASIL
# =========================
if "prediction_result" not in st.session_state:
    st.warning("Belum ada hasil prediksi. Silakan lakukan prediksi terlebih dahulu.")
    st.stop()

df_result = st.session_state["prediction_result"]
source = st.session_state.get("prediction_source", "unknown")

# =========================
# INFO SUMBER DATA
# =========================
if source == "csv":
    st.info("Sumber prediksi: Data CSV")
elif source == "manual":
    st.info("Sumber prediksi: Input Manual")
else:
    st.info("Sumber prediksi tidak diketahui")

# =========================
# PREVIEW
# =========================
st.subheader("Preview Hasil Prediksi")
st.dataframe(df_result.head(), use_container_width=True)

st.write(f"Jumlah data: **{len(df_result)} baris**")

# =========================
# DOWNLOAD
# =========================
csv = df_result.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Hasil Prediksi (CSV)",
    data=csv,
    file_name="hasil_prediksi.csv",
    mime="text/csv"
)
