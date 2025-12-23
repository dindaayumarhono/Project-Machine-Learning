import streamlit as st

st.set_page_config(
    page_title="Prediksi Risiko Stroke",
    layout="wide"
)

st.title("Aplikasi Prediksi Risiko Stroke")

st.markdown("""
Aplikasi ini merupakan sistem yang dibangun untuk Prediksi Risiko Stroke.

**Fitur Aplikasi:**
- Informasi program & tujuan
- Prediksi risiko stroke berbasis Machine Learning (XGBoost)
- Evaluasi performa model
- Unduh hasil prediksi
""")

st.markdown("""
**Tentang Program**

Aplikasi ini dibuat untuk membantu memprediksi risiko penyakit stroke
menggunakan Machine Learning (XGBoost) berdasarkan:

- Data demografi
- Riwayat kesehatan
- Gaya hidup

**Catatan Penting**
Aplikasi ini tidak bisa dijadikan pengganti diagnosis medis, melainkan hanya alat bantu analisis.
""")

st.markdown("""
**Dataset yang digunakan untuk pelatihan model**
dataset yang kami gunakan untuk pelatiham model merupakan data sekunder yang didapatkan dari website kaggle dengan judul Stroke Prediction Dataset.
data yang digunakan berjumlah 5110 dengan 11 jenis data yaitu 
1. gender : jenis kelamin
2. age : usia
3. hypertension : hipertensi
4. heart_disease : penyakit jantung
5. ever_married : status menikah
6. work_type: tipe pekerjaan
7 Residence_type: tipe tempat tinggal
8. avg_glucose_level: rata-rata kadar gula darah
9. bmi: indeks massa tubuh
10. smoking_status: status merokok
22. stroke: terkena stroke atau tidak
""")

st.markdown("""
**Tujuan**
- Memberikan prediksi risiko stroke
- Menampilkan performa model
- Menyediakan interpretasi hasil
- Memungkinkan unduhan hasil analisis
""")

st.markdown("""
**Nama Kelompok**
1. Devinka Marta Legawa (23081010142)
2. Azzahra Asti Khairunnissa (23081010157)
3. Dinda Ayu Puspaningrum Marhono (23081010175)
""")
