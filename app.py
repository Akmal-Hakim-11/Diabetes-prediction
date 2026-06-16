import streamlit as st
import pandas as pd
import pickle

# Konfigurasi Tampilan Halaman
st.set_page_config(page_title="Prediksi Risiko Diabetes", page_icon="🩸", layout="centered")

# Memuat Model dan Encoder
@st.cache_resource
def load_assets():
    with open('model_diabetes.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder_diabetes.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_assets()
le_gender = encoders['gender']
le_smoking = encoders['smoking']

# Kamus Terjemahan untuk Tampilan UI (Indonesia -> Inggris untuk Encoder)
gender_map = {
    "Perempuan": "Female", 
    "Laki-laki": "Male", 
    "Lainnya": "Other"
}

smoking_map = {
    "Tidak Ada Info": "No Info",
    "Tidak Pernah": "never",
    "Mantan Perokok": "former",
    "Perokok Aktif": "current",
    "Bukan Perokok Aktif": "not current",
    "Pernah Merokok": "ever"
}

# Header Dashboard yang Lebih Rapi
st.markdown("<h2 style='text-align: center; color: #1976D2;'>🩺 Sistem Cerdas Deteksi Risiko Diabetes</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Masukkan data rekam medis pasien di bawah ini untuk melihat estimasi risiko diabetes menggunakan algoritma Machine Learning.</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout Form Input
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👤 Data Demografi")
    age = st.number_input("Umur (Tahun)", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
    
    # Menampilkan pilihan dalam Bahasa Indonesia
    gender_indo = st.selectbox("Jenis Kelamin", list(gender_map.keys()))
    
    bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=80.0, value=25.0, step=0.1, help="Berat badan (kg) dibagi kuadrat tinggi badan (m). Normalnya 18.5 - 24.9")
    
    smoking_indo = st.selectbox("Riwayat Merokok", list(smoking_map.keys()))

with col2:
    st.markdown("#### 🩸 Data Klinis")
    hypertension = st.selectbox("Riwayat Darah Tinggi (Hipertensi)", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    heart_disease = st.selectbox("Riwayat Penyakit Jantung", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    hba1c_level = st.number_input("Level HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1, help="Rata-rata gula darah 2-3 bulan terakhir. Normalnya di bawah 5.7%")
    blood_glucose_level = st.number_input("Level Glukosa Darah (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=1.0, help="Gula darah saat tes dilakukan. Normalnya di bawah 140 mg/dL")

st.markdown("---")

# Logika Tombol Prediksi
if st.button("🔍 Analisis Risiko Sekarang", type="primary", use_container_width=True):
    # 1. Kembalikan input UI (Bahasa Indonesia) ke Bahasa Inggris asli
    gender_eng = gender_map[gender_indo]
    smoking_eng = smoking_map[smoking_indo]
    
    # 2. Ubah teks ke angka menggunakan Encoder
    gender_enc = le_gender.transform([gender_eng])[0]
    smoking_enc = le_smoking.transform([smoking_eng])[0]
    
    # 3. Susun data untuk model
    input_data = pd.DataFrame([[
        age, hypertension, heart_disease, bmi, hba1c_level, blood_glucose_level, gender_enc, smoking_enc
    ]], columns=['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender_encoded', 'smoking_history_encoded'])
    
    # 4. Jalankan Prediksi
    prediction = model.predict(input_data)[0]
    
    # 5. Tampilkan Hasil yang mudah dipahami
    if prediction == 1:
        st.error("### 🚨 KESIMPULAN: BERISIKO TINGGI DIABETES")
        st.write("Sistem mendeteksi adanya pola klinis yang mengarah pada **risiko diabetes**. Kami sangat menyarankan agar pasien segera berkonsultasi dengan dokter atau fasilitas kesehatan terdekat untuk pemeriksaan lebih lanjut.")
    else:
        st.success("### ✅ KESIMPULAN: SEHAT (RISIKO RENDAH)")
        st.write("Berdasarkan data yang dimasukkan, pasien memiliki **risiko rendah** terhadap diabetes. Tetap pertahankan gaya hidup sehat, jaga pola makan, dan rutin berolahraga!")
