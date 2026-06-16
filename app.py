import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Prediksi Risiko Diabetes", page_icon="🩸", layout="centered")

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

st.markdown("<h2 style='text-align: center; color: #1976D2;'>🩺 Sistem Prediksi Risiko Diabetes</h2>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Umur (Tahun)", min_value=0.0, max_value=120.0, value=30.0)
    gender = st.selectbox("Jenis Kelamin", le_gender.classes_)
    smoking_history = st.selectbox("Riwayat Merokok", le_smoking.classes_)
    bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=80.0, value=25.0)

with col2:
    hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    heart_disease = st.selectbox("Penyakit Jantung", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    hba1c_level = st.number_input("Level HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5)
    blood_glucose_level = st.number_input("Level Glukosa Darah", min_value=50.0, max_value=300.0, value=100.0)

if st.button("🔍 Analisis Risiko", type="primary", use_container_width=True):
    gender_enc = le_gender.transform([gender])[0]
    smoking_enc = le_smoking.transform([smoking_history])[0]
    
    input_data = pd.DataFrame([[
        age, hypertension, heart_disease, bmi, hba1c_level, blood_glucose_level, gender_enc, smoking_enc
    ]], columns=['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender_encoded', 'smoking_history_encoded'])
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("### 🚨 Hasil Analisis: BERISIKO TINGGI DIABETES")
    else:
        st.success("### ✅ Hasil Analisis: RISIKO RENDAH (SEHAT)")