# 🩺 Prediksi Risiko Penyakit Diabetes Berdasarkan Rekam Medis

---

## 📌 Project Overview
Proyek ini bertujuan memprediksi tingkat risiko seseorang terkena penyakit diabetes 
(Sehat / Berisiko Tinggi) berdasarkan data demografi dan indikator rekam medis klinis 
menggunakan algoritma *machine learning* dengan metodologi **CRISP-DM**.

🔗 **Live Demo:** [Prediksi-Diabetes di Hugging Face](https://huggingface.co/spaces/AkmalHakim21/Prediksi-Diabetes)  
📓 **Notebook:** [Google Colab](https://colab.research.google.com/drive/1rNA4Sju6GJtvPy97PnawQ1S0JbIaxxIf?usp=sharing)

---

## 👥 Tim

| Nama | NIM |
| :--- | :--- |
| Muhammad Akmal Hakim Purawijaya | 2330511085 |
| Mochammad Fattah Maulana Mansur | 2330511083 |

---

## 1. Business Understanding

### Latar Belakang
Penyakit diabetes sering kali terlambat disadari oleh penderitanya karena gejala 
awal yang cenderung samar. Keterlambatan diagnosis ini berisiko memunculkan komplikasi 
serius seperti kerusakan ginjal, gangguan penglihatan, dan penyakit kardiovaskular. 
Deteksi dini secara otomatis melalui sistem kecerdasan buatan sangat diperlukan agar 
pasien dapat segera menyesuaikan gaya hidup atau mencari penanganan medis profesional 
secepat mungkin.

### Problem Statement
- Dapatkah kita memprediksi risiko penyakit diabetes seseorang (0 = Sehat, 1 = Berisiko) 
  berdasarkan data demografi dan indikator kesehatan klinis menggunakan algoritma 
  *machine learning*?

### Goals
- Membangun model klasifikasi yang andal untuk memprediksi risiko penyakit diabetes.
- Mengidentifikasi indikator kesehatan mana (seperti glukosa atau HbA1c) yang paling 
  kuat pengaruhnya terhadap risiko diabetes.
- Men-*deploy* model menjadi aplikasi web interaktif yang mudah diakses pengguna awam.

### Solution Statement
- **Model Utama:** Random Forest Classifier
- **Metrik Evaluasi:** Accuracy Score, Confusion Matrix, Precision, Recall, dan F1-Score

---

## 2. Data Understanding

### Sumber Data
- **Dataset:** [Diabetes Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- **Jumlah data:** 100.000 baris observasi rekam medis
- **Jumlah fitur:** 9 kolom

### Deskripsi Fitur

| Fitur | Tipe | Deskripsi |
| :--- | :--- | :--- |
| `gender` | kategorikal | Jenis kelamin pasien (Female, Male, Other) |
| `age` | numerik | Usia pasien dalam hitungan tahun |
| `hypertension` | biner | Indikator penyakit darah tinggi (0 = Tidak, 1 = Ya) |
| `heart_disease` | biner | Indikator penyakit jantung (0 = Tidak, 1 = Ya) |
| `smoking_history` | kategorikal | Riwayat merokok (never, current, former, dll) |
| `bmi` | numerik | Indeks Massa Tubuh (Body Mass Index) |
| `HbA1c_level` | numerik | Kadar hemoglobin terglikasi rata-rata 2-3 bulan terakhir |
| `blood_glucose_level` | numerik | Kadar glukosa darah saat pengujian |
| `diabetes` | biner | **Target** — 0 (Sehat) / 1 (Berisiko Diabetes) |

### EDA Findings
- **Distribusi Kelas Target:** Dataset bersifat *imbalanced*, dengan rasio sekitar 91% 
  kelas Sehat dan 9% kelas Berisiko Diabetes.
- **Korelasi Medis:** Berdasarkan matriks korelasi, `HbA1c_level` dan 
  `blood_glucose_level` merupakan prediktor paling dominan terhadap risiko diabetes, 
  jauh di atas fitur lainnya seperti BMI dan usia.

---

## 3. Data Preparation
- **Pengecekan Missing Values:** Dataset divalidasi dan tidak ditemukan nilai kosong 
  (*null*) yang dapat mengganggu proses *training*.
- **Label Encoding:** Fitur bertipe teks (`gender` dan `smoking_history`) dikonversi 
  menjadi representasi numerik menggunakan `LabelEncoder`, dan encoder disimpan ke 
  file `encoder_diabetes.pkl` untuk konsistensi saat *deployment*.
- **Penanganan Imbalanced Data:** Ketidakseimbangan kelas ditangani dengan menggunakan 
  parameter `class_weight='balanced'` pada model, sehingga model tidak bias ke kelas 
  mayoritas.
- **Data Splitting:** Dataset dibagi menjadi *Training Set* (80%) dan *Testing Set* 
  (20%) secara acak dengan `random_state` tetap untuk hasil yang dapat direproduksi.

---

## 4. Modeling

### Random Forest Classifier
Model *ensemble* berbasis Decision Tree ini dipilih karena kemampuannya menangani 
kombinasi fitur kategorikal dan numerik tanpa memerlukan banyak preprocessing tambahan, 
serta tahan terhadap *overfitting* dibandingkan Decision Tree tunggal.

**Konfigurasi Hyperparameter:**
- `n_estimators=30` — jumlah pohon keputusan yang digunakan
- `max_depth=8` — kedalaman maksimum tiap pohon

Parameter ini dipilih melalui eksperimen untuk menyeimbangkan performa model dengan 
ukuran file `.pkl` yang harus di bawah 2 MB untuk kompatibilitas GitHub dan 
Hugging Face Spaces.

---

## 5. Evaluation

Model dievaluasi menggunakan *Testing Set* (20% dari total data = ±20.000 baris).

| Metrik | Nilai |
| :--- | :--- |
| **Accuracy** | ~97% |
| **Precision (kelas berisiko)** | - |
| **Recall (kelas berisiko)** | - |
| **F1-Score** | - |

> ⚠️ *Lengkapi tabel di atas dengan nilai aktual dari output notebook.*

**Confusion Matrix** menunjukkan bahwa model mampu mengklasifikasikan kelas Sehat 
dan Berisiko dengan tingkat kesalahan (*False Positive* dan *False Negative*) yang 
sangat rendah.

**Catatan penting:** Pada kasus medis seperti ini, metrik **Recall** untuk kelas 
Berisiko lebih krusial daripada Accuracy semata — karena *False Negative* (pasien 
berisiko yang terklasifikasi sehat) memiliki konsekuensi lebih serius daripada 
*False Positive*.

---

## 6. Deployment

Model dikemas menggunakan framework **Streamlit** dan di-*deploy* ke **Hugging Face 
Spaces**. Pengguna dapat mengisi data demografi dan klinis melalui antarmuka interaktif, 
lalu menekan tombol analisis untuk mendapatkan prediksi risiko diabetes secara real-time. 
Encoder (`encoder_diabetes.pkl`) dan model (`model_diabetes.pkl`) di-load sekali 
menggunakan `@st.cache_resource` untuk efisiensi performa.

**Tampilan Aplikasi:**
---<img width="512" height="436" alt="Screenshot 2026-06-16 204100" src="https://github.com/user-attachments/assets/28ac2d02-391a-4f70-bf8d-1749029bdd90" />

