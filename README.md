# 🩺 Prediksi Risiko Penyakit Diabetes Berdasarkan Rekam Medis

---

## 📌 Project Overview
Proyek ini bertujuan memprediksi tingkat risiko seseorang terkena penyakit diabetes **(Sehat / Berisiko Diabetes)** berdasarkan data demografi dan indikator rekam medis klinis menggunakan algoritma *machine learning* dengan metodologi **CRISP-DM**.

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
Penyakit diabetes sering kali terlambat disadari oleh penderitanya karena gejala awal yang cenderung samar dan tidak spesifik. Keterlambatan diagnosis ini berisiko memunculkan komplikasi serius di masa depan, seperti kerusakan ginjal, gangguan penglihatan, dan penyakit kardiovaskular. Deteksi dini secara otomatis melalui sistem kecerdasan buatan sangat diperlukan agar pasien dapat segera menyesuaikan gaya hidup atau mencari penanganan medis profesional secepat mungkin.

### Problem Statement
- Dapatkah kita memprediksi risiko penyakit diabetes seseorang **(0 = Sehat, 1 = Berisiko Diabetes)** berdasarkan data demografi dan indikator kesehatan klinis menggunakan algoritma *machine learning*?

### Goals
- Membangun model klasifikasi yang andal untuk memprediksi risiko penyakit diabetes.
- Mengidentifikasi indikator kesehatan mana (seperti kadar glukosa darah atau HbA1c) yang paling kuat pengaruhnya terhadap risiko diabetes.
- Men-*deploy* model yang telah dibangun menjadi sebuah aplikasi web interaktif yang mudah diakses dan digunakan oleh pengguna awam.

### Solution Statement
- **Model:** Random Forest Classifier
- **Metrik Evaluasi:** Accuracy Score & Confusion Matrix

---

## 2. Data Understanding

### Sumber Data
- **Dataset:** [Diabetes Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- **Jumlah data:** 100.000 baris observasi rekam medis
- **Jumlah fitur:** 9 kolom

### Sampel Data

| gender | age | hypertension | heart_disease | smoking_history | bmi | HbA1c_level | blood_glucose_level | diabetes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Female | 80.0 | 0 | 1 | never | 25.19 | 6.6 | 140 | 0 |
| Female | 54.0 | 0 | 0 | No Info | 27.32 | 6.6 | 80 | 0 |
| Male | 28.0 | 0 | 0 | never | 27.32 | 5.7 | 158 | 0 |

### Deskripsi Fitur

| Fitur | Tipe | Deskripsi |
| :--- | :--- | :--- |
| `gender` | Kategorikal | Jenis kelamin pasien (Female, Male, Other) |
| `age` | Numerik | Usia pasien dalam hitungan tahun |
| `hypertension` | Biner | Indikator penyakit darah tinggi (0 = Tidak, 1 = Ya) |
| `heart_disease` | Biner | Indikator penyakit jantung (0 = Tidak, 1 = Ya) |
| `smoking_history` | Kategorikal | Riwayat merokok (never, No Info, current, former, ever, not current) |
| `bmi` | Numerik | Indeks Massa Tubuh (Body Mass Index) |
| `HbA1c_level` | Numerik | Kadar hemoglobin terglikasi rata-rata dalam 2–3 bulan terakhir |
| `blood_glucose_level` | Numerik | Kadar glukosa darah saat pengujian |
| `diabetes` | Biner | **Target Utama** — 0 (Sehat) / 1 (Berisiko Diabetes) |

### EDA Findings

- **Distribusi Kelas Target:** Dataset bersifat *imbalanced* (tidak seimbang), di mana mayoritas pasien berada pada kelas Sehat (0) dan hanya sebagian kecil termasuk kelas Berisiko Diabetes (1). Kondisi ini umum ditemukan pada dataset medis di dunia nyata.
- **Korelasi Medis:** Berdasarkan analisis data, variabel `HbA1c_level` dan `blood_glucose_level` terbukti menjadi prediktor paling dominan terhadap risiko diabetes, jauh melebihi fitur lain seperti BMI dan usia.
- **Riwayat Merokok:** Distribusi antara berbagai kategori riwayat merokok terhadap status diabetes juga divisualisasikan untuk memahami pola hubungannya.

---

## 3. Data Preparation

- **Pengecekan Missing Values:** Dataset divalidasi dan tidak ditemukan nilai kosong (*null*) yang dapat mengganggu proses *training*.
- **Label Encoding:** Dua fitur bertipe teks dikonversi menjadi representasi numerik menggunakan `LabelEncoder`:
  - `gender` → dikodekan menjadi kolom `gender_encoded`
  - `smoking_history` → dikodekan menjadi kolom `smoking_history_encoded`
  
  Kedua encoder disimpan ke dalam file `encoder_diabetes.pkl` agar transformasi saat *deployment* konsisten dengan saat *training*.
  
- **Pembuangan Kolom Teks Lama:** Kolom `gender` dan `smoking_history` (versi teks asli) dihapus setelah encoding selesai, sehingga dataset bersih hanya berisi fitur numerik.
  
- **Data Splitting:** Dataset dibagi menjadi:
  - **Training Set (80%)** → 80.000 baris, digunakan untuk melatih model
  - **Testing Set (20%)** → 20.000 baris, digunakan untuk mengevaluasi performa model
  
  Pembagian dilakukan dengan `random_state=42` agar hasil dapat direproduksi (*reproducible*).

---

## 4. Modeling

### Random Forest Classifier

Model *ensemble* berbasis kumpulan *Decision Tree* ini dipilih karena beberapa alasan:
- Mampu menangani kombinasi fitur kategorikal (yang sudah di-encode) dan numerik dengan baik
- Tahan terhadap *overfitting* dibandingkan Decision Tree tunggal
- Tidak memerlukan banyak preprocessing tambahan seperti normalisasi fitur

**Konfigurasi Hyperparameter yang Digunakan:**

| Parameter | Nilai | Alasan |
| :--- | :--- | :--- |
| `n_estimators` | 30 | Membatasi jumlah pohon agar ukuran file `.pkl` tetap di bawah 2 MB |
| `max_depth` | 8 | Membatasi kedalaman pohon untuk mengurangi *overfitting* dan ukuran model |
| `random_state` | 42 | Memastikan hasil training dapat direproduksi |

> **Catatan:** Parameter `n_estimators=30` dan `max_depth=8` sengaja dipilih untuk menyeimbangkan antara performa prediksi dan ukuran file model, agar kompatibel saat diunggah ke GitHub (batas 100 MB) dan Hugging Face Spaces.

---

## 5. Evaluation

Model dievaluasi menggunakan *Testing Set* (20% dari total data = ±20.000 baris data yang belum pernah dilihat model sebelumnya).

### Hasil Akurasi

```
✅ Tingkat Akurasi Prediksi: 97.21%
```

### Confusion Matrix

Confusion Matrix mengevaluasi kemampuan model dalam mengklasifikasikan kedua kelas secara terpisah:

```
                  Prediksi Sehat    Prediksi Berisiko
Aktual Sehat          TN (✅)            FP (❌)
Aktual Berisiko       FN (❌)            TP (✅)
```

Hasil confusion matrix menunjukkan bahwa model mampu mengklasifikasikan pasien Sehat dan pasien Berisiko Diabetes dengan tingkat kesalahan (*False Positive* dan *False Negative*) yang sangat rendah.

> **Catatan Penting:** Dalam konteks medis, metrik **Recall** untuk kelas Berisiko (1) lebih krusial daripada sekadar Accuracy. Ini karena *False Negative* (pasien berisiko yang salah diprediksi sehat) memiliki konsekuensi lebih serius daripada *False Positive* (pasien sehat yang salah diprediksi berisiko).

---

## 6. Deployment

### Arsitektur Deployment

Model dikemas menggunakan framework **Streamlit** dan di-*deploy* ke **Hugging Face Spaces**. Terdapat dua file utama yang disimpan dan dimuat saat aplikasi berjalan:

| File | Isi | Fungsi |
| :--- | :--- | :--- |
| `model_diabetes.pkl` | Random Forest Classifier terlatih | Melakukan prediksi risiko |
| `encoder_diabetes.pkl` | `LabelEncoder` untuk `gender` dan `smoking_history` | Mengonversi input teks menjadi angka |

### Alur Kerja Aplikasi

1. Pengguna mengisi data medis (usia, BMI, kadar glukosa, dll.) melalui antarmuka Streamlit
2. Input teks (`gender`, `smoking_history`) dikonversi ke numerik menggunakan encoder yang tersimpan
3. Data dikirimkan ke model (`model_diabetes.pkl`) untuk prediksi
4. Hasil vonis risiko ditampilkan dalam hitungan detik: **Sehat** atau **Berisiko Diabetes**

### Contoh Prediksi

```python
# Input pasien contoh
pasien_gender       = 'Female'
pasien_age          = 50.0
pasien_hypertension = 1        # Ada hipertensi
pasien_heart_disease= 0        # Tidak ada penyakit jantung
pasien_smoking      = 'former'
pasien_bmi          = 28.5
pasien_hba1c        = 6.5
pasien_glucose      = 140

# Output model
# ✅ KESIMPULAN: PASIEN SEHAT (Risiko Rendah)!
```

### Tampilan Aplikasi

![Screenshot Aplikasi](https://github.com/user-attachments/assets/28ac2d02-391a-4f70-bf8d-1749029bdd90)

---

## 7. Struktur Repository

```
Diabetes-prediction/
├── app.py                          # Source code aplikasi Streamlit
├── model_diabetes.pkl              # Model Random Forest terlatih
├── encoder_diabetes.pkl            # LabelEncoder untuk fitur kategorikal
├── diabetes_prediction_dataset.csv # Dataset asli dari Kaggle
├── notebook.ipynb                  # Notebook eksplorasi & pelatihan model
├── requirements.txt                # Dependensi Python
└── README.md                       # Dokumentasi proyek
```

---

## 8. Cara Menjalankan Secara Lokal

```bash
# 1. Clone repository
git clone https://github.com/Akmal-Hakim-11/Diabetes-prediction.git
cd Diabetes-prediction

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan aplikasi
streamlit run app.py
```

---

## 9. Teknologi yang Digunakan

| Kategori | Tools / Library |
| :--- | :--- |
| Bahasa Pemrograman | Python 3 |
| Machine Learning | scikit-learn (RandomForestClassifier, LabelEncoder) |
| Data Processing | pandas |
| Visualisasi | matplotlib, seaborn |
| Serialisasi Model | pickle |
| Antarmuka Aplikasi | Streamlit |
| Platform Deployment | Hugging Face Spaces |
| Notebook | Google Colab |

**Tampilan Aplikasi:**
---<img width="512" height="436" alt="Screenshot 2026-06-16 204100" src="https://github.com/user-attachments/assets/28ac2d02-391a-4f70-bf8d-1749029bdd90" />

