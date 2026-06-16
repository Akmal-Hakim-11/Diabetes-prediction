# 🩺 Prediksi Risiko Penyakit Diabetes Berdasarkan Rekam Medis

---

## 📌 Project Overview
Proyek ini bertujuan memprediksi tingkat risiko seseorang terkena penyakit diabetes (Sehat / Berisiko Tinggi) berdasarkan data demografi dan indikator rekam medis klinis menggunakan algoritma *machine learning* dengan metodologi **CRISP-DM**.

🔗 **Live Demo:** [Link Live Demo](#)  
📓 **Notebook:** [Link Notebook](https://colab.research.google.com/drive/1rNA4Sju6GJtvPy97PnawQ1S0JbIaxxIf?usp=sharing)

---

## 👥 Tim

| Nama | NIM |
| :--- | :--- |
| [MUHAMMAD AKMAL HAKIM PURAWIJAYA] | [2330511085] |
| [Mochammad Fattah Maulana Mansur] | [2330511083] |

*(Catatan: Jangan lupa ganti isi tabel ini dengan nama dan NIM asli kelompokmu ya!)*

---


## 1. Business Understanding

### Latar Belakang
Penyakit diabetes sering kali terlambat disadari oleh penderitanya karena gejala awal yang cenderung samar. Keterlambatan diagnosis ini berisiko memunculkan komplikasi serius di masa depan. Deteksi dini secara otomatis melalui sistem cerdas sangat diperlukan agar pasien dapat segera menyesuaikan gaya hidup atau mencari penanganan medis profesional secepat mungkin.

### Problem Statement
* Dapatkah kita memprediksi risiko penyakit diabetes seseorang (0 = Sehat, 1 = Berisiko) berdasarkan data demografi dan indikator kesehatan klinis menggunakan algoritma *machine learning*?

### Goals
* Membangun model klasifikasi yang handal untuk memprediksi risiko penyakit diabetes.
* Mengidentifikasi indikator kesehatan mana (seperti glukosa atau BMI) yang paling kuat pengaruhnya terhadap risiko diabetes.
* Men-*deploy* model yang telah dibuat menjadi sebuah aplikasi web interaktif yang mudah diakses dan digunakan.

### Solution Statement
* **Model Utama:** Random Forest Classifier
* **Metrik Evaluasi:** Accuracy Score & Confusion Matrix

---

## 2. Data Understanding

### Sumber Data
* **Dataset:** [Diabetes Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
* **Jumlah data:** 100.000 baris observasi rekam medis
* **Jumlah fitur:** 9 kolom

### Deskripsi Fitur

| Fitur | Tipe | Deskripsi |
| :--- | :--- | :--- |
| `gender` | kategorikal | Jenis kelamin pasien (Female, Male, Other) |
| `age` | numerik | Usia pasien dalam hitungan tahun |
| `hypertension` | numerik | Indikator penyakit darah tinggi (0 = Tidak, 1 = Ya) |
| `heart_disease` | numerik | Indikator penyakit jantung (0 = Tidak, 1 = Ya) |
| `smoking_history` | kategorikal | Riwayat merokok (never, current, former, dll) |
| `bmi` | numerik | Indeks Massa Tubuh (Body Mass Index) |
| `HbA1c_level` | numerik | Tingkat rata-rata hemoglobin terikat glukosa (2-3 bulan terakhir) |
| `blood_glucose_level` | numerik | Tingkat kadar glukosa darah saat pengujian |
| `diabetes` | kategorikal | **Target Utama** — 0 (Sehat) / 1 (Berisiko Diabetes) |

### EDA Findings
* **Distribusi Kelas Target:** Data menunjukkan sifat ketidakseimbangan kelas (*imbalanced data*), di mana mayoritas pasien dalam dataset berada pada kelas sehat.
* **Korelasi Medis:** Berdasarkan matriks korelasi, variabel `HbA1c_level` dan `blood_glucose_level` merupakan faktor prediktor paling dominan yang memengaruhi risiko diabetes.

---

## 3. Data Preparation
* **Pengecekan Missing Values:** Memvalidasi dataset untuk memastikan tidak ada kekosongan data (*null*) yang dapat mengganggu proses *training*.
* **Label Encoding:** Melakukan transformasi data menggunakan `LabelEncoder` untuk mengubah fitur berupa teks (`gender` dan `smoking_history`) menjadi representasi numerik.
* **Data Splitting:** Dataset dibagi menjadi *Training Set* (80%) untuk bahan ajar model, dan *Testing Set* (20%) untuk simulasi ujian performa model.

---

## 4. Modeling

* **Model 1: Random Forest Classifier**  
  Model *ensemble* ini dipilih karena sangat kuat dalam menangani pola kombinasi data kategorikal dan numerik tanpa perlu banyak penyesuaian. Untuk mengoptimalkan kinerja *deployment*, dilakukan pembatasan cabang pohon keputusan (`n_estimators=30`, `max_depth=8`). Hal ini krusial untuk menekan potensi *overfitting* dan memastikan ukuran *file* model (`.pkl`) berada di bawah 2 MB agar tidak ditolak saat diunggah ke GitHub.

---

## 5. Evaluation

* **Accuracy Score:** Saat dilakukan pengujian buta menggunakan *Testing Set*, model terbukti stabil dan mencatatkan tingkat akurasi tebakan yang sangat tinggi.
* **Confusion Matrix:** Evaluasi berbasis kuadran (*True Positive, True Negative, False Positive, False Negative*) membuktikan bahwa model mampu mengklasifikasikan kelompok pasien sehat dan kelompok berisiko secara presisi, dengan tingkat kesalahan prediksi yang sangat minim.

---

## 6. Deployment
* **Tampilan Aplikasi:** Model dikemas menggunakan *framework* antarmuka **Streamlit**. Pada *dashboard* ini, pengguna dapat menggeser *slider* dan mengisi form data medis dasar. Setelah tombol analisis ditekan, *source code* akan mengirimkan data tersebut ke otak AI (`model_diabetes.pkl`) untuk mengeluarkan vonis risiko kesehatan dalam hitungan detik.

---

## ⚠️ Disclaimer
Proyek rekayasa perangkat lunak dan analisis data ini dikerjakan secara berkelompok untuk memenuhi tujuan akademis. Hasil prediksi sistem **tidak dapat dijadikan diagnosis medis profesional yang absolut**. Pengguna disarankan untuk tetap berkonsultasi dengan fasilitas kesehatan dan dokter spesialis terkait.
