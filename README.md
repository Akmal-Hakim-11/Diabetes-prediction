# Proyek Predictive Analysis - Deteksi Risiko Penyakit Diabetes

## Business Understanding

Penyakit diabetes sering kali terlambat ditangani karena minimnya kesadaran masyarakat terhadap gejala awal dan faktor risiko medis yang mereka miliki. Keterlambatan ini dapat berujung pada komplikasi kesehatan fatal. Oleh karena itu, diperlukan sebuah pendekatan komputasi untuk melakukan deteksi dini secara otomatis.

### Problem Statements
Berdasarkan latar belakang di atas, rumusan masalah dari proyek ini adalah:
* Bagaimana cara mendeteksi risiko penyakit diabetes pada pasien secara dini berdasarkan data rekam medis demografis dan kesehatan dasar?
* Faktor-faktor medis (fitur) apa saja yang paling berkontribusi signifikan dalam menentukan tingginya risiko diabetes pada seseorang?

### Goals
Tujuan dari proyek ini adalah:
* Membangun model *Machine Learning* yang dapat mengklasifikasikan pasien ke dalam kategori sehat (0) atau berisiko diabetes (1) dengan tingkat akurasi yang tinggi.
* Menganalisis dan mengetahui korelasi antara variabel medis seperti tingkat glukosa darah, BMI, dan riwayat hipertensi terhadap risiko penyakit diabetes.

### Solution Statement
Solusi yang diajukan untuk mencapai tujuan proyek adalah sebagai berikut:
* Menerapkan algoritma klasifikasi **Random Forest Classifier**. Algoritma ini dipilih karena kapabilitasnya yang sangat baik dalam menangani kombinasi data numerik dan kategorikal, serta kemampuannya dalam mencegah *overfitting* pada dataset berukuran besar.
* Melakukan tahapan *Exploratory Data Analysis* (EDA) untuk memvisualisasikan sebaran data dan mencari matriks korelasi antar fitur kesehatan.
* Mengevaluasi performa model menggunakan metrik *Accuracy Score* dan memvisualisasikan sebaran tebakan klasifikasi menggunakan *Confusion Matrix*.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini bersumber dari platform Kaggle dengan nama "Diabetes Prediction Dataset". Dataset ini memuat 100.000 baris data rekam medis historis pasien dengan berbagai kondisi dan kebiasaan hidup.

**Variabel-variabel pada dataset adalah sebagai berikut:**
* `gender` : Jenis kelamin pasien (Female, Male, Other).
* `age` : Usia pasien dalam hitungan tahun.
* `hypertension` : Indikator tekanan darah tinggi, di mana nilai 0 berarti tidak memiliki riwayat hipertensi dan 1 berarti memiliki riwayat hipertensi.
* `heart_disease` : Indikator penyakit jantung, di mana nilai 0 berarti tidak memiliki riwayat penyakit jantung dan 1 berarti memiliki riwayat penyakit jantung.
* `smoking_history` : Riwayat kebiasaan merokok pasien (seperti *never*, *current*, *former*, dll).
* `bmi` : *Body Mass Index* atau Indeks Massa Tubuh pasien, digunakan untuk mengukur tingkat obesitas.
* `HbA1c_level` : Tingkat Hemoglobin A1c dalam darah pasien, metrik utama penanda rata-rata gula darah selama 2-3 bulan terakhir.
* `blood_glucose_level` : Tingkat glukosa darah pasien pada saat pemeriksaan medis dilakukan.
* `diabetes` : Variabel target (label target), di mana angka 0 menandakan pasien diklasifikasikan sehat, dan angka 1 menandakan pasien terindikasi menderita diabetes.

---

## Data Preparation

Pada tahap ini, dilakukan persiapan dan pembersihan data sebelum dimasukkan ke dalam algoritma pelatihan:
* **Pengecekan Missing Values:** Memastikan tidak ada nilai kosong di dalam dataset yang dapat mengganggu proses komputasi model.
* **Label Encoding:** Melakukan transformasi data menggunakan `LabelEncoder` dari library `scikit-learn` untuk mengubah fitur dengan tipe data teks/kategorikal (`gender` dan `smoking_history`) menjadi format numerik (angka) agar dapat dipelajari oleh model mesin.
* **Train-Test Split:** Membagi keseluruhan dataset menjadi data latih (*training set*) sebesar 80% dan data uji (*testing set*) sebesar 20% untuk memvalidasi seberapa baik model memprediksi data baru yang belum pernah dilihat sebelumnya.
