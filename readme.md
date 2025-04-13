# 📱 Sentiment Analysis Review App - Google Play Store

Aplikasi ini dibuat untuk melakukan _scraping_ ulasan dari Google Play Store, melakukan _preprocessing_ data, memberi label sentimen secara manual maupun otomatis (menggunakan beberapa lexicon-based methods), hingga proses pemodelan dan evaluasi klasifikasi sentimen. Aplikasi ini cocok digunakan untuk riset maupun analisis bisnis berbasis opini pengguna.

## 🚀 Fitur Utama

### 1. **HOME**

Menu utama yang terdiri dari dua tab:

- **Main**: Menampilkan ringkasan dan tujuan aplikasi.
- **Scraping**: Melakukan pengambilan (scraping) ulasan aplikasi dari Google Play Store menggunakan nama paket (package name). Hasil scraping akan tersimpan dalam bentuk tabel dan bisa diunduh untuk analisis lebih lanjut.

### 2. **Data Preprocessing**

Menu untuk memproses data mentah hasil scraping agar siap dianalisis:

- **Text Preprocessing**:
  - Pembersihan teks (_cleaning_)
  - Penghapusan stopwords
  - Normalisasi, tokenisasi, dan proses lainnya.
- **TF-IDF**: Mengubah teks menjadi vektor numerik menggunakan teknik TF-IDF.
- **Labeling**:
  - **Manual Labeling**: Pengguna dapat memberi label secara manual.
  - **VADER**: Menggunakan lexicon berbasis aturan dari NLTK.
  - **INSET Lexicon**: Menggunakan kamus sentimen Bahasa Indonesia.
  - **TextBlob**: Analisis sentimen berbasis pustaka TextBlob.
- **SMOTE**: Penyeimbangan data menggunakan teknik oversampling (SMOTE).
- **Split Data**: Pembagian dataset ke dalam data latih dan uji.
- **Dataset Overview**: Menampilkan ringkasan dataset dan distribusi sentimen.

### 3. **Modelling**

Menu ini digunakan untuk pelatihan model dan evaluasi:

- **Model**: Melatih model klasifikasi menggunakan algoritma berikut:
  - Support Vector Machine (SVM)
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
- **Testing**: Menguji model menggunakan data uji.
- **Evaluation**: Menampilkan metrik evaluasi seperti akurasi, precision, recall, F1-score, dan confusion matrix.

---

## 🧰 Teknologi yang Digunakan

- **Python 3.x**
- **Streamlit** (untuk UI interaktif)
- **Pandas**, **NumPy**
- **Scikit-learn**
- **NLTK**, **TextBlob**
- **Matplotlib**, **Seaborn**
- **Google Play Scraper (e.g., `google-play-scraper` atau `play-scraper`)**

---

## 📦 Instalasi

1. **Clone repositori ini:**

```bash
git clone https://github.com/ahmadseloabadi/sentiment-analysis-review-app-google-play.git
cd sentiment-analysis-review-app-google-play
```

2. **Buat virtual environment (opsional tapi disarankan):**

```bash
python -m venv venv
source venv/bin/activate  # atau `venv\Scripts\activate` di Windows
```

3. **Install dependensi:**

```bash
pip install -r requirements.txt
```

4. **Jalankan aplikasi:**

```bash
streamlit run app.py
```

---

## 📁 Struktur Proyek

## 📂 Struktur Direktori Proyek

```text
sentiment-analysis-review-app-google-play/
├── data/
│   └── ...                # Berisi dataset, kamus kata, dan file pendukung lainnya
│
├── myModule/
│   ├── __init__.py        # Menandakan ini adalah package Python
│
├── app.py                 # Entry point aplikasi Streamlit untuk menjalankan sistem analisis
├── requirements.txt       # Daftar library dan dependensi yang digunakan dalam proyek
└── README.md              # Dokumentasi proyek (file ini)
```

## 🧑‍💻 Kontribusi

Kontribusi sangat terbuka! Silakan buat issue atau pull request untuk perbaikan atau penambahan fitur.

## 🔗 Tautan Terkait

GitHub Repository: https://github.com/ahmadseloabadi/sentiment-analysis-review-app-google-play
