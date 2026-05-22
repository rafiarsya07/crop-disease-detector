# 🌿 Crop Disease Detector

Aplikasi AI untuk mendeteksi penyakit tanaman dari foto daun.
Dibuat dengan Python, TensorFlow, dan Streamlit.

---

## 📁 Struktur Project

```
crop_disease_detector/
│
├── 01_explore_dataset.py   ← Step 1: Download & lihat dataset
├── 02_train_model.py       ← Step 2: Latih model AI
├── 03_test_model.py        ← Step 3: Test model dengan gambar
├── 04_app.py               ← Step 4: Web app (Streamlit)
│
├── requirements.txt        ← Daftar library yang dibutuhkan
├── README.md               ← File ini
│
├── dataset/                ← Taruh dataset di sini (setelah download)
│   └── PlantVillage/
│       ├── Apple___Apple_scab/
│       ├── Apple___healthy/
│       └── ... (38 folder kelas)
│
└── model/                  ← Model tersimpan di sini (otomatis dibuat)
    ├── crop_disease_model.h5
    └── class_labels.json
```

---

## 🚀 Cara Menjalankan

### 1. Install Library

Buka terminal di VS Code, jalankan:

```bash
python -m pip install -r requirements.txt
```

### 2. Download Dataset

- Pergi ke: https://www.kaggle.com/datasets/emmarex/plantdisease
- Klik **Download**
- Extract ZIP ke folder `dataset/PlantVillage/`

### 3. Jalankan Step by Step

```bash
# Step 1: Lihat dataset
python 01_explore_dataset.py

# Step 2: Latih model (butuh waktu 30-60 menit)
python 02_train_model.py

# Step 3: Test dengan gambar
python 03_test_model.py

# Step 4: Jalankan web app
streamlit run 04_app.py
```

---

## 🧠 Cara Kerja

```
Foto Daun
    ↓
Resize ke 224x224
    ↓
Normalize pixel (0-1)
    ↓
MobileNetV2 (Extract features)
    ↓
Dense Layers (Classifier)
    ↓
Softmax → 38 kelas
    ↓
Hasil: "Tomato - Late Blight (95.3%)"
```

---

## 🌱 Kelas yang Didukung (38 kelas)

| Tanaman | Penyakit |
|---------|----------|
| Apple | Apple scab, Black rot, Cedar rust, Healthy |
| Corn | Gray leaf spot, Common rust, Blight, Healthy |
| Tomato | Bacterial spot, Blight, Leaf mold, dll |
| Potato | Early blight, Late blight, Healthy |
| ... dan banyak lagi |

---

## 📝 Untuk Blog Post

Struktur artikel yang disarankan:

1. **Pendahuluan** — Masalah penyakit tanaman di Indonesia/Malaysia
2. **Dataset** — PlantVillage, 54k+ gambar, 38 kelas
3. **Transfer Learning** — Kenapa pakai MobileNetV2?
4. **Proses Training** — Data augmentasi, callbacks
5. **Hasil** — Akurasi, confusion matrix
6. **Demo** — Screenshot web app
7. **Kesimpulan** — Impact ke petani

---

## 🛠️ Teknologi

- **Python 3.x**
- **TensorFlow 2.x** — Deep learning framework
- **MobileNetV2** — Pre-trained model (Transfer Learning)
- **Streamlit** — Web app framework
- **Pillow** — Image processing
- **Matplotlib/Seaborn** — Visualisasi

---

## 💡 Pengembangan Selanjutnya

- [ ] Tambah info pengobatan untuk setiap penyakit
- [ ] Support lebih banyak jenis tanaman
- [ ] Deploy ke cloud (Streamlit Cloud / Heroku)
- [ ] Versi mobile app
- [ ] Integrasi dengan database petani

---

Dibuat untuk portfolio & blog | Dataset: PlantVillage Dataset
