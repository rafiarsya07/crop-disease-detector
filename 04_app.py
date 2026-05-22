# ============================================================
# STEP 4 - WEB APP (STREAMLIT)
# Jalankan dengan: streamlit run 04_app.py
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pathlib
import time

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title = "🌿 Crop Disease Detector",
    page_icon  = "🌿",
    layout     = "wide"
)


# ============================================================
# CSS CUSTOM
# ============================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2d6a4f;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-healthy {
        background-color: #d8f3dc;
        border-left: 5px solid #40916c;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-sick {
        background-color: #ffe5e5;
        border-left: 5px solid #e63946;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .info-box {
        background-color: #f0f7f4;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #b7e4c7;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH  = "model/crop_disease_model.h5"
LABELS_PATH = "model/class_labels.json"
IMG_SIZE    = (224, 224)

@st.cache_resource  # Cache supaya tidak reload setiap kali
def load_model():
    if not pathlib.Path(MODEL_PATH).exists():
        return None, None
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    
    return model, labels


# ============================================================
# FUNGSI PREDIKSI
# ============================================================

def predict(image, model, labels, top_k=3):
    """Prediksi penyakit dari gambar PIL"""
    
    # Preprocess
    img = image.convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_batch, verbose=0)[0]
    
    # Ambil top K
    top_indices = predictions.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        label = labels[str(idx)]
        confidence = float(predictions[idx] * 100)
        
        parts = label.split("___")
        plant = parts[0].replace("_", " ")
        condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        is_healthy = "healthy" in condition.lower()
        
        results.append({
            "label"      : label,
            "plant"      : plant,
            "condition"  : condition,
            "confidence" : confidence,
            "is_healthy" : is_healthy
        })
    
    return results


# ============================================================
# TAMPILAN UTAMA
# ============================================================

# Header
st.markdown('<p class="main-title">🌿 Crop Disease Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload foto daun tanaman untuk mendeteksi penyakit menggunakan AI</p>', unsafe_allow_html=True)

st.markdown("---")

# Load model
model, labels = load_model()

if model is None:
    st.error("⚠️ Model belum tersedia!")
    st.info("""
    **Cara setup:**
    1. Jalankan `01_explore_dataset.py` untuk download dataset
    2. Jalankan `02_train_model.py` untuk melatih model
    3. Jalankan app ini lagi: `streamlit run 04_app.py`
    """)
    st.stop()


# Layout dua kolom
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Upload Gambar")
    
    uploaded_file = st.file_uploader(
        "Pilih foto daun tanaman",
        type=["jpg", "jpeg", "png"],
        help="Format yang didukung: JPG, JPEG, PNG"
    )
    
    # Contoh cara pakai
    with st.expander("💡 Tips agar hasil akurat"):
        st.markdown("""
        - 📷 Foto dari jarak dekat (30-50 cm)
        - 🌞 Pencahayaan yang cukup
        - 🍃 Fokus pada bagian daun yang bermasalah
        - 📐 Gambar tidak terlalu buram atau gelap
        """)
    
    # Informasi tanaman yang didukung
    with st.expander("🌱 Tanaman yang didukung"):
        plants = [
            "Apple (Apel)", "Blueberry", "Cherry", "Corn (Jagung)",
            "Grape (Anggur)", "Orange (Jeruk)", "Peach (Persik)",
            "Pepper (Cabai)", "Potato (Kentang)", "Raspberry",
            "Soybean (Kedelai)", "Squash", "Strawberry", "Tomato (Tomat)"
        ]
        for plant in plants:
            st.write(f"• {plant}")


with col2:
    st.subheader("🔍 Hasil Analisis")
    
    if uploaded_file is None:
        st.markdown("""
        <div class="info-box">
        <p>👈 Upload gambar daun untuk memulai analisis</p>
        <p>App ini dapat mendeteksi <strong>38 jenis penyakit</strong> pada berbagai tanaman.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Tampilkan gambar yang diupload
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_column_width=True)
        
        # Tombol analisis
        if st.button("🔬 Analisis Sekarang", type="primary", use_container_width=True):
            
            with st.spinner("🧠 AI sedang menganalisis gambar..."):
                time.sleep(0.5)  # Sedikit delay biar keliatan prosesnya
                results = predict(image, model, labels)
            
            # Tampilkan hasil utama
            top = results[0]
            
            if top["is_healthy"]:
                st.markdown(f"""
                <div class="result-healthy">
                    <h3>✅ Daun Sehat!</h3>
                    <p><strong>Tanaman:</strong> {top['plant']}</p>
                    <p><strong>Kondisi:</strong> {top['condition']}</p>
                    <p><strong>Keyakinan AI:</strong> {top['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-sick">
                    <h3>⚠️ Terdeteksi Penyakit!</h3>
                    <p><strong>Tanaman:</strong> {top['plant']}</p>
                    <p><strong>Penyakit:</strong> {top['condition']}</p>
                    <p><strong>Keyakinan AI:</strong> {top['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Tampilkan top 3 prediksi
            st.markdown("#### 📊 Detail Prediksi")
            
            for i, result in enumerate(results):
                label = f"{'✅' if result['is_healthy'] else '⚠️'} {result['plant']} - {result['condition']}"
                st.progress(
                    result['confidence'] / 100,
                    text=f"{label} ({result['confidence']:.1f}%)"
                )
            
            # Disclaimer
            st.caption("⚠️ Hasil ini hanya prediksi AI. Konsultasikan dengan ahli pertanian untuk kepastian.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
    🌿 Crop Disease Detector | Dibuat dengan Python + TensorFlow + Streamlit<br>
    Dataset: PlantVillage | Model: MobileNetV2 (Transfer Learning)
</div>
""", unsafe_allow_html=True)
