# ============================================================
# STEP 3 - TEST MODEL
# Coba model dengan gambar baru
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import json
import sys
import pathlib

print("=" * 50)
print("CROP DISEASE DETECTOR - STEP 3")
print("Test Model")
print("=" * 50)


# ============================================================
# 3A. LOAD MODEL & LABELS
# ============================================================

MODEL_PATH  = "model/crop_disease_model.h5"
LABELS_PATH = "model/class_labels.json"
IMG_SIZE    = (224, 224)

# Cek model ada atau tidak
if not pathlib.Path(MODEL_PATH).exists():
    print("\n❌ Model tidak ditemukan!")
    print("Jalankan dulu: 02_train_model.py")
    exit()

print("\n📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    CLASS_LABELS = json.load(f)

print(f"✅ Model berhasil dimuat!")
print(f"   Jumlah kelas: {len(CLASS_LABELS)}")


# ============================================================
# 3B. FUNGSI PREDIKSI
# ============================================================

def predict_disease(image_path, top_k=3):
    """
    Predict penyakit dari gambar daun
    
    Args:
        image_path: path ke file gambar
        top_k: tampilkan top K prediksi
    
    Returns:
        dict berisi hasil prediksi
    """
    
    # Load dan preprocess gambar
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Prediksi
    predictions = model.predict(img_batch, verbose=0)[0]
    
    # Ambil top K prediksi
    top_indices = predictions.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        label = CLASS_LABELS[str(idx)]
        confidence = predictions[idx] * 100
        
        # Format nama lebih readable
        parts = label.split("___")
        plant = parts[0].replace("_", " ")
        condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        
        results.append({
            "label"       : label,
            "plant"       : plant,
            "condition"   : condition,
            "confidence"  : confidence,
            "is_healthy"  : "healthy" in condition.lower()
        })
    
    return results, img


def format_result(result):
    """Format hasil prediksi jadi teks yang mudah dibaca"""
    
    status = "✅ SEHAT" if result["is_healthy"] else "⚠️  SAKIT"
    
    return (
        f"{status}\n"
        f"Tanaman  : {result['plant']}\n"
        f"Kondisi  : {result['condition']}\n"
        f"Keyakinan: {result['confidence']:.1f}%"
    )


# ============================================================
# 3C. TEST DENGAN GAMBAR
# ============================================================

def test_single_image(image_path):
    """Test prediksi pada satu gambar"""
    
    if not pathlib.Path(image_path).exists():
        print(f"❌ Gambar tidak ditemukan: {image_path}")
        return
    
    print(f"\n🔍 Menganalisis: {image_path}")
    
    results, img = predict_disease(image_path, top_k=3)
    
    # Tampilkan gambar dan hasil
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gambar asli
    axes[0].imshow(img)
    axes[0].set_title("Gambar Input", fontsize=12)
    axes[0].axis('off')
    
    # Bar chart confidence
    labels = [f"{r['plant']}\n{r['condition'][:20]}" for r in results]
    confidences = [r['confidence'] for r in results]
    colors = ['green' if r['is_healthy'] else 'red' for r in results]
    
    bars = axes[1].barh(labels, confidences, color=colors, alpha=0.7)
    axes[1].set_xlabel("Tingkat Keyakinan (%)")
    axes[1].set_title("Hasil Prediksi", fontsize=12)
    axes[1].set_xlim(0, 100)
    
    # Tambah label persentase
    for bar, conf in zip(bars, confidences):
        axes[1].text(
            conf + 1, bar.get_y() + bar.get_height()/2,
            f'{conf:.1f}%', va='center', fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=100, bbox_inches='tight')
    plt.show()
    
    # Print hasil
    print("\n" + "=" * 40)
    print("HASIL ANALISIS")
    print("=" * 40)
    print(format_result(results[0]))
    print("\nAlternatif prediksi:")
    for i, r in enumerate(results[1:], 2):
        print(f"  {i}. {r['plant']} - {r['condition']} ({r['confidence']:.1f}%)")
    
    return results


# ============================================================
# 3D. TEST DENGAN MULTIPLE GAMBAR
# ============================================================

def test_multiple_images(image_folder):
    """Test prediksi pada banyak gambar sekaligus"""
    
    folder = pathlib.Path(image_folder)
    if not folder.exists():
        print(f"❌ Folder tidak ditemukan: {image_folder}")
        return
    
    # Cari semua gambar
    image_files = list(folder.glob("*.jpg")) + \
                  list(folder.glob("*.jpeg")) + \
                  list(folder.glob("*.png"))
    
    if not image_files:
        print("❌ Tidak ada gambar di folder ini")
        return
    
    print(f"\n🔍 Testing {len(image_files)} gambar...")
    
    # Plot semua hasil
    num_images = min(len(image_files), 9)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Hasil Prediksi Batch", fontsize=16, fontweight='bold')
    
    for ax, img_path in zip(axes.flatten(), image_files[:num_images]):
        results, img = predict_disease(str(img_path))
        top_result = results[0]
        
        ax.imshow(img)
        
        color = 'green' if top_result['is_healthy'] else 'red'
        title = f"{top_result['plant']}\n{top_result['condition'][:20]}\n{top_result['confidence']:.1f}%"
        ax.set_title(title, fontsize=8, color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("batch_prediction.png", dpi=100, bbox_inches='tight')
    plt.show()
    
    print("✅ Hasil batch disimpan ke 'batch_prediction.png'")


# ============================================================
# JALANKAN TEST
# ============================================================

# Ganti path ini dengan gambar yang mau kamu test
TEST_IMAGE = "test_image.jpg"  # Taruh gambar daun kamu di sini

if pathlib.Path(TEST_IMAGE).exists():
    test_single_image(TEST_IMAGE)
else:
    print(f"\n💡 Cara pakai:")
    print(f"   1. Taruh gambar daun di folder ini")
    print(f"   2. Ubah TEST_IMAGE di bawah ini:")
    print(f'      TEST_IMAGE = "nama_gambar_kamu.jpg"')
    print(f"   3. Jalankan script ini lagi")
    print(f"\n   Atau import fungsinya di script lain:")
    print(f"   from 03_test_model import predict_disease")
    print(f"   results, img = predict_disease('gambar.jpg')")

print("\n" + "=" * 50)
print("✅ Step 3 selesai!")
print("Lanjut ke: 04_app.py (Web App)")
print("=" * 50)
