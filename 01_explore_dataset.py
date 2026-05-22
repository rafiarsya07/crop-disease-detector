# ============================================================
# STEP 1 - DOWNLOAD & EXPLORE DATASET
# Jalankan file ini pertama kali
# ============================================================

import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ============================================================
# 1A. DOWNLOAD DATASET DARI KAGGLE
# ============================================================

# CARA DOWNLOAD MANUAL (Recommended untuk pemula):
# 1. Pergi ke https://www.kaggle.com/datasets/emmarex/plantdisease
# 2. Klik tombol "Download"
# 3. Extract ZIP ke folder "dataset" di dalam folder project ini

# ATAU pakai Kaggle API (kalau sudah setup):
# pip install kaggle
# kaggle datasets download -d emmarex/plantdisease
# unzip plantdisease.zip -d dataset/

print("=" * 50)
print("CROP DISEASE DETECTOR - STEP 1")
print("Download & Explore Dataset")
print("=" * 50)


# ============================================================
# 1B. CEK STRUKTUR FOLDER DATASET
# ============================================================

DATASET_PATH = pathlib.Path("dataset/PlantVillage")

# Cek apakah dataset sudah ada
if not DATASET_PATH.exists():
    print("\n⚠️  Dataset belum ada!")
    print("Silakan download dulu dari Kaggle:")
    print("https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("\nSetelah download, extract ke folder 'dataset/PlantVillage'")
else:
    # Hitung jumlah kelas dan gambar
    classes = sorted([d.name for d in DATASET_PATH.iterdir() if d.is_dir()])
    
    print(f"\n✅ Dataset ditemukan!")
    print(f"📁 Total kelas penyakit: {len(classes)}")
    
    total_images = 0
    class_counts = {}
    
    for cls in classes:
        cls_path = DATASET_PATH / cls
        images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.JPG")) + \
                 list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.png"))
        class_counts[cls] = len(images)
        total_images += len(images)
    
    print(f"🖼️  Total gambar: {total_images:,}")
    
    print("\n📊 10 Kelas Pertama:")
    print("-" * 50)
    for cls in classes[:10]:
        # Format nama kelas biar lebih readable
        name = cls.replace("___", " - ").replace("_", " ")
        print(f"  {name}: {class_counts[cls]} gambar")
    
    print(f"\n  ... dan {len(classes) - 10} kelas lainnya")


# ============================================================
# 1C. TAMPILKAN CONTOH GAMBAR
# ============================================================

def show_sample_images(dataset_path, num_samples=9):
    """Tampilkan contoh gambar dari beberapa kelas"""
    
    if not dataset_path.exists():
        print("Dataset belum ada, skip visualisasi")
        return
    
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle("Contoh Gambar Dataset PlantVillage", fontsize=16, fontweight='bold')
    
    selected_classes = np.random.choice(classes, num_samples, replace=False)
    
    for ax, cls in zip(axes.flatten(), selected_classes):
        cls_path = dataset_path / cls
        images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.JPG"))
        
        if images:
            img = mpimg.imread(str(images[0]))
            ax.imshow(img)
            
            # Format nama biar tidak terlalu panjang
            name = cls.replace("___", "\n").replace("_", " ")
            ax.set_title(name, fontsize=8)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("\n✅ Gambar contoh disimpan ke 'sample_images.png'")


# Tampilkan contoh gambar
show_sample_images(DATASET_PATH)


# ============================================================
# 1D. ANALISIS DISTRIBUSI DATASET
# ============================================================

def analyze_dataset(dataset_path):
    """Analisis distribusi data per kelas"""
    
    if not dataset_path.exists():
        print("Dataset belum ada, skip analisis")
        return
    
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    counts = []
    
    for cls in classes:
        cls_path = dataset_path / cls
        images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.JPG")) + \
                 list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.png"))
        counts.append(len(images))
    
    # Plot distribusi
    plt.figure(figsize=(16, 8))
    
    short_names = [c.replace("___", "\n").replace("_", " ")[:20] for c in classes]
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    
    bars = plt.bar(range(len(classes)), counts, color=colors)
    plt.xlabel("Kelas Penyakit", fontsize=12)
    plt.ylabel("Jumlah Gambar", fontsize=12)
    plt.title("Distribusi Gambar per Kelas", fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), short_names, rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig("dataset_distribution.png", dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Statistik Dataset:")
    print(f"   Rata-rata gambar per kelas : {np.mean(counts):.0f}")
    print(f"   Kelas dengan gambar terbanyak : {classes[np.argmax(counts)]} ({max(counts)})")
    print(f"   Kelas dengan gambar tersedikit : {classes[np.argmin(counts)]} ({min(counts)})")


analyze_dataset(DATASET_PATH)

print("\n" + "=" * 50)
print("✅ Step 1 selesai!")
print("Lanjut ke: 02_train_model.py")
print("=" * 50)
