# ============================================================
# STEP 2 - LATIH MODEL AI
# Jalankan setelah Step 1 (dataset sudah ada)
# ============================================================

import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

print("=" * 50)
print("CROP DISEASE DETECTOR - STEP 2")
print("Latih Model AI")
print("=" * 50)
print(f"\n🔧 TensorFlow version: {tf.__version__}")
print(f"💻 GPU tersedia: {len(tf.config.list_physical_devices('GPU')) > 0}")


# ============================================================
# 2A. KONFIGURASI
# ============================================================

# Ubah sesuai kebutuhan kamu
CONFIG = {
    "dataset_path"  : "dataset/PlantVillage",
    "img_size"      : (224, 224),      # Ukuran input MobileNetV2
    "batch_size"    : 32,              # Kurangi ke 16 kalau RAM terbatas
    "epochs"        : 15,              # Tambah kalau mau akurasi lebih tinggi
    "val_split"     : 0.2,             # 20% untuk validasi
    "learning_rate" : 0.001,
    "model_save"    : "model/crop_disease_model.h5",
    "labels_save"   : "model/class_labels.json",
}

# Buat folder model kalau belum ada
os.makedirs("model", exist_ok=True)

DATASET_PATH = pathlib.Path(CONFIG["dataset_path"])

if not DATASET_PATH.exists():
    print("\n❌ Dataset tidak ditemukan!")
    print("Jalankan dulu: 01_explore_dataset.py")
    exit()


# ============================================================
# 2B. LOAD & AUGMENTASI DATA
# ============================================================

print("\n📂 Loading dataset...")

# Data augmentasi untuk training
# Tujuan: perbanyak variasi data supaya model lebih robust
train_datagen = ImageDataGenerator(
    rescale          = 1./255,         # Normalize pixel 0-1
    validation_split = CONFIG["val_split"],
    rotation_range   = 20,             # Rotate gambar random
    width_shift_range= 0.2,            # Geser horizontal
    height_shift_range= 0.2,           # Geser vertikal
    shear_range      = 0.2,
    zoom_range       = 0.2,            # Zoom random
    horizontal_flip  = True,           # Flip horizontal
    fill_mode        = 'nearest'
)

# Untuk validasi, hanya normalize tanpa augmentasi
val_datagen = ImageDataGenerator(
    rescale          = 1./255,
    validation_split = CONFIG["val_split"]
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size = CONFIG["img_size"],
    batch_size  = CONFIG["batch_size"],
    class_mode  = 'categorical',
    subset      = 'training',
    shuffle     = True,
    seed        = 42
)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size = CONFIG["img_size"],
    batch_size  = CONFIG["batch_size"],
    class_mode  = 'categorical',
    subset      = 'validation',
    shuffle     = False,
    seed        = 42
)

NUM_CLASSES = len(train_generator.class_indices)
CLASS_LABELS = {v: k for k, v in train_generator.class_indices.items()}

print(f"\n✅ Data berhasil dimuat!")
print(f"   Training: {train_generator.samples:,} gambar")
print(f"   Validasi : {val_generator.samples:,} gambar")
print(f"   Jumlah kelas: {NUM_CLASSES}")

# Simpan label kelas
with open(CONFIG["labels_save"], "w") as f:
    json.dump(CLASS_LABELS, f, indent=2)
print(f"   Labels disimpan ke: {CONFIG['labels_save']}")


# ============================================================
# 2C. BANGUN MODEL (TRANSFER LEARNING)
# ============================================================

print("\n🏗️  Membangun model...")

# Pakai MobileNetV2 yang sudah pre-trained di ImageNet
# Ini jauh lebih efisien daripada bangun dari nol
base_model = MobileNetV2(
    input_shape = (*CONFIG["img_size"], 3),
    include_top = False,               # Hapus bagian classifier aslinya
    weights     = 'imagenet'           # Pakai bobot yang sudah dilatih
)

# Bekukan base model dulu
# Kita cuma mau latih bagian atas (classifier kita sendiri)
base_model.trainable = False

# Bangun model lengkap
model = models.Sequential([
    base_model,
    
    # Flatten feature map jadi 1D
    layers.GlobalAveragePooling2D(),
    
    # Fully connected layer
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),               # Dropout untuk hindari overfitting
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    
    # Output layer - jumlah neuron = jumlah kelas
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(
    optimizer = tf.keras.optimizers.Adam(CONFIG["learning_rate"]),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

model.summary()

print(f"\n📊 Total parameter: {model.count_params():,}")
print(f"   Parameter yang dilatih: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")


# ============================================================
# 2D. LATIH MODEL
# ============================================================

print("\n🚀 Mulai training...")
print(f"   Epochs: {CONFIG['epochs']}")
print(f"   Batch size: {CONFIG['batch_size']}")

# Callbacks - hal-hal yang terjadi selama training
callbacks = [
    # Simpan model terbaik secara otomatis
    tf.keras.callbacks.ModelCheckpoint(
        CONFIG["model_save"],
        monitor   = 'val_accuracy',
        save_best_only = True,
        verbose   = 1
    ),
    
    # Stop kalau tidak ada improvement selama 5 epoch
    tf.keras.callbacks.EarlyStopping(
        monitor   = 'val_accuracy',
        patience  = 5,
        restore_best_weights = True,
        verbose   = 1
    ),
    
    # Kurangi learning rate kalau stuck
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor   = 'val_loss',
        factor    = 0.2,
        patience  = 3,
        verbose   = 1
    )
]

# PHASE 1: Latih hanya bagian atas (lebih cepat)
print("\n--- PHASE 1: Latih classifier layer ---")
history_1 = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs          = CONFIG["epochs"],
    callbacks       = callbacks,
    verbose         = 1
)

# PHASE 2: Fine-tuning - buka sebagian base model
print("\n--- PHASE 2: Fine-tuning ---")
base_model.trainable = True

# Bekukan layer awal, buka layer akhir saja
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile dengan learning rate lebih kecil
model.compile(
    optimizer = tf.keras.optimizers.Adam(CONFIG["learning_rate"] / 10),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

history_2 = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs          = 5,              # Fine-tuning lebih sedikit epoch
    callbacks       = callbacks,
    verbose         = 1
)


# ============================================================
# 2E. EVALUASI MODEL
# ============================================================

print("\n📊 Evaluasi Model...")

# Gabungkan history kedua phase
def combine_histories(h1, h2):
    combined = {}
    for key in h1.history:
        combined[key] = h1.history[key] + h2.history[key]
    return combined

history = combine_histories(history_1, history_2)

# Plot akurasi dan loss
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Akurasi
axes[0].plot(history['accuracy'], label='Training', linewidth=2)
axes[0].plot(history['val_accuracy'], label='Validasi', linewidth=2)
axes[0].set_title('Akurasi Model', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Akurasi')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot Loss
axes[1].plot(history['loss'], label='Training', linewidth=2)
axes[1].plot(history['val_loss'], label='Validasi', linewidth=2)
axes[1].set_title('Loss Model', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_history.png", dpi=100, bbox_inches='tight')
plt.show()

# Evaluasi final
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"\n✅ Hasil Akhir:")
print(f"   Akurasi Validasi : {val_acc*100:.2f}%")
print(f"   Loss Validasi    : {val_loss:.4f}")
print(f"\n✅ Model disimpan ke: {CONFIG['model_save']}")

print("\n" + "=" * 50)
print("✅ Step 2 selesai!")
print("Lanjut ke: 03_test_model.py")
print("=" * 50)
