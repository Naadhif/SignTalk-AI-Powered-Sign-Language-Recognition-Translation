# config.py — Konfigurasi global project SIBI Recognition
# Semua model berbasis scikit-learn (tanpa PyTorch/TensorFlow)

import os

# ─── Path ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RAW_DIR     = os.path.join(DATA_DIR, "raw")
PROC_DIR    = os.path.join(DATA_DIR, "processed")
MODEL_DIR   = os.path.join(BASE_DIR, "models")

os.makedirs(RAW_DIR,   exist_ok=True)
os.makedirs(PROC_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Kamera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
FRAME_FPS      = 30

# ─── Hand Detector ────────────────────────────────────────────────────────────
# 'skincolor' = deteksi warna kulit (tanpa MediaPipe, tanpa YOLO)
# 'yolo'      = YOLOv8 jika terinstall (pip install ultralytics)
HAND_DETECT_METHOD = "skincolor"
MAX_HANDS          = 1

# ─── Ekstraksi Fitur ──────────────────────────────────────────────────────────
N_LANDMARKS      = 21
N_FEATURES       = 63        # 21 × 3 (x, y, z)
ANGLE_PAIRS      = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9,10), (9,10,11),(10,11,12),
    (0,13,14),(13,14,15),(14,15,16),
    (0,17,18),(17,18,19),(18,19,20),
]
N_ANGLE_FEATURES = len(ANGLE_PAIRS)   # 15

# ─── Label SIBI ───────────────────────────────────────────────────────────────
SIBI_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
N_CLASSES   = len(SIBI_LABELS)

# ─── Model ────────────────────────────────────────────────────────────────────
# 'random_forest' — paling cepat, direkomendasikan untuk mulai
# 'cnn'           — SVM + HOG (butuh dataset gambar di data/raw/)
# 'lstm'          — Gradient Boosting + fitur sekuens
ACTIVE_MODEL = "random_forest"

CNN_INPUT_SIZE = (64, 64)
LSTM_SEQ_LEN   = 30
LSTM_HIDDEN    = 128    # tidak dipakai, disimpan untuk kompatibilitas

BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 1e-3

MODEL_PATHS = {
    "random_forest": os.path.join(MODEL_DIR, "rf_sibi.joblib"),
    "cnn":           os.path.join(MODEL_DIR, "svm_hog_sibi.joblib"),
    "lstm":          os.path.join(MODEL_DIR, "gbm_seq_sibi.joblib"),
    "label_encoder": os.path.join(MODEL_DIR, "label_encoder.joblib"),
}

# ─── Prediksi & Confidence ────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70
PREDICTION_BUFFER    = 5
HOLD_FRAMES          = 15

# ─── TTS ──────────────────────────────────────────────────────────────────────
TTS_ENGINE = "pyttsx3"   # 'pyttsx3' (offline) | 'gtts' (online)
TTS_RATE   = 150
TTS_LANG   = "id"

# ─── Tampilan ─────────────────────────────────────────────────────────────────
FONT          = 2
FONT_SCALE    = 1.0
COLOR_GREEN   = (0, 200, 100)
COLOR_ORANGE  = (0, 165, 255)
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0, 0, 0)
COLOR_RED     = (0, 0, 220)
BOX_COLOR     = (60, 120, 200)
