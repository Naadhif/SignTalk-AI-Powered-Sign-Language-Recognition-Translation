#!/usr/bin/env python3
# train_cnn.py — Latih SVM+HOG dari gambar crop tangan
# (Pengganti CNN berbasis PyTorch — kompatibel Python 3.15)
#
# Struktur folder:
#   data/raw/
#     A/  img001.jpg  img002.jpg ...
#     B/  ...

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import RAW_DIR, SIBI_LABELS, CNN_INPUT_SIZE
from models.cnn_model import extract_hog_features, train, evaluate
import joblib


def load_image_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load semua gambar dari subfolder A-Z, ekstrak fitur HOG."""
    if not os.path.isdir(data_dir) or not os.listdir(data_dir):
        return np.array([]), np.array([])

    X, y = [], []
    folders = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))
                      and d.upper() in SIBI_LABELS])

    print(f"[SVM] Ditemukan folder: {folders}")

    for label in tqdm(folders, desc="Ekstrak HOG"):
        folder = os.path.join(data_dir, label)
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, fname)
            img  = cv2.imread(path)
            if img is None:
                continue
            feats = extract_hog_features(img, CNN_INPUT_SIZE)
            X.append(feats)
            y.append(label.upper())

    return np.array(X, dtype=np.float32), np.array(y)


def main():
    print("[SVM] Memuat dataset gambar...")
    X, y = load_image_dataset(RAW_DIR)

    if len(X) == 0:
        print(f"[!] Tidak ada gambar di {RAW_DIR}")
        print("    Buat subfolder A-Z dan isi dengan gambar tangan SIBI.")
        print("    Atau gunakan collect_data.py untuk data fitur (train_rf.py).")
        return

    print(f"[SVM] Total sampel: {len(X)}, kelas: {sorted(set(y))}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=42)

    pipeline = train(X_tr, y_tr, X_v, y_v)
    evaluate(pipeline, X_te, y_te)


if __name__ == "__main__":
    main()
