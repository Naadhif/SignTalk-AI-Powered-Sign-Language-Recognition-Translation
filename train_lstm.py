#!/usr/bin/env python3
# train_lstm.py — Latih Gradient Boosting (pengganti LSTM)
# dari fitur sekuens landmark — kompatibel Python 3.15

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

from config import PROC_DIR, MODEL_PATHS, SIBI_LABELS, LSTM_SEQ_LEN
from models.lstm_model import build_sequence_features, train, evaluate

CSV_PATH = os.path.join(PROC_DIR, "sibi_dataset.csv")


def make_sequence_dataset(X: np.ndarray, y: np.ndarray,
                           seq_len: int = LSTM_SEQ_LEN) -> tuple:
    """
    Buat fitur sekuens dari sampel statis dengan augmentasi noise.
    Setiap sampel di-augment menjadi beberapa 'pseudo-sequence'.
    """
    X_seq, y_seq = [], []
    n_aug = 5   # buat 5 versi sekuens per sampel

    for feat, label in zip(X, y):
        for _ in range(n_aug):
            # Buat sekuens dengan drift bertahap (simulasi gerakan)
            seq = np.zeros((seq_len, feat.shape[0]), dtype=np.float32)
            for t in range(seq_len):
                noise = np.random.normal(0, 0.005, feat.shape).astype(np.float32)
                drift = (t / seq_len) * np.random.normal(0, 0.01, feat.shape)
                seq[t] = feat + noise + drift.astype(np.float32)

            X_seq.append(build_sequence_features(seq))
            y_seq.append(label)

    return np.array(X_seq, dtype=np.float32), np.array(y_seq)


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[!] Dataset tidak ditemukan: {CSV_PATH}")
        print("    Jalankan collect_data.py terlebih dahulu.")
        return

    print("[GBM] Memuat dataset...")
    df    = pd.read_csv(CSV_PATH)
    X     = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df["label"].values

    mask  = np.isin(y_raw, SIBI_LABELS)
    X, y_raw = X[mask], y_raw[mask]
    print(f"[GBM] {len(X)} sampel, kelas: {sorted(set(y_raw))}")

    print("[GBM] Membuat fitur sekuens (augmentasi)...")
    X_seq, y_seq = make_sequence_dataset(X, y_raw)
    print(f"[GBM] Dataset augmented: {len(X_seq)} sampel, dim={X_seq.shape[1]}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=42)

    pipeline = train(X_tr, y_tr, X_v, y_v)
    evaluate(pipeline, X_te, y_te)


if __name__ == "__main__":
    main()
