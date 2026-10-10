#!/usr/bin/env python3
# train_rf.py — Latih model Random Forest dari dataset CSV

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from config import PROC_DIR, SIBI_LABELS

CSV_PATH = os.path.join(PROC_DIR, "sibi_dataset.csv")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[!] Dataset tidak ditemukan: {CSV_PATH}")
        print("    Jalankan collect_data.py terlebih dahulu.")
        return

    print("[RF] Memuat dataset...")
    # Dataset bisa datang dengan header (f0, f1, ..., label)
    # atau tanpa header (saat dikumpulkan dari collect_data.py).
    try:
        df = pd.read_csv(CSV_PATH)
        if "label" in df.columns:
            y_raw = df["label"].values
            X = df.drop(columns=["label"]).values.astype(np.float32)
        else:
            raise KeyError("label")
    except (KeyError, ValueError):
        df = pd.read_csv(CSV_PATH, header=None)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y_raw = df.iloc[:, -1].values

    # Filter hanya label yang ada di SIBI_LABELS
    mask = np.isin(y_raw, SIBI_LABELS)
    X, y = X[mask], y_raw[mask]

    print(f"    Jumlah sampel : {len(X)}")
    print(f"    Kelas tersedia: {sorted(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    from models.random_forest_model import train, evaluate
    print("\n[RF] Melatih model...")
    pipeline = train(X_train, y_train, X_val, y_val)

    print("\n[RF] Evaluasi pada data test:")
    evaluate(pipeline, X_test, y_test)


if __name__ == "__main__":
    main()
